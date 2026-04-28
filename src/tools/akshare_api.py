"""AKShare-based data provider for Chinese A-share stocks.

Provides the same interface as api.py but sources data from AKShare,
which supports Chinese A-shares listed on Shanghai (SH) and Shenzhen (SZ)
stock exchanges.

Ticker formats accepted: "600010", "600010.SH", "600010.SS", "600010.SZ"
"""

import logging
import re

import pandas as pd

from src.data.cache import get_cache
from src.data.models import (
    CompanyNews,
    FinancialMetrics,
    InsiderTrade,
    LineItem,
    Price,
)

logger = logging.getLogger(__name__)
_cache = get_cache()

# ---------------------------------------------------------------------------
# Ticker helpers
# ---------------------------------------------------------------------------

_A_SHARE_RE = re.compile(r"^\d{6}(\.(?:SH|SZ|SS|BJ))?$", re.IGNORECASE)


def is_a_share_ticker(ticker: str) -> bool:
    """Return True if *ticker* looks like a Chinese A-share code."""
    return bool(_A_SHARE_RE.match(ticker))


def _normalize(ticker: str) -> str:
    """Strip exchange suffix and return the bare 6-digit code."""
    return re.sub(r"\.(?:SH|SZ|SS|BJ)$", "", ticker, flags=re.IGNORECASE)


def _em_symbol(code: str) -> str:
    """Return East Money format: ``SH600010`` or ``SZ000001``."""
    return ("SH" if code.startswith(("6", "9")) else "SZ") + code


def _dot_symbol(code: str) -> str:
    """Return dot-separated format: ``600010.SH`` or ``000001.SZ``."""
    sfx = "SH" if code.startswith(("6", "9")) else "SZ"
    return f"{code}.{sfx}"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _safe_float(value, divisor: float = 1.0) -> float | None:
    """Safely convert *value* to float, returning ``None`` on failure."""
    if value is None:
        return None
    try:
        s = str(value).strip()
        if s in ("", "--", "nan", "None", "NaN", "-", "N/A"):
            return None
        return float(s.replace(",", "")) / divisor
    except (ValueError, TypeError):
        return None


def _pick(row: "pd.Series | None", *candidates: str) -> float | None:
    """Return the first non-``None`` float from *row* matching any candidate column."""
    if row is None:
        return None
    for col in candidates:
        if col in row.index:
            val = _safe_float(row[col])
            if val is not None:
                return val
    return None


def _pct_pick(row: "pd.Series | None", *candidates: str) -> float | None:
    """Like ``_pick`` but converts a percentage value to a decimal (÷ 100)."""
    v = _pick(row, *candidates)
    return v / 100.0 if v is not None else None


# ---------------------------------------------------------------------------
# Price data
# ---------------------------------------------------------------------------


def get_prices_akshare(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """Fetch daily OHLCV prices for an A-share stock via AKShare."""
    import akshare as ak

    symbol = _normalize(ticker)
    cache_key = f"{ticker}_{start_date}_{end_date}"

    if cached := _cache.get_prices(cache_key):
        return [Price(**p) for p in cached]

    start = start_date.replace("-", "")
    end = end_date.replace("-", "")

    try:
        df = ak.stock_zh_a_hist(
            symbol=symbol, period="daily", start_date=start, end_date=end, adjust="qfq"
        )
    except Exception as exc:
        logger.warning("AKShare price fetch failed for %s: %s", ticker, exc)
        return []

    if df is None or df.empty:
        return []

    prices: list[Price] = []
    for _, row in df.iterrows():
        try:
            prices.append(
                Price(
                    open=float(row.get("开盘", 0) or 0),
                    close=float(row.get("收盘", 0) or 0),
                    high=float(row.get("最高", 0) or 0),
                    low=float(row.get("最低", 0) or 0),
                    volume=int(float(row.get("成交量", 0) or 0)),
                    time=str(row.get("日期", "")),
                )
            )
        except Exception:
            continue

    if prices:
        _cache.set_prices(cache_key, [p.model_dump() for p in prices])
    return prices


# ---------------------------------------------------------------------------
# Market cap
# ---------------------------------------------------------------------------


def get_market_cap_akshare(ticker: str) -> float | None:
    """Return total market capitalisation (CNY) for an A-share stock."""
    import akshare as ak

    symbol = _normalize(ticker)
    try:
        df = ak.stock_individual_info_em(symbol=symbol)
    except Exception as exc:
        logger.warning("Failed to get stock info for %s: %s", symbol, exc)
        return None

    if df is None or df.empty:
        return None

    # DataFrame has columns ["item", "value"]
    for _, row in df.iterrows():
        key = str(row.iloc[0]).strip()
        if key == "总市值":
            try:
                # The API returns the raw number (e.g. 52_000_000_000)
                return float(str(row.iloc[1]).replace(",", ""))
            except (ValueError, TypeError):
                pass
    return None


def _get_stock_info(symbol: str) -> dict:
    """Return a {item: value} dict from stock_individual_info_em."""
    import akshare as ak

    try:
        df = ak.stock_individual_info_em(symbol=symbol)
        if df is None or df.empty:
            return {}
        return {str(r.iloc[0]).strip(): r.iloc[1] for _, r in df.iterrows()}
    except Exception as exc:
        logger.warning("Failed to get stock info for %s: %s", symbol, exc)
        return {}


# ---------------------------------------------------------------------------
# Valuation ratios (P/E, P/B, P/S) via Baidu
# ---------------------------------------------------------------------------


def _get_valuation_ratios(symbol: str, end_date: str) -> dict:
    """Return {'pe': ..., 'pb': ..., 'ps': ...} as of *end_date*."""
    import akshare as ak

    end_dt = pd.to_datetime(end_date)
    result: dict = {}

    indicator_map = {
        "市盈率(TTM)": "pe",
        "市净率": "pb",
    }
    for indicator, key in indicator_map.items():
        try:
            df = ak.stock_zh_valuation_baidu(
                symbol=symbol, indicator=indicator, period="近五年"
            )
            if df is None or df.empty:
                continue
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["date"] <= end_dt].sort_values("date")
            if not df.empty:
                val = _safe_float(df.iloc[-1]["value"])
                if val and val > 0:
                    result[key] = val
        except Exception as exc:
            logger.debug("Valuation ratio '%s' fetch failed for %s: %s", indicator, symbol, exc)

    return result


# ---------------------------------------------------------------------------
# Financial analysis indicators (Sina Finance)
# ---------------------------------------------------------------------------


def _get_sina_indicators(symbol: str, end_date: str, limit: int) -> list[dict]:
    """Return per-period financial ratio records from Sina Finance."""
    import akshare as ak

    try:
        start_year = str(max(2010, int(end_date[:4]) - limit))
        df = ak.stock_financial_analysis_indicator(symbol=symbol, start_year=start_year)
    except Exception as exc:
        logger.warning("Failed to get Sina financial indicators for %s: %s", symbol, exc)
        return []

    if df is None or df.empty:
        return []

    # Column "日期" contains the report date
    date_col = "日期" if "日期" in df.columns else df.columns[0]
    try:
        df[date_col] = pd.to_datetime(df[date_col])
        end_dt = pd.to_datetime(end_date)
        df = df[df[date_col] <= end_dt].sort_values(date_col, ascending=False)
    except Exception:
        pass

    return df.head(limit).to_dict("records")


# ---------------------------------------------------------------------------
# Financial metrics (FinancialMetrics model)
# ---------------------------------------------------------------------------


def _build_metrics(
    ticker: str,
    report_period: str,
    period: str,
    market_cap: float | None,
    val_ratios: dict,
    fa_record: dict,
    is_first: bool,
) -> FinancialMetrics:
    """Build a :class:`FinancialMetrics` from collected data."""

    def _r(row, *keys):
        """Get value from the fa_record dict (row is just fa_record here)."""
        for k in keys:
            v = _safe_float(row.get(k))
            if v is not None:
                return v
        return None

    def _rp(row, *keys):
        """Get percentage-as-decimal from fa_record."""
        v = _r(row, *keys)
        return v / 100.0 if v is not None else None

    rec = fa_record

    return FinancialMetrics(
        ticker=ticker,
        report_period=report_period,
        period=period,
        currency="CNY",
        market_cap=market_cap if is_first else None,
        enterprise_value=None,
        price_to_earnings_ratio=val_ratios.get("pe") if is_first else None,
        price_to_book_ratio=val_ratios.get("pb") if is_first else None,
        price_to_sales_ratio=val_ratios.get("ps") if is_first else None,
        enterprise_value_to_ebitda_ratio=None,
        enterprise_value_to_revenue_ratio=None,
        free_cash_flow_yield=None,
        peg_ratio=None,
        # Margins
        gross_margin=_rp(rec, "销售毛利率(%)", "毛利率(%)"),
        operating_margin=_rp(rec, "营业利润率(%)", "销售利润率(%)"),
        net_margin=_rp(rec, "销售净利率(%)", "净利润率(%)"),
        # Returns
        return_on_equity=_rp(rec, "净资产收益率(%)", "加权净资产收益率(%)"),
        return_on_assets=_rp(rec, "总资产净利润率(%)", "总资产收益率(%)"),
        return_on_invested_capital=None,
        # Turnover
        asset_turnover=_r(rec, "总资产周转率(次)", "总资产周转率"),
        inventory_turnover=_r(rec, "存货周转率(次)", "存货周转率"),
        receivables_turnover=_r(rec, "应收账款周转率(次)", "应收账款周转率"),
        days_sales_outstanding=_r(rec, "应收账款周转天数(天)"),
        operating_cycle=_r(rec, "营业周期(天)"),
        working_capital_turnover=None,
        # Liquidity
        current_ratio=_r(rec, "流动比率"),
        quick_ratio=_r(rec, "速动比率"),
        cash_ratio=_rp(rec, "现金比率(%)"),
        operating_cash_flow_ratio=None,
        # Solvency
        debt_to_equity=None,
        debt_to_assets=_rp(rec, "资产负债率(%)"),
        interest_coverage=_r(rec, "利息保障倍数(倍)"),
        # Growth
        revenue_growth=_rp(rec, "营业收入增长率(%)", "主营业务收入增长率(%)"),
        earnings_growth=_rp(rec, "净利润增长率(%)", "归属净利润增长率(%)"),
        book_value_growth=_rp(rec, "净资产增长率(%)"),
        earnings_per_share_growth=_rp(rec, "基本每股收益增长率(%)", "每股收益增长率(%)"),
        free_cash_flow_growth=None,
        operating_income_growth=None,
        ebitda_growth=None,
        # Per share
        payout_ratio=None,
        earnings_per_share=_r(rec, "基本每股收益(元)", "摊薄每股收益(元)"),
        book_value_per_share=_r(rec, "每股净资产(元)", "净资产每股(元)"),
        free_cash_flow_per_share=_r(
            rec,
            "每股经营活动产生的现金流量净额(元)",
            "每股经营现金流量(元)",
            "每股经营现金流(元)",
        ),
    )


def get_financial_metrics_akshare(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[FinancialMetrics]:
    """Return a list of :class:`FinancialMetrics` for an A-share stock."""
    symbol = _normalize(ticker)
    cache_key = f"{ticker}_{period}_{end_date}_{limit}"

    if cached := _cache.get_financial_metrics(cache_key):
        return [FinancialMetrics(**m) for m in cached]

    val_ratios = _get_valuation_ratios(symbol, end_date)
    market_cap = get_market_cap_akshare(ticker)
    fa_records = _get_sina_indicators(symbol, end_date, limit)

    if not fa_records:
        m = _build_metrics(
            ticker=ticker,
            report_period=end_date,
            period=period,
            market_cap=market_cap,
            val_ratios=val_ratios,
            fa_record={},
            is_first=True,
        )
        return [m]

    metrics_list: list[FinancialMetrics] = []
    for i, record in enumerate(fa_records):
        # Date column is always "日期" in Sina indicators
        date_col = "日期" if "日期" in record else list(record.keys())[0]
        try:
            report_period = str(record[date_col])[:10]
        except Exception:
            report_period = end_date

        m = _build_metrics(
            ticker=ticker,
            report_period=report_period,
            period=period,
            market_cap=market_cap,
            val_ratios=val_ratios,
            fa_record=record,
            is_first=(i == 0),
        )
        metrics_list.append(m)

    if metrics_list:
        _cache.set_financial_metrics(cache_key, [m.model_dump() for m in metrics_list])
    return metrics_list


# ---------------------------------------------------------------------------
# Financial statements (for line items)
# ---------------------------------------------------------------------------


def _fetch_statement(fn_name: str, em_sym: str) -> pd.DataFrame:
    """Call an AKShare financial statement function, returning an empty DataFrame on error."""
    import akshare as ak

    try:
        fn = getattr(ak, fn_name)
        df = fn(symbol=em_sym)
        return df if df is not None else pd.DataFrame()
    except Exception as exc:
        logger.warning("Failed to fetch %s for %s: %s", fn_name, em_sym, exc)
        return pd.DataFrame()


def _filter_df(df: pd.DataFrame, date_col: str, end_date: str, limit: int) -> pd.DataFrame:
    """Return the *limit* most-recent rows with date ≤ *end_date*."""
    if df.empty:
        return df
    try:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        end_dt = pd.to_datetime(end_date)
        df = df[df[date_col] <= end_dt].sort_values(date_col, ascending=False)
        return df.head(limit).reset_index(drop=True)
    except Exception:
        return df.head(limit)


def _row_or_none(df: pd.DataFrame, i: int) -> "pd.Series | None":
    return df.iloc[i] if not df.empty and i < len(df) else None


def _extract_line_item(
    inc: "pd.Series | None",
    bal: "pd.Series | None",
    cf: "pd.Series | None",
    fa: dict,
    field: str,
) -> float | None:
    """Map a standard field name to a value from one of the statement rows."""

    # Income statement columns (East Money EM API)
    # Balance sheet columns (East Money EM API)
    # Cash flow columns (East Money EM API)

    mapping: dict[str, callable] = {
        "revenue": lambda: _pick(inc,
            "TOTAL_OPERATE_INCOME", "OPERATE_INCOME", "TOTAL_REVENUE",
        ),
        "net_income": lambda: _pick(inc,
            "PARENT_NETPROFIT", "NET_PROFIT", "NETPROFIT",
        ),
        "gross_profit": lambda: _pick(inc, "GROSS_PROFIT") or _compute_gross_profit(inc),
        "operating_income": lambda: _pick(inc,
            "OPERATE_PROFIT", "OPERATING_PROFIT", "OPERATE_INCOME_TOTAL",
        ),
        "ebit": lambda: _pick(inc, "OPERATE_PROFIT", "OPERATING_PROFIT"),
        "interest_expense": lambda: _pick(inc,
            "INTEREST_EXPENSE", "FINANCE_EXPENSE", "FINANCIAL_EXPENSE",
        ),
        "total_assets": lambda: _pick(bal, "TOTAL_ASSETS"),
        "total_liabilities": lambda: _pick(bal, "TOTAL_LIABILITIES", "TOTAL_LIAB"),
        "shareholders_equity": lambda: _pick(bal,
            "PARENT_EQUITY", "TOTAL_EQUITY", "HOLDER_EQUITY",
        ),
        "current_assets": lambda: _pick(bal, "TOTAL_CURRENT_ASSETS", "CURRENT_ASSET"),
        "current_liabilities": lambda: _pick(bal,
            "TOTAL_CURRENT_LIABILITIES", "TOTAL_CURRENT_LIAB", "CURRENT_LIAB",
        ),
        "cash_and_equivalents": lambda: _pick(bal,
            "MONETARYARYFUNDS", "CASH_EQUIVALENTS", "MONEY_FUND", "CURRENCY_FUND",
        ),
        "total_debt": lambda: _compute_total_debt(bal),
        "working_capital": lambda: _compute_working_capital(bal),
        "free_cash_flow": lambda: _compute_fcf(cf),
        "capital_expenditure": lambda: _compute_capex(cf),
        "depreciation_and_amortization": lambda: _compute_da(cf),
        "ebitda": lambda: _compute_ebitda(inc, cf),
        "outstanding_shares": lambda: _safe_float(fa.get("_total_shares")),
        "dividends_and_other_cash_distributions": lambda: _compute_dividends(cf),
        "issuance_or_purchase_of_equity_shares": lambda: _pick(cf,
            "RECEIVE_EQUITY_ISSUE", "ISSUE_STOCK", "ISSUE_SHARE_CASH",
        ),
        # Per-share items fall back to Sina indicators
        "earnings_per_share": lambda: _safe_float(
            fa.get("基本每股收益(元)") or fa.get("摊薄每股收益(元)")
        ),
        "book_value_per_share": lambda: _safe_float(fa.get("每股净资产(元)")),
        "free_cash_flow_per_share": lambda: _safe_float(
            fa.get("每股经营活动产生的现金流量净额(元)")
            or fa.get("每股经营现金流量(元)")
        ),
    }

    extractor = mapping.get(field)
    if extractor:
        try:
            return extractor()
        except Exception:
            return None
    return None


def _compute_gross_profit(inc: "pd.Series | None") -> float | None:
    if inc is None:
        return None
    rev = _pick(inc, "TOTAL_OPERATE_INCOME", "OPERATE_INCOME")
    cost = _pick(inc, "OPERATE_COST", "COST_SELL")
    if rev is not None and cost is not None:
        return rev - cost
    return None


def _compute_total_debt(bal: "pd.Series | None") -> float | None:
    if bal is None:
        return None
    parts = [
        _pick(bal, "SHORT_LOAN", "SHORT_BORROW") or 0,
        _pick(bal, "LONG_LOAN", "LONG_BORROW") or 0,
        _pick(bal, "BOND_PAYABLE") or 0,
    ]
    total = sum(parts)
    return total if total > 0 else None


def _compute_working_capital(bal: "pd.Series | None") -> float | None:
    if bal is None:
        return None
    ca = _pick(bal, "TOTAL_CURRENT_ASSETS", "CURRENT_ASSET")
    cl = _pick(bal, "TOTAL_CURRENT_LIABILITIES", "TOTAL_CURRENT_LIAB", "CURRENT_LIAB")
    if ca is not None and cl is not None:
        return ca - cl
    return None


def _compute_fcf(cf: "pd.Series | None") -> float | None:
    if cf is None:
        return None
    op = _pick(cf, "NETCASH_OPERATE", "OPERATE_NETCASH")
    capex = _compute_capex(cf)
    if op is not None and capex is not None:
        return op - abs(capex)
    return op


def _compute_capex(cf: "pd.Series | None") -> float | None:
    if cf is None:
        return None
    v = _pick(cf,
        "FIX_INTAN_OTHER_ASSET_ACQUI",
        "CONSTRUCT_LONG_ASSET",
        "BUY_FIXED_ASSETS",
        "BUY_FILOGICAL_ASSET",
    )
    return abs(v) if v is not None else None


def _compute_da(cf: "pd.Series | None") -> float | None:
    if cf is None:
        return None
    d = _pick(cf, "DEPRECIATION_ASSET", "DEPRECIATION_AMORT_ASSET") or 0
    ia = _pick(cf, "AMORTIZE_INTANGIBLE", "AMORT_INTANGIBLE_ASSET") or 0
    la = _pick(cf, "AMORTIZE_LONGTERM_EXPENSE", "AMORT_LONGTERM_EXPENSE") or 0
    total = d + ia + la
    return total if total > 0 else None


def _compute_ebitda(inc: "pd.Series | None", cf: "pd.Series | None") -> float | None:
    op = _pick(inc, "OPERATE_PROFIT", "OPERATING_PROFIT") if inc is not None else None
    da = _compute_da(cf)
    if op is not None and da is not None:
        return op + da
    return op


def _compute_dividends(cf: "pd.Series | None") -> float | None:
    if cf is None:
        return None
    v = _pick(cf, "ASSIGN_DIVIDEND_PORFIT", "DIVIDEND_PAYMENT", "PAY_DIVIDEND_PORFIT")
    return abs(v) if v is not None else None


def search_line_items_akshare(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[LineItem]:
    """Return financial statement line items for an A-share stock."""
    symbol = _normalize(ticker)
    em_sym = _em_symbol(symbol)

    inc_df = _fetch_statement("stock_profit_sheet_by_report_em", em_sym)
    bal_df = _fetch_statement("stock_balance_sheet_by_report_em", em_sym)
    cf_df = _fetch_statement("stock_cash_flow_sheet_by_report_em", em_sym)

    def _find_date_col(df: pd.DataFrame) -> str | None:
        for col in df.columns:
            if "REPORT_DATE" in col.upper() or "DATE" in col.upper():
                return col
        return None

    inc_date = _find_date_col(inc_df)
    bal_date = _find_date_col(bal_df)
    cf_date = _find_date_col(cf_df)

    if inc_date:
        inc_df = _filter_df(inc_df, inc_date, end_date, limit)
    if bal_date:
        bal_df = _filter_df(bal_df, bal_date, end_date, limit)
    if cf_date:
        cf_df = _filter_df(cf_df, cf_date, end_date, limit)

    fa_records = _get_sina_indicators(symbol, end_date, limit)

    # Get outstanding shares from stock info (used for per-share calculations)
    info = _get_stock_info(symbol)
    total_shares_raw = info.get("总股本")
    total_shares = _safe_float(total_shares_raw)

    n = max(
        len(inc_df) if not inc_df.empty else 0,
        len(bal_df) if not bal_df.empty else 0,
        len(cf_df) if not cf_df.empty else 0,
        1,
    )
    n = min(n, limit)

    result: list[LineItem] = []
    for i in range(n):
        inc = _row_or_none(inc_df, i)
        bal = _row_or_none(bal_df, i)
        cf = _row_or_none(cf_df, i)
        fa = fa_records[i] if i < len(fa_records) else {}
        # Inject total shares so _extract_line_item can use it
        fa["_total_shares"] = total_shares

        # Determine report period
        report_period = end_date
        for row, dcol in [(inc, inc_date), (bal, bal_date), (cf, cf_date)]:
            if row is not None and dcol and dcol in row.index:
                try:
                    report_period = str(pd.to_datetime(row[dcol]).date())
                    break
                except Exception:
                    pass

        item_data: dict = {
            "ticker": ticker,
            "report_period": report_period,
            "period": period,
            "currency": "CNY",
        }

        for field in line_items:
            val = _extract_line_item(inc, bal, cf, fa, field)
            if val is not None:
                item_data[field] = val

        try:
            result.append(LineItem(**item_data))
        except Exception as exc:
            logger.warning("Failed to create LineItem for %s period %s: %s", ticker, report_period, exc)

    return result


# ---------------------------------------------------------------------------
# Insider trades (best-effort; limited disclosure in A-share market)
# ---------------------------------------------------------------------------


def get_insider_trades_akshare(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[InsiderTrade]:
    """Return insider trades for an A-share stock.

    A-share insider trading disclosures differ significantly from US SEC
    requirements.  This implementation returns an empty list rather than
    producing inaccurate results from an ill-fitting data source.
    """
    return []


# ---------------------------------------------------------------------------
# Company news
# ---------------------------------------------------------------------------


def get_company_news_akshare(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 100,
) -> list[CompanyNews]:
    """Return recent news articles for an A-share stock via AKShare."""
    import akshare as ak

    symbol = _normalize(ticker)
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"

    if cached := _cache.get_company_news(cache_key):
        return [CompanyNews(**n) for n in cached]

    try:
        df = ak.stock_news_em(symbol=symbol)
    except Exception as exc:
        logger.warning("Failed to get news for %s: %s", symbol, exc)
        return []

    if df is None or df.empty:
        return []

    end_dt = pd.to_datetime(end_date)
    start_dt = pd.to_datetime(start_date) if start_date else None

    news_list: list[CompanyNews] = []
    for _, row in df.iterrows():
        try:
            title = str(row.get("新闻标题", row.get("title", "")))
            source = str(row.get("文章来源", row.get("mediaName", "Unknown")))
            url = str(row.get("新闻链接", row.get("url", "")))
            date_raw = row.get("发布时间", row.get("datetime", row.get("date", "")))
            date_str = str(date_raw)[:10] if date_raw else end_date

            try:
                pub_dt = pd.to_datetime(date_str)
                if pub_dt > end_dt:
                    continue
                if start_dt and pub_dt < start_dt:
                    continue
            except Exception:
                pass

            news_list.append(
                CompanyNews(
                    ticker=ticker,
                    title=title,
                    source=source,
                    date=date_str,
                    url=url,
                    sentiment=None,
                )
            )
        except Exception:
            continue

    news_list = news_list[:limit]
    if news_list:
        _cache.set_company_news(cache_key, [n.model_dump() for n in news_list])
    return news_list
