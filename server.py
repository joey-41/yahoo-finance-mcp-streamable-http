import asyncio
import atexit
import json
import os
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Any, Callable

import pandas as pd
import uvicorn
import yfinance as yf
from mcp.server.fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


class FinancialType(str, Enum):
    income_stmt = "income_stmt"
    quarterly_income_stmt = "quarterly_income_stmt"
    balance_sheet = "balance_sheet"
    quarterly_balance_sheet = "quarterly_balance_sheet"
    cashflow = "cashflow"
    quarterly_cashflow = "quarterly_cashflow"


class HolderType(str, Enum):
    major_holders = "major_holders"
    institutional_holders = "institutional_holders"
    mutualfund_holders = "mutualfund_holders"
    insider_transactions = "insider_transactions"
    insider_purchases = "insider_purchases"
    insider_roster_holders = "insider_roster_holders"


class RecommendationType(str, Enum):
    recommendations = "recommendations"
    upgrades_downgrades = "upgrades_downgrades"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(minimum, parsed)


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return max(minimum, parsed)


def _normalize_mcp_path(path: str) -> tuple[str, str]:
    normalized = path.strip()
    if not normalized:
        normalized = "/mcp"
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"

    if normalized == "/":
        return "/", "/"
    path_without_slash = normalized.rstrip("/")
    path_with_slash = f"{path_without_slash}/"
    return path_without_slash, path_with_slash


MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("MCP_PORT", "8000"))
MCP_PATH = os.getenv("MCP_PATH", "/mcp")
MCP_LOG_LEVEL = os.getenv("MCP_LOG_LEVEL", "INFO").upper()
MCP_STATELESS_HTTP = _env_bool("MCP_STATELESS_HTTP", False)
MCP_API_KEY = os.getenv("MCP_API_KEY", "")
MCP_API_KEY_HEADER = os.getenv("MCP_API_KEY_HEADER", "X-API-Key")
MCP_MAX_CONCURRENT_YF_REQUESTS = _env_int("MCP_MAX_CONCURRENT_YF_REQUESTS", 32, minimum=1)
MCP_YF_THREAD_WORKERS = _env_int("MCP_YF_THREAD_WORKERS", 16, minimum=1)
MCP_YF_ACQUIRE_TIMEOUT_SECONDS = _env_float("MCP_YF_ACQUIRE_TIMEOUT_SECONDS", 2.0, minimum=0.1)
MCP_UVICORN_LIMIT_CONCURRENCY = _env_int("MCP_UVICORN_LIMIT_CONCURRENCY", 0, minimum=0)
MCP_PATH_NO_SLASH, MCP_PATH_WITH_SLASH = _normalize_mcp_path(MCP_PATH)


yfinance_server = FastMCP(
    "yfinance-streamable-http",
    instructions="""
# Yahoo Finance MCP Server (Streamable HTTP)

This server provides Yahoo Finance data over MCP Streamable HTTP transport.

Available tools:
- get_historical_stock_prices
- get_stock_info
- get_yahoo_finance_news
- get_stock_actions
- get_financial_statement
- get_holder_info
- get_option_expiration_dates
- get_option_chain
- get_recommendations
""",
    host=MCP_HOST,
    port=MCP_PORT,
    streamable_http_path=MCP_PATH_NO_SLASH,
    stateless_http=MCP_STATELESS_HTTP,
    log_level=MCP_LOG_LEVEL,
)


def _get_company(ticker: str) -> tuple[yf.Ticker | None, str | None]:
    normalized_ticker = ticker.strip().upper()
    if not normalized_ticker:
        return None, "Error: ticker is required."

    try:
        company = yf.Ticker(normalized_ticker)
    except Exception as exc:
        return None, f"Error: creating ticker client for {normalized_ticker}: {exc}"
    return company, None


def _safe_json(data: Any) -> str:
    return json.dumps(data, default=str)


def _extract_auth_bearer_token(authorization_header: str | None) -> str:
    if not authorization_header:
        return ""
    token_prefix = "Bearer "
    if authorization_header.startswith(token_prefix):
        return authorization_header[len(token_prefix) :].strip()
    return ""


def _is_empty_dataframe(data: Any) -> bool:
    return not isinstance(data, pd.DataFrame) or data.empty


YFINANCE_THREAD_POOL = ThreadPoolExecutor(
    max_workers=MCP_YF_THREAD_WORKERS,
    thread_name_prefix="yfinance",
)
YFINANCE_SEMAPHORE = asyncio.Semaphore(MCP_MAX_CONCURRENT_YF_REQUESTS)


@atexit.register
def _shutdown_thread_pool() -> None:
    YFINANCE_THREAD_POOL.shutdown(wait=False, cancel_futures=True)


async def _run_yfinance_task(task: Callable[[], str]) -> str:
    acquired = False
    try:
        await asyncio.wait_for(
            YFINANCE_SEMAPHORE.acquire(),
            timeout=MCP_YF_ACQUIRE_TIMEOUT_SECONDS,
        )
        acquired = True
    except asyncio.TimeoutError:
        return (
            "Error: server is busy with too many concurrent Yahoo Finance requests. "
            "Please retry in a few seconds."
        )

    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(YFINANCE_THREAD_POOL, task)
    except Exception as exc:
        return f"Error: yfinance request execution failed: {exc}"
    finally:
        if acquired:
            YFINANCE_SEMAPHORE.release()


class ApiKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        if request.url.path == MCP_PATH_NO_SLASH and MCP_PATH_NO_SLASH != MCP_PATH_WITH_SLASH:
            request.scope["path"] = MCP_PATH_WITH_SLASH
            request.scope["raw_path"] = MCP_PATH_WITH_SLASH.encode()

        if not MCP_API_KEY:
            return await call_next(request)

        if request.url.path == "/healthz":
            return await call_next(request)

        provided_key = request.headers.get(MCP_API_KEY_HEADER, "")
        if not provided_key:
            provided_key = _extract_auth_bearer_token(request.headers.get("Authorization"))

        if provided_key != MCP_API_KEY:
            return JSONResponse({"error": "Unauthorized"}, status_code=401)

        return await call_next(request)


@yfinance_server.custom_route("/healthz", methods=["GET"], name="healthz")
async def healthz(_: Request) -> Response:
    return JSONResponse(
        {
            "status": "ok",
            "service": "yfinance-streamable-http",
            "transport": "streamable-http",
            "mcp_path": MCP_PATH_NO_SLASH,
            "mcp_path_with_slash": MCP_PATH_WITH_SLASH,
            "stateless": MCP_STATELESS_HTTP,
            "max_concurrent_yfinance_requests": MCP_MAX_CONCURRENT_YF_REQUESTS,
            "yfinance_thread_workers": MCP_YF_THREAD_WORKERS,
        }
    )


@yfinance_server.tool(
    name="get_historical_stock_prices",
    description="""Get historical stock prices for a given ticker symbol from yahoo finance.
Include the following information: Date, Open, High, Low, Close, Volume, Adj Close.
Args:
    ticker: str
    period: str (default "1mo")
    interval: str (default "1d")
""",
)
async def get_historical_stock_prices(
    ticker: str, period: str = "1mo", interval: str = "1d"
) -> str:
    return await _run_yfinance_task(
        lambda: _get_historical_stock_prices_sync(ticker, period, interval)
    )


def _get_historical_stock_prices_sync(ticker: str, period: str, interval: str) -> str:
    company, error = _get_company(ticker)
    if error:
        return error
    try:
        hist_data = company.history(period=period, interval=interval)
        if _is_empty_dataframe(hist_data):
            return f"No historical price data found for ticker {ticker}."
        hist_data = hist_data.reset_index(names="Date")
        return hist_data.to_json(orient="records", date_format="iso")
    except Exception as exc:
        return f"Error: getting historical prices for {ticker}: {exc}"


@yfinance_server.tool(
    name="get_stock_info",
    description="""Get stock information for a given ticker symbol from yahoo finance.
Args:
    ticker: str
""",
)
async def get_stock_info(ticker: str) -> str:
    return await _run_yfinance_task(lambda: _get_stock_info_sync(ticker))


def _get_stock_info_sync(ticker: str) -> str:
    company, error = _get_company(ticker)
    if error:
        return error
    try:
        info = company.info
        if not info:
            return f"No stock info found for ticker {ticker}."
        return _safe_json(info)
    except Exception as exc:
        return f"Error: getting stock info for {ticker}: {exc}"


@yfinance_server.tool(
    name="get_yahoo_finance_news",
    description="""Get news for a given ticker symbol from yahoo finance.
Args:
    ticker: str
""",
)
async def get_yahoo_finance_news(ticker: str) -> str:
    return await _run_yfinance_task(lambda: _get_yahoo_finance_news_sync(ticker))


def _get_yahoo_finance_news_sync(ticker: str) -> str:
    company, error = _get_company(ticker)
    if error:
        return error
    try:
        raw_news = company.news
    except Exception as exc:
        return f"Error: getting news for {ticker}: {exc}"

    news_list = []
    for news_item in raw_news:
        content = news_item.get("content", {})
        if content.get("contentType", "") == "STORY":
            title = content.get("title", "")
            summary = content.get("summary", "")
            description = content.get("description", "")
            url = content.get("canonicalUrl", {}).get("url", "")
            news_list.append(
                f"Title: {title}\nSummary: {summary}\nDescription: {description}\nURL: {url}"
            )

    if not news_list:
        return f"No news found for company that searched with {ticker} ticker."
    return "\n\n".join(news_list)


@yfinance_server.tool(
    name="get_stock_actions",
    description="""Get stock dividends and stock splits for a given ticker symbol.
Args:
    ticker: str
""",
)
async def get_stock_actions(ticker: str) -> str:
    return await _run_yfinance_task(lambda: _get_stock_actions_sync(ticker))


def _get_stock_actions_sync(ticker: str) -> str:
    company, error = _get_company(ticker)
    if error:
        return error
    try:
        actions_df = company.actions
        if _is_empty_dataframe(actions_df):
            return f"No stock actions found for ticker {ticker}."
        actions_df = actions_df.reset_index(names="Date")
        return actions_df.to_json(orient="records", date_format="iso")
    except Exception as exc:
        return f"Error: getting stock actions for {ticker}: {exc}"


@yfinance_server.tool(
    name="get_financial_statement",
    description="""Get financial statement for a given ticker symbol from yahoo finance.
Args:
    ticker: str
    financial_type: str
""",
)
async def get_financial_statement(ticker: str, financial_type: str) -> str:
    return await _run_yfinance_task(
        lambda: _get_financial_statement_sync(ticker, financial_type)
    )


def _get_financial_statement_sync(ticker: str, financial_type: str) -> str:
    company, error = _get_company(ticker)
    if error:
        return error

    financial_attr_map = {
        FinancialType.income_stmt.value: "income_stmt",
        FinancialType.quarterly_income_stmt.value: "quarterly_income_stmt",
        FinancialType.balance_sheet.value: "balance_sheet",
        FinancialType.quarterly_balance_sheet.value: "quarterly_balance_sheet",
        FinancialType.cashflow.value: "cashflow",
        FinancialType.quarterly_cashflow.value: "quarterly_cashflow",
    }
    attr_name = financial_attr_map.get(financial_type)
    if attr_name is None:
        valid_types = ", ".join(ft.value for ft in FinancialType)
        return (
            f"Error: invalid financial type {financial_type}. "
            f"Please use one of: {valid_types}."
        )
    try:
        financial_statement = getattr(company, attr_name)
    except Exception as exc:
        return f"Error: getting financial statement for {ticker}: {exc}"

    if _is_empty_dataframe(financial_statement):
        return f"No financial statement data found for ticker {ticker} with type {financial_type}."

    result = []
    for column in financial_statement.columns:
        if isinstance(column, pd.Timestamp):
            date_str = column.strftime("%Y-%m-%d")
        else:
            date_str = str(column)
        date_obj = {"date": date_str}
        for index, value in financial_statement[column].items():
            date_obj[index] = None if pd.isna(value) else value
        result.append(date_obj)

    return _safe_json(result)


@yfinance_server.tool(
    name="get_holder_info",
    description="""Get holder information for a given ticker symbol from yahoo finance.
Args:
    ticker: str
    holder_type: str
""",
)
async def get_holder_info(ticker: str, holder_type: str) -> str:
    return await _run_yfinance_task(lambda: _get_holder_info_sync(ticker, holder_type))


def _get_holder_info_sync(ticker: str, holder_type: str) -> str:
    company, error = _get_company(ticker)
    if error:
        return error

    try:
        if holder_type == HolderType.major_holders.value:
            data = company.major_holders
            if _is_empty_dataframe(data):
                return f"No holder data found for ticker {ticker}."
            return data.reset_index(names="metric").to_json(orient="records")
        if holder_type == HolderType.institutional_holders.value:
            data = company.institutional_holders
            if _is_empty_dataframe(data):
                return f"No holder data found for ticker {ticker}."
            return data.to_json(orient="records")
        if holder_type == HolderType.mutualfund_holders.value:
            data = company.mutualfund_holders
            if _is_empty_dataframe(data):
                return f"No holder data found for ticker {ticker}."
            return data.to_json(orient="records", date_format="iso")
        if holder_type == HolderType.insider_transactions.value:
            data = company.insider_transactions
            if _is_empty_dataframe(data):
                return f"No holder data found for ticker {ticker}."
            return data.to_json(orient="records", date_format="iso")
        if holder_type == HolderType.insider_purchases.value:
            data = company.insider_purchases
            if _is_empty_dataframe(data):
                return f"No holder data found for ticker {ticker}."
            return data.to_json(orient="records", date_format="iso")
        if holder_type == HolderType.insider_roster_holders.value:
            data = company.insider_roster_holders
            if _is_empty_dataframe(data):
                return f"No holder data found for ticker {ticker}."
            return data.to_json(orient="records", date_format="iso")

        valid_types = ", ".join(ht.value for ht in HolderType)
        return (
            f"Error: invalid holder type {holder_type}. "
            f"Please use one of: {valid_types}."
        )
    except Exception as exc:
        return f"Error: getting holder info for {ticker}: {exc}"


@yfinance_server.tool(
    name="get_option_expiration_dates",
    description="""Fetch available options expiration dates for a given ticker symbol.
Args:
    ticker: str
""",
)
async def get_option_expiration_dates(ticker: str) -> str:
    return await _run_yfinance_task(lambda: _get_option_expiration_dates_sync(ticker))


def _get_option_expiration_dates_sync(ticker: str) -> str:
    company, error = _get_company(ticker)
    if error:
        return error
    try:
        options = company.options
        if not options:
            return f"No option expiration dates found for ticker {ticker}."
        return _safe_json(options)
    except Exception as exc:
        return f"Error: getting option expiration dates for {ticker}: {exc}"


@yfinance_server.tool(
    name="get_option_chain",
    description="""Fetch option chain for a given ticker symbol, expiration date, and option type.
Args:
    ticker: str
    expiration_date: str
    option_type: str
""",
)
async def get_option_chain(ticker: str, expiration_date: str, option_type: str) -> str:
    return await _run_yfinance_task(
        lambda: _get_option_chain_sync(ticker, expiration_date, option_type)
    )


def _get_option_chain_sync(ticker: str, expiration_date: str, option_type: str) -> str:
    company, error = _get_company(ticker)
    if error:
        return error

    try:
        option_dates = company.options
    except Exception as exc:
        return f"Error: getting option expiration dates for {ticker}: {exc}"

    if expiration_date not in option_dates:
        return (
            f"Error: No options available for the date {expiration_date}. "
            "You can use `get_option_expiration_dates` to get available dates."
        )
    if option_type not in {"calls", "puts"}:
        return "Error: Invalid option type. Please use 'calls' or 'puts'."

    try:
        option_chain = company.option_chain(expiration_date)
        option_df = option_chain.calls if option_type == "calls" else option_chain.puts
        if _is_empty_dataframe(option_df):
            return (
                f"No option chain data found for ticker {ticker} "
                f"on {expiration_date} ({option_type})."
            )
        return option_df.to_json(orient="records", date_format="iso")
    except Exception as exc:
        return f"Error: getting option chain for {ticker}: {exc}"


@yfinance_server.tool(
    name="get_recommendations",
    description="""Get recommendations or upgrades/downgrades for a ticker symbol.
Args:
    ticker: str
    recommendation_type: str
    months_back: int = 12
""",
)
async def get_recommendations(ticker: str, recommendation_type: str, months_back: int = 12) -> str:
    return await _run_yfinance_task(
        lambda: _get_recommendations_sync(ticker, recommendation_type, months_back)
    )


def _get_recommendations_sync(ticker: str, recommendation_type: str, months_back: int = 12) -> str:
    company, error = _get_company(ticker)
    if error:
        return error
    try:
        if recommendation_type == RecommendationType.recommendations.value:
            recommendations = company.recommendations
            if _is_empty_dataframe(recommendations):
                return f"No recommendations found for ticker {ticker}."
            return recommendations.to_json(orient="records")
        if recommendation_type == RecommendationType.upgrades_downgrades.value:
            upgrades_downgrades = company.upgrades_downgrades
            if _is_empty_dataframe(upgrades_downgrades):
                return f"No upgrades/downgrades data found for ticker {ticker}."
            upgrades_downgrades = upgrades_downgrades.reset_index()
            if "GradeDate" not in upgrades_downgrades.columns:
                return f"Error: unexpected upgrades/downgrades schema for ticker {ticker}."
            cutoff_date = pd.Timestamp.now() - pd.DateOffset(months=months_back)
            upgrades_downgrades = upgrades_downgrades[
                upgrades_downgrades["GradeDate"] >= cutoff_date
            ]
            if upgrades_downgrades.empty:
                return (
                    f"No upgrades/downgrades found for ticker {ticker} "
                    f"in the last {months_back} months."
                )
            upgrades_downgrades = upgrades_downgrades.sort_values("GradeDate", ascending=False)
            latest_by_firm = (
                upgrades_downgrades.drop_duplicates(subset=["Firm"])
                if "Firm" in upgrades_downgrades.columns
                else upgrades_downgrades
            )
            return latest_by_firm.to_json(orient="records", date_format="iso")
        valid_types = ", ".join(rt.value for rt in RecommendationType)
        return (
            f"Error: invalid recommendation type {recommendation_type}. "
            f"Please use one of: {valid_types}."
        )
    except Exception as exc:
        return f"Error: getting recommendations for {ticker}: {exc}"


if __name__ == "__main__":
    app = yfinance_server.streamable_http_app()
    app.add_middleware(ApiKeyMiddleware)

    uvicorn_limit_concurrency = (
        MCP_UVICORN_LIMIT_CONCURRENCY if MCP_UVICORN_LIMIT_CONCURRENCY > 0 else None
    )

    print(
        "Starting Yahoo Finance MCP Streamable HTTP server "
        f"on {MCP_HOST}:{MCP_PORT}{MCP_PATH_WITH_SLASH}; "
        f"api_key_enabled={'yes' if MCP_API_KEY else 'no'}; "
        f"yf_max_concurrency={MCP_MAX_CONCURRENT_YF_REQUESTS}; "
        f"yf_thread_workers={MCP_YF_THREAD_WORKERS}; "
        f"uvicorn_limit_concurrency={uvicorn_limit_concurrency or 'unlimited'}"
    )
    config = uvicorn.Config(
        app,
        host=MCP_HOST,
        port=MCP_PORT,
        log_level=MCP_LOG_LEVEL.lower(),
        limit_concurrency=uvicorn_limit_concurrency,
    )
    uvicorn.Server(config).run()
