import json
import os
from enum import Enum
from typing import Any

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


MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("MCP_PORT", "8000"))
MCP_PATH = os.getenv("MCP_PATH", "/mcp")
MCP_LOG_LEVEL = os.getenv("MCP_LOG_LEVEL", "INFO").upper()
MCP_STATELESS_HTTP = _env_bool("MCP_STATELESS_HTTP", False)
MCP_API_KEY = os.getenv("MCP_API_KEY", "")
MCP_API_KEY_HEADER = os.getenv("MCP_API_KEY_HEADER", "X-API-Key")


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
    streamable_http_path=MCP_PATH,
    stateless_http=MCP_STATELESS_HTTP,
    log_level=MCP_LOG_LEVEL,
)


def _get_company(ticker: str) -> tuple[yf.Ticker | None, str | None]:
    company = yf.Ticker(ticker)
    try:
        if company.isin is None:
            return None, f"Company ticker {ticker} not found."
    except Exception as exc:
        return None, f"Error: validating ticker {ticker}: {exc}"
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


class ApiKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
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
            "mcp_path": MCP_PATH,
            "stateless": MCP_STATELESS_HTTP,
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
    company, error = _get_company(ticker)
    if error:
        return error
    hist_data = company.history(period=period, interval=interval)
    hist_data = hist_data.reset_index(names="Date")
    return hist_data.to_json(orient="records", date_format="iso")


@yfinance_server.tool(
    name="get_stock_info",
    description="""Get stock information for a given ticker symbol from yahoo finance.
Args:
    ticker: str
""",
)
async def get_stock_info(ticker: str) -> str:
    company, error = _get_company(ticker)
    if error:
        return error
    return _safe_json(company.info)


@yfinance_server.tool(
    name="get_yahoo_finance_news",
    description="""Get news for a given ticker symbol from yahoo finance.
Args:
    ticker: str
""",
)
async def get_yahoo_finance_news(ticker: str) -> str:
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
    company, error = _get_company(ticker)
    if error:
        return error
    actions_df = company.actions.reset_index(names="Date")
    return actions_df.to_json(orient="records", date_format="iso")


@yfinance_server.tool(
    name="get_financial_statement",
    description="""Get financial statement for a given ticker symbol from yahoo finance.
Args:
    ticker: str
    financial_type: str
""",
)
async def get_financial_statement(ticker: str, financial_type: str) -> str:
    company, error = _get_company(ticker)
    if error:
        return error

    if financial_type == FinancialType.income_stmt:
        financial_statement = company.income_stmt
    elif financial_type == FinancialType.quarterly_income_stmt:
        financial_statement = company.quarterly_income_stmt
    elif financial_type == FinancialType.balance_sheet:
        financial_statement = company.balance_sheet
    elif financial_type == FinancialType.quarterly_balance_sheet:
        financial_statement = company.quarterly_balance_sheet
    elif financial_type == FinancialType.cashflow:
        financial_statement = company.cashflow
    elif financial_type == FinancialType.quarterly_cashflow:
        financial_statement = company.quarterly_cashflow
    else:
        return (
            f"Error: invalid financial type {financial_type}. "
            f"Please use one of: {FinancialType.income_stmt}, "
            f"{FinancialType.quarterly_income_stmt}, {FinancialType.balance_sheet}, "
            f"{FinancialType.quarterly_balance_sheet}, {FinancialType.cashflow}, "
            f"{FinancialType.quarterly_cashflow}."
        )

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
    company, error = _get_company(ticker)
    if error:
        return error

    if holder_type == HolderType.major_holders:
        return company.major_holders.reset_index(names="metric").to_json(orient="records")
    if holder_type == HolderType.institutional_holders:
        return company.institutional_holders.to_json(orient="records")
    if holder_type == HolderType.mutualfund_holders:
        return company.mutualfund_holders.to_json(orient="records", date_format="iso")
    if holder_type == HolderType.insider_transactions:
        return company.insider_transactions.to_json(orient="records", date_format="iso")
    if holder_type == HolderType.insider_purchases:
        return company.insider_purchases.to_json(orient="records", date_format="iso")
    if holder_type == HolderType.insider_roster_holders:
        return company.insider_roster_holders.to_json(orient="records", date_format="iso")
    return (
        f"Error: invalid holder type {holder_type}. "
        f"Please use one of: {HolderType.major_holders}, {HolderType.institutional_holders}, "
        f"{HolderType.mutualfund_holders}, {HolderType.insider_transactions}, "
        f"{HolderType.insider_purchases}, {HolderType.insider_roster_holders}."
    )


@yfinance_server.tool(
    name="get_option_expiration_dates",
    description="""Fetch available options expiration dates for a given ticker symbol.
Args:
    ticker: str
""",
)
async def get_option_expiration_dates(ticker: str) -> str:
    company, error = _get_company(ticker)
    if error:
        return error
    return _safe_json(company.options)


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
    company, error = _get_company(ticker)
    if error:
        return error

    if expiration_date not in company.options:
        return (
            f"Error: No options available for the date {expiration_date}. "
            "You can use `get_option_expiration_dates` to get available dates."
        )
    if option_type not in {"calls", "puts"}:
        return "Error: Invalid option type. Please use 'calls' or 'puts'."

    option_chain = company.option_chain(expiration_date)
    if option_type == "calls":
        return option_chain.calls.to_json(orient="records", date_format="iso")
    return option_chain.puts.to_json(orient="records", date_format="iso")


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
    company, error = _get_company(ticker)
    if error:
        return error
    try:
        if recommendation_type == RecommendationType.recommendations:
            return company.recommendations.to_json(orient="records")
        if recommendation_type == RecommendationType.upgrades_downgrades:
            upgrades_downgrades = company.upgrades_downgrades.reset_index()
            cutoff_date = pd.Timestamp.now() - pd.DateOffset(months=months_back)
            upgrades_downgrades = upgrades_downgrades[
                upgrades_downgrades["GradeDate"] >= cutoff_date
            ]
            upgrades_downgrades = upgrades_downgrades.sort_values("GradeDate", ascending=False)
            latest_by_firm = upgrades_downgrades.drop_duplicates(subset=["Firm"])
            return latest_by_firm.to_json(orient="records", date_format="iso")
        return (
            f"Error: invalid recommendation type {recommendation_type}. "
            f"Please use one of: {RecommendationType.recommendations}, "
            f"{RecommendationType.upgrades_downgrades}."
        )
    except Exception as exc:
        return f"Error: getting recommendations for {ticker}: {exc}"


if __name__ == "__main__":
    app = yfinance_server.streamable_http_app()
    app.add_middleware(ApiKeyMiddleware)

    print(
        "Starting Yahoo Finance MCP Streamable HTTP server "
        f"on {MCP_HOST}:{MCP_PORT}{MCP_PATH}; "
        f"api_key_enabled={'yes' if MCP_API_KEY else 'no'}"
    )
    config = uvicorn.Config(
        app,
        host=MCP_HOST,
        port=MCP_PORT,
        log_level=MCP_LOG_LEVEL.lower(),
    )
    uvicorn.Server(config).run()
