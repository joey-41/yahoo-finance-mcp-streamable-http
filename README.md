# Yahoo Finance MCP (Streamable HTTP)

[中文说明](./README.zh.md)

This project is a Streamable HTTP version of [Alex2Yang97/yahoo-finance-mcp](https://github.com/Alex2Yang97/yahoo-finance-mcp), designed for server deployment and remote MCP calls via URL.

## Features

- MCP Streamable HTTP transport (default path: `/mcp/`, compatible with `/mcp`)
- Yahoo Finance toolset (prices, fundamentals, options, news, recommendations, holders)
- Health check endpoint: `/healthz`
- Environment-based configuration for host, port, path, log level, and stateless mode
- API Key authentication via request header (optional)
- Concurrency protection for blocking yfinance calls (thread pool + semaphore + overload fast-fail)

## Available Tools

- `get_historical_stock_prices`
- `get_stock_info`
- `get_yahoo_finance_news`
- `get_stock_actions`
- `get_financial_statement`
- `get_holder_info`
- `get_option_expiration_dates`
- `get_option_chain`
- `get_recommendations`

## Requirements

- Python `>=3.11`

## Quick Start (Local)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python server.py
```

Default runtime:

- Host: `0.0.0.0`
- Port: `8000`
- MCP Path: `/mcp`
- Health Check: `http://127.0.0.1:8000/healthz`

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `MCP_HOST` | `0.0.0.0` | Bind address |
| `MCP_PORT` | `8000` | Service port |
| `MCP_PATH` | `/mcp` | Streamable HTTP MCP base path (server accepts both with/without trailing slash) |
| `MCP_LOG_LEVEL` | `INFO` | Log level |
| `MCP_STATELESS_HTTP` | `false` | Enable stateless HTTP mode |
| `MCP_API_KEY` | empty | API Key for authentication; empty means disabled |
| `MCP_API_KEY_HEADER` | `X-API-Key` | Header name used for API Key |
| `MCP_MAX_CONCURRENT_YF_REQUESTS` | `32` | Maximum in-flight yfinance tasks allowed by semaphore |
| `MCP_YF_THREAD_WORKERS` | `16` | Worker count for the dedicated yfinance thread pool |
| `MCP_YF_ACQUIRE_TIMEOUT_SECONDS` | `2.0` | Timeout while waiting for yfinance concurrency slot |
| `MCP_UVICORN_LIMIT_CONCURRENCY` | `0` | Uvicorn global concurrency cap (`0` = unlimited) |

Example:

```bash
MCP_HOST=0.0.0.0 MCP_PORT=9000 MCP_PATH=/mcp python server.py
```

Enable API Key:

```bash
MCP_HOST=0.0.0.0 \
MCP_PORT=9000 \
MCP_PATH=/mcp \
MCP_API_KEY='your-strong-key' \
MCP_API_KEY_HEADER='X-API-Key' \
python server.py
```

## Authentication

When `MCP_API_KEY` is set, requests to MCP endpoints must include one of:

- `X-API-Key: <your-key>` (or your custom `MCP_API_KEY_HEADER`)
- `Authorization: Bearer <your-key>`

`/healthz` remains public by default.

## Docker

```bash
docker build -t yahoo-finance-mcp-http .
docker run -d \
  --name yahoo-finance-mcp-http \
  -p 8000:8000 \
  -e MCP_HOST=0.0.0.0 \
  -e MCP_PORT=8000 \
  -e MCP_PATH=/mcp \
  -e MCP_API_KEY=your-strong-key \
  -e MCP_API_KEY_HEADER=X-API-Key \
  yahoo-finance-mcp-http
```

## Nginx Reverse Proxy Example

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location /mcp {
        proxy_pass http://127.0.0.1:8000/mcp;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    location /healthz {
        proxy_pass http://127.0.0.1:8000/healthz;
    }
}
```

## MCP Endpoint

After deployment, use the following URL in MCP clients:

- `https://your-domain.com/mcp/` (recommended)

## cURL Example (Initialize)

```bash
curl -i https://your-domain.com/mcp \
  -H 'Accept: application/json, text/event-stream' \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: your-strong-key' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"curl","version":"0.1"}}}'
```

## Credits

- Original project: [Alex2Yang97/yahoo-finance-mcp](https://github.com/Alex2Yang97/yahoo-finance-mcp)
