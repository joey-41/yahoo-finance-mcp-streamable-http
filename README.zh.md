# Yahoo Finance MCP（Streamable HTTP）

[English](./README.md)

这个项目是 [Alex2Yang97/yahoo-finance-mcp](https://github.com/Alex2Yang97/yahoo-finance-mcp) 的 Streamable HTTP 版本，适合部署在服务器上，通过 URL 远程调用 MCP 服务。

## 功能特性

- 支持 MCP Streamable HTTP 传输（默认路径 `/mcp/`，兼容 `/mcp`）
- 提供 Yahoo Finance 常用能力（行情、财报、期权、新闻、评级、持仓）
- 提供健康检查端点：`/healthz`
- 支持通过环境变量配置监听地址、端口、路径、日志级别和无状态模式
- 支持 API Key 鉴权（可选）
- 针对高并发场景增加保护（线程池隔离 + 并发信号量 + 快速失败）

## 可用工具

- `get_historical_stock_prices`
- `get_stock_info`
- `get_yahoo_finance_news`
- `get_stock_actions`
- `get_financial_statement`
- `get_holder_info`
- `get_option_expiration_dates`
- `get_option_chain`
- `get_recommendations`

## 运行要求

- Python `>=3.11`

## 本地快速启动

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python server.py
```

默认运行参数：

- Host：`0.0.0.0`
- Port：`8000`
- MCP 路径：`/mcp`
- 健康检查：`http://127.0.0.1:8000/healthz`

## 环境变量

| 变量名 | 默认值 | 说明 |
| --- | --- | --- |
| `MCP_HOST` | `0.0.0.0` | 服务监听地址 |
| `MCP_PORT` | `8000` | 服务端口 |
| `MCP_PATH` | `/mcp` | Streamable HTTP MCP 基础路径（服务端兼容有无尾斜杠） |
| `MCP_LOG_LEVEL` | `INFO` | 日志级别 |
| `MCP_STATELESS_HTTP` | `false` | 是否启用无状态 HTTP 模式 |
| `MCP_API_KEY` | 空 | API Key；为空表示不启用鉴权 |
| `MCP_API_KEY_HEADER` | `X-API-Key` | API Key 请求头名称 |
| `MCP_MAX_CONCURRENT_YF_REQUESTS` | `32` | 允许同时执行的 yfinance 任务上限（信号量） |
| `MCP_YF_THREAD_WORKERS` | `16` | 专用 yfinance 线程池工作线程数 |
| `MCP_YF_ACQUIRE_TIMEOUT_SECONDS` | `2.0` | 等待并发槽位的超时时间（秒） |
| `MCP_UVICORN_LIMIT_CONCURRENCY` | `0` | Uvicorn 全局并发上限（`0` 表示不限制） |

基础示例：

```bash
MCP_HOST=0.0.0.0 MCP_PORT=9000 MCP_PATH=/mcp python server.py
```

启用 API Key 示例：

```bash
MCP_HOST=0.0.0.0 \
MCP_PORT=9000 \
MCP_PATH=/mcp \
MCP_API_KEY='your-strong-key' \
MCP_API_KEY_HEADER='X-API-Key' \
python server.py
```

## 鉴权说明

当设置 `MCP_API_KEY` 后，访问 MCP 端点需携带以下任一请求头：

- `X-API-Key: <your-key>`（或你自定义的 `MCP_API_KEY_HEADER`）
- `Authorization: Bearer <your-key>`

默认情况下，`/healthz` 端点不做鉴权。

## Docker 部署

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

## Nginx 反向代理示例

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

## MCP 服务地址

部署完成后，在 MCP 客户端中使用：

- `https://your-domain.com/mcp/`（推荐）

## cURL 初始化示例

```bash
curl -i https://your-domain.com/mcp \
  -H 'Accept: application/json, text/event-stream' \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: your-strong-key' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"curl","version":"0.1"}}}'
```

## 致谢

- 原始项目：[Alex2Yang97/yahoo-finance-mcp](https://github.com/Alex2Yang97/yahoo-finance-mcp)
