# Deployment Guide

This guide covers how to deploy NEURAX in development and production environments.

---

## Table of Contents

1. [Quick Deploy (Docker Compose)](#quick-deploy-docker-compose)
2. [Development Setup](#development-setup)
3. [Production Deployment](#production-deployment)
4. [Environment Variables](#environment-variables)
5. [Health Checks](#health-checks)
6. [Troubleshooting](#troubleshooting)

---

## Quick Deploy (Docker Compose)

The fastest way to run all NEURAX services locally is via Docker Compose.

### Prerequisites

- Docker Engine 24+
- Docker Compose v2+

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/rustnew/NEURAX.git
cd NEURAX

# 2. Configure environment files
cp neurax-service/.env.example neurax-service/.env
cp neurax-ui/.env.example neurax-ui/.env
cp neurax-agent/.env neurax-agent/.env  # Already exists, adjust as needed

# 3. Start all services
docker compose up

# 4. Verify
curl http://localhost:9098/health
curl http://localhost:8099/health
# Open http://localhost:8081 in your browser
```

### Services

| Service | Port | Dockerfile | Description |
|---|---|---|---|
| `service` | 9098 | `Dockerfile` | Rust actix-web backend (38 routes) |
| `ui` | 8081 | `Dockerfile.ui` | React 18 + TypeScript frontend |
| `agent` | 8099 | `Dockerfile.agent` | Python FastAPI + LangChain agent |

### Stopping

```bash
docker compose down
```

---

## Development Setup

For active development, run each service locally without Docker.

### Option A: All-in-One Script

```bash
# Clone and enter the repo
git clone https://github.com/rustnew/NEURAX.git
cd NEURAX

# Run the development startup script
chmod +x start-dev.sh
./start-dev.sh
```

The `start-dev.sh` script launches all three services in the background:

- **Rust backend** (`neurax-service`) on port 9098
- **Python agent** (`neurax-agent`) on port 8099
- **React frontend** (`neurax-ui`) on port 8081

### Option B: Manual Setup

#### 1. Rust Backend (neurax-service)

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Run the service
cargo run -p neurax-service
# → http://localhost:9098
```

#### 2. Python Agent (neurax-agent)

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r neurax-agent/requirements.txt

# Set environment variables
export OPENAI_API_KEY="sk-..."  # Or ANTHROPIC_API_KEY
export NEURAX_AGENT_HOST=127.0.0.1
export NEURAX_AGENT_PORT=8099

# Run the agent
cd neurax-agent
python3 -m uvicorn app:app --host 127.0.0.1 --port 8099 --reload
# → http://localhost:8099
```

#### 3. React Frontend (neurax-ui)

```bash
# Install dependencies
cd neurax-ui
npm install  # or: pnpm install

# Configure environment
cp .env.example .env
# Edit .env:
#   VITE_SUPABASE_DISABLED=true
#   VITE_NEURAX_API_URL=http://127.0.0.1:9098
#   VITE_AGENT_BASE_URL=http://127.0.0.1:8099

# Start the dev server
npm run dev
# → http://localhost:8081
```

### 4. MCP Server (neurax-mcp)

```bash
# Install
pip install -e neurax-mcp

# Run (stdio mode, configured in Claude Desktop's claude_desktop_config.json)
# See: https://modelcontextprotocol.io/docs/develop/server
```

---

## Production Deployment

### Docker Compose (Recommended)

```bash
# 1. Create production .env files
cp neurax-service/.env.example neurax-service/.env
# Edit with real Supabase URL, Stripe keys, etc.

cp neurax-ui/.env.example neurax-ui/.env
# Set VITE_NEURAX_API_URL and VITE_AGENT_BASE_URL to production URLs

cp neurax-agent/.env neurax-agent/.env
# Set OPENAI_API_KEY or ANTHROPIC_API_KEY

# 2. Build and start
docker compose up -d

# 3. Verify health
./healthcheck.sh
```

### Docker Build (Individual Services)

```bash
# Build the Rust service
docker build -t neurax-service -f Dockerfile .

# Build the UI
docker build -t neurax-ui -f Dockerfile.ui .

# Build the agent
docker build -t neurax-agent -f Dockerfile.agent .
```

### Kubernetes

A Kubernetes deployment manifest is planned. For now, use Docker Compose with a reverse proxy (nginx/Caddy) for TLS termination and load balancing.

Example nginx reverse proxy config:

```nginx
server {
    listen 443 ssl http2;
    server_name neurax.example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8081;
        proxy_set_header Host $host;
    }

    location /api/ {
        proxy_pass http://localhost:9098/;
        proxy_set_header Host $host;
    }

    location /agent/ {
        proxy_pass http://localhost:8099/;
        proxy_set_header Host $host;
    }
}
```

---

## Environment Variables

### neurax-service

| Variable | Default | Description |
|---|---|---|
| `NEURAX_BIND` | `0.0.0.0:9098` | Bind address for the Actix server |
| `RUST_LOG` | `info` | Logging level |
| `NEURAX_DEBUG_NOAUTH` | `false` | Bypass auth for development |
| `NEURAX_MOCK_PLAN` | `elite` | Mock subscription plan for development |
| `SUPABASE_URL` | — | Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | — | Supabase service role key (backend only) |
| `STRIPE_SECRET_KEY` | — | Stripe secret key |
| `STRIPE_WEBHOOK_SECRET` | — | Stripe webhook signing secret |
| `STRIPE_PRICE_ESSENTIAL_MONTHLY` | — | Stripe price ID |
| `STRIPE_PRICE_ESSENTIAL_ANNUAL` | — | Stripe price ID |
| `STRIPE_PRICE_ARCHITECT_MONTHLY` | — | Stripe price ID |
| `STRIPE_PRICE_ARCHITECT_ANNUAL` | — | Stripe price ID |
| `STRIPE_PRICE_ELITE_MONTHLY` | — | Stripe price ID |
| `STRIPE_PRICE_ELITE_ANNUAL` | — | Stripe price ID |
| `STRIPE_PORTAL_RETURN_URL` | — | Stripe portal return URL |

### neurax-agent

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key (for GPT models) |
| `ANTHROPIC_API_KEY` | — | Anthropic API key (for Claude models) |
| `NEURAX_AGENT_HOST` | `127.0.0.1` | Bind address |
| `NEURAX_AGENT_PORT` | `8099` | Port number |
| `NEURAX_SERVICE_URL` | `http://127.0.0.1:9098` | Backend service URL |

### neurax-ui

| Variable | Default | Description |
|---|---|---|
| `VITE_NEURAX_API_URL` | `http://localhost:9098` | Backend API URL |
| `VITE_AGENT_BASE_URL` | `http://localhost:8099` | Agent API URL |
| `VITE_SUPABASE_DISABLED` | `false` | Disable Supabase auth for development |

---

## Health Checks

Use the included health check script to verify all services are running:

```bash
./healthcheck.sh
```

Expected output:

```
[✓] neurax-service (port 9098): healthy
[✓] neurax-agent (port 8099): healthy
[✓] neurax-ui (port 8081): healthy
```

---

## Troubleshooting

### Port Already in Use

```bash
# Check what's using a port
lsof -i :9098

# Kill the process
kill -9 <PID>
```

### Agent LLM Not Responding

Ensure `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` is set in `neurax-agent/.env`.

### Supabase Connection Issues

Verify `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` are set correctly in `neurax-service/.env`.