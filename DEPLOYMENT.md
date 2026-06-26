# NEURAX Deployment Guide

## Quick Start (Development)

### Prerequisites

- Rust 1.70+ (with LLVM 18 for MLIR support)
- Node.js 18+ and pnpm
- Python 3.11+ (for AI agent)
- Docker and Docker Compose (for containerized deployment)

### 1. Start the Backend

```bash
# From project root
cargo run -p neurax-service

# Or with debug auth bypass (no Supabase needed)
NEURAX_DEBUG_NOAUTH=true cargo run -p neurax-service
```

The service starts on port 9098 by default.

### 2. Start the Frontend

```bash
cd neurax-ui
pnpm install
pnpm dev
```

The UI starts on port 8081 by default.

### 3. Start the AI Agent (Optional)

```bash
cd neurax-agent
pip install -r requirements.txt
uvicorn main:app --port 8099
```

---

## Docker Deployment

### Build and Run

```bash
# Build and start all services
docker compose up --build

# Or build individually
docker build -t neurax-service -f Dockerfile .
docker build -t neurax-ui -f Dockerfile.ui .
docker build -t neurax-agent -f Dockerfile.agent .
```

### Docker Compose Services

| Service | Port | Description |
|---------|------|-------------|
| `service` | 9098 | Rust actix-web API server |
| `ui` | 8081 | React frontend (nginx) |
| `agent` | 8099 | Python FastAPI AI agent |

### Environment Variables

Create a `.env` file in the project root:

```env
# Authentication
NEURAX_SUPABASE_URL=https://your-project.supabase.co
NEURAX_SUPABASE_ANON_KEY=your-anon-key
NEURAX_SUPABASE_JWT_SECRET=your-jwt-secret

# Debug mode (skip auth)
NEURAX_DEBUG_NOAUTH=true

# Stripe billing
STRIPE_SECRET_KEY=sk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...
NEURAX_ESSENTIAL_PRICE_ID=price_...
NEURAX_ARCHITECT_PRICE_ID=price_...
NEURAX_ELITE_PRICE_ID=price_...

# AI Agent
OPENAI_API_KEY=sk-...
```

---

## Production Deployment

### Backend (Rust Service)

The Dockerfile uses a multi-stage build:

1. **Builder stage**: Compiles the Rust workspace in release mode
2. **Runtime stage**: Copies the binary to `debian:bookworm-slim`

```bash
# Build release binary
cargo build --release -p neurax-service

# Or use Docker
docker build -t neurax-service -f Dockerfile .
```

**Health check**: `GET /health` returns `{"status":"ok"}`

### Frontend (React SPA)

The Dockerfile.ui uses a multi-stage build:

1. **Builder stage**: Runs `pnpm build` to produce static assets
2. **Runtime stage**: Serves via nginx with SPA routing and API proxy

```bash
# Build static assets
cd neurax-ui && pnpm build

# Or use Docker
docker build -t neurax-ui -f Dockerfile.ui .
```

The nginx configuration proxies `/api` requests to the backend service.

### AI Agent (Python)

```bash
# Build Docker image
docker build -t neurax-agent -f Dockerfile.agent .

# Run directly
cd neurax-agent
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8099
```

**Health check**: `GET /health` returns `{"status":"ok"}`

---

## Reverse Proxy Configuration

### nginx

```nginx
server {
    listen 80;
    server_name neurax.example.com;

    # Frontend
    location / {
        proxy_pass http://ui:8081;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # API
    location /api/ {
        proxy_pass http://service:9098/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # SSE support
    location /api/analyze/stream/ {
        proxy_pass http://service:9098/analyze/stream/;
        proxy_set_header Connection '';
        proxy_http_version 1.1;
        chunked_transfer_encoding off;
        proxy_buffering off;
        proxy_cache off;
    }
}
```

### Caddy

```
neurax.example.com {
    reverse_proxy /api/* service:9098
    reverse_proxy ui:8081
}
```

---

## SSL/TLS

For production, use Let's Encrypt with certbot or Caddy's automatic TLS:

```bash
# With certbot
certbot --nginx -d neurax.example.com

# With Caddy (automatic)
# Caddy handles TLS automatically
```

---

## Monitoring

### Health Checks

All services expose `/health` endpoints:

- Backend: `http://service:9098/health`
- AI Agent: `http://agent:8099/health`

### Docker Compose Health Checks

The `docker-compose.yml` includes health checks for all services:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:9098/health"]
  interval: 30s
  timeout: 10s
  retries: 3
```

---

## Scaling Considerations

### Horizontal Scaling

The backend service is stateless (analysis results are computed on-the-fly). For horizontal scaling:

1. Put multiple instances behind a load balancer
2. Use sticky sessions for SSE streaming endpoints
3. For project persistence, replace DashMap with a database (PostgreSQL via Supabase)

### Stateful Components

- **Projects** (`DashMap<ProjectKey, Project>`): In-memory, lost on restart. Migrate to Supabase/PostgreSQL for persistence.
- **Credits** (`DashMap<String, CreditInfo>`): In-memory, lost on restart. Migrate to database for persistence.
- **Streaming jobs** (`DashMap<String, JobInfo>`): Ephemeral by design, no persistence needed.

### Database Migration

For production, replace DashMap with Supabase/PostgreSQL:

1. Create `projects` and `credits` tables in Supabase
2. Replace `DashMap` operations with SQL queries via `sqlx` or Supabase REST API
3. Add connection pooling with `deadpool` or `sqlx::PgPool`

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `NEURAX_DEBUG_NOAUTH` not working | Ensure the env var is set to `true` (string, not boolean) |
| SSE connection drops | Increase nginx/proxy timeout, disable buffering for `/analyze/stream/` |
| CORS errors | The service allows all origins in dev mode. Configure `NEURAX_CORS_ORIGINS` for production |
| Stripe webhooks failing | Verify `STRIPE_WEBHOOK_SECRET` matches your Stripe dashboard |
| MLIR compilation fails | Ensure LLVM 18 is installed. Use `--no-default-features` to skip MLIR |

### Logs

```bash
# Backend logs
cargo run -p neurax-service 2>&1 | tee neurax.log

# Docker logs
docker compose logs -f service
docker compose logs -f ui
docker compose logs -f agent
```