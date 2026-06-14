# Neurax UI

This directory contains the Vite frontend for Neurax.

For the full local development flow, including the Rust backend and the Python agent, see the repo-level [README](../README.md).

## Run the frontend

Create `neurax-ui/.env` if you do not already have it:

```bash
cp .env.example .env
```

For local development without Supabase, use:

```dotenv
VITE_SUPABASE_DISABLED=true
VITE_NEURAX_API_URL=http://127.0.0.1:9098
VITE_AGENT_BASE_URL=http://127.0.0.1:8099
```

Install dependencies and start Vite:

```bash
npm install
npm run dev
```

The frontend runs on `http://localhost:8081` by default.

## Related services

- Backend: `cargo run -p neurax-service`
- Agent: `cd ../neurax-agent && python3 -m uvicorn app:app --host 127.0.0.1 --port 8099 --reload`
