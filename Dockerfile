# ─── NEURAX Service — Multi-stage Docker Build ──────────────────────
# Build:  docker build -t neurax-service .
# Run:    docker run -p 9098:9098 --env-file neurax-service/.env neurax-service

# ── Stage 1: Build ──────────────────────────────────────────────────
FROM rust:1.78-bookworm AS builder

WORKDIR /app

# Copy workspace manifests first (cache dependencies)
COPY Cargo.toml Cargo.lock ./
COPY neurax-core/Cargo.toml neurax-core/Cargo.toml
COPY neurax-parser/Cargo.toml neurax-parser/Cargo.toml
COPY neurax-ir/Cargo.toml neurax-ir/Cargo.toml
COPY neurax-formulas/Cargo.toml neurax-formulas/Cargo.toml
COPY neurax-hardware-db/Cargo.toml neurax-hardware-db/Cargo.toml
COPY neurax-tui/Cargo.toml neurax-tui/Cargo.toml
COPY neurax-service/Cargo.toml neurax-service/Cargo.toml
COPY neurax-opspec/Cargo.toml neurax-opspec/Cargo.toml

# Create dummy src files so cargo can resolve the workspace. Every member
# needs one, whether or not neurax-service depends on it — resolution reads
# the whole workspace.
RUN mkdir -p neurax-core/src && echo "fn main(){}" > neurax-core/src/lib.rs \
    && mkdir -p neurax-parser/src && echo "fn main(){}" > neurax-parser/src/lib.rs \
    && mkdir -p neurax-ir/src && echo "fn main(){}" > neurax-ir/src/lib.rs \
    && mkdir -p neurax-formulas/src && echo "fn main(){}" > neurax-formulas/src/lib.rs \
    && mkdir -p neurax-hardware-db/src && echo "fn main(){}" > neurax-hardware-db/src/lib.rs \
    && mkdir -p neurax-tui/src && echo "fn main(){}" > neurax-tui/src/main.rs \
    && mkdir -p neurax-service/src && echo "fn main(){}" > neurax-service/src/main.rs \
    && mkdir -p neurax-opspec/src && echo "fn main(){}" > neurax-opspec/src/lib.rs

# Build dependencies only (cached layer)
RUN cargo build --release -p neurax-service 2>/dev/null || true

# Copy actual source code
COPY neurax-core/ neurax-core/
COPY neurax-parser/ neurax-parser/
COPY neurax-ir/ neurax-ir/
COPY neurax-formulas/ neurax-formulas/
COPY neurax-hardware-db/ neurax-hardware-db/
COPY neurax-tui/ neurax-tui/
COPY neurax-service/ neurax-service/

# Touch source files to invalidate cache
RUN find neurax-*/src -name "*.rs" -exec touch {} +

# Build the service binary
RUN cargo build --release -p neurax-service

# ── Stage 2: Runtime ────────────────────────────────────────────────
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash neurax

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/target/release/neurax-service /app/neurax-service

RUN chown -R neurax:neurax /app

USER neurax

# Default env vars (override with --env-file or -e). NEURAX_DEBUG_NOAUTH is
# deliberately NOT set here — it defaults to off in the service itself
# (see neurax-service/src/lib.rs's noauth_enabled doc comment: it used to
# default to on, which meant a deployment that never set it anonymously
# served every endpoint). Setting it here would silently reintroduce that.
ENV NEURAX_BIND=0.0.0.0:9098
ENV RUST_LOG=info

EXPOSE 9098

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:9098/health || exit 1

ENTRYPOINT ["/app/neurax-service"]