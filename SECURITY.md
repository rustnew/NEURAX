# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.6.x   | ✅ Active development |
| 0.5.x   | ✅ Maintained |
| < 0.5   | ❌ Not supported |

## Reporting a Vulnerability

**Please do not open public issues for security vulnerabilities.**

Instead, report privately via GitHub's **Security Advisories**:

1. Go to https://github.com/rustnew/NEURAX/security/advisories
2. Click **"New draft security advisory"**
3. Describe the vulnerability, affected versions, and impact

You can also email the maintainer directly (address available on the
repository profile page).

We aim to acknowledge reports within **48 hours** and publish a fix within
**7 days** for confirmed vulnerabilities.

## Security Notes

- NEURAX analyzes **untrusted model JSON files** — the parser is fuzz-tested,
  but treat configs from unknown sources with care.
- API keys (Stripe, Supabase) must only be set via environment variables,
  never committed.
