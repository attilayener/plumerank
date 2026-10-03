# Security Policy

## Supported Versions
| Version | Supported |
|---|---|
| latest (`main`) | ✅ |

## Reporting a Vulnerability
Please **do not** open a public issue for security vulnerabilities. Instead,
report them privately:

- Email **edy.cu@live.com**, or
- Use GitHub's [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability) (Security → Report a vulnerability).

You'll get an acknowledgment within 48 hours and a resolution timeline after
triage. Please give us a reasonable window to patch before public disclosure.

## Scope notes specific to this project
- The live endpoint is a single-tenant research box with no user accounts, no
  upload path, and no dispatch/actuation capability by design (`specs/spec.md`
  §7, §9 in the project kitchen). The security-relevant surface is small:
  four read-only API routes (`/build-info`, `/health`, `/api/watchlist`,
  `/api/benchmark`) and a Next.js static frontend.
- The AWS instance profile attached to the live box carries **read-only**
  statements only; it cannot write or delete anything in the corpus bucket.
- Camera imagery is public wildland surveillance (HPWREN FIgLib); the pipeline
  persists derived statistics, not frames, and no personal data is processed.
