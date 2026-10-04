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
  upload path, and no dispatch/actuation capability by design. The
  security-relevant surface is small:
  four read-only API routes (`/build-info`, `/health`, `/api/watchlist`,
  `/api/benchmark`) and a Next.js static frontend.
- The AWS instance profile attached to the live box has read-only access to the
  corpus bucket (it cannot write or delete anything there) plus
  `cloudwatch:PutMetricData` scoped to the `PlumeRank` namespace.
- Camera imagery is public HPWREN FIgLib footage (long-range landscape views); the box writes only
  small evidence crops and thumbnails of it for the screen, and no personal data
  is processed.
