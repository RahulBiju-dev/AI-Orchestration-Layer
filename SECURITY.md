# Security Policy

## Supported Versions

Only the latest release version on the `main` branch is actively supported with security updates.

| Version | Supported |
| --- | --- |
| v2.x | :white_check_mark: Yes |
| < v2.0 | :x: No |

## Reporting a Vulnerability

We take the security of AI CLI Agent seriously. If you find any security vulnerability, please do not report it through public issues. Instead, please report it via one of the following methods:

- **Email**: Send a detailed report to rnair5603@gmail.com.
- **GitHub Private Report**: If available, use the private vulnerability reporting feature on the GitHub repository.

Please include:
1. A description of the vulnerability and its potential impact.
2. Steps to reproduce the vulnerability (proof of concept code, command, or environment details).
3. Any suggested remediation steps.

We will acknowledge receipt of your report within 48 hours and work with you to resolve the issue before making any public announcements.

## Sandboxing Notice
AI CLI Agent is designed as a local terminal assistant. The model can call local tools (like filesystem operations or web browsing tools). By running this software, you recognize that the model has access to execute actions locally on your machine within the scope of the registered python handlers. Never run the agent on sensitive environments without appropriate monitoring and controls.
