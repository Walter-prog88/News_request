# Security notes

- Never commit `.env` or a real `TWOSHARE_TOKEN`.
- Use only `https://jiaoch.site`; the client rejects plaintext HTTP.
- Rotate the token if it is exposed outside the intended course/research group.
- Snapshot receipts deliberately omit request credentials.
- Before every push, scan tracked files for credentials and unexpectedly large files.

