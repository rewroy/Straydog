# straydog-gguf

GGUF v3 structure checks with size limits and checked arithmetic, which the engine runs on
each shard. Structure checks alone prove neither authenticity nor correctness; the engine
authenticates every file by SHA-256 first.

```bash
cargo test -p straydog-gguf
```
