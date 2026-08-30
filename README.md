# Veyra v0.1

Veyra is a local coding-agent runtime written in Rust. It connects to an
OpenAI-compatible `llama-server`, streams model output, and lets the model inspect
a configured workspace. File changes and process execution are always bound to an
exact, one-time user approval.

## Requirements

- Windows 11 with WSL2 Ubuntu 24.04 (the reference environment), or Linux
- Rust 1.85.0; `rust-toolchain.toml` installs the required formatter and clippy
- Git for the read-only Git tools
- A running OpenAI-compatible server for interactive use

The repository uses Cargo's Rust-version fallback resolver and commits
`Cargo.lock` so dependencies remain compatible with the MSRV.

## Build and verify

```bash
cargo build --workspace
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

The tests include a local mock HTTP/SSE server; downloading a model is not needed
for the automated test suite.

## Start llama-server

Download `Qwen3-Coder-30B-A3B-Instruct-UD-Q3_K_XL.gguf`, build llama.cpp with
CUDA support, and run from WSL2:

```bash
./llama-server \
  -m ./models/Qwen3-Coder-30B-A3B-Instruct-UD-Q3_K_XL.gguf \
  --host 127.0.0.1 --port 8080 --ctx-size 32768 \
  --n-gpu-layers 999 --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --batch-size 2048 --ubatch-size 512 --parallel 1
```

Check connectivity with:

```bash
cargo run -p agent-cli -- models status
```

## Configuration

The default file is [`config/agent.toml`](config/agent.toml). Precedence from
highest to lowest is CLI flags, `VEYRA_` environment variables, the selected TOML
file, the default TOML file, and built-in safe defaults.

Supported environment overrides are `VEYRA_MODEL_BASE_URL`, `VEYRA_MODEL_NAME`,
`VEYRA_WORKSPACE_ROOT`, and `VEYRA_LOG_LEVEL`. Use another file or workspace with
`--config <path>` and `--workspace <path>`. Relative workspace paths are resolved
from the process working directory.

## CLI

```bash
veyra                                      # same as `veyra chat`
veyra chat
veyra run "inspect this project and fix the failing test"
veyra models status
veyra tools list
veyra config check
```

`chat` starts a prompt loop; use `/quit`, `/exit`, or EOF to leave. A task first
checks model health. Model tokens are printed as they arrive, tool observations
are sent back to the model, and the loop stops at the configured iteration,
tool-call, or consecutive-error limit.

Console input is decoded as UTF-8 first and falls back to Windows-949/CP949 for
Korean terminals. To inspect a directory outside the default `workspace`, select
it when starting Veyra, for example `veyra --workspace ~/algorithm chat`; paths
requested by the model remain confined to that selected workspace.

## Tools and approval policy

Read-only tools run automatically: `list_directory`, `read_file`,
`read_file_range`, `glob`, `grep`, `git_status`, and `git_diff`.

State-changing tools require `Allow once` or `Deny` every time:

- `patch_file` applies a unified patch and can bind it to `expected_sha256`.
- `write_file` creates files; replacing an existing file additionally requires
  `overwrite=true` and can verify `expected_sha256`.
- `run_command` accepts a program, argument array, workspace-relative working
  directory, bounded timeout, and explicit environment map. It never invokes a
  shell implicitly.

Risk 3 commands such as deletion, privilege changes, destructive Git operations,
or network mutation show a prominent warning but can run only after exact
one-time approval. Approval is bound to the tool, complete JSON arguments, and
canonical workspace. Changing any of them invalidates the decision.

All file targets and symlink destinations must stay within the canonical
workspace. Commands have no interactive stdin or background-daemon support.
Timeout cancellation kills the direct child and performs best-effort cleanup;
platform-specific descendant processes may survive if they detach themselves.

## Logs and audit

Human-readable and structured rolling logs are written under `logs/`. Every tool
request creates correlated append-only records in `logs/audit.jsonl`, including
session/task/call IDs, redacted arguments, risk, approval, duration, truncation,
and final status. Keys containing tokens, passwords, API keys, authorization, or
secrets and Bearer values are masked.

## Architecture

- `agent-core`: bounded state machine, events, observations, and orchestration
- `agent-model`: provider contract and OpenAI-compatible SSE adapter
- `agent-tools`: registry and the ten native v0.1 tools
- `agent-security`: IDs, workspace guard, risk/approval, redaction, JSONL audit
- `agent-cli`: configuration, composition root, rendering, and approval prompt

Session persistence, Web/TUI, MCP/browser automation, document/vision support,
Git write workflows, `Allow Always`, and advanced retrieval are intentionally
outside v0.1.

## Verification status

On 2026-08-30 the workspace passed build, formatting, strict clippy, all automated
tests, `config check`, and `tools list` under WSL2 with Rust 1.85.0. The mock SSE
path is automated. A real Qwen/llama-server smoke test remains conditional on the
model runtime being available.
