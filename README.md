# Veyra v0.3

Veyra is a safe local coding-agent runtime written in Rust. It connects to an
OpenAI-compatible `llama-server`, streams model output, inspects a configured
workspace, applies approved edits, classifies Rust build/test failures, replans,
and reviews the final Git diff before completing a changed task. A bounded Context
Manager retrieves relevant source ranges and keeps every model request within an
explicit 32K or 65K profile.

## Requirements

- Windows 11 with WSL2 Ubuntu 24.04 (reference environment), or Linux
- Rust 1.85.0; `rust-toolchain.toml` installs rustfmt and clippy
- Git, Cargo, and preferably ripgrep for coding and automatic retrieval
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

The automated suite uses a mock HTTP/SSE server, temporary Git repositories,
and a temporary Rust fixture. Downloading a model is not required.

## Start llama-server

From WSL2, start the configured Qwen3-Coder model:

```bash
./llama-server \
  -m ./models/Qwen3-Coder-30B-A3B-Instruct-UD-Q3_K_XL.gguf \
  --host 127.0.0.1 --port 8080 --ctx-size 32768 \
  --n-gpu-layers 999 --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --batch-size 2048 --ubatch-size 512 --parallel 1
```

Check connectivity with `cargo run -p agent-cli -- models status`.

For an explicitly selected large-context run, restart the server with the matching
context and KV cache profile:

```bash
./llama-server \
  -m ./models/Qwen3-Coder-30B-A3B-Instruct-UD-Q3_K_XL.gguf \
  --host 127.0.0.1 --port 8080 --ctx-size 65536 \
  --n-gpu-layers 999 --flash-attn on \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --batch-size 2048 --ubatch-size 512 --parallel 1
```

Veyra selects request budgets but does not restart or reconfigure `llama-server`.

## Configuration

The default file is [`config/agent.toml`](config/agent.toml). Precedence from
highest to lowest is CLI flags, supported `VEYRA_` environment variables, the
selected TOML file, the default file, and built-in safe defaults.

```toml
[agent]
max_iterations = 30
max_consecutive_errors = 3
max_tool_calls = 50
max_identical_failures = 3

[context]
profile = "default"

[tools]
command_timeout_seconds = 120
stdout_limit_bytes = 1048576
stderr_limit_bytes = 1048576

[tools.command_profiles.cargo_build]
timeout_seconds = 300

[tools.command_profiles.cargo_test]
timeout_seconds = 600

[tools.command_profiles.git]
timeout_seconds = 60
```

Each profile can set `timeout_seconds`, `stdout_limit_bytes`, and
`stderr_limit_bytes`. A Tool request may lower but never raise its configured
timeout. Existing v0.1 configuration files remain valid. Supported environment
overrides are `VEYRA_MODEL_BASE_URL`, `VEYRA_MODEL_NAME`,
`VEYRA_WORKSPACE_ROOT`, and `VEYRA_LOG_LEVEL`.

The built-in `default` profile uses a 32,768-token context with a 2,048-token
output reserve. `large` uses 65,536 and reserves 4,096. If an older config has no
`[context]` section, its existing `[model].context_size` remains a safe legacy cap.

## CLI

```bash
veyra
veyra chat
veyra run "inspect this project and fix the failing test"
veyra --context-profile large run "analyze the repository architecture"
veyra models status
veyra tools list
veyra config check
```

`chat` starts a prompt loop; use `/quit`, `/exit`, or EOF to leave. Console input
supports UTF-8 and Windows-949/CP949 Korean text. Model tokens stream immediately,
while workflow, Tool, failure-classification, context selection, estimated usage,
actual server usage, and overflow-retry events are shown on stderr. The global
`--context-profile` option overrides `[context].profile` for one invocation.

## v0.3 context optimization

Before the first model request, Veyra extracts task terms and uses `rg --files` and
`rg -n` to select related files and bounded line ranges. A workspace-confined Rust
fallback is used when ripgrep is unavailable. Git-ignored paths, build output, large
or binary files are excluded from automatic retrieval.

If context diagnostics show `retrieval=rust_fallback`, verify that ripgrep is
available inside the same WSL/Linux environment that runs Veyra:

```bash
command -v rg
rg --version
```

On Ubuntu, install it with `sudo apt update && sudo apt install ripgrep`, then rerun
the task. The next `[context]` line should show `retrieval=ripgrep`. The Rust fallback
is safe and functional, but installing ripgrep gives faster search and full Git-ignore
handling on larger repositories.

Every request is rebuilt from the system prompt, current task and plan, retrieved
source ranges, recent relevant conversation, and Tool observations. Full Agent
state remains intact; only the model-facing view is trimmed. Assistant Tool calls
stay paired with their results, while failures, fingerprints, approval denials,
verification, and final diff reviews receive higher retention priority. Long Tool
observations are compressed deterministically with their summary, failure metadata,
and bounded head/tail content preserved.

The conservative provider-independent estimator accounts for messages and Tool
schemas before dispatch. Unused category budget can be borrowed by categories that
have input, but the output reserve is never used for prompt content. Server-reported
usage is requested with `stream_options.include_usage` and logged to expose estimation
error. A recognized HTTP context-overflow error
triggers one more aggressive rebuild and exactly one retry; a second overflow fails
the task without entering the general transient retry loop.

## v0.2 coding workflow

For tasks that change files, Veyra tracks:

```text
Discovery → Editing → Verifying → Recovering (when needed) → Reviewing → Completed
```

After the last file change, a successful `cargo_build`, `cargo_test`, or recognized
structured build/test/lint command is required. A later `git_diff` review is also
required before the Core accepts the model's final response. Compiler diagnostics,
test failures, timeouts, patch conflicts, policy violations, and generic command
failures are normalized. A repeated fingerprint requires replanning on its second
occurrence and terminates safely on its third occurrence by default.

The final response must identify changed files, checks performed, and remaining
risks. Approval denial is treated as a user decision, not a failure.

In a non-Git workspace, `git_diff` returns a successful `review_unavailable`
observation instead of trapping the workflow; the final response must disclose that
no repository diff could be reviewed.

## Tools and approval policy

Read-only tools run automatically:

- Workspace: `list_directory`, `read_file`, `read_file_range`, `glob`, `grep`
- Git: `git_status`, `git_diff`, `git_log`, `git_show`, and branch listing

State-changing and process tools require an exact one-time approval:

- `patch_file` and `write_file`
- `cargo_build` and `cargo_test`
- structured `run_command`
- branch creation/switching, `git_commit`, and `git_checkpoint`

`git_diff` supports working, staged, base-revision, and path-scoped views. The
default working-tree review adds bounded pseudo-diffs for untracked, non-ignored
text files and identifies binary untracked files, so an unborn repository cannot
silently pass review with an empty diff.
`git_branch` only lists or creates branches. `git_checkout` only switches branches
and refuses a dirty worktree. `git_commit` commits explicit paths only and disables
repository hooks; it does not amend or include unrelated staged changes.

`git_checkpoint` saves tracked staged/unstaged changes under
`refs/veyra/checkpoints/<UUID>` without changing HEAD, the current branch, index,
or worktree. Untracked files are deliberately excluded and reported. It refuses an
empty tracked snapshot.

Shell interpreters and Git writes through `run_command` are rejected. Remote push,
reset, clean, path checkout, forced branch operations, and other destructive Git
automation are not supported. Structured command timeout/cancellation
terminates a Unix process group or the Windows process tree, then reports a bounded
head/tail of stdout and stderr.

All paths and symlink destinations must remain inside the canonical workspace.
Approval is bound to Tool name, complete JSON arguments, and workspace; changing
any of them invalidates the decision.

## Dirty worktrees and recovery

Inspect `git_status` before editing. If tracked user changes exist, create an
approved checkpoint before modifying them, never as a finalization step. Do not
commit unless the user explicitly requested it. Do not assume untracked files are
recoverable from a checkpoint. Branch switching is intentionally blocked until
the worktree is clean. Patch hash/context conflicts are returned as recoverable
observations so the agent can reread the target and create a fresh minimal patch.

## Logs and architecture

Human-readable and structured rolling logs are written under `logs/`. Every Tool
request creates correlated append-only records in `logs/audit.jsonl`, with secrets
redacted from arguments and summaries.

- `agent-core`: bounded loop, workflow evaluator, failure fingerprints, events
- `agent-context`: token budgets, retrieval, trimming, observation compression
- `agent-model`: provider contract and OpenAI-compatible SSE adapter
- `agent-tools`: workspace, Git, Cargo, command, output, and process-tree adapters
- `agent-security`: workspace guard, risk/approval, redaction, JSONL audit
- `agent-cli`: configuration, composition root, rendering, approval prompt

Session persistence, Web/TUI, MCP/browser automation, document/vision support,
remote Git operations, `Allow Always`, embeddings, vector databases, semantic
reranking, and long-term memory retrieval remain out of scope.

## Verification status

Veyra v0.3.0 has 42 automated tests covering the earlier approval, audit, workspace,
workflow, Cargo, Git, and process contracts plus 32K/65K budgets, Unicode and long
input bounds, Tool message pairing, observation compression, retrieval ranges and
fallback exclusions, profile precedence, usage events, and one-shot overflow
recovery. The workspace quality gates pass under WSL2 Ubuntu 24.04 with Rust 1.85.0.
The live Qwen3-Coder/llama-server smoke test also passed with the default profile:
automatic retrieval used ripgrep and server-reported usage produced
`estimate=2417 actual=2373 delta=-44 retry=false`.

See [`docs/releases/v0.3.0.md`](docs/releases/v0.3.0.md) for release details. The
`docs/` directory is intentionally local and Git-ignored.
