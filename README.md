# Veyra v0.9

Veyra is a safe local coding-agent runtime written in Rust. It connects to an
OpenAI-compatible `llama-server`, streams model output, inspects a configured
workspace, applies approved edits, classifies Rust build/test failures, replans,
and reviews the final Git diff before completing a changed task. A bounded Context
Manager retrieves relevant source ranges and durable workspace memories while
keeping every model request within an explicit 32K or 65K profile. SQLite-backed
sessions preserve task, plan, message, Tool, approval, event, and audit history.
Veyra v0.9 adds a server-owned runtime, versioned Axum API, replayable SSE events,
a React control surface, and a ratatui client. CLI, TUI, and Web clients can operate
the same durable session and resolve each approval exactly once. Repository selection
remains confined to the configured workspace root.

## Requirements

- Windows 11 with WSL2 Ubuntu 24.04 (reference environment), or Linux
- Rust 1.88.0; `rust-toolchain.toml` installs rustfmt and clippy
- Git, Cargo, and preferably ripgrep for coding and automatic retrieval
- Poppler `pdftoppm` for scanned-PDF fallback
- A running llama.cpp router for interactive coding/vision use
- Node.js 22 or newer for the Web frontend or Playwright MCP server

The repository uses Cargo's Rust-version fallback resolver and commits
`Cargo.lock` so dependencies remain compatible with the MSRV.

## Build and verify

```bash
cargo build --workspace
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace

cd frontend
npm install
npm run lint
npm run typecheck
npm run test
npm run build
```

The automated suite uses a mock HTTP/SSE server, temporary Git repositories,
and a temporary Rust fixture. Downloading a model is not required.

## Start llama-server

From WSL2, start llama.cpp in router mode. `--models-max 1` keeps at most one model
loaded, which is the reference policy for a 16 GB GPU:

```bash
./llama-server \
  --models-preset ./config/llama-models.ini \
  --models-max 1 \
  --host 127.0.0.1 --port 8080
```

Check connectivity with `cargo run -p agent-cli -- models status`.

The preset defines `coding-default`, `coding-large`, and `vision` aliases. Veyra calls
the router's `/models` and `/models/load` endpoints and waits for readiness, but never
starts, stops, or owns the router process. Existing single-model `[model].model`
configuration remains a compatible default/large fallback.

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

[model.routes]
default = "coding-default"
large = "coding-large"
vision = "vision"
load_timeout_seconds = 300

[storage]
database_path = "data/veyra.sqlite3"

[documents]
max_file_bytes = 26214400
max_uncompressed_bytes = 104857600
max_documents_per_request = 100
max_chunks_per_document = 10000
chunk_target_chars = 2000
chunk_overlap_chars = 200
default_search_limit = 10
max_search_limit = 50

[vision]
max_file_bytes = 10485760
max_images_per_request = 8
max_pixels_per_image = 16000000
max_total_pixels = 32000000
max_pdf_pages = 50
pdf_dpi = 144
render_timeout_seconds = 120
max_output_chars = 65536
pdftoppm_command = "pdftoppm"

[research]
searxng_base_url = "http://127.0.0.1:8888/"
request_timeout_seconds = 20
max_redirects = 5
max_response_bytes = 2097152
max_results = 10
user_agent = "Veyra/0.9"

[server]
bind = "127.0.0.1:3000"
allow_remote = false
frontend_directory = "frontend/dist"

[mcp]
connect_timeout_seconds = 30
call_timeout_seconds = 60
max_result_bytes = 1048576

[mcp.servers.playwright]
enabled = false
kind = "playwright"
command = "npx"
args = ["-y", "@playwright/mcp@0.0.80", "--headless", "--output-dir", ".veyra/browser"]
pass_env = []

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
overrides are `VEYRA_MODEL_BASE_URL`, `VEYRA_MODEL_NAME`, `VEYRA_VISION_MODEL_NAME`,
`VEYRA_WORKSPACE_ROOT`, `VEYRA_LOG_LEVEL`, and `VEYRA_SEARXNG_BASE_URL`.

The built-in `default` profile uses a 32,768-token context with a 2,048-token
output reserve. `large` uses 65,536 and reserves 4,096. If an older config has no
`[context]` section, its existing `[model].context_size` remains a safe legacy cap.

## CLI

```bash
veyra
veyra chat
veyra run "inspect this project and fix the failing test"
veyra --context-profile large run "analyze the repository architecture"
veyra sessions list
veyra sessions show <session-id> --json --all
veyra sessions show <session-id> --research
veyra sessions resume <session-id>
veyra sessions prune --older-than 90
veyra documents add docs/spec.pdf notes.md
veyra documents list --json
veyra documents show <document-id> --chunks
veyra documents search "security requirements" --limit 10
veyra vision analyze screenshots/error.png --prompt "오류 원인을 설명해 줘"
veyra vision analyze architecture.png diagram.webp --prompt "구성 요소를 비교해 줘" --json
veyra models status
veyra tools list
veyra config check
veyra serve
veyra tui
veyra --server-url http://127.0.0.1:3000 run "inspect the workspace"
```

`chat` starts one durable session and records each prompt as a task in that session;
use `/quit`, `/exit`, or EOF to leave. `run` creates a new single-task session and
prints its ID. Console input
supports UTF-8 and Windows-949/CP949 Korean text. Model tokens stream immediately,
while workflow, Tool, failure-classification, context selection, estimated usage,
actual server usage, and overflow-retry events are shown on stderr. The global
`--context-profile` option overrides `[context].profile` for one invocation.
The bundled system prompt requires all assistant prose, including progress and final
answers, to remain in Korean while preserving code, commands, URLs, identifiers, and
verbatim errors where translation would reduce accuracy.

## v0.9 Web API and TUI

Build `frontend/`, then start the server with `veyra serve` or
`cargo run -p agent-server`. Open `http://127.0.0.1:3000` for the Web UI. The server
serves the production frontend and the `/api/v1` session, task, approval, model,
Tool, document, research, audit, and workspace APIs from one origin. The OpenAPI
description is available at `/api/v1/openapi.json`.

Agent events use SSE. `Last-Event-ID` or the `after` query parameter replays persisted
events before the connection switches to live delivery. Message submission and
approval decisions use ordinary HTTP POST requests. Concurrent approval decisions
are resolved transactionally; only the first decision can resume the Tool.

The default bind is loopback-only. A non-loopback bind requires both
`server.allow_remote = true` and a non-empty `VEYRA_SERVER_TOKEN`; clients send it as
`Authorization: Bearer ...`. The Web UI stores a supplied token only in browser
session storage. A session may select the configured workspace root or a canonical
subdirectory, never an arbitrary path or symlink escape.

Run `cargo run -p agent-tui` (or `veyra tui` when the companion binary is installed)
for the terminal interface. It provides session navigation, conversation, plan and
context panels, activity refresh, message entry, and allow/deny shortcuts. Existing
local `veyra chat` and `veyra run` remain compatible; `--server-url` selects client
mode for a shared server-owned session.

## v0.8 Vision and scanned PDFs

`vision analyze` and the read-only `vision_analyze` Agent Tool accept workspace-local
PNG, JPEG, and WebP files. Each result includes exact source citations, the loaded
model ID, extracted content, `high`/`medium`/`low`/`unknown` confidence, and limitations.
Paths are canonicalized through the workspace guard; symlink escape, remote URLs,
MIME/extension mismatch, corrupt files, more than 8 inputs, files over 10 MiB, images
over 16 MP, or requests over 32 MP are rejected rather than resized silently.

PDF parsing still tries the text layer first, page by page. Only pages with fewer than
20 alphanumeric characters are rendered by `pdftoppm` at 144 DPI and sent sequentially
to Vision, up to 50 pages. Successful pages survive individual failures; mixed results
are `partial`, total failure is `failed`, and an absent/unavailable Vision route retains
the legacy `unsupported_scanned` behavior. Chunks persist extraction method, confidence,
and limitations. Reuse requires both the file hash and a model/rendering pipeline
fingerprint to match. Vision output is an untrusted observation, never an instruction.

## v0.7 document analysis

`documents add` accepts workspace-relative files or directories. Directories are
walked recursively for PDF, DOCX, HTML, Markdown, and TXT files. The persistent index
is workspace-scoped; unchanged content hashes are reused and changed files are
replaced atomically. `documents list/show/search` support human output and `--json`.

The Agent exposes `document_index`, `document_list`, and `document_search`. Search
results include a stable document ID, BM25 score, bounded excerpt, page when known,
heading, UTF-8 byte offsets, and a citation label that must be reproduced in a
document-analysis answer. A failure or unsupported input is recorded independently,
so other documents remain searchable.

Completion-gated answers are buffered until validation succeeds, so a rejected draft
is not printed as if it were a final answer. If a citation is missing, the retry
receives valid labels to copy verbatim instead of regenerating the same uncited text.
For an explicit document-analysis task, Core rejects `read_file` calls and directs
the model back to `document_list`, `document_index`, and `document_search`.

Text PDFs are extracted page by page; encrypted PDFs are marked separately. When the
v0.8 Vision route is unavailable, scanned PDFs retain `unsupported_scanned` for backward
compatibility. Embeddings, vector search, exact layout restoration, and document editing
remain later-version work. DOCX archive expansion and all input,
chunk, collection, and result sizes are bounded by `[documents]`.

## v0.6 MCP and browser

MCP is disabled by default. Only entries with `enabled = true` are started, and
`config check` validates their names, limits, environment allowlists, and command
shape without launching a process. `tools list`, `chat`, and `run` connect enabled
servers, discover all paginated Tool definitions, and report an individual server
or Tool failure without removing native Tools or healthy MCP servers.

Each discovered Tool is exposed as `mcp__<server>__<remote-tool>`; long names receive
a stable hash suffix. Server commands are an executable plus argv, run with the
canonical workspace as cwd and a minimal environment. Add only environment variable
names—not values—to `pass_env` when a server needs one. Generic MCP Tools are Risk 3
unless their exact remote name is listed under `risk.read`, `risk.modify`, or
`risk.execute` for that server. Server annotations may raise, but never lower, risk.

The bundled Playwright example pins `@playwright/mcp@0.0.80`, uses stdio, is headless,
and writes artifacts beneath `.veyra/browser`. Snapshot and inspection actions are
read-only; navigation, tab/window state, resize, screenshot, and PDF generation are
Risk 1; click, typing, form actions, upload/drop, script execution, storage/cookie or
network mutation, and unknown actions are Risk 3. Upload inputs and output filenames
are revalidated against the workspace immediately before execution. A downloaded
file is data only—executing it requires a separate approved command.

The research order is Search → static `http_fetch` → Browser. Browser use is reserved
for an explicit interaction request or a JavaScript/dynamic page that static fetch
could not verify. An explicit Browser request is never replaced with static research
solely because the target happens to be static, and cannot complete until a Browser
snapshot succeeds. Negated Tool names such as “do not use web_search” do not create a
research requirement. A successful `browser_snapshot` final URL is accepted by the same
source/citation gate as `http_fetch`. MCP text and embedded text resources are bounded
and marked as untrusted external data; image and binary blocks retain metadata only.
There is no automatic restart or call replay after timeout, cancellation, disconnect,
or malformed output.

If Playwright discovery times out in WSL, verify `command -v node`, `node --version`,
and `command -v npx` inside that same distribution. Install Node.js 18+ natively in
WSL; a Windows `npx` path inherited through interop can print its version yet fail to
provide a reliable bidirectional stdio server. Also confirm the pinned package can run
with `npx -y @playwright/mcp@0.0.80 --version` and install the required browser runtime
according to the Playwright MCP diagnostics before retrying `veyra tools list`.

## v0.5 web research

The repository includes a local-only SearXNG and Valkey Compose stack under
[`searXNG/`](searXNG/README.md). From that directory, run `./setup.ps1` once and
then use `docker compose up -d` and `docker compose down` to start and stop it.
The bundled settings enable the JSON response format and bind SearXNG to
`127.0.0.1:8888` by default. Its tested image versions are pinned in `.env`;
updates are explicit and the Compose healthchecks cover both services.

For another SearXNG instance, enable JSON in its `search.formats` setting, then
set `research.searxng_base_url` or `VEYRA_SEARXNG_BASE_URL`. SearXNG is not
contacted at startup; an unavailable instance becomes a structured `web_search`
Tool error only when research is requested.

The Agent uses `web_search` to collect candidate URLs and `http_fetch` to verify
static sources. Research completion requires at least one fetched final URL in the
answer. Search snippets and page bodies are explicitly marked as untrusted external
data, are retained in session Tool results, and cannot grant permission or direct
Tool execution. Audit records keep bounded query/source metadata but do not duplicate
the extracted page body. Tasks that explicitly request web search or combine a
search/research action with source, citation, URL, or freshness requirements cannot
complete until `web_search` succeeds; prior session memories are omitted from those
requests and cannot substitute for current evidence. After the same normalized query
has already led to a successful fetch in the current task, an identical search is
skipped with guidance to use the verified evidence or refine the query. `sessions show
<id> --research` prints a bounded research timeline without page bodies or snippets;
add `--json` for compact machine-readable output.

`http_fetch` accepts only HTTP/HTTPS GET requests and supports `text/html`,
`application/xhtml+xml`, and `text/plain`. It disables proxies and automatic
redirects, validates and pins DNS results on every redirect, rejects local/private,
link-local, reserved, and metadata-service addresses, follows at most five redirects,
and enforces the configured timeout and byte limit while streaming. JavaScript
rendering is delegated to the explicitly enabled browser workflow. Credential
storage/automatic input, captcha bypass, background crawling, and automatic execution
of downloaded files are not supported.
Operators remain responsible for each site's terms and robots policy.

## v0.4 sessions, memory, and audit

SQLite at `storage.database_path` is the canonical durable store. Versioned,
transactional migrations create normalized sessions, tasks, plans, messages, Tool
calls, approvals, memories, events, and audit records. The legacy
`logs/audit.jsonl` remains an append-only compatibility mirror with the same secret
redaction policy.

`sessions resume` requires the configured canonical workspace to match the stored
workspace. A pending approval is cancelled on recovery and a requested or running
Tool is marked interrupted. Veyra adds a Tool observation describing the interruption
but never replays the Tool or reuses an `Allow once` decision automatically. Terminal
sessions continue as a new task in the same conversation.

Ctrl+C at an approval prompt performs a normal cancellation and exits the current
CLI invocation without executing the Tool. To test crash recovery rather than normal
cancellation, terminate the `veyra` process from another terminal (for example with
`kill -9`) while the approval prompt is pending, then run `sessions resume`.
`sessions show` reports normalized `tool_calls` and `approvals` rows as well as the
event and audit streams, so recovered rows can be inspected as `interrupted` and
`cancelled`. Token-usage counters remain visible; only credential-shaped token keys
such as `api_token` are redacted.

Successful tasks produce a deterministic summary memory without another model call.
Only memories from the same canonical workspace with matching task terms are eligible
for context. The default/large profiles reserve 1,024/2,048 tokens for memory; an
overflow retry halves that allowance before rebuilding the request.

Sessions are retained indefinitely. `sessions prune --older-than <days>` previews and
confirms deletion of terminal SQLite sessions only; `--yes` enables non-interactive
execution. Running sessions and append-only JSONL audit files are never pruned by this
command.

For backup, stop Veyra and copy the SQLite database together with any `-wal` and `-shm`
files, or use SQLite's online `.backup` command. Restore all files while Veyra is
stopped, then run `veyra config check`; startup applies pending migrations
transactionally.

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
- Web: `web_search` and static GET-only `http_fetch`
- MCP: exact generic read allowlists and Playwright snapshot/inspection actions

State-changing and process tools require an exact one-time approval:

- `patch_file` and `write_file`
- `cargo_build` and `cargo_test`
- structured `run_command`
- branch creation/switching, `git_commit`, and `git_checkpoint`
- MCP actions classified as Risk 1–3, including all Playwright interaction actions

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
- `agent-research`: SearXNG search, SSRF-resistant fetch, source DTOs, extraction
- `agent-document`: format parsers, normalization, chunking, and retrieval contracts
- `agent-vision`: validated image DTOs, OpenAI multipart adapter, and Poppler fallback
- `agent-mcp`: stdio lifecycle, discovery, Tool adapter, result and browser policy
- `agent-storage`: SQLite migrations, session snapshots, memory and audit queries
- `agent-model`: provider contract, router manager, capabilities, and SSE adapter
- `agent-tools`: workspace, Git, Cargo, command, output, and process-tree adapters
- `agent-security`: workspace guard, risk/approval, redaction, JSONL audit
- `agent-app`: shared server runtime, composition, active task and approval broker
- `agent-server`: versioned HTTP/SSE API and production frontend hosting
- `agent-tui`: terminal client for the shared API
- `agent-cli`: compatible local workflow plus server/client launch commands

Remote/HTTP MCP transports, Browser takeover, credential automation,
background crawling, image generation/editing, video/audio, cloud Vision fallback,
professional OCR/layout restoration, remote Git operations, `Allow Always`, embeddings,
vector databases, semantic reranking, and long-term memory retrieval
remain out of scope.

## Verification status

Veyra v0.9.0 has 116 Rust tests plus two frontend component tests. They
cover the v0.1-v0.8 regression suite together with workspace-confined sessions,
remote-bind policy, Bearer authentication, atomic approval resolution, redacted event
replay, frontend reconnect deduplication, and approval focus trapping. Rust build, format, strict Clippy, and
test gates pass under WSL2 Ubuntu 24.04 with Rust 1.88.0; frontend lint, typecheck,
test, and production build pass with Node.js 24. A localhost smoke test returned the
production UI, OpenAPI v0.9.0, and consistent session creation/listing responses.

See [`docs/releases/v0.9.0.md`](docs/releases/v0.9.0.md) for release details. The
`docs/` directory is intentionally local and Git-ignored.
