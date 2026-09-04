use agent_context::MemorySnippet;
use agent_core::{AgentEvent, AgentState, AgentStatus, Observation, SessionRepository};
use agent_document::{
    DocumentChunk, DocumentError, DocumentFormat, DocumentRepository, DocumentSearchHit,
    DocumentSearchQuery, DocumentSource, DocumentStatus, DocumentSummary, IndexResult,
    NormalizedDocument, term_frequencies, tokenize,
};
use agent_model::Message;
use agent_security::{
    ApprovalDecision, AuditEvent, AuditSink, SecurityError, ToolCallId, redact_value,
};
use async_trait::async_trait;
use chrono::{Duration, Utc};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use uuid::Uuid;

const MIGRATION_1: &str = r#"
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    workspace TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_event_id INTEGER
);
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    request TEXT NOT NULL,
    status TEXT NOT NULL,
    snapshot_json TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    tool_calls INTEGER NOT NULL,
    consecutive_errors INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS plans (
    task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    step_id TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL,
    PRIMARY KEY(task_id, ordinal)
);
CREATE TABLE IF NOT EXISTS messages (
    task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    message_json TEXT NOT NULL,
    PRIMARY KEY(task_id, ordinal)
);
CREATE TABLE IF NOT EXISTS tool_calls (
    call_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    model_call_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    arguments_json TEXT NOT NULL,
    risk TEXT NOT NULL,
    status TEXT NOT NULL,
    result_json TEXT,
    error TEXT,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS approvals (
    approval_id TEXT PRIMARY KEY,
    call_id TEXT NOT NULL REFERENCES tool_calls(call_id) ON DELETE CASCADE,
    request_json TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    decision_json TEXT,
    status TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    workspace TEXT NOT NULL,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    summary TEXT NOT NULL,
    search_text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(task_id)
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    event_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    call_id TEXT NOT NULL,
    phase TEXT NOT NULL,
    event_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(session_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_workspace ON memories(workspace, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_events(session_id, id);
"#;

const MIGRATION_2: &str = r#"
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    workspace TEXT NOT NULL,
    path TEXT NOT NULL,
    format TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    title TEXT,
    status TEXT NOT NULL,
    error TEXT,
    metadata_json TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    byte_size INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    indexed_at TEXT NOT NULL,
    UNIQUE(workspace, path)
);
CREATE TABLE IF NOT EXISTS document_chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    text TEXT NOT NULL,
    page INTEGER,
    heading TEXT,
    start_offset INTEGER NOT NULL,
    end_offset INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    UNIQUE(document_id, ordinal)
);
CREATE TABLE IF NOT EXISTS document_terms (
    chunk_id TEXT NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
    term TEXT NOT NULL,
    frequency INTEGER NOT NULL,
    PRIMARY KEY(chunk_id, term)
);
CREATE INDEX IF NOT EXISTS idx_documents_workspace ON documents(workspace, indexed_at DESC);
CREATE INDEX IF NOT EXISTS idx_document_chunks_document ON document_chunks(document_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_document_terms_term ON document_terms(term, chunk_id);
"#;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    pub id: String,
    pub status: String,
    pub workspace: String,
    pub updated_at: String,
    pub recent_task: String,
}

#[derive(Debug, Clone)]
pub struct SqliteSessionRepository {
    path: PathBuf,
}

impl SqliteSessionRepository {
    pub async fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref().to_path_buf();
        let migration_path = path.clone();
        tokio::task::spawn_blocking(move || migrate(&migration_path))
            .await
            .map_err(|error| error.to_string())??;
        Ok(Self { path })
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub async fn list_sessions(&self, limit: usize) -> Result<Vec<SessionSummary>, String> {
        let path = self.path.clone();
        tokio::task::spawn_blocking(move || {
            let connection = connect(&path)?;
            let mut statement = connection.prepare(
                "SELECT s.id, s.status, s.workspace, s.updated_at,
                    COALESCE((SELECT request FROM tasks t WHERE t.session_id=s.id ORDER BY t.updated_at DESC LIMIT 1), '')
                 FROM sessions s ORDER BY s.updated_at DESC LIMIT ?1",
            ).map_err(display)?;
            let rows = statement.query_map([i64::try_from(limit).unwrap_or(i64::MAX)], |row| {
                Ok(SessionSummary {
                    id: row.get(0)?, status: row.get(1)?, workspace: row.get(2)?,
                    updated_at: row.get(3)?, recent_task: row.get(4)?,
                })
            }).map_err(display)?;
            rows.collect::<Result<Vec<_>, _>>().map_err(display)
        }).await.map_err(|error| error.to_string())?
    }

    pub async fn show_session(&self, id: &str, limit: Option<usize>) -> Result<Value, String> {
        let path = self.path.clone();
        let id = id.to_owned();
        tokio::task::spawn_blocking(move || show(&path, &id, limit))
            .await
            .map_err(|error| error.to_string())?
    }

    pub async fn show_research(&self, id: &str, limit: Option<usize>) -> Result<Value, String> {
        let path = self.path.clone();
        let id = id.to_owned();
        tokio::task::spawn_blocking(move || research_view(&path, &id, limit))
            .await
            .map_err(|error| error.to_string())?
    }

    pub async fn load_latest(&self, id: &str) -> Result<AgentState, String> {
        let path = self.path.clone();
        let id = id.to_owned();
        tokio::task::spawn_blocking(move || load_and_normalize(&path, &id))
            .await
            .map_err(|error| error.to_string())?
    }

    pub async fn prune_count(&self, older_than_days: i64) -> Result<usize, String> {
        let path = self.path.clone();
        tokio::task::spawn_blocking(move || {
            let cutoff = (Utc::now() - Duration::days(older_than_days)).to_rfc3339();
            let connection = connect(&path)?;
            let count: i64 = connection.query_row(
                "SELECT COUNT(*) FROM sessions WHERE updated_at < ?1 AND status IN ('completed','failed','cancelled')",
                [cutoff], |row| row.get(0),
            ).map_err(display)?;
            usize::try_from(count).map_err(|error| error.to_string())
        }).await.map_err(|error| error.to_string())?
    }

    pub async fn prune(&self, older_than_days: i64) -> Result<usize, String> {
        let path = self.path.clone();
        tokio::task::spawn_blocking(move || {
            let cutoff = (Utc::now() - Duration::days(older_than_days)).to_rfc3339();
            let connection = connect(&path)?;
            connection.execute(
                "DELETE FROM sessions WHERE updated_at < ?1 AND status IN ('completed','failed','cancelled')",
                [cutoff],
            ).map_err(display)
        }).await.map_err(|error| error.to_string())?
    }
}

#[async_trait]
impl DocumentRepository for SqliteSessionRepository {
    async fn upsert(&self, document: &NormalizedDocument) -> Result<IndexResult, DocumentError> {
        let path = self.path.clone();
        let document = document.clone();
        tokio::task::spawn_blocking(move || upsert_document(&path, &document))
            .await
            .map_err(|e| DocumentError::Storage(e.to_string()))?
            .map_err(DocumentError::Storage)
    }

    async fn list(
        &self,
        workspace: &str,
        status: Option<DocumentStatus>,
        limit: usize,
    ) -> Result<Vec<DocumentSummary>, DocumentError> {
        let path = self.path.clone();
        let workspace = workspace.to_owned();
        tokio::task::spawn_blocking(move || list_documents(&path, &workspace, status, limit))
            .await
            .map_err(|e| DocumentError::Storage(e.to_string()))?
            .map_err(DocumentError::Storage)
    }

    async fn get(
        &self,
        workspace: &str,
        id: &str,
        chunks: bool,
    ) -> Result<Option<NormalizedDocument>, DocumentError> {
        let path = self.path.clone();
        let workspace = workspace.to_owned();
        let id = id.to_owned();
        tokio::task::spawn_blocking(move || get_document(&path, &workspace, &id, chunks))
            .await
            .map_err(|e| DocumentError::Storage(e.to_string()))?
            .map_err(DocumentError::Storage)
    }

    async fn search(
        &self,
        query: DocumentSearchQuery,
    ) -> Result<Vec<DocumentSearchHit>, DocumentError> {
        let path = self.path.clone();
        tokio::task::spawn_blocking(move || search_documents(&path, &query))
            .await
            .map_err(|e| DocumentError::Storage(e.to_string()))?
            .map_err(DocumentError::Storage)
    }
}

#[async_trait]
impl SessionRepository for SqliteSessionRepository {
    async fn checkpoint(
        &self,
        state: &AgentState,
        event: Option<&AgentEvent>,
    ) -> Result<(), String> {
        let path = self.path.clone();
        let state = state.clone();
        let event = event.cloned();
        tokio::task::spawn_blocking(move || checkpoint_sync(&path, &state, event.as_ref()))
            .await
            .map_err(|error| error.to_string())?
    }

    async fn relevant_memories(
        &self,
        workspace: &str,
        task: &str,
        limit: usize,
    ) -> Result<Vec<MemorySnippet>, String> {
        let path = self.path.clone();
        let workspace = workspace.to_owned();
        let terms = terms(task);
        tokio::task::spawn_blocking(move || {
            let connection = connect(&path)?;
            let mut statement = connection.prepare(
                "SELECT summary, search_text, created_at FROM memories WHERE workspace=?1 ORDER BY created_at DESC LIMIT 200"
            ).map_err(display)?;
            let rows = statement.query_map([workspace], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?))
            }).map_err(display)?;
            let mut ranked = rows.collect::<Result<Vec<_>, _>>().map_err(display)?
                .into_iter().map(|(summary, search, created)| {
                    let score = terms.iter().filter(|term| search.contains(term.as_str())).count();
                    (score, created, summary)
                }).filter(|(score, _, _)| *score > 0).collect::<Vec<_>>();
            ranked.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| right.1.cmp(&left.1)));
            Ok(ranked.into_iter().take(limit).map(|(score, _, summary)| MemorySnippet {
                summary, reason: format!("{score} task terms matched"),
            }).collect())
        }).await.map_err(|error| error.to_string())?
    }

    async fn store_memory(&self, state: &AgentState, answer: &str) -> Result<(), String> {
        let path = self.path.clone();
        let state = state.clone();
        let answer = answer.to_owned();
        tokio::task::spawn_blocking(move || {
            let connection = connect(&path)?;
            let observations = state.observations.iter().rev().take(6)
                .map(|item| item.summary.as_str()).collect::<Vec<_>>().join("; ");
            let summary = truncate(&format!("Task: {}\nResult: {}\nObservations: {}", state.task, answer, observations), 8_000);
            let search = format!("{} {}", state.task, summary).to_lowercase();
            connection.execute(
                "INSERT INTO memories(id,workspace,session_id,task_id,summary,search_text,created_at)
                 VALUES(?1,?2,?3,?4,?5,?6,?7) ON CONFLICT(task_id) DO NOTHING",
                params![Uuid::new_v4().to_string(), state.workspace, state.session_id.to_string(),
                    state.task_id.to_string(), summary, search, Utc::now().to_rfc3339()],
            ).map_err(display)?;
            Ok(())
        }).await.map_err(|error| error.to_string())?
    }
}

#[async_trait]
impl AuditSink for SqliteSessionRepository {
    async fn record(&self, mut event: AuditEvent) -> Result<(), SecurityError> {
        redact_value(&mut event.arguments);
        if let Some(metadata) = &mut event.metadata {
            redact_value(metadata);
        }
        let path = self.path.clone();
        tokio::task::spawn_blocking(move || {
            let connection = connect(&path)?;
            let json = serde_json::to_string(&event).map_err(display)?;
            connection.execute(
                "INSERT INTO audit_events(session_id,task_id,call_id,phase,event_json,created_at) VALUES(?1,?2,?3,?4,?5,?6)",
                params![event.session_id.to_string(), event.task_id.to_string(), event.call_id.to_string(),
                    enum_name(&event.phase)?, json, event.timestamp.to_rfc3339()],
            ).map_err(display)?;
            Ok::<_, String>(())
        }).await.map_err(|error| SecurityError::AuditStorage(error.to_string()))?
          .map_err(SecurityError::AuditStorage)
    }
}

fn migrate(path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(display)?;
    }
    let mut connection = connect(path)?;
    let transaction = connection.transaction().map_err(display)?;
    transaction.execute_batch(MIGRATION_1).map_err(display)?;
    transaction.execute_batch(MIGRATION_2).map_err(display)?;
    transaction
        .execute(
            "INSERT OR IGNORE INTO schema_migrations(version,applied_at) VALUES(1,?1)",
            [Utc::now().to_rfc3339()],
        )
        .map_err(display)?;
    transaction
        .execute(
            "INSERT OR IGNORE INTO schema_migrations(version,applied_at) VALUES(2,?1)",
            [Utc::now().to_rfc3339()],
        )
        .map_err(display)?;
    transaction.commit().map_err(display)
}

fn connect(path: &Path) -> Result<Connection, String> {
    let connection = Connection::open(path).map_err(display)?;
    connection
        .busy_timeout(std::time::Duration::from_secs(5))
        .map_err(display)?;
    connection
        .execute_batch("PRAGMA foreign_keys=ON; PRAGMA journal_mode=WAL;")
        .map_err(display)?;
    Ok(connection)
}

fn checkpoint_sync(
    path: &Path,
    state: &AgentState,
    event: Option<&AgentEvent>,
) -> Result<(), String> {
    let mut connection = connect(path)?;
    let transaction = connection.transaction().map_err(display)?;
    let now = Utc::now().to_rfc3339();
    let status = enum_name(&state.status)?;
    transaction.execute(
        "INSERT INTO sessions(id,workspace,status,created_at,updated_at) VALUES(?1,?2,?3,?4,?4)
         ON CONFLICT(id) DO UPDATE SET workspace=excluded.workspace,status=excluded.status,updated_at=excluded.updated_at",
        params![state.session_id.to_string(), state.workspace, status, now],
    ).map_err(display)?;
    transaction.execute(
        "INSERT INTO tasks(id,session_id,request,status,snapshot_json,iteration,tool_calls,consecutive_errors,started_at,updated_at)
         VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?9)
         ON CONFLICT(id) DO UPDATE SET status=excluded.status,snapshot_json=excluded.snapshot_json,
         iteration=excluded.iteration,tool_calls=excluded.tool_calls,consecutive_errors=excluded.consecutive_errors,updated_at=excluded.updated_at",
        params![state.task_id.to_string(), state.session_id.to_string(), state.task, status,
            serde_json::to_string(state).map_err(display)?, i64_value(state.iteration), i64_value(state.tool_calls),
            i64_value(state.consecutive_errors), now],
    ).map_err(display)?;
    transaction
        .execute(
            "DELETE FROM plans WHERE task_id=?1",
            [state.task_id.to_string()],
        )
        .map_err(display)?;
    for (ordinal, step) in state.plan.iter().enumerate() {
        transaction.execute(
            "INSERT INTO plans(task_id,ordinal,step_id,description,status) VALUES(?1,?2,?3,?4,?5)",
            params![state.task_id.to_string(), i64_value(ordinal), step.id.to_string(), step.description, enum_name(&step.status)?],
        ).map_err(display)?;
    }
    transaction
        .execute(
            "DELETE FROM messages WHERE task_id=?1",
            [state.task_id.to_string()],
        )
        .map_err(display)?;
    for (ordinal, message) in state.messages.iter().enumerate() {
        transaction
            .execute(
                "INSERT INTO messages(task_id,ordinal,message_json) VALUES(?1,?2,?3)",
                params![
                    state.task_id.to_string(),
                    i64_value(ordinal),
                    serde_json::to_string(message).map_err(display)?
                ],
            )
            .map_err(display)?;
    }
    if let Some(event) = event {
        let mut event_json = serde_json::to_value(event).map_err(display)?;
        redact_value(&mut event_json);
        transaction
            .execute(
                "INSERT INTO events(session_id,task_id,event_json,created_at) VALUES(?1,?2,?3,?4)",
                params![
                    state.session_id.to_string(),
                    state.task_id.to_string(),
                    event_json.to_string(),
                    now
                ],
            )
            .map_err(display)?;
        let event_id = transaction.last_insert_rowid();
        transaction
            .execute(
                "UPDATE sessions SET last_event_id=?1 WHERE id=?2",
                params![event_id, state.session_id.to_string()],
            )
            .map_err(display)?;
        apply_event(&transaction, state, event, &now)?;
    }
    transaction.commit().map_err(display)
}

fn apply_event(
    connection: &Connection,
    state: &AgentState,
    event: &AgentEvent,
    now: &str,
) -> Result<(), String> {
    match event {
        AgentEvent::ToolRequested {
            call_id,
            model_call_id,
            name,
            arguments,
            risk,
        } => {
            let mut args = arguments.clone();
            redact_value(&mut args);
            connection.execute(
                "INSERT OR REPLACE INTO tool_calls(call_id,task_id,model_call_id,tool_name,arguments_json,risk,status,updated_at)
                 VALUES(?1,?2,?3,?4,?5,?6,'requested',?7)",
                params![call_id.to_string(), state.task_id.to_string(), model_call_id, name, args.to_string(), enum_name(risk)?, now],
            ).map_err(display)?;
        }
        AgentEvent::ApprovalRequested { request } => {
            let value = serde_json::to_string(request).map_err(display)?;
            connection.execute(
                "INSERT OR REPLACE INTO approvals(approval_id,call_id,request_json,fingerprint,status,updated_at) VALUES(?1,?2,?3,?4,'pending',?5)",
                params![request.approval_id.to_string(), request.call_id.to_string(), value, request.fingerprint, now],
            ).map_err(display)?;
            connection.execute("UPDATE tool_calls SET status='awaiting_approval',updated_at=?1 WHERE call_id=?2", params![now, request.call_id.to_string()]).map_err(display)?;
        }
        AgentEvent::ApprovalResolved {
            approval_id,
            decision,
        } => {
            connection.execute(
                "UPDATE approvals SET decision_json=?1,status='resolved',updated_at=?2 WHERE approval_id=?3",
                params![serde_json::to_string(decision).map_err(display)?, now, approval_id.to_string()],
            ).map_err(display)?;
        }
        AgentEvent::ToolStarted { call_id } => {
            connection
                .execute(
                    "UPDATE tool_calls SET status='started',updated_at=?1 WHERE call_id=?2",
                    params![now, call_id.to_string()],
                )
                .map_err(display)?;
        }
        AgentEvent::ToolCompleted { call_id, result } => {
            connection.execute("UPDATE tool_calls SET status='completed',result_json=?1,updated_at=?2 WHERE call_id=?3", params![serde_json::to_string(result).map_err(display)?, now, call_id.to_string()]).map_err(display)?;
        }
        AgentEvent::ToolFailed { call_id, error } => {
            connection
                .execute(
                    "UPDATE tool_calls SET status='failed',error=?1,updated_at=?2 WHERE call_id=?3",
                    params![error, now, call_id.to_string()],
                )
                .map_err(display)?;
        }
        _ => {}
    }
    Ok(())
}

fn load_and_normalize(path: &Path, id: &str) -> Result<AgentState, String> {
    let mut connection = connect(path)?;
    let snapshot: String = connection
        .query_row(
            "SELECT snapshot_json FROM tasks WHERE session_id=?1 ORDER BY updated_at DESC LIMIT 1",
            [id],
            |row| row.get(0),
        )
        .optional()
        .map_err(display)?
        .ok_or_else(|| format!("session not found: {id}"))?;
    let mut state: AgentState = serde_json::from_str(&snapshot).map_err(display)?;
    if matches!(
        state.status,
        AgentStatus::Completed | AgentStatus::Failed | AgentStatus::Cancelled
    ) {
        return Ok(state);
    }
    let transaction = connection.transaction().map_err(display)?;
    let pending = {
        let mut statement = transaction.prepare(
            "SELECT call_id,model_call_id,status FROM tool_calls WHERE task_id=?1 AND status IN ('requested','awaiting_approval','started')"
        ).map_err(display)?;
        let rows = statement
            .query_map([state.task_id.to_string()], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .map_err(display)?;
        rows.collect::<Result<Vec<_>, _>>().map_err(display)?
    };
    let had_pending = !pending.is_empty();
    for (call, model_call, previous) in pending {
        let uuid = Uuid::parse_str(&call).map_err(display)?;
        let call_id = ToolCallId(uuid);
        let summary = format!(
            "tool call was interrupted during {previous}; it was not automatically re-executed"
        );
        let observation = Observation {
            tool_call_id: call_id,
            tool_name: String::new(),
            summary: summary.clone(),
            content: json!({"interrupted":true,"previous_status":previous}),
            truncated: false,
            is_error: false,
            workflow_phase: state.workflow_phase,
            failure: None,
        };
        state.messages.push(Message::tool(
            model_call,
            serde_json::to_string(&observation).map_err(display)?,
        ));
        state.observations.push(observation);
        transaction.execute("UPDATE tool_calls SET status='interrupted',error=?1,updated_at=?2 WHERE call_id=?3", params![summary, Utc::now().to_rfc3339(), call]).map_err(display)?;
        transaction.execute(
            "UPDATE approvals SET status='cancelled',decision_json=?1,updated_at=?2 WHERE call_id=?3 AND status='pending'",
            params![serde_json::to_string(&ApprovalDecision::Cancelled { decided_at: Utc::now() }).map_err(display)?, Utc::now().to_rfc3339(), call],
        ).map_err(display)?;
    }
    if had_pending {
        state.status = AgentStatus::Recovering;
        let state_json = serde_json::to_string(&state).map_err(display)?;
        transaction
            .execute(
                "UPDATE tasks SET snapshot_json=?1,status='recovering',updated_at=?2 WHERE id=?3",
                params![
                    state_json,
                    Utc::now().to_rfc3339(),
                    state.task_id.to_string()
                ],
            )
            .map_err(display)?;
        transaction
            .execute(
                "UPDATE sessions SET status='recovering',updated_at=?1 WHERE id=?2",
                params![Utc::now().to_rfc3339(), id],
            )
            .map_err(display)?;
    }
    transaction.commit().map_err(display)?;
    Ok(state)
}

fn show(path: &Path, id: &str, limit: Option<usize>) -> Result<Value, String> {
    let connection = connect(path)?;
    let session = connection.query_row(
        "SELECT id,workspace,status,created_at,updated_at,last_event_id FROM sessions WHERE id=?1", [id], |row| {
            Ok(json!({"id":row.get::<_,String>(0)?,"workspace":row.get::<_,String>(1)?,"status":row.get::<_,String>(2)?,
                "created_at":row.get::<_,String>(3)?,"updated_at":row.get::<_,String>(4)?,"last_event_id":row.get::<_,Option<i64>>(5)?}))
        }
    ).optional().map_err(display)?.ok_or_else(|| format!("session not found: {id}"))?;
    let row_limit = limit
        .and_then(|value| i64::try_from(value).ok())
        .unwrap_or(i64::MAX);
    let tasks = json_rows(
        &connection,
        "SELECT snapshot_json FROM tasks WHERE session_id=?1 ORDER BY updated_at DESC LIMIT ?2",
        id,
        row_limit,
    )?;
    let events = json_rows(
        &connection,
        "SELECT event_json FROM events WHERE session_id=?1 ORDER BY id DESC LIMIT ?2",
        id,
        row_limit,
    )?;
    let audit = json_rows(
        &connection,
        "SELECT event_json FROM audit_events WHERE session_id=?1 ORDER BY id DESC LIMIT ?2",
        id,
        row_limit,
    )?;
    let memories = text_rows(
        &connection,
        "SELECT summary FROM memories WHERE session_id=?1 ORDER BY created_at DESC LIMIT ?2",
        id,
        row_limit,
    )?;
    let tool_calls = tool_call_rows(&connection, id, row_limit)?;
    let approvals = approval_rows(&connection, id, row_limit)?;
    Ok(json!({
        "session":session,
        "tasks":tasks,
        "tool_calls":tool_calls,
        "approvals":approvals,
        "events":events,
        "audit":audit,
        "memories":memories
    }))
}

fn research_view(path: &Path, id: &str, limit: Option<usize>) -> Result<Value, String> {
    let connection = connect(path)?;
    let session = connection
        .query_row(
            "SELECT id,status,updated_at FROM sessions WHERE id=?1",
            [id],
            |row| {
                Ok(json!({
                    "id":row.get::<_,String>(0)?,
                    "status":row.get::<_,String>(1)?,
                    "updated_at":row.get::<_,String>(2)?
                }))
            },
        )
        .optional()
        .map_err(display)?
        .ok_or_else(|| format!("session not found: {id}"))?;
    let row_limit = limit
        .and_then(|value| i64::try_from(value).ok())
        .unwrap_or(i64::MAX);
    let mut entries = research_rows(&connection, id, row_limit)?;
    entries.reverse();
    Ok(json!({"session":session,"research":entries}))
}

fn research_rows(connection: &Connection, id: &str, limit: i64) -> Result<Vec<Value>, String> {
    let mut statement = connection
        .prepare(
            "SELECT tc.call_id,tc.tool_name,tc.arguments_json,tc.status,tc.result_json,tc.error,tc.updated_at
             FROM tool_calls tc JOIN tasks t ON t.id=tc.task_id
             WHERE t.session_id=?1 AND tc.tool_name IN ('web_search','http_fetch')
             ORDER BY tc.updated_at DESC LIMIT ?2",
        )
        .map_err(display)?;
    let rows = statement
        .query_map(params![id, limit], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, Option<String>>(4)?,
                row.get::<_, Option<String>>(5)?,
                row.get::<_, String>(6)?,
            ))
        })
        .map_err(display)?;
    rows.map(|row| {
        let (call_id, tool_name, arguments, status, result, error, updated_at) =
            row.map_err(display)?;
        let arguments = parse_json(&arguments)?;
        let result = parse_optional_json(result.as_deref())?;
        if tool_name == "web_search" {
            let sources = result["content"]["sources"]
                .as_array()
                .map(|values| {
                    values
                        .iter()
                        .map(|source| {
                            json!({
                                "rank":source["rank"],
                                "title":source["title"],
                                "url":source["url"],
                                "provider":source["provider"],
                                "engine":source["engine"],
                                "searched_at":source["searched_at"]
                            })
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            Ok(json!({
                "call_id":call_id,
                "kind":"web_search",
                "status":status,
                "updated_at":updated_at,
                "query":arguments["query"],
                "requested_limit":arguments["limit"],
                "provider":result["metadata"]["provider"],
                "result_count":result["metadata"]["result_count"],
                "limit_reached":result["metadata"]["limit_reached"],
                "searched_at":result["metadata"]["searched_at"],
                "skipped_duplicate":result["metadata"]["skipped_duplicate"],
                "sources":sources,
                "error":error
            }))
        } else {
            let source = if result["metadata"]["source"].is_object() {
                &result["metadata"]["source"]
            } else {
                &result["content"]["source"]
            };
            Ok(json!({
                "call_id":call_id,
                "kind":"http_fetch",
                "status":status,
                "updated_at":updated_at,
                "requested_url":arguments["url"],
                "final_url":source["final_url"],
                "title":source["title"],
                "fetched_at":source["fetched_at"],
                "content_type":source["content_type"],
                "redirects":source["redirects"],
                "received_bytes":result["metadata"]["received_bytes"],
                "error":error
            }))
        }
    })
    .collect()
}

fn tool_call_rows(connection: &Connection, id: &str, limit: i64) -> Result<Vec<Value>, String> {
    let mut statement = connection
        .prepare(
            "SELECT tc.call_id,tc.task_id,tc.model_call_id,tc.tool_name,tc.arguments_json,tc.risk,
                tc.status,tc.result_json,tc.error,tc.updated_at
         FROM tool_calls tc JOIN tasks t ON t.id=tc.task_id
         WHERE t.session_id=?1 ORDER BY tc.updated_at DESC LIMIT ?2",
        )
        .map_err(display)?;
    let rows = statement
        .query_map(params![id, limit], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, String>(6)?,
                row.get::<_, Option<String>>(7)?,
                row.get::<_, Option<String>>(8)?,
                row.get::<_, String>(9)?,
            ))
        })
        .map_err(display)?;
    rows.map(|row| {
        let (
            call_id,
            task_id,
            model_call_id,
            tool_name,
            arguments,
            risk,
            status,
            result,
            error,
            updated_at,
        ) = row.map_err(display)?;
        Ok(json!({
            "call_id":call_id, "task_id":task_id, "model_call_id":model_call_id,
            "tool_name":tool_name, "arguments":parse_json(&arguments)?, "risk":risk,
            "status":status, "result":parse_optional_json(result.as_deref())?,
            "error":error, "updated_at":updated_at
        }))
    })
    .collect()
}

fn approval_rows(connection: &Connection, id: &str, limit: i64) -> Result<Vec<Value>, String> {
    let mut statement = connection.prepare(
        "SELECT a.approval_id,a.call_id,a.request_json,a.fingerprint,a.decision_json,a.status,a.updated_at
         FROM approvals a JOIN tool_calls tc ON tc.call_id=a.call_id JOIN tasks t ON t.id=tc.task_id
         WHERE t.session_id=?1 ORDER BY a.updated_at DESC LIMIT ?2"
    ).map_err(display)?;
    let rows = statement
        .query_map(params![id, limit], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, Option<String>>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, String>(6)?,
            ))
        })
        .map_err(display)?;
    rows.map(|row| {
        let (approval_id, call_id, request, fingerprint, decision, status, updated_at) =
            row.map_err(display)?;
        Ok(json!({
            "approval_id":approval_id, "call_id":call_id, "request":parse_json(&request)?,
            "fingerprint":fingerprint, "decision":parse_optional_json(decision.as_deref())?,
            "status":status, "updated_at":updated_at
        }))
    })
    .collect()
}

fn parse_json(value: &str) -> Result<Value, String> {
    serde_json::from_str(value).map_err(display)
}

fn parse_optional_json(value: Option<&str>) -> Result<Value, String> {
    value
        .map(parse_json)
        .transpose()
        .map(|value| value.unwrap_or(Value::Null))
}

fn json_rows(
    connection: &Connection,
    sql: &str,
    id: &str,
    limit: i64,
) -> Result<Vec<Value>, String> {
    let mut statement = connection.prepare(sql).map_err(display)?;
    let rows = statement
        .query_map(params![id, limit], |row| row.get::<_, String>(0))
        .map_err(display)?;
    rows.map(|row| {
        row.map_err(display)
            .and_then(|text| serde_json::from_str(&text).map_err(display))
    })
    .collect()
}

fn text_rows(
    connection: &Connection,
    sql: &str,
    id: &str,
    limit: i64,
) -> Result<Vec<String>, String> {
    let mut statement = connection.prepare(sql).map_err(display)?;
    let rows = statement
        .query_map(params![id, limit], |row| row.get::<_, String>(0))
        .map_err(display)?;
    rows.collect::<Result<Vec<_>, _>>().map_err(display)
}

fn enum_name<T: Serialize>(value: &T) -> Result<String, String> {
    serde_json::to_string(value)
        .map(|value| value.trim_matches('"').to_owned())
        .map_err(display)
}

fn upsert_document(path: &Path, document: &NormalizedDocument) -> Result<IndexResult, String> {
    let mut connection = connect(path)?;
    let unchanged: bool=connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM documents WHERE workspace=?1 AND path=?2 AND content_hash=?3 AND status=?4)",
        params![document.workspace,document.source.path,document.source.content_hash,enum_name(&document.status)?], |row| row.get(0)).map_err(display)?;
    if unchanged {
        let summary = list_documents(path, &document.workspace, None, usize::MAX)?
            .into_iter()
            .find(|d| d.id == document.id)
            .ok_or_else(|| "unchanged document disappeared".to_owned())?;
        return Ok(IndexResult {
            document: summary,
            unchanged: true,
        });
    }
    let transaction = connection.transaction().map_err(display)?;
    let now = Utc::now().to_rfc3339();
    transaction
        .execute(
            "DELETE FROM documents WHERE workspace=?1 AND path=?2",
            params![document.workspace, document.source.path],
        )
        .map_err(display)?;
    transaction.execute("INSERT INTO documents(id,workspace,path,format,content_hash,title,status,error,metadata_json,normalized_text,byte_size,chunk_count,indexed_at) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13)",
        params![document.id,document.workspace,document.source.path,enum_name(&document.source.format)?,document.source.content_hash,document.title,
        enum_name(&document.status)?,document.error,serde_json::to_string(&document.metadata).map_err(display)?,document.text,
        i64_value(document.metadata.byte_size),i64_value(document.chunks.len()),now]).map_err(display)?;
    for chunk in &document.chunks {
        transaction.execute("INSERT INTO document_chunks(id,document_id,ordinal,text,page,heading,start_offset,end_offset,token_count) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9)",
            params![chunk.id,document.id,i64_value(chunk.ordinal),chunk.text,chunk.page.map(i64::from),chunk.heading,i64_value(chunk.start_offset),i64_value(chunk.end_offset),i64_value(chunk.token_count)]).map_err(display)?;
        for (term, frequency) in term_frequencies(&chunk.text) {
            transaction
                .execute(
                    "INSERT INTO document_terms(chunk_id,term,frequency) VALUES(?1,?2,?3)",
                    params![chunk.id, term, i64_value(frequency)],
                )
                .map_err(display)?;
        }
    }
    transaction.commit().map_err(display)?;
    Ok(IndexResult {
        document: DocumentSummary {
            id: document.id.clone(),
            path: document.source.path.clone(),
            format: document.source.format.clone(),
            title: document.title.clone(),
            status: document.status.clone(),
            error: document.error.clone(),
            byte_size: document.metadata.byte_size,
            chunk_count: document.chunks.len(),
            indexed_at: now,
        },
        unchanged: false,
    })
}

fn list_documents(
    path: &Path,
    workspace: &str,
    status: Option<DocumentStatus>,
    limit: usize,
) -> Result<Vec<DocumentSummary>, String> {
    let connection = connect(path)?;
    let status_name = status.as_ref().map(enum_name).transpose()?;
    let mut statement=connection.prepare("SELECT id,path,format,title,status,error,byte_size,chunk_count,indexed_at FROM documents WHERE workspace=?1 AND (?2 IS NULL OR status=?2) ORDER BY indexed_at DESC,path ASC LIMIT ?3").map_err(display)?;
    let rows = statement
        .query_map(params![workspace, status_name, i64_value(limit)], |row| {
            let found: DocumentStatus =
                serde_json::from_str(&format!("\"{}\"", row.get::<_, String>(4)?)).map_err(
                    |e| {
                        rusqlite::Error::FromSqlConversionFailure(
                            4,
                            rusqlite::types::Type::Text,
                            Box::new(e),
                        )
                    },
                )?;
            let format: DocumentFormat =
                serde_json::from_str(&format!("\"{}\"", row.get::<_, String>(2)?)).map_err(
                    |e| {
                        rusqlite::Error::FromSqlConversionFailure(
                            2,
                            rusqlite::types::Type::Text,
                            Box::new(e),
                        )
                    },
                )?;
            Ok(DocumentSummary {
                id: row.get(0)?,
                path: row.get(1)?,
                format,
                title: row.get(3)?,
                status: found,
                error: row.get(5)?,
                byte_size: usize_value(row.get(6)?),
                chunk_count: usize_value(row.get(7)?),
                indexed_at: row.get(8)?,
            })
        })
        .map_err(display)?;
    rows.collect::<Result<Vec<_>, _>>().map_err(display)
}

fn get_document(
    path: &Path,
    workspace: &str,
    id: &str,
    include_chunks: bool,
) -> Result<Option<NormalizedDocument>, String> {
    let connection = connect(path)?;
    let row=connection.query_row("SELECT path,format,content_hash,title,status,error,metadata_json,normalized_text FROM documents WHERE workspace=?1 AND id=?2",params![workspace,id],|row| Ok((row.get::<_,String>(0)?,row.get::<_,String>(1)?,row.get::<_,String>(2)?,row.get::<_,Option<String>>(3)?,row.get::<_,String>(4)?,row.get::<_,Option<String>>(5)?,row.get::<_,String>(6)?,row.get::<_,String>(7)?))).optional().map_err(display)?;
    let Some((source_path, format, status_hash, title, status, error, metadata, text)) = row else {
        return Ok(None);
    };
    let chunks = if include_chunks {
        let mut statement=connection.prepare("SELECT id,ordinal,text,page,heading,start_offset,end_offset,token_count FROM document_chunks WHERE document_id=?1 ORDER BY ordinal").map_err(display)?;
        statement
            .query_map([id], |row| {
                Ok(DocumentChunk {
                    id: row.get(0)?,
                    document_id: id.to_owned(),
                    ordinal: usize_value(row.get(1)?),
                    text: row.get(2)?,
                    page: row
                        .get::<_, Option<i64>>(3)?
                        .and_then(|v| u32::try_from(v).ok()),
                    heading: row.get(4)?,
                    start_offset: usize_value(row.get(5)?),
                    end_offset: usize_value(row.get(6)?),
                    token_count: usize_value(row.get(7)?),
                })
            })
            .map_err(display)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(display)?
    } else {
        Vec::new()
    };
    Ok(Some(NormalizedDocument {
        id: id.to_owned(),
        workspace: workspace.to_owned(),
        source: DocumentSource {
            path: source_path,
            format: parse_enum(&format)?,
            content_hash: status_hash,
        },
        title,
        metadata: serde_json::from_str(&metadata).map_err(display)?,
        status: parse_enum(&status)?,
        error,
        text,
        chunks,
    }))
}

fn search_documents(
    path: &Path,
    query: &DocumentSearchQuery,
) -> Result<Vec<DocumentSearchHit>, String> {
    let terms = tokenize(&query.query);
    if terms.is_empty() {
        return Ok(Vec::new());
    }
    let filters = query.document_ids.iter().cloned().collect::<HashSet<_>>();
    let connection = connect(path)?;
    let mut statement=connection.prepare("SELECT c.id,c.document_id,c.ordinal,c.text,c.page,c.heading,c.start_offset,c.end_offset,c.token_count,d.path FROM document_chunks c JOIN documents d ON d.id=c.document_id WHERE d.workspace=?1 AND d.status IN ('ready','partial')").map_err(display)?;
    #[derive(Clone)]
    struct Row {
        cid: String,
        did: String,
        ord: usize,
        text: String,
        page: Option<u32>,
        heading: Option<String>,
        start: usize,
        end: usize,
        len: usize,
        path: String,
        tf: HashMap<String, usize>,
    }
    let rows = statement
        .query_map([&query.workspace], |row| {
            let text: String = row.get(3)?;
            Ok(Row {
                cid: row.get(0)?,
                did: row.get(1)?,
                ord: usize_value(row.get(2)?),
                tf: term_frequencies(&text).into_iter().collect(),
                text,
                page: row
                    .get::<_, Option<i64>>(4)?
                    .and_then(|v| u32::try_from(v).ok()),
                heading: row.get(5)?,
                start: usize_value(row.get(6)?),
                end: usize_value(row.get(7)?),
                len: usize_value(row.get(8)?),
                path: row.get(9)?,
            })
        })
        .map_err(display)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(display)?
        .into_iter()
        .filter(|r| filters.is_empty() || filters.contains(&r.did))
        .collect::<Vec<_>>();
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    let n = rows.len() as f64;
    let avgdl = rows.iter().map(|r| r.len).sum::<usize>() as f64 / n;
    let mut df = HashMap::new();
    for term in &terms {
        df.insert(
            term.clone(),
            rows.iter().filter(|r| r.tf.contains_key(term)).count() as f64,
        );
    }
    let query_lower = query.query.to_lowercase();
    let mut hits = rows
        .into_iter()
        .filter_map(|r| {
            let mut score = 0.0;
            for term in &terms {
                let tf = *r.tf.get(term).unwrap_or(&0) as f64;
                if tf > 0.0 {
                    let d = *df.get(term).unwrap_or(&0.0);
                    let idf = ((n - d + 0.5) / (d + 0.5) + 1.0).ln();
                    score += idf * (tf * 2.2)
                        / (tf + 1.2 * (1.0 - 0.75 + 0.75 * r.len as f64 / avgdl.max(1.0)));
                }
            }
            if r.text.to_lowercase().contains(&query_lower) {
                score += 1.0
            }
            if score <= 0.0 {
                return None;
            }
            let citation = format!(
                "[{}{}{} @{}-{}]",
                r.path,
                r.page.map(|p| format!(" p.{p}")).unwrap_or_default(),
                r.heading
                    .as_ref()
                    .map(|h| format!(" § {h}"))
                    .unwrap_or_default(),
                r.start,
                r.end
            );
            Some(DocumentSearchHit {
                document_id: r.did,
                path: r.path,
                chunk_id: r.cid,
                ordinal: r.ord,
                page: r.page,
                heading: r.heading,
                start_offset: r.start,
                end_offset: r.end,
                score,
                excerpt: truncate(&r.text, 500),
                citation,
            })
        })
        .collect::<Vec<_>>();
    hits.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.path.cmp(&b.path))
            .then_with(|| a.page.cmp(&b.page))
            .then_with(|| a.ordinal.cmp(&b.ordinal))
    });
    hits.truncate(query.limit);
    Ok(hits)
}

fn parse_enum<T: for<'de> Deserialize<'de>>(value: &str) -> Result<T, String> {
    serde_json::from_str(&format!("\"{value}\"")).map_err(display)
}
fn usize_value(value: i64) -> usize {
    usize::try_from(value).unwrap_or(0)
}

fn terms(text: &str) -> BTreeSet<String> {
    text.to_lowercase()
        .split(|character: char| !character.is_alphanumeric() && character != '_')
        .filter(|term| term.chars().count() >= 2)
        .map(ToOwned::to_owned)
        .collect()
}

fn truncate(value: &str, max: usize) -> String {
    value.chars().take(max).collect()
}

fn i64_value(value: usize) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}
fn display(error: impl std::fmt::Display) -> String {
    error.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_core::WorkflowPhase;
    use agent_core::{PlanStep, StepStatus};
    use agent_security::{ApprovalRequest, AuditPhase, RiskLevel, SessionId, TaskId};
    use agent_tools::ToolResult;
    use std::collections::BTreeMap;

    #[tokio::test]
    async fn document_migration_upsert_and_bm25_search_are_workspace_scoped()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let repository = SqliteSessionRepository::open(temp.path().join("documents.db")).await?;
        let service = agent_document::DocumentService::new(Default::default())?;
        let first = service.parse(
            "workspace-a",
            "guide.md",
            "# Rust\nRust safety ownership ownership memory.".as_bytes(),
        )?;
        let second = service.parse(
            "workspace-a",
            "other.txt",
            "Cooking recipe ingredients and oven.".as_bytes(),
        )?;
        assert!(!repository.upsert(&first).await?.unchanged);
        assert!(repository.upsert(&first).await?.unchanged);
        repository.upsert(&second).await?;
        let hits = repository
            .search(DocumentSearchQuery {
                workspace: "workspace-a".into(),
                query: "Rust ownership".into(),
                document_ids: Vec::new(),
                limit: 10,
            })
            .await?;
        assert_eq!(
            hits.first().map(|hit| hit.document_id.as_str()),
            Some(first.id.as_str())
        );
        assert!(
            hits.first()
                .is_some_and(|hit| hit.citation.contains("guide.md"))
        );
        assert!(
            repository
                .search(DocumentSearchQuery {
                    workspace: "workspace-b".into(),
                    query: "Rust".into(),
                    document_ids: Vec::new(),
                    limit: 10
                })
                .await?
                .is_empty()
        );
        let versions = connect(repository.path())?.query_row(
            "SELECT COUNT(*) FROM schema_migrations",
            [],
            |row| row.get::<_, i64>(0),
        )?;
        assert_eq!(versions, 2);
        Ok(())
    }

    fn state(workspace: &Path) -> AgentState {
        AgentState {
            session_id: SessionId::new(),
            task_id: TaskId::new(),
            workspace: workspace.display().to_string(),
            task: "fix durable session".to_owned(),
            status: AgentStatus::Completed,
            plan: vec![PlanStep {
                id: Uuid::new_v4(),
                description: "fix".to_owned(),
                status: StepStatus::Completed,
            }],
            messages: vec![Message::user("fix durable session")],
            observations: Vec::new(),
            iteration: 1,
            tool_calls: 0,
            consecutive_errors: 0,
            workflow_phase: WorkflowPhase::Completed,
            change_sequence: 0,
            last_successful_verification: None,
            last_diff_review: None,
            failure_counts: BTreeMap::new(),
        }
    }

    #[tokio::test]
    async fn migration_checkpoint_memory_and_prune_are_consistent()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let repository = SqliteSessionRepository::open(temp.path().join("sessions.db")).await?;
        let value = state(temp.path());
        repository
            .checkpoint(
                &value,
                Some(&AgentEvent::TaskCompleted {
                    task_id: value.task_id,
                    answer: "done".to_owned(),
                }),
            )
            .await?;
        repository
            .store_memory(&value, "durable session fixed")
            .await?;
        assert_eq!(repository.list_sessions(10).await?.len(), 1);
        assert_eq!(
            repository
                .relevant_memories(&value.workspace, "durable session", 8)
                .await?
                .len(),
            1
        );
        assert_eq!(
            repository
                .load_latest(&value.session_id.to_string())
                .await?
                .task_id,
            value.task_id
        );
        SqliteSessionRepository::open(repository.path()).await?;
        {
            let connection = connect(repository.path())?;
            connection.execute(
                "UPDATE sessions SET updated_at='2000-01-01T00:00:00Z' WHERE id=?1",
                [value.session_id.to_string()],
            )?;
        }
        assert_eq!(repository.prune_count(30).await?, 1);
        assert_eq!(repository.prune(30).await?, 1);
        assert!(repository.list_sessions(10).await?.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn research_source_metadata_is_queryable_without_audit_body_duplication()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let repository = SqliteSessionRepository::open(temp.path().join("sessions.db")).await?;
        let value = state(temp.path());
        let search_call_id = ToolCallId::new();
        let search_arguments = json!({"query":"Rust 1.85","limit":2});
        repository
            .checkpoint(
                &value,
                Some(&AgentEvent::ToolRequested {
                    call_id: search_call_id,
                    model_call_id: "search-call".to_owned(),
                    name: "web_search".to_owned(),
                    arguments: search_arguments,
                    risk: RiskLevel::Read,
                }),
            )
            .await?;
        repository
            .checkpoint(
                &value,
                Some(&AgentEvent::ToolCompleted {
                    call_id: search_call_id,
                    result: ToolResult {
                        content: json!({
                            "kind":"web_search",
                            "query":"Rust 1.85",
                            "sources":[{
                                "rank":1,
                                "title":"Rust 1.85",
                                "url":"https://example.com/article",
                                "provider":"searxng",
                                "engine":"example",
                                "searched_at":"2026-09-02T00:00:00Z",
                                "snippet":"OMITTED-FROM-RESEARCH-VIEW"
                            }]
                        }),
                        summary: "web search returned 1 source".to_owned(),
                        truncated: false,
                        metadata: json!({
                            "kind":"web_search",
                            "query":"Rust 1.85",
                            "provider":"searxng",
                            "result_count":1,
                            "limit_reached":false,
                            "searched_at":"2026-09-02T00:00:00Z"
                        }),
                    },
                }),
            )
            .await?;
        let call_id = ToolCallId::new();
        let arguments = json!({"url":"https://example.com/article"});
        repository
            .checkpoint(
                &value,
                Some(&AgentEvent::ToolRequested {
                    call_id,
                    model_call_id: "fetch-call".to_owned(),
                    name: "http_fetch".to_owned(),
                    arguments: arguments.clone(),
                    risk: RiskLevel::Read,
                }),
            )
            .await?;
        let source = json!({
            "requested_url":"https://example.com/article",
            "final_url":"https://example.com/final",
            "fetched_at":"2026-09-02T00:00:00Z",
            "content_type":"text/html"
        });
        repository
            .checkpoint(
                &value,
                Some(&AgentEvent::ToolCompleted {
                    call_id,
                    result: ToolResult {
                        content: json!({"kind":"http_fetch","source":source,"text":"BODY-MUST-NOT-BE-IN-AUDIT"}),
                        summary: "fetched https://example.com/final".to_owned(),
                        truncated: false,
                        metadata: json!({"kind":"http_fetch","source":source}),
                    },
                }),
            )
            .await?;
        repository
            .record(AuditEvent {
                timestamp: Utc::now(),
                session_id: value.session_id,
                task_id: value.task_id,
                call_id,
                tool_name: "http_fetch".to_owned(),
                arguments,
                risk: RiskLevel::Read,
                phase: AuditPhase::Completed,
                approval: Some(ApprovalDecision::NotRequired),
                duration_ms: Some(1),
                summary: Some("fetched https://example.com/final".to_owned()),
                metadata: Some(json!({"kind":"http_fetch","source":source})),
                truncated: false,
                error: None,
            })
            .await?;
        let shown = repository
            .show_session(&value.session_id.to_string(), None)
            .await?;
        assert_eq!(
            shown["tool_calls"][0]["result"]["content"]["source"]["final_url"],
            "https://example.com/final"
        );
        assert!(
            shown["audit"]
                .to_string()
                .contains("https://example.com/final")
        );
        assert!(
            !shown["audit"]
                .to_string()
                .contains("BODY-MUST-NOT-BE-IN-AUDIT")
        );
        let research = repository
            .show_research(&value.session_id.to_string(), None)
            .await?;
        assert_eq!(research["research"].as_array().map(Vec::len), Some(2));
        assert!(research.to_string().contains("Rust 1.85"));
        assert!(research.to_string().contains("https://example.com/final"));
        assert!(!research.to_string().contains("BODY-MUST-NOT-BE-IN-AUDIT"));
        assert!(!research.to_string().contains("OMITTED-FROM-RESEARCH-VIEW"));
        Ok(())
    }

    #[tokio::test]
    async fn pending_approval_is_interrupted_once_without_execution()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let repository = SqliteSessionRepository::open(temp.path().join("sessions.db")).await?;
        let mut value = state(temp.path());
        value.status = AgentStatus::AwaitingApproval;
        value.workflow_phase = WorkflowPhase::Editing;
        value.plan[0].status = StepStatus::InProgress;
        let call_id = ToolCallId::new();
        let arguments =
            json!({"path":"src/lib.rs","content":"secret","api_token":"must-not-persist"});
        repository
            .checkpoint(
                &value,
                Some(&AgentEvent::ToolRequested {
                    call_id,
                    model_call_id: "model-call".to_owned(),
                    name: "write_file".to_owned(),
                    arguments: arguments.clone(),
                    risk: RiskLevel::Modify,
                }),
            )
            .await?;
        let request = ApprovalRequest::for_tool(
            call_id,
            "write_file",
            RiskLevel::Modify,
            &arguments,
            temp.path(),
        );
        repository
            .checkpoint(&value, Some(&AgentEvent::ApprovalRequested { request }))
            .await?;

        let loaded = repository
            .load_latest(&value.session_id.to_string())
            .await?;
        assert_eq!(loaded.status, AgentStatus::Recovering);
        assert_eq!(
            loaded
                .observations
                .iter()
                .filter(|item| item.content["interrupted"] == true)
                .count(),
            1
        );
        let loaded_again = repository
            .load_latest(&value.session_id.to_string())
            .await?;
        assert_eq!(
            loaded_again
                .observations
                .iter()
                .filter(|item| item.content["interrupted"] == true)
                .count(),
            1
        );
        let shown = repository
            .show_session(&value.session_id.to_string(), None)
            .await?;
        assert!(!shown.to_string().contains("must-not-persist"));
        assert_eq!(shown["tool_calls"][0]["status"], "interrupted");
        assert_eq!(shown["approvals"][0]["status"], "cancelled");
        Ok(())
    }
}
