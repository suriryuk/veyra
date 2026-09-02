use agent_context::MemorySnippet;
use agent_core::{AgentEvent, AgentState, AgentStatus, Observation, SessionRepository};
use agent_model::Message;
use agent_security::{
    ApprovalDecision, AuditEvent, AuditSink, SecurityError, ToolCallId, redact_value,
};
use async_trait::async_trait;
use chrono::{Duration, Utc};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeSet;
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
    transaction
        .execute(
            "INSERT OR IGNORE INTO schema_migrations(version,applied_at) VALUES(1,?1)",
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
    use agent_security::{ApprovalRequest, RiskLevel, SessionId, TaskId};
    use std::collections::BTreeMap;

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
