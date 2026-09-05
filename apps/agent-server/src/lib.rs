use agent_app::{AppError, ApplicationService, ContextProfileChoice};
use agent_document::DocumentSearchQuery;
use agent_security::{ApprovalId, SessionId, WorkspaceGuard};
use agent_storage::StoredEvent;
use axum::body::Body;
use axum::extract::{DefaultBodyLimit, Multipart, Path, Query, State};
use axum::http::{HeaderMap, Request, StatusCode};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::stream::{self, Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::Path as FsPath;
use tokio_stream::wrappers::BroadcastStream;
use tower_http::services::{ServeDir, ServeFile};
use uuid::Uuid;

#[derive(Clone)]
pub struct ApiState {
    pub app: ApplicationService,
    auth: AuthState,
}
#[derive(Clone)]
struct AuthState {
    required: bool,
    token: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ApiErrorBody {
    pub code: &'static str,
    pub message: String,
}
#[derive(Debug)]
pub struct ApiError(StatusCode, &'static str, String);
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.0,
            Json(ApiErrorBody {
                code: self.1,
                message: self.2,
            }),
        )
            .into_response()
    }
}
impl From<AppError> for ApiError {
    fn from(value: AppError) -> Self {
        match value {
            AppError::SessionNotFound(message) => {
                Self(StatusCode::NOT_FOUND, "session_not_found", message)
            }
            AppError::SessionBusy => Self(StatusCode::CONFLICT, "session_busy", value.to_string()),
            AppError::ApprovalNotFound => Self(
                StatusCode::NOT_FOUND,
                "approval_not_found",
                value.to_string(),
            ),
            AppError::ApprovalConflict => {
                Self(StatusCode::CONFLICT, "approval_conflict", value.to_string())
            }
            AppError::Config(message) | AppError::Security(message) => {
                Self(StatusCode::BAD_REQUEST, "invalid_request", message)
            }
            AppError::Storage(message) | AppError::Runtime(message) => {
                Self(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", message)
            }
        }
    }
}

#[derive(Debug, Deserialize)]
struct Paging {
    limit: Option<usize>,
}
#[derive(Debug, Deserialize)]
struct SessionQuery {
    session_id: String,
    path: Option<String>,
    limit: Option<usize>,
}
#[derive(Debug, Deserialize)]
struct EventQuery {
    after: Option<i64>,
}
#[derive(Debug, Deserialize)]
struct AuditQuery {
    session_id: Option<String>,
    limit: Option<usize>,
}
#[derive(Debug, Deserialize)]
pub struct CreateSession {
    pub workspace: Option<String>,
}
#[derive(Debug, Serialize)]
pub struct SessionCreated {
    pub id: String,
    pub workspace: String,
}
#[derive(Debug, Deserialize)]
pub struct SendMessage {
    pub message: String,
    pub context_profile: Option<ContextProfileChoice>,
    pub model: Option<String>,
}
#[derive(Debug, Deserialize)]
struct SearchDocuments {
    session_id: String,
    query: String,
    #[serde(default)]
    document_ids: Vec<String>,
    limit: Option<usize>,
}

pub fn validate_bind(
    bind: &str,
    allow_remote: bool,
    token: Option<String>,
) -> Result<(SocketAddr, Option<String>), AppError> {
    let address: SocketAddr = bind
        .parse()
        .map_err(|error| AppError::Config(format!("invalid server bind: {error}")))?;
    if !address.ip().is_loopback() {
        if !allow_remote {
            return Err(AppError::Config(
                "non-loopback bind requires server.allow_remote=true".into(),
            ));
        }
        if token.as_deref().is_none_or(str::is_empty) {
            return Err(AppError::Config(
                "non-loopback bind requires VEYRA_SERVER_TOKEN".into(),
            ));
        }
    }
    Ok((address, token))
}

pub fn router(
    app: ApplicationService,
    token: Option<String>,
    frontend: impl AsRef<FsPath>,
) -> Router {
    let state = ApiState {
        app,
        auth: AuthState {
            required: token.is_some(),
            token,
        },
    };
    let api = Router::new()
        .route("/sessions", post(create_session).get(list_sessions))
        .route("/sessions/{id}", get(show_session))
        .route("/sessions/{id}/messages", post(send_message))
        .route("/sessions/{id}/events", get(events))
        .route("/sessions/{id}/research", get(research))
        .route("/tasks/{id}", get(show_task))
        .route("/tasks/{id}/plan", get(task_plan))
        .route("/tasks/{id}/tool-calls", get(task_tool_calls))
        .route("/tasks/{id}/cancel", post(cancel_task))
        .route("/approvals/{id}/allow", post(allow_approval))
        .route("/approvals/{id}/deny", post(deny_approval))
        .route("/models", get(models))
        .route("/models/status", get(model_status))
        .route("/tools", get(tools))
        .route("/documents", get(documents).post(upload_document))
        .route("/documents/search", post(search_documents))
        .route("/audit", get(audit))
        .route("/workspace/tree", get(workspace_tree))
        .route("/workspace/file", get(workspace_file))
        .route("/workspace/image", get(workspace_image))
        .route("/workspace/git/status", get(git_status))
        .route("/workspace/git/diff", get(git_diff))
        .route("/openapi.json", get(openapi))
        .layer(DefaultBodyLimit::max(26 * 1024 * 1024))
        .layer(middleware::from_fn_with_state(
            state.auth.clone(),
            authorize,
        ));
    let index = frontend.as_ref().join("index.html");
    Router::new()
        .nest("/api/v1", api)
        .fallback_service(ServeDir::new(frontend).fallback(ServeFile::new(index)))
        .with_state(state)
}

async fn authorize(
    State(auth): State<AuthState>,
    headers: HeaderMap,
    request: Request<Body>,
    next: Next,
) -> Response {
    if !auth.required {
        return next.run(request).await;
    }
    let expected = auth.token.as_deref().map(|value| format!("Bearer {value}"));
    let actual = headers.get("authorization").and_then(|v| v.to_str().ok());
    if actual == expected.as_deref() {
        next.run(request).await
    } else {
        ApiError(
            StatusCode::UNAUTHORIZED,
            "unauthorized",
            "a valid bearer token is required".into(),
        )
        .into_response()
    }
}

async fn create_session(
    State(state): State<ApiState>,
    Json(input): Json<CreateSession>,
) -> Result<(StatusCode, Json<SessionCreated>), ApiError> {
    let id = state.app.create_session(input.workspace.as_deref()).await?;
    let value = state
        .app
        .database()
        .show_session(&id, Some(1))
        .await
        .map_err(internal)?;
    Ok((
        StatusCode::CREATED,
        Json(SessionCreated {
            id,
            workspace: value["session"]["workspace"]
                .as_str()
                .unwrap_or_default()
                .to_owned(),
        }),
    ))
}
async fn list_sessions(
    State(state): State<ApiState>,
    Query(query): Query<Paging>,
) -> Result<Json<Value>, ApiError> {
    let values = state
        .app
        .database()
        .list_sessions(query.limit.unwrap_or(50).min(200))
        .await
        .map_err(internal)?;
    Ok(Json(json!({"items":values})))
}
async fn show_session(
    State(state): State<ApiState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    state
        .app
        .database()
        .show_session(&id, Some(500))
        .await
        .map(Json)
        .map_err(|_| ApiError(StatusCode::NOT_FOUND, "session_not_found", id))
}
async fn send_message(
    State(state): State<ApiState>,
    Path(id): Path<String>,
    Json(input): Json<SendMessage>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    let started = state
        .app
        .start_task_with_model(
            parse_session(&id)?,
            input.message,
            input.context_profile,
            input.model,
        )
        .await?;
    Ok((StatusCode::ACCEPTED, Json(json!(started))))
}
async fn show_task(
    State(state): State<ApiState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    state
        .app
        .database()
        .show_task(&id)
        .await
        .map_err(internal)?
        .map(Json)
        .ok_or(ApiError(StatusCode::NOT_FOUND, "task_not_found", id))
}
async fn task_plan(
    State(state): State<ApiState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let task = task_value(&state, &id).await?;
    Ok(Json(json!({"task_id":id,"plan":task["plan"]})))
}
async fn task_tool_calls(
    State(state): State<ApiState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let task = task_value(&state, &id).await?;
    Ok(Json(
        json!({"task_id":id,"observations":task["observations"]}),
    ))
}
async fn cancel_task(
    State(state): State<ApiState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    state.app.cancel_task(&id).await?;
    Ok(StatusCode::ACCEPTED)
}
async fn allow_approval(
    State(state): State<ApiState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    state
        .app
        .resolve_approval(parse_approval(&id)?, true)
        .await?;
    Ok(StatusCode::NO_CONTENT)
}
async fn deny_approval(
    State(state): State<ApiState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    state
        .app
        .resolve_approval(parse_approval(&id)?, false)
        .await?;
    Ok(StatusCode::NO_CONTENT)
}
async fn models(State(state): State<ApiState>) -> Result<Json<Value>, ApiError> {
    let value = &state.app.config().model.routes;
    let items = state.app.selectable_models()?;
    let default = value
        .default
        .as_ref()
        .filter(|id| items.contains(id))
        .unwrap_or(&items[0]);
    Ok(Json(
        json!({"default":default,"large":value.large,"vision":value.vision,"items":items}),
    ))
}
async fn model_status(State(state): State<ApiState>) -> Result<Json<Value>, ApiError> {
    serde_json::to_value(state.app.model_status().await?)
        .map(Json)
        .map_err(internal)
}
async fn tools(State(state): State<ApiState>) -> Result<Json<Value>, ApiError> {
    serde_json::to_value(state.app.tools().await?)
        .map(|v| Json(json!({"items":v})))
        .map_err(internal)
}

async fn events(
    State(state): State<ApiState>,
    Path(id): Path<String>,
    Query(query): Query<EventQuery>,
    headers: HeaderMap,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, ApiError> {
    let after = query
        .after
        .or_else(|| {
            headers
                .get("last-event-id")
                .and_then(|v| v.to_str().ok())?
                .parse()
                .ok()
        })
        .unwrap_or(0);
    let replay = state
        .app
        .database()
        .events_after(&id, after, 10_000)
        .await
        .map_err(internal)?;
    let last = replay.last().map_or(after, |event| event.id);
    let session = id.clone();
    let live = BroadcastStream::new(state.app.subscribe()).filter_map(move |value| {
        let session = session.clone();
        async move {
            value
                .ok()
                .filter(|event| event.session_id == session && event.id > last)
        }
    });
    Ok(Sse::new(
        stream::iter(replay)
            .chain(live)
            .map(|value| Ok(stored_event(value))),
    )
    .keep_alive(KeepAlive::default()))
}
fn stored_event(value: StoredEvent) -> Event {
    let name = value.event["type"].as_str().unwrap_or("agent_event");
    let envelope = json!({"id":value.id,"type":name,"occurred_at":value.created_at,"session_id":value.session_id,"task_id":value.task_id,"payload":value.event});
    Event::default()
        .id(value.id.to_string())
        .event(name)
        .json_data(envelope)
        .unwrap_or_else(|_| Event::default().event("serialization_error"))
}

async fn documents(
    State(state): State<ApiState>,
    Query(query): Query<SessionQuery>,
) -> Result<Json<Value>, ApiError> {
    let workspace = session_workspace(&state, &query.session_id).await?;
    let values = state
        .app
        .documents(&workspace, None, query.limit.unwrap_or(50).min(200))
        .await?;
    Ok(Json(json!({"items":values})))
}
async fn upload_document(
    State(state): State<ApiState>,
    mut multipart: Multipart,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    let mut session_id = None;
    let mut files = Vec::new();
    while let Some(field) = multipart.next_field().await.map_err(bad_request)? {
        if field.name() == Some("session_id") {
            session_id = Some(field.text().await.map_err(bad_request)?);
            continue;
        }
        let name = field
            .file_name()
            .map(safe_filename)
            .unwrap_or_else(|| format!("upload-{}.bin", Uuid::new_v4()));
        let bytes = field.bytes().await.map_err(bad_request)?;
        files.push((name, bytes));
    }
    let session_id = session_id.ok_or_else(|| {
        ApiError(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "session_id is required".into(),
        )
    })?;
    let guard = session_guard(&state, &session_id).await?;
    let upload_dir = guard.resolve_new(".veyra/documents").map_err(security)?;
    tokio::fs::create_dir_all(&upload_dir)
        .await
        .map_err(internal)?;
    let mut indexed = Vec::new();
    for (name, bytes) in files {
        let relative = format!(".veyra/documents/{}-{name}", Uuid::new_v4());
        let target = guard.resolve_new(&relative).map_err(security)?;
        tokio::fs::write(&target, bytes).await.map_err(internal)?;
        indexed.push(
            state
                .app
                .index_document(guard.root().to_string_lossy().as_ref(), &relative)
                .await?,
        );
    }
    Ok((StatusCode::CREATED, Json(json!({"items":indexed}))))
}
async fn search_documents(
    State(state): State<ApiState>,
    Json(input): Json<SearchDocuments>,
) -> Result<Json<Value>, ApiError> {
    let workspace = session_workspace(&state, &input.session_id).await?;
    let values = state
        .app
        .search_documents(DocumentSearchQuery {
            workspace,
            query: input.query,
            document_ids: input.document_ids,
            limit: input.limit.unwrap_or(10).min(50),
        })
        .await?;
    Ok(Json(json!({"items":values})))
}
async fn research(
    State(state): State<ApiState>,
    Path(id): Path<String>,
    Query(query): Query<Paging>,
) -> Result<Json<Value>, ApiError> {
    state
        .app
        .database()
        .show_research(&id, Some(query.limit.unwrap_or(100).min(500)))
        .await
        .map(Json)
        .map_err(internal)
}
async fn audit(
    State(state): State<ApiState>,
    Query(query): Query<AuditQuery>,
) -> Result<Json<Value>, ApiError> {
    let items = state
        .app
        .database()
        .audit_events(
            query.session_id.as_deref(),
            query.limit.unwrap_or(100).min(500),
        )
        .await
        .map_err(internal)?;
    Ok(Json(json!({"items":items})))
}

async fn workspace_tree(
    State(state): State<ApiState>,
    Query(query): Query<SessionQuery>,
) -> Result<Json<Value>, ApiError> {
    let guard = session_guard(&state, &query.session_id).await?;
    let root = match query.path {
        Some(path) => guard.resolve_existing(path).map_err(security)?,
        None => guard.root().to_path_buf(),
    };
    if !root.is_dir() {
        return Err(ApiError(
            StatusCode::BAD_REQUEST,
            "not_directory",
            "path is not a directory".into(),
        ));
    }
    let mut entries = Vec::new();
    for entry in std::fs::read_dir(root).map_err(internal)?.take(500) {
        let entry = entry.map_err(internal)?;
        let ty = entry.file_type().map_err(internal)?;
        entries.push(json!({"name":entry.file_name().to_string_lossy(),"directory":ty.is_dir()}));
    }
    Ok(Json(json!({"items":entries})))
}
async fn workspace_image(
    State(state): State<ApiState>,
    Query(query): Query<SessionQuery>,
) -> Result<Response, ApiError> {
    use tokio::io::AsyncReadExt;
    let path = query.path.ok_or_else(|| bad_request("path is required"))?;
    let guard = session_guard(&state, &query.session_id).await?;
    let resolved = guard.resolve_existing(&path).map_err(security)?;
    let limits = state.app.config().vision.clone();
    let file = tokio::fs::File::open(resolved).await.map_err(internal)?;
    let mut bytes = Vec::new();
    file.take(limits.max_file_bytes as u64 + 1)
        .read_to_end(&mut bytes)
        .await
        .map_err(internal)?;
    if bytes.len() > limits.max_file_bytes {
        return Err(ApiError(
            StatusCode::PAYLOAD_TOO_LARGE,
            "image_too_large",
            path,
        ));
    }
    let decoded =
        tokio::task::spawn_blocking(move || agent_vision::decode_image(path, None, bytes, &limits))
            .await
            .map_err(internal)?
            .map_err(|e| {
                ApiError(
                    StatusCode::UNSUPPORTED_MEDIA_TYPE,
                    "invalid_image",
                    e.to_string(),
                )
            })?;
    Ok((
        [
            ("content-type", decoded.mime_type),
            ("x-content-type-options", "nosniff".into()),
            ("cache-control", "private, no-store".into()),
        ],
        decoded.bytes,
    )
        .into_response())
}

async fn workspace_file(
    State(state): State<ApiState>,
    Query(query): Query<SessionQuery>,
) -> Result<Json<Value>, ApiError> {
    let path = query.path.ok_or_else(|| {
        ApiError(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "path is required".into(),
        )
    })?;
    let guard = session_guard(&state, &query.session_id).await?;
    let resolved = guard.resolve_existing(&path).map_err(security)?;
    let bytes = tokio::fs::read(&resolved).await.map_err(internal)?;
    if bytes.len() > state.app.config().tools.file_read_limit_bytes {
        return Err(ApiError(
            StatusCode::PAYLOAD_TOO_LARGE,
            "file_too_large",
            path,
        ));
    }
    let content = String::from_utf8(bytes).map_err(|_| {
        ApiError(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "binary_file",
            path.clone(),
        )
    })?;
    Ok(Json(json!({"path":path,"content":content})))
}
async fn git_status(
    State(state): State<ApiState>,
    Query(query): Query<SessionQuery>,
) -> Result<Json<Value>, ApiError> {
    git_output(&state, &query.session_id, &["status", "--short"]).await
}
async fn git_diff(
    State(state): State<ApiState>,
    Query(query): Query<SessionQuery>,
) -> Result<Json<Value>, ApiError> {
    git_output(&state, &query.session_id, &["diff", "--no-ext-diff"]).await
}
async fn git_output(
    state: &ApiState,
    session: &str,
    args: &[&str],
) -> Result<Json<Value>, ApiError> {
    let workspace = session_workspace(state, session).await?;
    let output = tokio::process::Command::new("git")
        .args(args)
        .current_dir(workspace)
        .output()
        .await
        .map_err(internal)?;
    Ok(Json(
        json!({"success":output.status.success(),"output":String::from_utf8_lossy(&output.stdout)}),
    ))
}

async fn openapi() -> Json<Value> {
    Json(openapi_document())
}
const OPENAPI_OPERATIONS: &[(&str, &str, &str)] = &[
    (
        "get",
        "/api/v1/workspace/image",
        "Read a bounded workspace image",
    ),
    (
        "post",
        "/api/v1/sessions",
        "Create a workspace-confined session",
    ),
    ("get", "/api/v1/sessions", "List sessions"),
    ("get", "/api/v1/sessions/{id}", "Read session history"),
    ("post", "/api/v1/sessions/{id}/messages", "Start a task"),
    (
        "get",
        "/api/v1/sessions/{id}/events",
        "Replay and stream SSE events",
    ),
    ("get", "/api/v1/tasks/{id}", "Read a task"),
    ("get", "/api/v1/tasks/{id}/plan", "Read a task plan"),
    ("get", "/api/v1/tasks/{id}/tool-calls", "Read Tool activity"),
    ("post", "/api/v1/tasks/{id}/cancel", "Cancel an active task"),
    (
        "post",
        "/api/v1/approvals/{id}/allow",
        "Allow an approval once",
    ),
    ("post", "/api/v1/approvals/{id}/deny", "Deny an approval"),
    ("get", "/api/v1/models/status", "Read model fleet status"),
    ("get", "/api/v1/tools", "List registered Tools"),
    ("get", "/api/v1/documents", "List indexed documents"),
    ("post", "/api/v1/documents", "Upload and index documents"),
    ("post", "/api/v1/documents/search", "Search documents"),
    ("get", "/api/v1/audit", "Read redacted audit events"),
    ("get", "/api/v1/workspace/tree", "Browse the workspace"),
    (
        "get",
        "/api/v1/workspace/file",
        "Preview a bounded text file",
    ),
    ("get", "/api/v1/workspace/git/status", "Read Git status"),
    ("get", "/api/v1/workspace/git/diff", "Read Git diff"),
];

pub fn openapi_document() -> Value {
    let mut paths = serde_json::Map::new();
    for (method, path, summary) in OPENAPI_OPERATIONS {
        paths.entry((*path).to_owned()).or_insert_with(|| json!({}))[*method] = json!({
            "summary":summary,
            "responses":{"200":{"description":"Success"},"400":{"description":"Invalid request"},"401":{"description":"Unauthorized"},"409":{"description":"Conflict"}}
        });
    }
    json!({
        "openapi":"3.1.0",
        "info":{"title":"Veyra Agent API","version":"0.9.0"},
        "components":{"securitySchemes":{"bearerAuth":{"type":"http","scheme":"bearer"}},"schemas":{"AgentEventEnvelope":{"type":"object","required":["id","type","occurred_at","session_id","payload"],"properties":{"id":{"type":"integer"},"type":{"type":"string"},"occurred_at":{"type":"string","format":"date-time"},"session_id":{"type":"string","format":"uuid"},"task_id":{"type":["string","null"],"format":"uuid"},"payload":{"type":"object"}}},"ApiError":{"type":"object","required":["code","message"],"properties":{"code":{"type":"string"},"message":{"type":"string"}}}}},
        "paths":paths
    })
}

async fn task_value(state: &ApiState, id: &str) -> Result<Value, ApiError> {
    state
        .app
        .database()
        .show_task(id)
        .await
        .map_err(internal)?
        .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, "task_not_found", id.to_owned()))
}
async fn session_workspace(state: &ApiState, id: &str) -> Result<String, ApiError> {
    let value = state
        .app
        .database()
        .show_session(id, Some(1))
        .await
        .map_err(|_| ApiError(StatusCode::NOT_FOUND, "session_not_found", id.into()))?;
    value["session"]["workspace"]
        .as_str()
        .map(ToOwned::to_owned)
        .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, "session_not_found", id.into()))
}
async fn session_guard(state: &ApiState, id: &str) -> Result<WorkspaceGuard, ApiError> {
    WorkspaceGuard::new(session_workspace(state, id).await?).map_err(security)
}
fn parse_session(value: &str) -> Result<SessionId, ApiError> {
    Uuid::parse_str(value).map(SessionId).map_err(bad_request)
}
fn parse_approval(value: &str) -> Result<ApprovalId, ApiError> {
    Uuid::parse_str(value).map(ApprovalId).map_err(bad_request)
}
fn safe_filename(value: &str) -> String {
    value
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect()
}
fn internal(error: impl std::fmt::Display) -> ApiError {
    ApiError(
        StatusCode::INTERNAL_SERVER_ERROR,
        "internal_error",
        error.to_string(),
    )
}
fn bad_request(error: impl std::fmt::Display) -> ApiError {
    ApiError(
        StatusCode::BAD_REQUEST,
        "invalid_request",
        error.to_string(),
    )
}
fn security(error: impl std::fmt::Display) -> ApiError {
    ApiError(
        StatusCode::BAD_REQUEST,
        "workspace_violation",
        error.to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_app::AppConfig;
    use tower::ServiceExt;
    #[test]
    fn remote_bind_requires_policy_and_token() {
        assert!(validate_bind("0.0.0.0:3000", false, Some("x".into())).is_err());
        assert!(validate_bind("0.0.0.0:3000", true, None).is_err());
        assert!(validate_bind("0.0.0.0:3000", true, Some("secret".into())).is_ok());
        assert!(validate_bind("127.0.0.1:3000", false, None).is_ok());
    }
    #[test]
    fn uploaded_names_are_confined() {
        assert_eq!(safe_filename("../../secret.md"), ".._.._secret.md");
    }

    #[tokio::test]
    async fn image_endpoint_rejects_escape_and_non_images() -> Result<(), Box<dyn std::error::Error>>
    {
        let temp = tempfile::tempdir()?;
        let mut config = AppConfig::default();
        config.security.workspace_root = temp.path().join("workspace");
        config.storage.database_path = temp.path().join("veyra.db");
        config.logging.directory = temp.path().join("logs");
        let service = ApplicationService::open(config).await?;
        let session = service.create_session(Some(".")).await?;
        std::fs::write(temp.path().join("workspace/fake.png"), b"not an image")?;
        let app = router(service, Some("secret".into()), temp.path());
        for (path, expected) in [
            ("..%2Fveyra.db", StatusCode::BAD_REQUEST),
            ("fake.png", StatusCode::UNSUPPORTED_MEDIA_TYPE),
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri(format!(
                            "/api/v1/workspace/image?session_id={session}&path={path}"
                        ))
                        .header("authorization", "Bearer secret")
                        .body(Body::empty())?,
                )
                .await?;
            assert_eq!(response.status(), expected);
        }
        let response = app
            .oneshot(
                Request::builder()
                    .uri(format!(
                        "/api/v1/workspace/image?session_id={session}&path=fake.png"
                    ))
                    .body(Body::empty())?,
            )
            .await?;
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        Ok(())
    }

    #[test]
    fn generated_openapi_contains_versioned_operations() {
        let document = openapi_document();
        assert_eq!(document["info"]["version"], "0.9.0");
        assert!(document["paths"]["/api/v1/sessions"]["post"].is_object());
        assert!(document["components"]["schemas"]["AgentEventEnvelope"].is_object());
    }

    #[tokio::test]
    async fn bearer_auth_protects_the_api() -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let mut config = AppConfig::default();
        config.security.workspace_root = temp.path().join("workspace");
        config.storage.database_path = temp.path().join("veyra.db");
        config.logging.directory = temp.path().join("logs");
        let service = ApplicationService::open(config).await?;
        let app = router(service, Some("secret".into()), temp.path());
        let denied = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/v1/sessions")
                    .body(Body::empty())?,
            )
            .await?;
        assert_eq!(denied.status(), StatusCode::UNAUTHORIZED);
        let allowed = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/sessions")
                    .header("authorization", "Bearer secret")
                    .body(Body::empty())?,
            )
            .await?;
        assert_eq!(allowed.status(), StatusCode::OK);
        Ok(())
    }
}
