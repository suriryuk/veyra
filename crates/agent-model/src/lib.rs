use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use reqwest::{Client, Url};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use thiserror::Error;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<WireToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    #[must_use]
    pub fn system(content: impl Into<String>) -> Self {
        Self::plain(Role::System, content)
    }
    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self::plain(Role::User, content)
    }
    #[must_use]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::plain(Role::Assistant, content)
    }
    #[must_use]
    pub fn tool(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: Some(id.into()),
        }
    }
    #[must_use]
    pub fn assistant_response(content: Option<String>, calls: &[RequestedToolCall]) -> Self {
        let tool_calls =
            (!calls.is_empty()).then(|| calls.iter().map(WireToolCall::from).collect());
        Self {
            role: Role::Assistant,
            content,
            tool_calls,
            tool_call_id: None,
        }
    }
    fn plain(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: WireFunction,
}

impl From<&RequestedToolCall> for WireToolCall {
    fn from(call: &RequestedToolCall) -> Self {
        Self {
            id: call.id.clone(),
            kind: "function".to_owned(),
            function: WireFunction {
                name: call.name.clone(),
                arguments: call.arguments.to_string(),
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionDefinition,
}

impl ToolDefinition {
    #[must_use]
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
    ) -> Self {
        Self {
            kind: "function".to_owned(),
            function: FunctionDefinition {
                name: name.into(),
                description: description.into(),
                parameters,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub repeat_penalty: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.3,
            top_p: 0.8,
            top_k: 20,
            repeat_penalty: 1.05,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelRequest {
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub sampling: SamplingConfig,
    pub max_output_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestedToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TokenUsage {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    ToolCalls,
    Length,
    Other(String),
}

#[derive(Debug, Clone)]
pub struct ModelResponse {
    pub content: Option<String>,
    pub tool_calls: Vec<RequestedToolCall>,
    pub usage: Option<TokenUsage>,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelHealth {
    pub available: bool,
    pub detail: String,
}

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("invalid model URL: {0}")]
    InvalidUrl(String),
    #[error("model request failed: {0}")]
    Transport(String),
    #[error("model returned HTTP {status}: {body}")]
    Http { status: u16, body: String },
    #[error("model stream was idle for too long")]
    IdleTimeout,
    #[error("malformed model stream: {0}")]
    MalformedStream(String),
    #[error("malformed tool arguments for {tool}: {detail}")]
    MalformedToolArguments { tool: String, detail: String },
    #[error("model router returned malformed data: {0}")]
    MalformedRouter(String),
    #[error("model is not configured for route {0}")]
    RouteUnavailable(String),
    #[error("model {model} did not become ready within {seconds} seconds")]
    LoadTimeout { model: String, seconds: u64 },
    #[error("model {model} does not support required modality {modality}")]
    Capability { model: String, modality: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ModelId(pub String);

impl ModelId {
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelProfile {
    Default,
    Large,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelRoute {
    Default,
    Large,
    Vision,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ModelCapabilities {
    pub text: bool,
    pub image: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RoutedModelStatus {
    pub route: ModelRoute,
    pub model: Option<ModelId>,
    pub status: String,
    pub capabilities: ModelCapabilities,
    pub failed: bool,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFleetHealth {
    pub available: bool,
    pub routes: Vec<RoutedModelStatus>,
}

#[derive(Debug, Clone)]
pub struct ModelRoutes {
    pub default: ModelId,
    pub large: ModelId,
    pub vision: Option<ModelId>,
}

#[async_trait]
pub trait ModelManager: Send + Sync {
    async fn health(&self) -> Result<ModelFleetHealth, ModelError>;
    async fn switch_model(&self, model: &ModelId) -> Result<(), ModelError>;
    async fn switch_profile(&self, profile: ModelProfile) -> Result<ModelId, ModelError>;
    fn model_for(&self, route: ModelRoute) -> Option<ModelId>;
}

impl ModelError {
    #[must_use]
    pub fn is_context_overflow(&self) -> bool {
        let Self::Http { status, body } = self else {
            return false;
        };
        if !matches!(*status, 400 | 413) {
            return false;
        }
        let body = body.to_ascii_lowercase();
        [
            "context length",
            "context window",
            "context size",
            "too many tokens",
            "maximum context",
            "prompt is too long",
            "exceeds the available context",
        ]
        .iter()
        .any(|marker| body.contains(marker))
    }
}

#[async_trait]
pub trait ModelEventSink: Send + Sync {
    async fn token(&self, text: &str);
}

#[async_trait]
pub trait ModelProvider: Send + Sync {
    async fn complete(
        &self,
        request: ModelRequest,
        events: &dyn ModelEventSink,
    ) -> Result<ModelResponse, ModelError>;
    async fn health(&self) -> Result<ModelHealth, ModelError>;
}

#[derive(Clone)]
pub struct OpenAiCompatibleProvider {
    client: Client,
    base_url: Url,
    model: String,
    idle_timeout: Duration,
}

impl OpenAiCompatibleProvider {
    pub fn new(
        base_url: &str,
        model: impl Into<String>,
        request_timeout: Duration,
    ) -> Result<Self, ModelError> {
        let base_url =
            Url::parse(base_url).map_err(|error| ModelError::InvalidUrl(error.to_string()))?;
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(request_timeout)
            .build()
            .map_err(|error| ModelError::Transport(error.to_string()))?;
        Ok(Self {
            client,
            base_url,
            model: model.into(),
            idle_timeout: Duration::from_secs(30),
        })
    }

    fn endpoint(&self, suffix: &str) -> Result<Url, ModelError> {
        let mut base = self.base_url.clone();
        let path = format!(
            "{}/{}",
            base.path().trim_end_matches('/'),
            suffix.trim_start_matches('/')
        );
        base.set_path(&path);
        Ok(base)
    }

    fn health_url(&self) -> Url {
        let mut url = self.base_url.clone();
        url.set_path("/health");
        url
    }
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [Message],
    tools: &'a [ToolDefinition],
    stream: bool,
    stream_options: StreamOptions,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    repeat_penalty: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

#[derive(Serialize)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(Debug, Deserialize)]
struct StreamEnvelope {
    choices: Vec<StreamChoice>,
    usage: Option<TokenUsage>,
}
#[derive(Debug, Deserialize)]
struct StreamChoice {
    delta: StreamDelta,
    finish_reason: Option<String>,
}
#[derive(Debug, Deserialize)]
struct StreamDelta {
    content: Option<String>,
    tool_calls: Option<Vec<ToolCallDelta>>,
}
#[derive(Debug, Deserialize)]
struct ToolCallDelta {
    index: usize,
    id: Option<String>,
    function: Option<FunctionDelta>,
}
#[derive(Debug, Deserialize)]
struct FunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Default)]
struct ToolAccumulator {
    id: String,
    name: String,
    arguments: String,
}

#[async_trait]
impl ModelProvider for OpenAiCompatibleProvider {
    async fn complete(
        &self,
        request: ModelRequest,
        events: &dyn ModelEventSink,
    ) -> Result<ModelResponse, ModelError> {
        let body = ChatRequest {
            model: &self.model,
            messages: &request.messages,
            tools: &request.tools,
            stream: true,
            stream_options: StreamOptions {
                include_usage: true,
            },
            temperature: request.sampling.temperature,
            top_p: request.sampling.top_p,
            top_k: request.sampling.top_k,
            repeat_penalty: request.sampling.repeat_penalty,
            max_tokens: request.max_output_tokens,
        };
        let endpoint = self.endpoint("chat/completions")?;
        let mut attempt = 0_u32;
        let response = loop {
            match self.client.post(endpoint.clone()).json(&body).send().await {
                Ok(response) if response.status().is_success() => break response,
                Ok(response) if response.status().is_server_error() && attempt < 2 => {
                    attempt += 1;
                    tokio::time::sleep(Duration::from_millis(100 * u64::from(1_u32 << attempt)))
                        .await;
                }
                Ok(response) => return http_error(response).await,
                Err(error) if (error.is_connect() || error.is_timeout()) && attempt < 2 => {
                    attempt += 1;
                    tokio::time::sleep(Duration::from_millis(100 * u64::from(1_u32 << attempt)))
                        .await;
                }
                Err(error) => return Err(ModelError::Transport(error.to_string())),
            }
        };
        let mut stream = response.bytes_stream().eventsource();
        let mut content = String::new();
        let mut calls: BTreeMap<usize, ToolAccumulator> = BTreeMap::new();
        let mut finish = FinishReason::Stop;
        let mut usage = None;
        loop {
            let next = tokio::time::timeout(self.idle_timeout, stream.next())
                .await
                .map_err(|_| ModelError::IdleTimeout)?;
            let Some(event) = next else { break };
            let event = event.map_err(|error| ModelError::MalformedStream(error.to_string()))?;
            if event.data.trim() == "[DONE]" {
                break;
            }
            let envelope: StreamEnvelope = serde_json::from_str(&event.data)
                .map_err(|error| ModelError::MalformedStream(error.to_string()))?;
            if envelope.usage.is_some() {
                usage = envelope.usage;
            }
            for choice in envelope.choices {
                if let Some(text) = choice.delta.content {
                    events.token(&text).await;
                    content.push_str(&text);
                }
                if let Some(tool_deltas) = choice.delta.tool_calls {
                    for delta in tool_deltas {
                        let item = calls.entry(delta.index).or_default();
                        if let Some(id) = delta.id {
                            item.id.push_str(&id);
                        }
                        if let Some(function) = delta.function {
                            if let Some(name) = function.name {
                                item.name.push_str(&name);
                            }
                            if let Some(arguments) = function.arguments {
                                item.arguments.push_str(&arguments);
                            }
                        }
                    }
                }
                if let Some(reason) = choice.finish_reason {
                    finish = parse_finish_reason(&reason);
                }
            }
        }
        let tool_calls = calls
            .into_values()
            .map(|call| {
                let arguments = serde_json::from_str(&call.arguments).map_err(|error| {
                    ModelError::MalformedToolArguments {
                        tool: call.name.clone(),
                        detail: error.to_string(),
                    }
                })?;
                Ok(RequestedToolCall {
                    id: call.id,
                    name: call.name,
                    arguments,
                })
            })
            .collect::<Result<Vec<_>, ModelError>>()?;
        Ok(ModelResponse {
            content: (!content.is_empty()).then_some(content),
            tool_calls,
            usage,
            finish_reason: finish,
        })
    }

    async fn health(&self) -> Result<ModelHealth, ModelError> {
        let response = self
            .client
            .get(self.health_url())
            .send()
            .await
            .map_err(|error| ModelError::Transport(error.to_string()))?;
        let status = response.status();
        let detail = response.text().await.unwrap_or_else(|_| status.to_string());
        if status.is_success() {
            Ok(ModelHealth {
                available: true,
                detail,
            })
        } else {
            Err(ModelError::Http {
                status: status.as_u16(),
                body: truncate(&detail, 1024),
            })
        }
    }
}

#[derive(Clone)]
pub struct RouterModelManager {
    client: Client,
    base_url: Url,
    routes: ModelRoutes,
    load_timeout: Duration,
    current_profile: Arc<RwLock<ModelProfile>>,
}

impl RouterModelManager {
    pub fn new(
        base_url: &str,
        routes: ModelRoutes,
        request_timeout: Duration,
        load_timeout: Duration,
    ) -> Result<Self, ModelError> {
        let base_url =
            Url::parse(base_url).map_err(|error| ModelError::InvalidUrl(error.to_string()))?;
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(request_timeout)
            .build()
            .map_err(|error| ModelError::Transport(error.to_string()))?;
        Ok(Self {
            client,
            base_url,
            routes,
            load_timeout,
            current_profile: Arc::new(RwLock::new(ModelProfile::Default)),
        })
    }

    fn router_endpoint(&self, path: &str) -> Url {
        let mut url = self.base_url.clone();
        url.set_path(path);
        url.set_query(None);
        url
    }

    pub fn provider(
        &self,
        route: ModelRoute,
        request_timeout: Duration,
    ) -> Result<OpenAiCompatibleProvider, ModelError> {
        let model = self
            .model_for(route)
            .ok_or_else(|| ModelError::RouteUnavailable(format!("{route:?}").to_lowercase()))?;
        OpenAiCompatibleProvider::new(self.base_url.as_str(), model.0, request_timeout)
    }

    async fn raw_models(&self) -> Result<Vec<Value>, ModelError> {
        let response = self
            .client
            .get(self.router_endpoint("/models"))
            .send()
            .await
            .map_err(|error| ModelError::Transport(error.to_string()))?;
        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|error| ModelError::Transport(error.to_string()))?;
        if !status.is_success() {
            return Err(ModelError::Http {
                status: status.as_u16(),
                body: truncate(&body, 4096),
            });
        }
        let value: Value = serde_json::from_str(&body)
            .map_err(|error| ModelError::MalformedRouter(error.to_string()))?;
        value
            .get("data")
            .and_then(Value::as_array)
            .cloned()
            .ok_or_else(|| ModelError::MalformedRouter("missing models.data array".to_owned()))
    }

    fn status_for(
        route: ModelRoute,
        model: Option<ModelId>,
        entries: &[Value],
    ) -> RoutedModelStatus {
        let Some(model) = model else {
            return RoutedModelStatus {
                route,
                model: None,
                status: "unconfigured".to_owned(),
                capabilities: ModelCapabilities::default(),
                failed: false,
                detail: None,
            };
        };
        let Some(entry) = entries
            .iter()
            .find(|entry| entry["id"].as_str() == Some(model.as_str()))
        else {
            return RoutedModelStatus {
                route,
                model: Some(model),
                status: "missing".to_owned(),
                capabilities: ModelCapabilities::default(),
                failed: true,
                detail: Some("model is not present in the router catalog".to_owned()),
            };
        };
        let status = entry["status"]["value"]
            .as_str()
            .unwrap_or("unknown")
            .to_owned();
        let modalities = entry["architecture"]["input_modalities"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let has = |wanted: &str| {
            modalities
                .iter()
                .any(|value| value.as_str() == Some(wanted))
        };
        RoutedModelStatus {
            route,
            model: Some(model),
            status,
            capabilities: ModelCapabilities {
                text: has("text"),
                image: has("image"),
            },
            failed: entry["status"]["failed"].as_bool().unwrap_or(false),
            detail: entry["status"]["exit_code"]
                .as_i64()
                .map(|code| format!("router child exited with code {code}")),
        }
    }
}

#[async_trait]
impl ModelManager for RouterModelManager {
    async fn health(&self) -> Result<ModelFleetHealth, ModelError> {
        let entries = self.raw_models().await?;
        let routes = [ModelRoute::Default, ModelRoute::Large, ModelRoute::Vision]
            .into_iter()
            .map(|route| Self::status_for(route, self.model_for(route), &entries))
            .collect::<Vec<_>>();
        Ok(ModelFleetHealth {
            available: routes
                .iter()
                .any(|route| !route.failed && route.model.is_some()),
            routes,
        })
    }

    async fn switch_model(&self, model: &ModelId) -> Result<(), ModelError> {
        let entries = self.raw_models().await?;
        let current = Self::status_for(ModelRoute::Default, Some(model.clone()), &entries);
        if current.status == "missing" {
            return Err(ModelError::RouteUnavailable(model.0.clone()));
        }
        if !current.failed && matches!(current.status.as_str(), "loaded" | "sleeping") {
            return Ok(());
        }
        let response = self
            .client
            .post(self.router_endpoint("/models/load"))
            .json(&serde_json::json!({"model":model.as_str()}))
            .send()
            .await
            .map_err(|error| ModelError::Transport(error.to_string()))?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_else(|_| status.to_string());
            if status.as_u16() == 400 && body.contains("model is already running") {
                return Ok(());
            }
            return Err(ModelError::Http {
                status: status.as_u16(),
                body: truncate(&body, 4096),
            });
        }
        let started = std::time::Instant::now();
        loop {
            let entries = self.raw_models().await?;
            let status = Self::status_for(ModelRoute::Default, Some(model.clone()), &entries);
            if status.failed {
                return Err(ModelError::Transport(
                    status
                        .detail
                        .unwrap_or_else(|| format!("model {} failed to load", model.0)),
                ));
            }
            if matches!(status.status.as_str(), "loaded" | "sleeping") {
                return Ok(());
            }
            if started.elapsed() >= self.load_timeout {
                return Err(ModelError::LoadTimeout {
                    model: model.0.clone(),
                    seconds: self.load_timeout.as_secs(),
                });
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }

    async fn switch_profile(&self, profile: ModelProfile) -> Result<ModelId, ModelError> {
        let route = match profile {
            ModelProfile::Default => ModelRoute::Default,
            ModelProfile::Large => ModelRoute::Large,
        };
        let model = self
            .model_for(route)
            .ok_or_else(|| ModelError::RouteUnavailable(format!("{route:?}").to_lowercase()))?;
        self.switch_model(&model).await?;
        let mut current = self
            .current_profile
            .write()
            .map_err(|_| ModelError::Transport("model profile lock poisoned".to_owned()))?;
        *current = profile;
        Ok(model)
    }

    fn model_for(&self, route: ModelRoute) -> Option<ModelId> {
        match route {
            ModelRoute::Default => Some(self.routes.default.clone()),
            ModelRoute::Large => Some(self.routes.large.clone()),
            ModelRoute::Vision => self.routes.vision.clone(),
        }
    }
}

async fn http_error(response: reqwest::Response) -> Result<ModelResponse, ModelError> {
    let status = response.status();
    let body = response.text().await.unwrap_or_else(|_| status.to_string());
    Err(ModelError::Http {
        status: status.as_u16(),
        body: truncate(&body, 4096),
    })
}

fn parse_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "stop" => FinishReason::Stop,
        "tool_calls" => FinishReason::ToolCalls,
        "length" => FinishReason::Length,
        other => FinishReason::Other(other.to_owned()),
    }
}

fn truncate(value: &str, max: usize) -> String {
    value.chars().take(max).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    struct Tokens(tokio::sync::Mutex<String>);
    #[async_trait]
    impl ModelEventSink for Tokens {
        async fn token(&self, text: &str) {
            self.0.lock().await.push_str(text);
        }
    }

    #[test]
    fn messages_preserve_tool_call_protocol() {
        let call = RequestedToolCall {
            id: "c1".to_owned(),
            name: "read_file".to_owned(),
            arguments: serde_json::json!({"path":"a"}),
        };
        let message = Message::assistant_response(None, &[call]);
        let json = serde_json::to_value(message).unwrap_or(Value::Null);
        assert_eq!(json["tool_calls"][0]["function"]["name"], "read_file");
    }

    #[test]
    fn finish_reason_is_normalized() {
        assert_eq!(parse_finish_reason("tool_calls"), FinishReason::ToolCalls);
        assert_eq!(
            parse_finish_reason("new_reason"),
            FinishReason::Other("new_reason".to_owned())
        );
    }

    #[test]
    fn only_context_limit_http_errors_are_overflows() {
        assert!(
            ModelError::Http {
                status: 400,
                body: "maximum context length exceeded".to_owned(),
            }
            .is_context_overflow()
        );
        assert!(
            !ModelError::Http {
                status: 500,
                body: "maximum context length exceeded".to_owned(),
            }
            .is_context_overflow()
        );
        assert!(
            !ModelError::Http {
                status: 400,
                body: "invalid tool schema".to_owned(),
            }
            .is_context_overflow()
        );
    }

    #[tokio::test]
    async fn mock_openai_server_streams_content_and_tool_arguments()
    -> Result<(), Box<dyn std::error::Error>> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await?;
            let mut request = vec![0_u8; 8192];
            let read = socket.read(&mut request).await?;
            let request = String::from_utf8_lossy(&request[..read]);
            assert!(request.contains("\"stream_options\":{\"include_usage\":true}"));
            let events = concat!(
                "data: {\"choices\":[{\"delta\":{\"content\":\"hello \"},\"finish_reason\":null}]}\n\n",
                "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\"}}]},\"finish_reason\":null}]}\n\n",
                "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"a.rs\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
                "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":3,\"total_tokens\":15}}\n\n",
                "data: [DONE]\n\n"
            );
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                events.len(),
                events
            );
            socket.write_all(response.as_bytes()).await?;
            Ok::<(), std::io::Error>(())
        });
        let provider = OpenAiCompatibleProvider::new(
            &format!("http://{address}/v1"),
            "mock",
            Duration::from_secs(5),
        )?;
        let tokens = Arc::new(Tokens(tokio::sync::Mutex::new(String::new())));
        let response = provider
            .complete(
                ModelRequest {
                    messages: vec![Message::user("test")],
                    tools: vec![],
                    sampling: SamplingConfig::default(),
                    max_output_tokens: None,
                },
                tokens.as_ref(),
            )
            .await?;
        server.await??;
        assert_eq!(*tokens.0.lock().await, "hello ");
        assert_eq!(response.tool_calls[0].name, "read_file");
        assert_eq!(
            response
                .usage
                .as_ref()
                .and_then(|usage| usage.prompt_tokens),
            Some(12)
        );
        assert_eq!(response.tool_calls[0].arguments["path"], "a.rs");
        assert_eq!(response.finish_reason, FinishReason::ToolCalls);
        Ok(())
    }

    #[tokio::test]
    async fn router_manager_loads_and_reports_route_capabilities()
    -> Result<(), Box<dyn std::error::Error>> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let server = tokio::spawn(async move {
            for index in 0..4 {
                let (mut socket, _) = listener.accept().await?;
                let mut request = vec![0_u8; 8192];
                let read = socket.read(&mut request).await?;
                let request = String::from_utf8_lossy(&request[..read]);
                let body = if index == 1 {
                    assert!(request.starts_with("POST /models/load"));
                    assert!(request.contains("\"model\":\"vision\""));
                    "{\"success\":true}".to_owned()
                } else {
                    assert!(request.starts_with("GET /models"));
                    let status = if index == 0 { "unloaded" } else { "loaded" };
                    format!(
                        "{{\"data\":[{{\"id\":\"coding\",\"status\":{{\"value\":\"unloaded\"}},\"architecture\":{{\"input_modalities\":[\"text\"]}}}},{{\"id\":\"vision\",\"status\":{{\"value\":\"{status}\"}},\"architecture\":{{\"input_modalities\":[\"text\",\"image\"]}}}}]}}"
                    )
                };
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                socket.write_all(response.as_bytes()).await?;
            }
            Ok::<(), std::io::Error>(())
        });
        let manager = RouterModelManager::new(
            &format!("http://{address}/v1"),
            ModelRoutes {
                default: ModelId("coding".to_owned()),
                large: ModelId("coding".to_owned()),
                vision: Some(ModelId("vision".to_owned())),
            },
            Duration::from_secs(5),
            Duration::from_secs(5),
        )?;
        manager.switch_model(&ModelId("vision".to_owned())).await?;
        let health = manager.health().await?;
        server.await??;
        let vision = health
            .routes
            .iter()
            .find(|route| route.route == ModelRoute::Vision)
            .ok_or("missing vision route")?;
        assert_eq!(vision.status, "loaded");
        assert!(vision.capabilities.image);
        assert_eq!(
            manager.model_for(ModelRoute::Large).map(|id| id.0),
            Some("coding".to_owned())
        );
        Ok(())
    }

    #[tokio::test]
    async fn router_health_reports_missing_and_child_failure()
    -> Result<(), Box<dyn std::error::Error>> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await?;
            let mut request = vec![0_u8; 8192];
            let read = socket.read(&mut request).await?;
            assert!(String::from_utf8_lossy(&request[..read]).starts_with("GET /models"));
            let body = r#"{"data":[{"id":"vision","status":{"value":"unloaded","failed":true,"exit_code":137},"architecture":{"input_modalities":["text","image"]}}]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            socket.write_all(response.as_bytes()).await?;
            Ok::<(), std::io::Error>(())
        });
        let manager = RouterModelManager::new(
            &format!("http://{address}/v1"),
            ModelRoutes {
                default: ModelId("missing".to_owned()),
                large: ModelId("missing".to_owned()),
                vision: Some(ModelId("vision".to_owned())),
            },
            Duration::from_secs(5),
            Duration::from_secs(5),
        )?;
        let health = manager.health().await?;
        server.await??;
        assert!(!health.available);
        assert_eq!(health.routes[0].status, "missing");
        let vision = &health.routes[2];
        assert!(vision.failed);
        assert_eq!(
            vision.detail.as_deref(),
            Some("router child exited with code 137")
        );
        Ok(())
    }

    #[tokio::test]
    async fn router_load_timeout_is_bounded() -> Result<(), Box<dyn std::error::Error>> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let server = tokio::spawn(async move {
            for index in 0..3 {
                let (mut socket, _) = listener.accept().await?;
                let mut request = vec![0_u8; 8192];
                let read = socket.read(&mut request).await?;
                let request = String::from_utf8_lossy(&request[..read]);
                let body = if index == 1 {
                    assert!(request.starts_with("POST /models/load"));
                    r#"{"success":true}"#
                } else {
                    assert!(request.starts_with("GET /models"));
                    if index == 0 {
                        r#"{"data":[{"id":"vision","status":{"value":"unloaded"},"architecture":{"input_modalities":["text","image"]}}]}"#
                    } else {
                        r#"{"data":[{"id":"vision","status":{"value":"loading"},"architecture":{"input_modalities":["text","image"]}}]}"#
                    }
                };
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                socket.write_all(response.as_bytes()).await?;
            }
            Ok::<(), std::io::Error>(())
        });
        let manager = RouterModelManager::new(
            &format!("http://{address}/v1"),
            ModelRoutes {
                default: ModelId("coding".to_owned()),
                large: ModelId("coding".to_owned()),
                vision: Some(ModelId("vision".to_owned())),
            },
            Duration::from_secs(5),
            Duration::ZERO,
        )?;
        let error = match manager.switch_model(&ModelId("vision".to_owned())).await {
            Ok(()) => return Err("loading unexpectedly completed".into()),
            Err(error) => error,
        };
        server.await??;
        assert!(matches!(error, ModelError::LoadTimeout { .. }));
        Ok(())
    }
}
