use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use reqwest::{Client, Url};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::time::Duration;
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
    temperature: f32,
    top_p: f32,
    top_k: u32,
    repeat_penalty: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
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

    #[tokio::test]
    async fn mock_openai_server_streams_content_and_tool_arguments()
    -> Result<(), Box<dyn std::error::Error>> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await?;
            let mut request = vec![0_u8; 8192];
            let _ = socket.read(&mut request).await?;
            let events = concat!(
                "data: {\"choices\":[{\"delta\":{\"content\":\"hello \"},\"finish_reason\":null}]}\n\n",
                "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\"}}]},\"finish_reason\":null}]}\n\n",
                "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"a.rs\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
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
        assert_eq!(response.tool_calls[0].arguments["path"], "a.rs");
        assert_eq!(response.finish_reason, FinishReason::ToolCalls);
        Ok(())
    }
}
