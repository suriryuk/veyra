use async_trait::async_trait;
use chrono::{DateTime, Utc};
use encoding_rs::Encoding;
use futures_util::StreamExt;
use reqwest::header::{CONTENT_LENGTH, CONTENT_TYPE, LOCATION};
use reqwest::{Client, StatusCode};
use scraper::{ElementRef, Html, Selector};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::time::Duration;
use thiserror::Error;
use tokio::net::lookup_host;
use tokio_util::sync::CancellationToken;
use url::Url;

const MAX_URL_LENGTH: usize = 4096;
const MAX_QUERY_LENGTH: usize = 1024;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchResult {
    pub url: String,
    pub title: String,
    pub snippet: Option<String>,
    pub provider: String,
    pub engine: Option<String>,
    pub rank: usize,
    pub searched_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SourceMetadata {
    pub requested_url: String,
    pub final_url: String,
    pub title: Option<String>,
    pub content_type: String,
    pub fetched_at: DateTime<Utc>,
    pub redirects: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FetchedDocument {
    pub source: SourceMetadata,
    pub text: String,
    pub received_bytes: usize,
}

#[derive(Debug, Error)]
pub enum SearchError {
    #[error("invalid search configuration: {0}")]
    Configuration(String),
    #[error("search query is invalid: {0}")]
    InvalidQuery(String),
    #[error("search request failed: {0}")]
    Request(String),
    #[error("SearXNG JSON format is disabled (HTTP 403)")]
    JsonDisabled,
    #[error("search provider rate limited the request (HTTP 429)")]
    RateLimited,
    #[error("search provider returned HTTP {0}")]
    Http(u16),
    #[error("search provider returned malformed JSON: {0}")]
    Malformed(String),
}

#[derive(Debug, Error)]
pub enum FetchError {
    #[error("invalid URL: {0}")]
    InvalidUrl(String),
    #[error("network policy denied URL: {0}")]
    Policy(String),
    #[error("DNS resolution failed: {0}")]
    Dns(String),
    #[error("HTTP request failed: {0}")]
    Request(String),
    #[error("HTTP request timed out after {0} seconds")]
    Timeout(u64),
    #[error("HTTP request was cancelled")]
    Cancelled,
    #[error("redirect policy failed: {0}")]
    Redirect(String),
    #[error("response exceeded {limit} bytes")]
    TooLarge { limit: usize },
    #[error("unsupported content type: {0}")]
    UnsupportedContentType(String),
    #[error("content extraction failed: {0}")]
    Extraction(String),
    #[error("server returned HTTP {0}")]
    Http(u16),
}

#[async_trait]
pub trait SearchProvider: Send + Sync {
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>, SearchError>;
}

#[derive(Clone)]
pub struct SearxngProvider {
    endpoint: Url,
    client: Client,
}

impl SearxngProvider {
    pub fn new(base_url: &str, timeout: Duration, user_agent: &str) -> Result<Self, SearchError> {
        if timeout.is_zero() || user_agent.trim().is_empty() {
            return Err(SearchError::Configuration(
                "timeout and user agent must be non-empty and positive".to_owned(),
            ));
        }
        let normalized_base = if base_url.ends_with('/') {
            base_url.to_owned()
        } else {
            format!("{base_url}/")
        };
        let base = validate_http_url(&normalized_base).map_err(SearchError::Configuration)?;
        let endpoint = base
            .join("search")
            .map_err(|error| SearchError::Configuration(error.to_string()))?;
        let client = Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .timeout(timeout)
            .user_agent(user_agent)
            .build()
            .map_err(|error| SearchError::Configuration(error.to_string()))?;
        Ok(Self { endpoint, client })
    }
}

#[derive(Deserialize)]
struct SearxngResponse {
    results: Vec<SearxngItem>,
}

#[derive(Deserialize)]
struct SearxngItem {
    url: String,
    #[serde(default)]
    title: String,
    content: Option<String>,
    engine: Option<String>,
}

#[async_trait]
impl SearchProvider for SearxngProvider {
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>, SearchError> {
        let query = query.trim();
        if query.is_empty() || query.len() > MAX_QUERY_LENGTH {
            return Err(SearchError::InvalidQuery(format!(
                "query must contain 1..={MAX_QUERY_LENGTH} bytes"
            )));
        }
        if limit == 0 {
            return Err(SearchError::InvalidQuery(
                "result limit must be positive".to_owned(),
            ));
        }
        let response = self
            .client
            .get(self.endpoint.clone())
            .query(&[("q", query), ("format", "json")])
            .send()
            .await
            .map_err(|error| SearchError::Request(error.to_string()))?;
        match response.status() {
            StatusCode::FORBIDDEN => return Err(SearchError::JsonDisabled),
            StatusCode::TOO_MANY_REQUESTS => return Err(SearchError::RateLimited),
            status if !status.is_success() => return Err(SearchError::Http(status.as_u16())),
            _ => {}
        }
        let payload: SearxngResponse = response
            .json()
            .await
            .map_err(|error| SearchError::Malformed(error.to_string()))?;
        let searched_at = Utc::now();
        let mut seen = HashSet::new();
        let mut results = Vec::new();
        for item in payload.results {
            if results.len() >= limit {
                break;
            }
            let Ok(url) = validate_http_url(&item.url) else {
                continue;
            };
            let normalized = url.to_string();
            if !seen.insert(normalized.clone()) {
                continue;
            }
            results.push(SearchResult {
                url: normalized,
                title: normalize_whitespace(&item.title),
                snippet: item.content.map(|value| normalize_whitespace(&value)),
                provider: "searxng".to_owned(),
                engine: item.engine,
                rank: results.len() + 1,
                searched_at,
            });
        }
        Ok(results)
    }
}

#[derive(Debug, Clone)]
pub struct FetchPolicy {
    pub timeout: Duration,
    pub max_redirects: usize,
    pub max_response_bytes: usize,
    pub user_agent: String,
    allow_private: bool,
}

impl FetchPolicy {
    #[must_use]
    pub fn production(
        timeout: Duration,
        max_redirects: usize,
        max_response_bytes: usize,
        user_agent: String,
    ) -> Self {
        Self {
            timeout,
            max_redirects,
            max_response_bytes,
            user_agent,
            allow_private: false,
        }
    }

    #[cfg(test)]
    fn fixture() -> Self {
        Self {
            timeout: Duration::from_secs(2),
            max_redirects: 2,
            max_response_bytes: 4096,
            user_agent: "Veyra-test".to_owned(),
            allow_private: true,
        }
    }
}

#[derive(Clone)]
pub struct HttpFetcher {
    policy: FetchPolicy,
}

impl HttpFetcher {
    pub fn new(policy: FetchPolicy) -> Result<Self, FetchError> {
        if policy.timeout.is_zero()
            || policy.max_redirects == 0
            || policy.max_response_bytes == 0
            || policy.user_agent.trim().is_empty()
        {
            return Err(FetchError::Policy(
                "fetch limits and user agent must be non-empty and positive".to_owned(),
            ));
        }
        Ok(Self { policy })
    }

    pub async fn fetch(
        &self,
        requested: &str,
        cancellation: &CancellationToken,
    ) -> Result<FetchedDocument, FetchError> {
        let requested_url = validate_http_url(requested).map_err(FetchError::InvalidUrl)?;
        let mut current = requested_url.clone();
        let mut visited = HashSet::new();
        let mut redirects = 0;
        loop {
            if !visited.insert(current.to_string()) {
                return Err(FetchError::Redirect("redirect loop detected".to_owned()));
            }
            let client = self.client_for_url(&current, cancellation).await?;
            let response = tokio::select! {
                () = cancellation.cancelled() => return Err(FetchError::Cancelled),
                result = client.get(current.clone()).send() => {
                    result.map_err(|error| {
                        if error.is_timeout() {
                            FetchError::Timeout(self.policy.timeout.as_secs())
                        } else {
                            FetchError::Request(error.to_string())
                        }
                    })?
                }
            };
            if response.status().is_redirection() {
                if redirects >= self.policy.max_redirects {
                    return Err(FetchError::Redirect(format!(
                        "more than {} redirects",
                        self.policy.max_redirects
                    )));
                }
                let location = response
                    .headers()
                    .get(LOCATION)
                    .and_then(|value| value.to_str().ok())
                    .ok_or_else(|| FetchError::Redirect("missing Location header".to_owned()))?;
                current = current
                    .join(location)
                    .map_err(|error| FetchError::Redirect(error.to_string()))?;
                validate_url_parts(&current).map_err(FetchError::Policy)?;
                redirects += 1;
                continue;
            }
            if !response.status().is_success() {
                return Err(FetchError::Http(response.status().as_u16()));
            }
            if let Some(length) = response
                .headers()
                .get(CONTENT_LENGTH)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.parse::<usize>().ok())
            {
                if length > self.policy.max_response_bytes {
                    return Err(FetchError::TooLarge {
                        limit: self.policy.max_response_bytes,
                    });
                }
            }
            let declared_type = response
                .headers()
                .get(CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .map(str::to_owned);
            if let Some(value) = declared_type.as_deref() {
                ensure_supported_content_type(value)?;
            }
            let mut bytes = Vec::new();
            let mut stream = response.bytes_stream();
            loop {
                let next = tokio::select! {
                    () = cancellation.cancelled() => return Err(FetchError::Cancelled),
                    value = stream.next() => value,
                };
                let Some(chunk) = next else { break };
                let chunk = chunk.map_err(|error| FetchError::Request(error.to_string()))?;
                if bytes.len().saturating_add(chunk.len()) > self.policy.max_response_bytes {
                    return Err(FetchError::TooLarge {
                        limit: self.policy.max_response_bytes,
                    });
                }
                bytes.extend_from_slice(&chunk);
            }
            let content_type = declared_type.unwrap_or_else(|| sniff_content_type(&bytes));
            ensure_supported_content_type(&content_type)?;
            let decoded = decode_text(&bytes, &content_type);
            let media_type = media_type(&content_type);
            let (title, text) = if media_type == "text/plain" {
                (None, normalize_plain_text(&decoded))
            } else {
                extract_html(&decoded)?
            };
            return Ok(FetchedDocument {
                source: SourceMetadata {
                    requested_url: requested_url.to_string(),
                    final_url: current.to_string(),
                    title,
                    content_type,
                    fetched_at: Utc::now(),
                    redirects,
                },
                text,
                received_bytes: bytes.len(),
            });
        }
    }

    async fn client_for_url(
        &self,
        url: &Url,
        cancellation: &CancellationToken,
    ) -> Result<Client, FetchError> {
        validate_url_parts(url).map_err(FetchError::Policy)?;
        let host = url
            .host_str()
            .ok_or_else(|| FetchError::InvalidUrl("URL has no host".to_owned()))?;
        let port = url
            .port_or_known_default()
            .ok_or_else(|| FetchError::InvalidUrl("URL has no usable port".to_owned()))?;
        let resolution = tokio::select! {
            () = cancellation.cancelled() => return Err(FetchError::Cancelled),
            value = tokio::time::timeout(self.policy.timeout, lookup_host((host, port))) => value,
        };
        let addrs: Vec<SocketAddr> = resolution
            .map_err(|_| FetchError::Timeout(self.policy.timeout.as_secs()))?
            .map_err(|error| FetchError::Dns(error.to_string()))?
            .collect();
        if addrs.is_empty() {
            return Err(FetchError::Dns("host resolved to no addresses".to_owned()));
        }
        if !self.policy.allow_private && addrs.iter().any(|address| !is_public_ip(address.ip())) {
            return Err(FetchError::Policy(format!(
                "{host} resolved to a non-public address"
            )));
        }
        Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .timeout(self.policy.timeout)
            .user_agent(&self.policy.user_agent)
            .resolve_to_addrs(host, &addrs)
            .build()
            .map_err(|error| FetchError::Request(error.to_string()))
    }
}

fn validate_http_url(value: &str) -> Result<Url, String> {
    if value.len() > MAX_URL_LENGTH {
        return Err(format!("URL exceeds {MAX_URL_LENGTH} bytes"));
    }
    let url = Url::parse(value).map_err(|error| error.to_string())?;
    validate_url_parts(&url)?;
    Ok(url)
}

fn validate_url_parts(url: &Url) -> Result<(), String> {
    if !matches!(url.scheme(), "http" | "https") {
        return Err("only HTTP and HTTPS URLs are supported".to_owned());
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err("URL user-info is not allowed".to_owned());
    }
    if url.host_str().is_none() {
        return Err("URL host is required".to_owned());
    }
    if url.as_str().len() > MAX_URL_LENGTH {
        return Err(format!("URL exceeds {MAX_URL_LENGTH} bytes"));
    }
    Ok(())
}

fn is_public_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(value) => is_public_ipv4(value),
        IpAddr::V6(value) => is_public_ipv6(value),
    }
}

fn is_public_ipv4(ip: Ipv4Addr) -> bool {
    let [a, b, c, _] = ip.octets();
    !(a == 0
        || a == 10
        || a == 127
        || (a == 100 && (64..=127).contains(&b))
        || (a == 169 && b == 254)
        || (a == 172 && (16..=31).contains(&b))
        || (a == 192 && b == 0 && c == 0)
        || (a == 192 && b == 0 && c == 2)
        || (a == 192 && b == 88 && c == 99)
        || (a == 192 && b == 168)
        || (a == 198 && (b == 18 || b == 19))
        || (a == 198 && b == 51 && c == 100)
        || (a == 203 && b == 0 && c == 113)
        || a >= 224)
}

fn is_public_ipv6(ip: Ipv6Addr) -> bool {
    if let Some(mapped) = ip.to_ipv4_mapped() {
        return is_public_ipv4(mapped);
    }
    let segments = ip.segments();
    (segments[0] & 0xe000) == 0x2000
        && !(ip.is_unspecified()
            || ip.is_loopback()
            || ip.is_multicast()
            || (segments[0] & 0xfe00) == 0xfc00
            || (segments[0] & 0xffc0) == 0xfe80
            || (segments[0] == 0x2001 && segments[1] == 0x0db8))
}

fn media_type(content_type: &str) -> String {
    content_type
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase()
}

fn ensure_supported_content_type(value: &str) -> Result<(), FetchError> {
    match media_type(value).as_str() {
        "text/html" | "application/xhtml+xml" | "text/plain" => Ok(()),
        _ => Err(FetchError::UnsupportedContentType(value.to_owned())),
    }
}

fn sniff_content_type(bytes: &[u8]) -> String {
    if bytes.iter().take(8192).any(|byte| *byte == 0) {
        return "application/octet-stream".to_owned();
    }
    let prefix = String::from_utf8_lossy(&bytes[..bytes.len().min(256)]).to_ascii_lowercase();
    if prefix.contains("<!doctype html") || prefix.contains("<html") {
        "text/html; charset=utf-8".to_owned()
    } else {
        "text/plain; charset=utf-8".to_owned()
    }
}

fn decode_text(bytes: &[u8], content_type: &str) -> String {
    let encoding = content_type.split(';').skip(1).find_map(|part| {
        let (name, value) = part.trim().split_once('=')?;
        name.eq_ignore_ascii_case("charset")
            .then(|| Encoding::for_label(value.trim_matches(['\'', '"']).as_bytes()))
            .flatten()
    });
    encoding
        .unwrap_or(encoding_rs::UTF_8)
        .decode(bytes)
        .0
        .into_owned()
}

fn extract_html(source: &str) -> Result<(Option<String>, String), FetchError> {
    let document = Html::parse_document(source);
    let title = first_text(&document, "meta[property='og:title']", Some("content"))
        .or_else(|| first_text(&document, "title", None))
        .filter(|value| !value.is_empty());
    let root_selector = Selector::parse("article, main, [role='main'], body")
        .map_err(|error| FetchError::Extraction(error.to_string()))?;
    let block_selector = Selector::parse("h1, h2, h3, h4, h5, h6, p, li, pre, blockquote, td, th")
        .map_err(|error| FetchError::Extraction(error.to_string()))?;
    let root = document
        .select(&root_selector)
        .next()
        .ok_or_else(|| FetchError::Extraction("HTML has no readable body".to_owned()))?;
    let mut blocks = Vec::new();
    for element in root.select(&block_selector) {
        if has_excluded_ancestor(element) {
            continue;
        }
        let text = normalize_whitespace(&element.text().collect::<Vec<_>>().join(" "));
        if !text.is_empty() && blocks.last() != Some(&text) {
            blocks.push(text);
        }
    }
    if blocks.is_empty() {
        let text = normalize_whitespace(&root.text().collect::<Vec<_>>().join(" "));
        if !text.is_empty() {
            blocks.push(text);
        }
    }
    if blocks.is_empty() {
        return Err(FetchError::Extraction(
            "HTML contains no readable text".to_owned(),
        ));
    }
    Ok((title, blocks.join("\n\n")))
}

fn first_text(document: &Html, selector: &str, attribute: Option<&str>) -> Option<String> {
    let selector = Selector::parse(selector).ok()?;
    let element = document.select(&selector).next()?;
    let value = attribute
        .and_then(|name| element.value().attr(name).map(str::to_owned))
        .unwrap_or_else(|| element.text().collect::<Vec<_>>().join(" "));
    Some(normalize_whitespace(&value))
}

fn has_excluded_ancestor(element: ElementRef<'_>) -> bool {
    element
        .ancestors()
        .filter_map(ElementRef::wrap)
        .any(|item| {
            matches!(
                item.value().name(),
                "script" | "style" | "nav" | "header" | "footer" | "form" | "svg" | "noscript"
            )
        })
}

fn normalize_plain_text(value: &str) -> String {
    value
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn normalize_whitespace(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;

    async fn fixture(response: &'static str) -> Result<String, Box<dyn std::error::Error>> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        tokio::spawn(async move {
            if let Ok((mut socket, _)) = listener.accept().await {
                let mut request = vec![0; 4096];
                let _ = socket.read(&mut request).await;
                let _ = socket.write_all(response.as_bytes()).await;
            }
        });
        Ok(format!("http://{address}/"))
    }

    async fn fixture_with_request(
        response: &'static str,
    ) -> Result<(String, oneshot::Receiver<String>), Box<dyn std::error::Error>> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let (sender, receiver) = oneshot::channel();
        tokio::spawn(async move {
            if let Ok((mut socket, _)) = listener.accept().await {
                let mut request = vec![0; 4096];
                let read = socket.read(&mut request).await.unwrap_or_default();
                let _ = sender.send(String::from_utf8_lossy(&request[..read]).into_owned());
                let _ = socket.write_all(response.as_bytes()).await;
            }
        });
        Ok((format!("http://{address}/"), receiver))
    }

    async fn redirect_fixture() -> Result<String, Box<dyn std::error::Error>> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        tokio::spawn(async move {
            for _ in 0..2 {
                if let Ok((mut socket, _)) = listener.accept().await {
                    let mut request = vec![0; 4096];
                    let read = socket.read(&mut request).await.unwrap_or_default();
                    let request = String::from_utf8_lossy(&request[..read]);
                    let response = if request.starts_with("GET /start ") {
                        "HTTP/1.1 302 Found\r\nLocation: /final\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_owned()
                    } else {
                        let body = "redirected";
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                            body.len()
                        )
                    };
                    let _ = socket.write_all(response.as_bytes()).await;
                }
            }
        });
        Ok(format!("http://{address}/start"))
    }

    #[test]
    fn public_address_policy_blocks_internal_and_reserved_ranges() {
        assert!(!is_public_ip(
            "127.0.0.1"
                .parse()
                .unwrap_or(IpAddr::V4(Ipv4Addr::LOCALHOST))
        ));
        assert!(!is_public_ip(
            "169.254.169.254"
                .parse()
                .unwrap_or(IpAddr::V4(Ipv4Addr::LOCALHOST))
        ));
        assert!(!is_public_ip(
            "10.0.0.1"
                .parse()
                .unwrap_or(IpAddr::V4(Ipv4Addr::LOCALHOST))
        ));
        assert!(!is_public_ip(
            "::1".parse().unwrap_or(IpAddr::V6(Ipv6Addr::LOCALHOST))
        ));
        assert!(is_public_ip(
            "1.1.1.1".parse().unwrap_or(IpAddr::V4(Ipv4Addr::LOCALHOST))
        ));
    }

    #[test]
    fn extracts_main_content_and_ignores_navigation() -> Result<(), FetchError> {
        let html = "<html><head><title> Example </title></head><body><nav><p>Ignore me</p></nav><main><h1>Heading</h1><p>Main   text</p></main></body></html>";
        let (title, text) = extract_html(html)?;
        assert_eq!(title.as_deref(), Some("Example"));
        assert_eq!(text, "Heading\n\nMain text");
        Ok(())
    }

    #[test]
    fn decodes_declared_non_utf8_charset() {
        assert_eq!(
            decode_text(&[0xe9], "text/plain; charset=windows-1252"),
            "é"
        );
    }

    #[tokio::test]
    async fn searxng_encodes_query_deduplicates_and_skips_invalid_urls()
    -> Result<(), Box<dyn std::error::Error>> {
        let body = r#"{"results":[
            {"url":"https://example.com/a","title":" First ","content":"one","engine":"brave"},
            {"url":"https://example.com/a","title":"Duplicate"},
            {"url":"file:///etc/passwd","title":"Invalid"},
            {"url":"https://example.com/b","title":"Second"}
        ]}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let response: &'static str = Box::leak(response.into_boxed_str());
        let (base, request) = fixture_with_request(response).await?;
        let provider = SearxngProvider::new(&base, Duration::from_secs(2), "test")?;
        let results = provider.search("Rust async", 10).await?;
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].rank, 1);
        let request = request.await?;
        assert!(request.starts_with("GET /search?"));
        assert!(request.contains("q=Rust+async"));
        assert!(request.contains("format=json"));
        Ok(())
    }

    #[tokio::test]
    async fn searxng_reports_disabled_json_and_rate_limits()
    -> Result<(), Box<dyn std::error::Error>> {
        for (status, expected_disabled) in [(403, true), (429, false)] {
            let response = format!(
                "HTTP/1.1 {status} Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            );
            let response: &'static str = Box::leak(response.into_boxed_str());
            let base = fixture(response).await?;
            let provider = SearxngProvider::new(&base, Duration::from_secs(2), "test")?;
            let error = provider.search("test", 1).await.err();
            assert_eq!(
                error
                    .as_ref()
                    .is_some_and(|value| matches!(value, SearchError::JsonDisabled)),
                expected_disabled
            );
            assert_eq!(
                error
                    .as_ref()
                    .is_some_and(|value| matches!(value, SearchError::RateLimited)),
                !expected_disabled
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn searxng_reports_malformed_and_server_errors() -> Result<(), Box<dyn std::error::Error>>
    {
        let malformed = fixture(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 1\r\nConnection: close\r\n\r\n{",
        )
        .await?;
        let provider = SearxngProvider::new(&malformed, Duration::from_secs(2), "test")?;
        assert!(matches!(
            provider.search("test", 1).await,
            Err(SearchError::Malformed(_))
        ));

        let unavailable =
            fixture("HTTP/1.1 503 Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                .await?;
        let provider = SearxngProvider::new(&unavailable, Duration::from_secs(2), "test")?;
        assert!(matches!(
            provider.search("test", 1).await,
            Err(SearchError::Http(503))
        ));
        Ok(())
    }

    #[tokio::test]
    async fn fetches_html_from_fixture() -> Result<(), Box<dyn std::error::Error>> {
        let body = "<html><title>Fixture</title><article><p>Hello web</p></article></html>";
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let response: &'static str = Box::leak(response.into_boxed_str());
        let url = fixture(response).await?;
        let fetcher = HttpFetcher::new(FetchPolicy::fixture())?;
        let result = fetcher.fetch(&url, &CancellationToken::new()).await?;
        assert_eq!(result.source.title.as_deref(), Some("Fixture"));
        assert_eq!(result.text, "Hello web");
        Ok(())
    }

    #[tokio::test]
    async fn rejects_declared_oversized_response() -> Result<(), Box<dyn std::error::Error>> {
        let url = fixture(
            "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 99999\r\nConnection: close\r\n\r\nx",
        )
        .await?;
        let fetcher = HttpFetcher::new(FetchPolicy::fixture())?;
        assert!(matches!(
            fetcher.fetch(&url, &CancellationToken::new()).await,
            Err(FetchError::TooLarge { .. })
        ));
        Ok(())
    }

    #[tokio::test]
    async fn rejects_streaming_oversized_response_without_length()
    -> Result<(), Box<dyn std::error::Error>> {
        let body = "x".repeat(5000);
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nConnection: close\r\n\r\n{body}"
        );
        let response: &'static str = Box::leak(response.into_boxed_str());
        let url = fixture(response).await?;
        let fetcher = HttpFetcher::new(FetchPolicy::fixture())?;
        assert!(matches!(
            fetcher.fetch(&url, &CancellationToken::new()).await,
            Err(FetchError::TooLarge { .. })
        ));
        Ok(())
    }

    #[tokio::test]
    async fn follows_relative_redirect_and_records_final_url()
    -> Result<(), Box<dyn std::error::Error>> {
        let url = redirect_fixture().await?;
        let fetcher = HttpFetcher::new(FetchPolicy::fixture())?;
        let result = fetcher.fetch(&url, &CancellationToken::new()).await?;
        assert_eq!(result.text, "redirected");
        assert_eq!(result.source.redirects, 1);
        assert!(result.source.final_url.ends_with("/final"));
        Ok(())
    }

    #[tokio::test]
    async fn observes_cancellation_before_dns_or_request() -> Result<(), Box<dyn std::error::Error>>
    {
        let fetcher = HttpFetcher::new(FetchPolicy::fixture())?;
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        assert!(matches!(
            fetcher
                .fetch("http://example.invalid/", &cancellation)
                .await,
            Err(FetchError::Cancelled)
        ));
        Ok(())
    }

    #[tokio::test]
    async fn request_timeout_is_structured() -> Result<(), Box<dyn std::error::Error>> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        tokio::spawn(async move {
            if let Ok((_socket, _)) = listener.accept().await {
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        });
        let fetcher = HttpFetcher::new(FetchPolicy {
            timeout: Duration::from_millis(50),
            max_redirects: 1,
            max_response_bytes: 1024,
            user_agent: "test".to_owned(),
            allow_private: true,
        })?;
        assert!(matches!(
            fetcher
                .fetch(&format!("http://{address}/"), &CancellationToken::new())
                .await,
            Err(FetchError::Timeout(_))
        ));
        Ok(())
    }

    #[tokio::test]
    async fn rejects_missing_type_when_body_is_binary() -> Result<(), Box<dyn std::error::Error>> {
        let url = fixture("HTTP/1.1 200 OK\r\nContent-Length: 3\r\nConnection: close\r\n\r\na\0b")
            .await?;
        let fetcher = HttpFetcher::new(FetchPolicy::fixture())?;
        assert!(matches!(
            fetcher.fetch(&url, &CancellationToken::new()).await,
            Err(FetchError::UnsupportedContentType(_))
        ));
        Ok(())
    }

    #[tokio::test]
    async fn detects_relative_redirect_loop() -> Result<(), Box<dyn std::error::Error>> {
        let url = fixture(
            "HTTP/1.1 302 Found\r\nLocation: /\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        )
        .await?;
        let fetcher = HttpFetcher::new(FetchPolicy::fixture())?;
        assert!(matches!(
            fetcher.fetch(&url, &CancellationToken::new()).await,
            Err(FetchError::Redirect(message)) if message.contains("loop")
        ));
        Ok(())
    }
}
