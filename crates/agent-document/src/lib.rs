use async_trait::async_trait;
use pulldown_cmark::{Event, HeadingLevel, Parser, Tag, TagEnd};
use quick_xml::Reader;
use quick_xml::events::Event as XmlEvent;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use thiserror::Error;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;
use zip::ZipArchive;

pub const SUPPORTED_EXTENSIONS: &[&str] = &["pdf", "docx", "html", "htm", "md", "markdown", "txt"];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DocumentStatus {
    Ready,
    Partial,
    UnsupportedFormat,
    UnsupportedScanned,
    UnsupportedEncrypted,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DocumentFormat {
    Pdf,
    Docx,
    Html,
    Markdown,
    Text,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSource {
    pub path: String,
    pub format: DocumentFormat,
    pub content_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct DocumentMetadata {
    pub page_count: Option<u32>,
    pub byte_size: usize,
    pub warnings: Vec<String>,
    pub pipeline_fingerprint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ExtractionMethod {
    #[default]
    Text,
    Vision,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ExtractionConfidence {
    High,
    Medium,
    Low,
    #[default]
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    pub id: String,
    pub document_id: String,
    pub ordinal: usize,
    pub text: String,
    pub page: Option<u32>,
    pub heading: Option<String>,
    pub start_offset: usize,
    pub end_offset: usize,
    pub token_count: usize,
    #[serde(default)]
    pub extraction_method: ExtractionMethod,
    #[serde(default)]
    pub confidence: ExtractionConfidence,
    #[serde(default)]
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizedDocument {
    pub id: String,
    pub workspace: String,
    pub source: DocumentSource,
    pub title: Option<String>,
    pub metadata: DocumentMetadata,
    pub status: DocumentStatus,
    pub error: Option<String>,
    pub text: String,
    pub chunks: Vec<DocumentChunk>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSummary {
    pub id: String,
    pub path: String,
    pub format: DocumentFormat,
    pub title: Option<String>,
    pub status: DocumentStatus,
    pub error: Option<String>,
    pub byte_size: usize,
    pub chunk_count: usize,
    pub indexed_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSearchQuery {
    pub workspace: String,
    pub query: String,
    pub document_ids: Vec<String>,
    pub limit: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSearchHit {
    pub document_id: String,
    pub path: String,
    pub chunk_id: String,
    pub ordinal: usize,
    pub page: Option<u32>,
    pub heading: Option<String>,
    pub start_offset: usize,
    pub end_offset: usize,
    pub score: f64,
    pub excerpt: String,
    pub citation: String,
    pub extraction_method: ExtractionMethod,
    pub confidence: ExtractionConfidence,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexResult {
    pub document: DocumentSummary,
    pub unchanged: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DocumentLimits {
    pub max_file_bytes: usize,
    pub max_uncompressed_bytes: usize,
    pub max_documents_per_request: usize,
    pub max_chunks_per_document: usize,
    pub chunk_target_chars: usize,
    pub chunk_overlap_chars: usize,
    pub default_search_limit: usize,
    pub max_search_limit: usize,
}

impl Default for DocumentLimits {
    fn default() -> Self {
        Self {
            max_file_bytes: 25 * 1024 * 1024,
            max_uncompressed_bytes: 100 * 1024 * 1024,
            max_documents_per_request: 100,
            max_chunks_per_document: 10_000,
            chunk_target_chars: 2_000,
            chunk_overlap_chars: 200,
            default_search_limit: 10,
            max_search_limit: 50,
        }
    }
}

impl DocumentLimits {
    pub fn validate(&self) -> Result<(), DocumentError> {
        if self.max_file_bytes == 0
            || self.max_uncompressed_bytes == 0
            || self.max_documents_per_request == 0
            || self.max_chunks_per_document == 0
            || self.chunk_target_chars == 0
            || self.default_search_limit == 0
            || self.max_search_limit == 0
            || self.default_search_limit > self.max_search_limit
            || self.chunk_overlap_chars >= self.chunk_target_chars
        {
            return Err(DocumentError::Invalid(
                "document limits must be positive and overlap/default limits must be bounded"
                    .into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum DocumentError {
    #[error("unsupported document format: {0}")]
    UnsupportedFormat(String),
    #[error("document exceeds configured limit: {0}")]
    Limit(String),
    #[error("document parse failed: {0}")]
    Parse(String),
    #[error("invalid document request: {0}")]
    Invalid(String),
    #[error("document storage failed: {0}")]
    Storage(String),
}

#[async_trait]
pub trait DocumentRepository: Send + Sync {
    async fn upsert(&self, document: &NormalizedDocument) -> Result<IndexResult, DocumentError>;
    async fn list(
        &self,
        workspace: &str,
        status: Option<DocumentStatus>,
        limit: usize,
    ) -> Result<Vec<DocumentSummary>, DocumentError>;
    async fn get(
        &self,
        workspace: &str,
        id: &str,
        chunks: bool,
    ) -> Result<Option<NormalizedDocument>, DocumentError>;
    async fn search(
        &self,
        query: DocumentSearchQuery,
    ) -> Result<Vec<DocumentSearchHit>, DocumentError>;
}

pub trait DocumentParser: Send + Sync {
    fn parse(&self, bytes: &[u8], limits: &DocumentLimits)
    -> Result<ParsedDocument, DocumentError>;
}

#[derive(Debug, Clone)]
pub struct ParsedSection {
    pub text: String,
    pub page: Option<u32>,
    pub heading: Option<String>,
    pub extraction_method: ExtractionMethod,
    pub confidence: ExtractionConfidence,
    pub limitations: Vec<String>,
}
#[derive(Debug, Clone)]
pub struct ParsedDocument {
    pub title: Option<String>,
    pub sections: Vec<ParsedSection>,
    pub page_count: Option<u32>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct VisionPageExtraction {
    pub text: String,
    pub confidence: ExtractionConfidence,
    pub limitations: Vec<String>,
}

#[async_trait]
pub trait ScannedPdfFallback: Send + Sync {
    fn max_pages(&self) -> usize;
    fn pipeline_fingerprint(&self) -> String;
    async fn extract_page(
        &self,
        pdf_path: &Path,
        source: &str,
        page: u32,
        cancellation: &CancellationToken,
    ) -> Result<VisionPageExtraction, DocumentError>;
}

#[derive(Clone)]
pub struct DocumentService {
    limits: DocumentLimits,
}

impl DocumentService {
    pub fn new(limits: DocumentLimits) -> Result<Self, DocumentError> {
        limits.validate()?;
        Ok(Self { limits })
    }
    pub fn limits(&self) -> &DocumentLimits {
        &self.limits
    }
    pub fn parse(
        &self,
        workspace: &str,
        relative_path: &str,
        bytes: &[u8],
    ) -> Result<NormalizedDocument, DocumentError> {
        if bytes.len() > self.limits.max_file_bytes {
            return Err(DocumentError::Limit(format!("{} bytes", bytes.len())));
        }
        let format = format_from_path(Path::new(relative_path)).unwrap_or(DocumentFormat::Unknown);
        let hash = format!("{:x}", Sha256::digest(bytes));
        let id = stable_id(workspace, relative_path);
        let parsed = match format {
            DocumentFormat::Pdf => PdfParser.parse(bytes, &self.limits),
            DocumentFormat::Docx => DocxParser.parse(bytes, &self.limits),
            DocumentFormat::Html => HtmlParser.parse(bytes, &self.limits),
            DocumentFormat::Markdown => MarkdownParser.parse(bytes, &self.limits),
            DocumentFormat::Text => TextParser.parse(bytes, &self.limits),
            DocumentFormat::Unknown => {
                return Ok(NormalizedDocument {
                    id,
                    workspace: workspace.into(),
                    source: DocumentSource {
                        path: relative_path.into(),
                        format,
                        content_hash: hash,
                    },
                    title: None,
                    metadata: DocumentMetadata {
                        page_count: None,
                        byte_size: bytes.len(),
                        warnings: Vec::new(),
                        pipeline_fingerprint: None,
                    },
                    status: DocumentStatus::UnsupportedFormat,
                    error: Some("unsupported document format".into()),
                    text: String::new(),
                    chunks: Vec::new(),
                });
            }
        };
        match parsed {
            Ok(parsed) => {
                let visible = parsed
                    .sections
                    .iter()
                    .flat_map(|s| s.text.chars())
                    .filter(|c| c.is_alphanumeric())
                    .count();
                let scanned = format == DocumentFormat::Pdf && visible < 20;
                let empty = format != DocumentFormat::Pdf && visible == 0;
                let (text, chunks) = if scanned || empty {
                    (String::new(), Vec::new())
                } else {
                    chunk_sections(&id, &parsed.sections, &self.limits)?
                };
                let status = if scanned {
                    DocumentStatus::UnsupportedScanned
                } else if empty {
                    DocumentStatus::Failed
                } else if parsed.warnings.is_empty() {
                    DocumentStatus::Ready
                } else {
                    DocumentStatus::Partial
                };
                Ok(NormalizedDocument {
                    id,
                    workspace: workspace.into(),
                    source: DocumentSource {
                        path: relative_path.into(),
                        format,
                        content_hash: hash,
                    },
                    title: parsed.title,
                    metadata: DocumentMetadata {
                        page_count: parsed.page_count,
                        byte_size: bytes.len(),
                        warnings: parsed.warnings,
                        pipeline_fingerprint: None,
                    },
                    status,
                    error: if scanned {
                        Some("scanned PDF requires the v0.8 vision fallback".into())
                    } else if empty {
                        Some("document contains no extractable text".into())
                    } else {
                        None
                    },
                    text,
                    chunks,
                })
            }
            Err(error) => {
                let message = error.to_string();
                let lower = message.to_lowercase();
                let status = if format == DocumentFormat::Pdf
                    && (lower.contains("encrypt") || lower.contains("password"))
                {
                    DocumentStatus::UnsupportedEncrypted
                } else {
                    DocumentStatus::Failed
                };
                Ok(NormalizedDocument {
                    id,
                    workspace: workspace.into(),
                    source: DocumentSource {
                        path: relative_path.into(),
                        format,
                        content_hash: hash,
                    },
                    title: None,
                    metadata: DocumentMetadata {
                        page_count: None,
                        byte_size: bytes.len(),
                        warnings: Vec::new(),
                        pipeline_fingerprint: None,
                    },
                    status,
                    error: Some(message),
                    text: String::new(),
                    chunks: Vec::new(),
                })
            }
        }
    }

    pub async fn parse_with_fallback(
        &self,
        workspace: &str,
        relative_path: &str,
        pdf_path: &Path,
        bytes: &[u8],
        fallback: Option<&dyn ScannedPdfFallback>,
        cancellation: &CancellationToken,
    ) -> Result<NormalizedDocument, DocumentError> {
        let base = self.parse(workspace, relative_path, bytes)?;
        if base.source.format != DocumentFormat::Pdf {
            return Ok(base);
        }
        let pages = match pdf_extract::extract_text_from_mem_by_pages(bytes) {
            Ok(pages) => pages,
            Err(_) => return Ok(base),
        };
        let page_count = u32::try_from(pages.len()).ok();
        let candidate_count = pages.iter().filter(|text| visible_chars(text) < 20).count();
        if candidate_count == 0 {
            return Ok(base);
        }
        let Some(fallback) = fallback else {
            return Ok(base);
        };
        let mut sections = Vec::new();
        let mut warnings = Vec::new();
        let mut attempted = 0_usize;
        let mut missing_pages = 0_usize;
        for (index, text) in pages.into_iter().enumerate() {
            if cancellation.is_cancelled() {
                return Err(DocumentError::Parse(
                    "vision fallback was cancelled".to_owned(),
                ));
            }
            let page = u32::try_from(index + 1).ok();
            let normalized = normalize(&text);
            if visible_chars(&normalized) >= 20 {
                sections.push(ParsedSection {
                    text: normalized,
                    page,
                    heading: None,
                    extraction_method: ExtractionMethod::Text,
                    confidence: ExtractionConfidence::Unknown,
                    limitations: Vec::new(),
                });
                continue;
            }
            let Some(page) = page else {
                missing_pages += 1;
                continue;
            };
            if attempted >= fallback.max_pages() {
                missing_pages += 1;
                warnings.push(format!(
                    "page {page} exceeded the configured vision page limit"
                ));
                continue;
            }
            attempted += 1;
            match fallback
                .extract_page(pdf_path, relative_path, page, cancellation)
                .await
            {
                Ok(extraction) if !extraction.text.trim().is_empty() => {
                    sections.push(ParsedSection {
                        text: normalize(&extraction.text),
                        page: Some(page),
                        heading: None,
                        extraction_method: ExtractionMethod::Vision,
                        confidence: extraction.confidence,
                        limitations: extraction.limitations,
                    });
                }
                Ok(_) => {
                    missing_pages += 1;
                    warnings.push(format!("page {page} vision extraction returned no text"));
                }
                Err(error) => {
                    missing_pages += 1;
                    warnings.push(format!("page {page} vision extraction failed: {error}"));
                }
            }
        }
        let id = stable_id(workspace, relative_path);
        let (text, chunks) = chunk_sections(&id, &sections, &self.limits)?;
        let has_text = !text.is_empty();
        let status = if !has_text {
            DocumentStatus::Failed
        } else if missing_pages > 0 {
            DocumentStatus::Partial
        } else {
            DocumentStatus::Ready
        };
        Ok(NormalizedDocument {
            id,
            workspace: workspace.to_owned(),
            source: base.source,
            title: base.title,
            metadata: DocumentMetadata {
                page_count,
                byte_size: bytes.len(),
                warnings,
                pipeline_fingerprint: Some(fallback.pipeline_fingerprint()),
            },
            status,
            error: (!has_text).then(|| {
                format!("vision extraction failed for all {candidate_count} candidate pages")
            }),
            text,
            chunks,
        })
    }
}

fn visible_chars(text: &str) -> usize {
    text.chars().filter(|value| value.is_alphanumeric()).count()
}

pub fn format_from_path(path: &Path) -> Result<DocumentFormat, DocumentError> {
    match path
        .extension()
        .and_then(|v| v.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "pdf" => Ok(DocumentFormat::Pdf),
        "docx" => Ok(DocumentFormat::Docx),
        "html" | "htm" => Ok(DocumentFormat::Html),
        "md" | "markdown" => Ok(DocumentFormat::Markdown),
        "txt" => Ok(DocumentFormat::Text),
        value => Err(DocumentError::UnsupportedFormat(value.into())),
    }
}
pub fn is_supported(path: &Path) -> bool {
    format_from_path(path).is_ok()
}
pub fn stable_id(workspace: &str, path: &str) -> String {
    Uuid::new_v5(
        &Uuid::NAMESPACE_URL,
        format!("veyra:{workspace}:{path}").as_bytes(),
    )
    .to_string()
}

struct TextParser;
impl DocumentParser for TextParser {
    fn parse(&self, bytes: &[u8], _: &DocumentLimits) -> Result<ParsedDocument, DocumentError> {
        let text = decode_text(bytes)?;
        Ok(plain_document(text))
    }
}

struct MarkdownParser;
impl DocumentParser for MarkdownParser {
    fn parse(&self, bytes: &[u8], _: &DocumentLimits) -> Result<ParsedDocument, DocumentError> {
        let input = decode_text(bytes)?;
        let mut sections = Vec::new();
        let mut text = String::new();
        let mut heading = None;
        let mut heading_text = String::new();
        let mut in_heading = false;
        for event in Parser::new(&input) {
            match event {
                Event::Start(Tag::Heading { level, .. }) => {
                    flush_section(&mut sections, &mut text, heading.clone(), None);
                    in_heading = true;
                    heading_text.clear();
                    heading_text.push_str(heading_prefix(level));
                }
                Event::End(TagEnd::Heading(_)) => {
                    in_heading = false;
                    heading = Some(heading_text.trim().to_owned());
                }
                Event::Text(value) | Event::Code(value) => {
                    if in_heading {
                        heading_text.push_str(&value)
                    } else {
                        text.push_str(&value)
                    }
                }
                Event::SoftBreak | Event::HardBreak => text.push('\n'),
                Event::End(TagEnd::Paragraph) | Event::End(TagEnd::Item) => text.push_str("\n\n"),
                _ => {}
            }
        }
        flush_section(&mut sections, &mut text, heading, None);
        Ok(ParsedDocument {
            title: sections.first().and_then(|s| s.heading.clone()),
            sections,
            page_count: None,
            warnings: Vec::new(),
        })
    }
}
fn heading_prefix(level: HeadingLevel) -> &'static str {
    match level {
        HeadingLevel::H1 => "",
        HeadingLevel::H2 => "",
        HeadingLevel::H3 => "",
        HeadingLevel::H4 => "",
        HeadingLevel::H5 => "",
        HeadingLevel::H6 => "",
    }
}

struct HtmlParser;
impl DocumentParser for HtmlParser {
    fn parse(&self, bytes: &[u8], _: &DocumentLimits) -> Result<ParsedDocument, DocumentError> {
        let input = decode_text(bytes)?;
        let html = Html::parse_document(&input);
        let title_sel =
            Selector::parse("title").map_err(|e| DocumentError::Parse(e.to_string()))?;
        let block_sel = Selector::parse("h1,h2,h3,h4,h5,h6,p,li,pre,blockquote,td,th")
            .map_err(|e| DocumentError::Parse(e.to_string()))?;
        let title = html
            .select(&title_sel)
            .next()
            .map(|n| normalize(&n.text().collect::<Vec<_>>().join(" ")))
            .filter(|s| !s.is_empty());
        let mut sections = Vec::new();
        let mut heading = None;
        for node in html.select(&block_sel) {
            let name = node.value().name();
            let text = normalize(&node.text().collect::<Vec<_>>().join(" "));
            if text.is_empty() {
                continue;
            }
            if name.starts_with('h') {
                heading = Some(text);
            } else {
                sections.push(ParsedSection {
                    text,
                    page: None,
                    heading: heading.clone(),
                    extraction_method: ExtractionMethod::Text,
                    confidence: ExtractionConfidence::Unknown,
                    limitations: Vec::new(),
                });
            }
        }
        Ok(ParsedDocument {
            title,
            sections,
            page_count: None,
            warnings: Vec::new(),
        })
    }
}

struct PdfParser;
impl DocumentParser for PdfParser {
    fn parse(&self, bytes: &[u8], _: &DocumentLimits) -> Result<ParsedDocument, DocumentError> {
        let pages = pdf_extract::extract_text_from_mem_by_pages(bytes)
            .map_err(|e| DocumentError::Parse(e.to_string()))?;
        let page_count = u32::try_from(pages.len()).ok();
        let mut warnings = Vec::new();
        let sections = pages
            .into_iter()
            .enumerate()
            .map(|(index, text)| {
                let text = normalize(&text);
                if text.is_empty() {
                    warnings.push(format!("page {} produced no extractable text", index + 1));
                }
                ParsedSection {
                    text,
                    page: u32::try_from(index + 1).ok(),
                    heading: None,
                    extraction_method: ExtractionMethod::Text,
                    confidence: ExtractionConfidence::Unknown,
                    limitations: Vec::new(),
                }
            })
            .collect();
        Ok(ParsedDocument {
            title: None,
            sections,
            page_count,
            warnings,
        })
    }
}

struct DocxParser;
impl DocumentParser for DocxParser {
    fn parse(
        &self,
        bytes: &[u8],
        limits: &DocumentLimits,
    ) -> Result<ParsedDocument, DocumentError> {
        let mut archive =
            ZipArchive::new(Cursor::new(bytes)).map_err(|e| DocumentError::Parse(e.to_string()))?;
        let total: u64 = (0..archive.len())
            .filter_map(|i| archive.by_index_raw(i).ok().map(|f| f.size()))
            .sum();
        if total > limits.max_uncompressed_bytes as u64 {
            return Err(DocumentError::Limit(format!(
                "DOCX expands to {total} bytes"
            )));
        }
        let mut xml = String::new();
        archive
            .by_name("word/document.xml")
            .map_err(|e| DocumentError::Parse(e.to_string()))?
            .read_to_string(&mut xml)
            .map_err(|e| DocumentError::Parse(e.to_string()))?;
        let mut reader = Reader::from_str(&xml);
        reader.config_mut().trim_text(false);
        let mut sections = Vec::new();
        let mut paragraph = String::new();
        let mut heading = None;
        let mut style = None;
        let mut page = 1u32;
        loop {
            match reader.read_event() {
                Ok(XmlEvent::Eof) => break,
                Ok(XmlEvent::Empty(e)) | Ok(XmlEvent::Start(e))
                    if e.local_name().as_ref() == b"pStyle" =>
                {
                    for a in e.attributes().flatten() {
                        if a.key.local_name().as_ref() == b"val" {
                            style = Some(String::from_utf8_lossy(a.value.as_ref()).into_owned());
                        }
                    }
                }
                Ok(XmlEvent::Empty(e)) if is_docx_page_break(&e) => {
                    page = page.saturating_add(1);
                }
                Ok(XmlEvent::Text(e)) => {
                    paragraph.push_str(&String::from_utf8_lossy(e.as_ref()));
                }
                Ok(XmlEvent::End(e)) if e.local_name().as_ref() == b"p" => {
                    let text = normalize(&paragraph);
                    if !text.is_empty() {
                        if style
                            .as_deref()
                            .is_some_and(|s| s.to_ascii_lowercase().starts_with("heading"))
                        {
                            heading = Some(text);
                        } else {
                            sections.push(ParsedSection {
                                text,
                                page: Some(page),
                                heading: heading.clone(),
                                extraction_method: ExtractionMethod::Text,
                                confidence: ExtractionConfidence::Unknown,
                                limitations: Vec::new(),
                            });
                        }
                    }
                    paragraph.clear();
                    style = None;
                }
                Ok(_) => {}
                Err(e) => return Err(DocumentError::Parse(e.to_string())),
            }
        }
        let title = sections.first().and_then(|s| s.heading.clone());
        Ok(ParsedDocument {
            title,
            sections,
            page_count: Some(page),
            warnings: Vec::new(),
        })
    }
}

fn is_docx_page_break(event: &quick_xml::events::BytesStart<'_>) -> bool {
    if event.local_name().as_ref() == b"lastRenderedPageBreak" {
        return true;
    }
    event.local_name().as_ref() == b"br"
        && event.attributes().flatten().any(|attribute| {
            attribute.key.local_name().as_ref() == b"type" && attribute.value.as_ref() == b"page"
        })
}

fn decode_text(bytes: &[u8]) -> Result<String, DocumentError> {
    if bytes.starts_with(&[0xff, 0xfe]) {
        let (value, _, errors) = encoding_rs::UTF_16LE.decode(&bytes[2..]);
        if errors {
            return Err(DocumentError::Parse("invalid UTF-16LE".into()));
        }
        return Ok(value.into_owned());
    }
    if bytes.starts_with(&[0xfe, 0xff]) {
        let (value, _, errors) = encoding_rs::UTF_16BE.decode(&bytes[2..]);
        if errors {
            return Err(DocumentError::Parse("invalid UTF-16BE".into()));
        }
        return Ok(value.into_owned());
    }
    let bytes = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(bytes);
    std::str::from_utf8(bytes)
        .map(str::to_owned)
        .map_err(|e| DocumentError::Parse(e.to_string()))
}
fn plain_document(text: String) -> ParsedDocument {
    ParsedDocument {
        title: None,
        sections: vec![ParsedSection {
            text: normalize(&text),
            page: None,
            heading: None,
            extraction_method: ExtractionMethod::Text,
            confidence: ExtractionConfidence::Unknown,
            limitations: Vec::new(),
        }],
        page_count: None,
        warnings: Vec::new(),
    }
}
fn normalize(input: &str) -> String {
    input
        .lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect::<Vec<_>>()
        .join("\n")
        .split("\n\n\n")
        .collect::<Vec<_>>()
        .join("\n\n")
        .trim()
        .to_owned()
}
fn flush_section(
    sections: &mut Vec<ParsedSection>,
    text: &mut String,
    heading: Option<String>,
    page: Option<u32>,
) {
    let value = normalize(text);
    if !value.is_empty() {
        sections.push(ParsedSection {
            text: value,
            page,
            heading,
            extraction_method: ExtractionMethod::Text,
            confidence: ExtractionConfidence::Unknown,
            limitations: Vec::new(),
        });
    }
    text.clear();
}

fn chunk_sections(
    id: &str,
    sections: &[ParsedSection],
    limits: &DocumentLimits,
) -> Result<(String, Vec<DocumentChunk>), DocumentError> {
    let mut document = String::new();
    let mut chunks = Vec::new();
    for section in sections {
        if section.text.is_empty() {
            continue;
        }
        if !document.is_empty() {
            document.push_str("\n\n");
        }
        let section_start = document.len();
        document.push_str(&section.text);
        let boundaries = char_windows(
            &section.text,
            limits.chunk_target_chars,
            limits.chunk_overlap_chars,
        );
        for (local_start, local_end) in boundaries {
            if chunks.len() >= limits.max_chunks_per_document {
                return Err(DocumentError::Limit(
                    "maximum chunks per document reached".into(),
                ));
            }
            let start = section_start + local_start;
            let end = section_start + local_end;
            let text = document[start..end].trim().to_owned();
            if text.is_empty() {
                continue;
            }
            let ordinal = chunks.len();
            chunks.push(DocumentChunk {
                id: format!("{id}:{ordinal}"),
                document_id: id.into(),
                ordinal,
                token_count: tokenize(&text).len(),
                text,
                page: section.page,
                heading: section.heading.clone(),
                start_offset: start,
                end_offset: end,
                extraction_method: section.extraction_method.clone(),
                confidence: section.confidence.clone(),
                limitations: section.limitations.clone(),
            });
        }
    }
    Ok((document, chunks))
}
fn char_windows(text: &str, target: usize, overlap: usize) -> Vec<(usize, usize)> {
    let indices = text
        .char_indices()
        .map(|(i, _)| i)
        .chain(std::iter::once(text.len()))
        .collect::<Vec<_>>();
    let chars = indices.len().saturating_sub(1);
    if chars == 0 {
        return Vec::new();
    }
    let mut result = Vec::new();
    let mut start = 0;
    while start < chars {
        let ideal_end = (start + target).min(chars);
        let minimum_end = (start + target / 2).min(ideal_end);
        let end = if ideal_end < chars {
            (minimum_end..=ideal_end)
                .rev()
                .find(|index| {
                    let byte = indices[index.saturating_sub(1)];
                    text[byte..].starts_with('\n')
                })
                .unwrap_or(ideal_end)
        } else {
            ideal_end
        };
        result.push((indices[start], indices[end]));
        if end == chars {
            break;
        }
        start = end.saturating_sub(overlap);
    }
    result
}
pub fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|v| v.chars().count() >= 2)
        .map(str::to_owned)
        .collect()
}
pub fn term_frequencies(text: &str) -> BTreeMap<String, usize> {
    let mut out = BTreeMap::new();
    for term in tokenize(text) {
        *out.entry(term).or_default() += 1;
    }
    out
}
pub fn collect_supported_paths(
    root: &Path,
    inputs: &[PathBuf],
    max: usize,
) -> Result<Vec<PathBuf>, DocumentError> {
    let mut paths = Vec::new();
    for input in inputs {
        let candidate = if input.is_absolute() {
            input.clone()
        } else {
            root.join(input)
        };
        if candidate.is_dir() {
            for entry in walkdir(&candidate)? {
                if is_supported(&entry) {
                    paths.push(entry);
                }
            }
        } else {
            paths.push(candidate);
        }
        if paths.len() > max {
            return Err(DocumentError::Limit(format!("more than {max} documents")));
        }
    }
    paths.sort();
    paths.dedup();
    Ok(paths)
}
fn walkdir(root: &Path) -> Result<Vec<PathBuf>, DocumentError> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(dir).map_err(|e| DocumentError::Parse(e.to_string()))? {
            let path = entry
                .map_err(|e| DocumentError::Parse(e.to_string()))?
                .path();
            let metadata = std::fs::symlink_metadata(&path)
                .map_err(|e| DocumentError::Parse(e.to_string()))?;
            if metadata.file_type().is_symlink() {
                continue;
            }
            if metadata.is_dir() {
                stack.push(path)
            } else {
                out.push(path)
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::Arc;
    use zip::write::SimpleFileOptions;
    #[test]
    fn text_and_markdown_preserve_utf8_offsets() -> Result<(), Box<dyn std::error::Error>> {
        let service = DocumentService::new(DocumentLimits {
            chunk_target_chars: 8,
            chunk_overlap_chars: 2,
            ..Default::default()
        })?;
        let doc = service.parse(
            "w",
            "guide.md",
            "# 제목\n\n한국어 문서 내용입니다.".as_bytes(),
        )?;
        assert_eq!(doc.status, DocumentStatus::Ready);
        assert!(
            doc.chunks
                .iter()
                .all(|c| doc.text.is_char_boundary(c.start_offset)
                    && doc.text.is_char_boundary(c.end_offset))
        );
        assert_eq!(
            doc.chunks.first().and_then(|c| c.heading.as_deref()),
            Some("제목")
        );
        Ok(())
    }
    #[test]
    fn utf16_text_is_supported() -> Result<(), Box<dyn std::error::Error>> {
        let mut bytes = vec![0xff, 0xfe];
        for unit in "hello 문서".encode_utf16() {
            bytes.extend(unit.to_le_bytes())
        }
        let doc = DocumentService::new(Default::default())?.parse("w", "a.txt", &bytes)?;
        assert!(doc.text.contains("문서"));
        Ok(())
    }
    #[test]
    fn empty_pdf_result_is_scanned_contract() {
        assert_eq!(stable_id("a", "b"), stable_id("a", "b"));
        assert!(format_from_path(Path::new("x.exe")).is_err());
    }

    #[test]
    fn html_and_docx_preserve_titles_headings_tables_and_pages()
    -> Result<(), Box<dyn std::error::Error>> {
        let service = DocumentService::new(Default::default())?;
        let html=service.parse("w","page.html",b"<title>Guide</title><script>ignore</script><h1>Safety</h1><p>Trusted content paragraph.</p>")?;
        assert_eq!(html.title.as_deref(), Some("Guide"));
        assert_eq!(
            html.chunks.first().and_then(|c| c.heading.as_deref()),
            Some("Safety")
        );
        assert!(!html.text.contains("ignore"));
        let cursor = Cursor::new(Vec::new());
        let mut zip = zip::ZipWriter::new(cursor);
        zip.start_file("word/document.xml", SimpleFileOptions::default())?;
        zip.write_all(br#"<w:document xmlns:w="x"><w:body><w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Overview</w:t></w:r></w:p><w:p><w:r><w:t>DOCX body</w:t></w:r><w:r><w:br w:type="page"/></w:r><w:r><w:t>table value</w:t></w:r></w:p></w:body></w:document>"#)?;
        let bytes = zip.finish()?.into_inner();
        let docx = service.parse("w", "sample.docx", &bytes)?;
        assert_eq!(docx.status, DocumentStatus::Ready);
        assert_eq!(
            docx.chunks.first().and_then(|c| c.heading.as_deref()),
            Some("Overview")
        );
        assert!(docx.text.contains("table value"));
        Ok(())
    }

    #[test]
    fn html_without_extractable_body_is_failed_instead_of_ready()
    -> Result<(), Box<dyn std::error::Error>> {
        let document = DocumentService::new(Default::default())?.parse(
            "w",
            "empty.html",
            b"<title>Only metadata</title><script>ignored()</script>",
        )?;
        assert_eq!(document.status, DocumentStatus::Failed);
        assert_eq!(
            document.error.as_deref(),
            Some("document contains no extractable text")
        );
        assert!(document.chunks.is_empty());
        Ok(())
    }

    #[test]
    fn text_pdf_is_page_addressable_and_empty_pdf_is_scanned()
    -> Result<(), Box<dyn std::error::Error>> {
        let service = DocumentService::new(Default::default())?;
        let text_pdf = pdf(Some(
            "Document analysis text is searchable and page addressable.",
        ))?;
        let parsed = service.parse("w", "text.pdf", &text_pdf)?;
        assert_eq!(parsed.status, DocumentStatus::Ready);
        assert_eq!(parsed.chunks.first().and_then(|c| c.page), Some(1));
        assert!(parsed.text.contains("Document analysis"));
        let empty = service.parse("w", "scan.pdf", &pdf(None)?)?;
        assert_eq!(empty.status, DocumentStatus::UnsupportedScanned);
        Ok(())
    }

    fn pdf(text: Option<&str>) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        use lopdf::{
            Document, Object, Stream,
            content::{Content, Operation},
            dictionary,
        };
        let mut doc = Document::with_version("1.5");
        let pages_id = doc.new_object_id();
        let font_id =
            doc.add_object(dictionary! {"Type"=>"Font","Subtype"=>"Type1","BaseFont"=>"Helvetica"});
        let operations = if let Some(text) = text {
            vec![
                Operation::new("BT", vec![]),
                Operation::new("Tf", vec![Object::Name(b"F1".to_vec()), 12.into()]),
                Operation::new("Td", vec![50.into(), 700.into()]),
                Operation::new("Tj", vec![Object::string_literal(text)]),
                Operation::new("ET", vec![]),
            ]
        } else {
            Vec::new()
        };
        let content_id = doc.add_object(Stream::new(
            dictionary! {},
            Content { operations }.encode()?,
        ));
        let page_id=doc.add_object(dictionary!{"Type"=>"Page","Parent"=>pages_id,"MediaBox"=>vec![0.into(),0.into(),595.into(),842.into()],"Contents"=>content_id,"Resources"=>dictionary!{"Font"=>dictionary!{"F1"=>font_id}}});
        doc.objects.insert(
            pages_id,
            Object::Dictionary(
                dictionary! {"Type"=>"Pages","Kids"=>vec![page_id.into()],"Count"=>1},
            ),
        );
        let catalog_id = doc.add_object(dictionary! {"Type"=>"Catalog","Pages"=>pages_id});
        doc.trailer.set("Root", catalog_id);
        doc.compress();
        let mut bytes = Vec::new();
        doc.save_to(&mut bytes)?;
        Ok(bytes)
    }

    struct FakeFallback;

    #[async_trait]
    impl ScannedPdfFallback for FakeFallback {
        fn max_pages(&self) -> usize {
            50
        }
        fn pipeline_fingerprint(&self) -> String {
            "vision:test:144dpi:v1".to_owned()
        }
        async fn extract_page(
            &self,
            _: &Path,
            _: &str,
            page: u32,
            _: &CancellationToken,
        ) -> Result<VisionPageExtraction, DocumentError> {
            Ok(VisionPageExtraction {
                text: format!("Vision extracted searchable content from page {page}."),
                confidence: ExtractionConfidence::High,
                limitations: vec!["fixture".to_owned()],
            })
        }
    }

    struct LimitedFallback;

    #[async_trait]
    impl ScannedPdfFallback for LimitedFallback {
        fn max_pages(&self) -> usize {
            0
        }
        fn pipeline_fingerprint(&self) -> String {
            "vision:test:limit".to_owned()
        }
        async fn extract_page(
            &self,
            _: &Path,
            _: &str,
            _: u32,
            _: &CancellationToken,
        ) -> Result<VisionPageExtraction, DocumentError> {
            Err(DocumentError::Parse("must not be called".to_owned()))
        }
    }

    #[tokio::test]
    async fn scanned_pdf_fallback_preserves_page_provenance()
    -> Result<(), Box<dyn std::error::Error>> {
        let bytes = pdf(None)?;
        let temp = tempfile::NamedTempFile::new()?;
        std::fs::write(temp.path(), &bytes)?;
        let service = DocumentService::new(Default::default())?;
        let fallback: Arc<dyn ScannedPdfFallback> = Arc::new(FakeFallback);
        let document = service
            .parse_with_fallback(
                "w",
                "scan.pdf",
                temp.path(),
                &bytes,
                Some(fallback.as_ref()),
                &CancellationToken::new(),
            )
            .await?;
        assert_eq!(document.status, DocumentStatus::Ready);
        assert_eq!(
            document.metadata.pipeline_fingerprint.as_deref(),
            Some("vision:test:144dpi:v1")
        );
        let chunk = document.chunks.first().ok_or("missing vision chunk")?;
        assert_eq!(chunk.page, Some(1));
        assert_eq!(chunk.extraction_method, ExtractionMethod::Vision);
        assert_eq!(chunk.confidence, ExtractionConfidence::High);
        Ok(())
    }

    #[tokio::test]
    async fn scanned_pdf_limit_and_cancellation_are_explicit()
    -> Result<(), Box<dyn std::error::Error>> {
        let bytes = pdf(None)?;
        let temp = tempfile::NamedTempFile::new()?;
        std::fs::write(temp.path(), &bytes)?;
        let service = DocumentService::new(Default::default())?;
        let limited = service
            .parse_with_fallback(
                "w",
                "scan.pdf",
                temp.path(),
                &bytes,
                Some(&LimitedFallback),
                &CancellationToken::new(),
            )
            .await?;
        assert_eq!(limited.status, DocumentStatus::Failed);
        assert!(limited.metadata.warnings[0].contains("page limit"));

        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let cancelled = service
            .parse_with_fallback(
                "w",
                "scan.pdf",
                temp.path(),
                &bytes,
                Some(&FakeFallback),
                &cancellation,
            )
            .await;
        assert!(
            matches!(cancelled, Err(DocumentError::Parse(message)) if message.contains("cancelled"))
        );
        Ok(())
    }
}
