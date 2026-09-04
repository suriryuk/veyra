use crate::{Tool, ToolContext, ToolError, ToolRegistry, ToolResult};
use agent_document::{
    DocumentRepository, DocumentSearchQuery, DocumentService, ScannedPdfFallback,
    collect_supported_paths,
};
use agent_model::ToolDefinition;
use agent_security::RiskLevel;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::Arc;

pub fn register_document_tools(
    registry: &mut ToolRegistry,
    repository: Arc<dyn DocumentRepository>,
    service: DocumentService,
    fallback: Option<Arc<dyn ScannedPdfFallback>>,
) -> Result<(), ToolError> {
    registry.register(DocumentIndex {
        repository: repository.clone(),
        service: service.clone(),
        fallback,
    })?;
    registry.register(DocumentList {
        repository: repository.clone(),
    })?;
    registry.register(DocumentSearch {
        repository,
        service,
    })
}

struct DocumentIndex {
    repository: Arc<dyn DocumentRepository>,
    service: DocumentService,
    fallback: Option<Arc<dyn ScannedPdfFallback>>,
}
struct DocumentList {
    repository: Arc<dyn DocumentRepository>,
}
struct DocumentSearch {
    repository: Arc<dyn DocumentRepository>,
    service: DocumentService,
}

#[derive(Deserialize)]
struct IndexArgs {
    paths: Vec<PathBuf>,
}
#[derive(Deserialize)]
struct ListArgs {
    limit: Option<usize>,
}
#[derive(Deserialize)]
struct SearchArgs {
    query: String,
    #[serde(default)]
    document_ids: Vec<String>,
    limit: Option<usize>,
}

fn definition(
    name: &str,
    description: &str,
    properties: Value,
    required: Vec<&str>,
) -> ToolDefinition {
    ToolDefinition::function(
        name,
        description,
        json!({"type":"object","properties":properties,"required":required,"additionalProperties":false}),
    )
}
fn decode<T: for<'de> Deserialize<'de>>(value: Value) -> Result<T, ToolError> {
    serde_json::from_value(value).map_err(|e| ToolError::InvalidArguments(e.to_string()))
}

#[async_trait]
impl Tool for DocumentIndex {
    fn definition(&self) -> ToolDefinition {
        definition(
            "document_index",
            "Parse and persistently index supported workspace documents.",
            json!({"paths":{"type":"array","items":{"type":"string"},"minItems":1}}),
            vec!["paths"],
        )
    }
    fn risk(&self, _: &Value) -> Result<RiskLevel, ToolError> {
        Ok(RiskLevel::Read)
    }
    fn validate(&self, args: &Value) -> Result<(), ToolError> {
        let value: IndexArgs = decode(args.clone())?;
        if value.paths.is_empty() {
            return Err(ToolError::InvalidArguments(
                "paths must not be empty".into(),
            ));
        }
        Ok(())
    }
    async fn execute(&self, context: &ToolContext, args: Value) -> Result<ToolResult, ToolError> {
        let args: IndexArgs = decode(args)?;
        let safe_inputs = args
            .paths
            .into_iter()
            .map(|path| context.workspace.resolve_existing(path))
            .collect::<Result<Vec<_>, _>>()?;
        let paths = collect_supported_paths(
            context.workspace.root(),
            &safe_inputs,
            self.service.limits().max_documents_per_request,
        )
        .map_err(|e| ToolError::InvalidArguments(e.to_string()))?;
        let mut results = Vec::new();
        let mut audit_documents = Vec::new();
        for path in paths {
            if context.cancellation.is_cancelled() {
                return Err(ToolError::Cancelled);
            }
            let resolved = context.workspace.resolve_existing(&path)?;
            let relative = resolved
                .strip_prefix(context.workspace.root())
                .map_err(|e| ToolError::Policy(e.to_string()))?
                .to_string_lossy()
                .replace('\\', "/");
            let bytes = tokio::fs::read(&resolved)
                .await
                .map_err(|e| ToolError::Io(e.to_string()))?;
            let document = self
                .service
                .parse_with_fallback(
                    &context.workspace.root().display().to_string(),
                    &relative,
                    &resolved,
                    &bytes,
                    self.fallback.as_deref(),
                    &context.cancellation,
                )
                .await
                .map_err(|e| ToolError::Execution(e.to_string()))?;
            let vision_pages = document
                .chunks
                .iter()
                .filter(|chunk| chunk.extraction_method == agent_document::ExtractionMethod::Vision)
                .filter_map(|chunk| chunk.page)
                .collect::<std::collections::BTreeSet<_>>();
            audit_documents.push(json!({
                "path": relative,
                "status": document.status,
                "pipeline_fingerprint": document.metadata.pipeline_fingerprint,
                "vision_pages_succeeded": vision_pages,
                "page_failures_or_limits": document.metadata.warnings,
            }));
            results.push(
                self.repository
                    .upsert(&document)
                    .await
                    .map_err(|e| ToolError::Execution(e.to_string()))?,
            );
        }
        let count = results.len();
        Ok(ToolResult {
            content: json!({"kind":"document_index","documents":results}),
            summary: format!("indexed {count} documents"),
            truncated: false,
            metadata: json!({
                "kind":"document_index",
                "document_count":count,
                "vision_route":"vision",
                "pdf_rendering":"pdftoppm",
                "documents":audit_documents,
            }),
        })
    }
}

#[async_trait]
impl Tool for DocumentList {
    fn definition(&self) -> ToolDefinition {
        definition(
            "document_list",
            "List persistently indexed documents in the current workspace.",
            json!({"limit":{"type":"integer","minimum":1,"maximum":100}}),
            vec![],
        )
    }
    fn risk(&self, _: &Value) -> Result<RiskLevel, ToolError> {
        Ok(RiskLevel::Read)
    }
    fn validate(&self, args: &Value) -> Result<(), ToolError> {
        let value: ListArgs = decode(args.clone())?;
        if value.limit == Some(0) {
            return Err(ToolError::InvalidArguments("limit must be positive".into()));
        }
        Ok(())
    }
    async fn execute(&self, context: &ToolContext, args: Value) -> Result<ToolResult, ToolError> {
        let args: ListArgs = decode(args)?;
        let limit = args.limit.unwrap_or(50).min(100);
        let docs = self
            .repository
            .list(&context.workspace.root().display().to_string(), None, limit)
            .await
            .map_err(|e| ToolError::Execution(e.to_string()))?;
        let count = docs.len();
        Ok(ToolResult {
            content: json!({"kind":"document_list","documents":docs}),
            summary: format!("listed {count} documents"),
            truncated: false,
            metadata: json!({"kind":"document_list","document_count":count}),
        })
    }
}

#[async_trait]
impl Tool for DocumentSearch {
    fn definition(&self) -> ToolDefinition {
        definition(
            "document_search",
            "Search indexed document chunks with keyword and BM25 ranking and return source citations.",
            json!({"query":{"type":"string","minLength":1},"document_ids":{"type":"array","items":{"type":"string"}},"limit":{"type":"integer","minimum":1,"maximum":50}}),
            vec!["query"],
        )
    }
    fn risk(&self, _: &Value) -> Result<RiskLevel, ToolError> {
        Ok(RiskLevel::Read)
    }
    fn validate(&self, args: &Value) -> Result<(), ToolError> {
        let value: SearchArgs = decode(args.clone())?;
        if value.query.trim().is_empty() {
            return Err(ToolError::InvalidArguments(
                "query must not be empty".into(),
            ));
        }
        if value
            .limit
            .is_some_and(|v| v == 0 || v > self.service.limits().max_search_limit)
        {
            return Err(ToolError::InvalidArguments(
                "limit is outside configured range".into(),
            ));
        }
        Ok(())
    }
    async fn execute(&self, context: &ToolContext, args: Value) -> Result<ToolResult, ToolError> {
        let args: SearchArgs = decode(args)?;
        let limit = args
            .limit
            .unwrap_or(self.service.limits().default_search_limit)
            .min(self.service.limits().max_search_limit);
        let hits = self
            .repository
            .search(DocumentSearchQuery {
                workspace: context.workspace.root().display().to_string(),
                query: args.query.clone(),
                document_ids: args.document_ids,
                limit,
            })
            .await
            .map_err(|e| ToolError::Execution(e.to_string()))?;
        let count = hits.len();
        Ok(ToolResult {
            content: json!({"kind":"document_search","query":args.query,"hits":hits}),
            summary: format!("document search returned {count} chunks"),
            truncated: false,
            metadata: json!({"kind":"document_search","result_count":count}),
        })
    }
}
