use crate::{Tool, ToolContext, ToolError, ToolRegistry, ToolResult};
use agent_model::ToolDefinition;
use agent_security::RiskLevel;
use agent_vision::VisionService;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::Arc;

pub fn register_vision_tool(
    registry: &mut ToolRegistry,
    service: Arc<VisionService>,
) -> Result<(), ToolError> {
    registry.register(VisionAnalyze { service })
}

struct VisionAnalyze {
    service: Arc<VisionService>,
}

#[derive(Deserialize)]
struct VisionArgs {
    paths: Vec<PathBuf>,
    prompt: String,
}

#[async_trait]
impl Tool for VisionAnalyze {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "vision_analyze",
            "Analyze workspace PNG, JPEG, or WebP images, screenshots, and diagrams with the configured local vision model.",
            json!({
                "type":"object",
                "properties":{
                    "paths":{"type":"array","items":{"type":"string"},"minItems":1,"maxItems":8},
                    "prompt":{"type":"string","minLength":1}
                },
                "required":["paths","prompt"],
                "additionalProperties":false
            }),
        )
    }

    fn risk(&self, _: &Value) -> Result<RiskLevel, ToolError> {
        Ok(RiskLevel::Read)
    }

    fn validate(&self, arguments: &Value) -> Result<(), ToolError> {
        let args: VisionArgs = serde_json::from_value(arguments.clone())
            .map_err(|error| ToolError::InvalidArguments(error.to_string()))?;
        if args.paths.is_empty() || args.paths.len() > self.service.limits().max_images_per_request
        {
            return Err(ToolError::InvalidArguments(
                "paths count is outside configured vision limits".to_owned(),
            ));
        }
        if args.prompt.trim().is_empty() {
            return Err(ToolError::InvalidArguments(
                "prompt must not be empty".to_owned(),
            ));
        }
        Ok(())
    }

    async fn execute(
        &self,
        context: &ToolContext,
        arguments: Value,
    ) -> Result<ToolResult, ToolError> {
        let args: VisionArgs = serde_json::from_value(arguments)
            .map_err(|error| ToolError::InvalidArguments(error.to_string()))?;
        let mut paths = Vec::with_capacity(args.paths.len());
        for path in args.paths {
            let resolved = context.workspace.resolve_existing(&path)?;
            if !resolved.is_file() {
                return Err(ToolError::InvalidArguments(format!(
                    "{} is not a file",
                    path.display()
                )));
            }
            let relative = resolved
                .strip_prefix(context.workspace.root())
                .map_err(|error| ToolError::Policy(error.to_string()))?
                .to_string_lossy()
                .replace('\\', "/");
            paths.push((relative, resolved));
        }
        let result = self
            .service
            .analyze_paths(&paths, &args.prompt, &context.cancellation)
            .await
            .map_err(|error| ToolError::Execution(error.to_string()))?;
        let count = result.sources.len();
        let model_id = result.model_id.clone();
        let citations = result
            .sources
            .iter()
            .map(|source| source.citation.clone())
            .collect::<Vec<_>>();
        Ok(ToolResult {
            content: json!({"kind":"vision","result":result}),
            summary: format!("analyzed {count} images with {model_id}"),
            truncated: false,
            metadata: json!({
                "kind":"vision",
                "source_count":count,
                "model_id":model_id,
                "model_route":"vision",
                "model_transition":"loaded",
                "citations":citations,
            }),
        })
    }
}
