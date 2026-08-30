use crate::process::run_process;
use crate::{CommandProfile, Tool, ToolContext, ToolError, ToolRegistry, ToolResult};
use agent_model::ToolDefinition;
use agent_security::RiskLevel;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub(crate) fn register(registry: &mut ToolRegistry) -> Result<(), ToolError> {
    registry.register(CargoBuild)?;
    registry.register(CargoTest)?;
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CargoDiagnostic {
    pub level: String,
    pub code: Option<String>,
    pub message: String,
    pub file: Option<String>,
    pub line: Option<usize>,
    pub column: Option<usize>,
    pub rendered: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CargoArg {
    package: Option<String>,
    #[serde(default)]
    features: Vec<String>,
    #[serde(default)]
    all_features: bool,
    #[serde(default)]
    no_default_features: bool,
    target: Option<String>,
    timeout_seconds: Option<u64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CargoTestArg {
    package: Option<String>,
    test: Option<String>,
    test_filter: Option<String>,
    #[serde(default)]
    features: Vec<String>,
    #[serde(default)]
    all_features: bool,
    #[serde(default)]
    no_default_features: bool,
    target: Option<String>,
    timeout_seconds: Option<u64>,
}

struct CargoBuild;
struct CargoTest;

fn schema(name: &str, description: &str, test: bool) -> ToolDefinition {
    let mut properties = serde_json::Map::from_iter([
        ("package".to_owned(), json!({"type":"string"})),
        (
            "features".to_owned(),
            json!({"type":"array","items":{"type":"string"}}),
        ),
        ("all_features".to_owned(), json!({"type":"boolean"})),
        ("no_default_features".to_owned(), json!({"type":"boolean"})),
        ("target".to_owned(), json!({"type":"string"})),
        (
            "timeout_seconds".to_owned(),
            json!({"type":"integer","minimum":1}),
        ),
    ]);
    if test {
        properties.insert("test".to_owned(), json!({"type":"string"}));
        properties.insert("test_filter".to_owned(), json!({"type":"string"}));
    }
    ToolDefinition::function(
        name,
        description,
        json!({"type":"object","properties":properties,"additionalProperties":false}),
    )
}

fn validate_name(value: &str, kind: &str) -> Result<(), ToolError> {
    if value.is_empty() || value.starts_with('-') || value.contains(char::is_whitespace) {
        Err(ToolError::InvalidArguments(format!(
            "unsafe {kind}: {value}"
        )))
    } else {
        Ok(())
    }
}

fn add_common_args(
    command: &mut Vec<String>,
    package: Option<String>,
    features: Vec<String>,
    all_features: bool,
    no_default_features: bool,
    target: Option<String>,
) {
    if let Some(package) = package {
        command.extend(["--package".to_owned(), package]);
    }
    if !features.is_empty() {
        command.extend(["--features".to_owned(), features.join(",")]);
    }
    if all_features {
        command.push("--all-features".to_owned());
    }
    if no_default_features {
        command.push("--no-default-features".to_owned());
    }
    if let Some(target) = target {
        command.extend(["--target".to_owned(), target]);
    }
    command.push("--message-format=json-diagnostic-rendered-ansi".to_owned());
}

#[async_trait]
impl Tool for CargoBuild {
    fn definition(&self) -> ToolDefinition {
        schema(
            "cargo_build",
            "Build a Rust workspace and return structured diagnostics",
            false,
        )
    }
    fn risk(&self, _: &Value) -> Result<RiskLevel, ToolError> {
        Ok(RiskLevel::Execute)
    }
    fn validate(&self, value: &Value) -> Result<(), ToolError> {
        let args: CargoArg = serde_json::from_value(value.clone())
            .map_err(|error| ToolError::InvalidArguments(error.to_string()))?;
        validate_common(
            args.package.as_deref(),
            &args.features,
            args.target.as_deref(),
            args.timeout_seconds,
        )
    }
    async fn execute(&self, context: &ToolContext, value: Value) -> Result<ToolResult, ToolError> {
        let args: CargoArg = serde_json::from_value(value)
            .map_err(|error| ToolError::InvalidArguments(error.to_string()))?;
        let mut command = vec!["build".to_owned()];
        add_common_args(
            &mut command,
            args.package,
            args.features,
            args.all_features,
            args.no_default_features,
            args.target,
        );
        run_cargo(
            context,
            command,
            CommandProfile::CargoBuild,
            args.timeout_seconds,
            "cargo_build",
        )
        .await
    }
}

#[async_trait]
impl Tool for CargoTest {
    fn definition(&self) -> ToolDefinition {
        schema(
            "cargo_test",
            "Run Rust tests and return structured compiler and test failures",
            true,
        )
    }
    fn risk(&self, _: &Value) -> Result<RiskLevel, ToolError> {
        Ok(RiskLevel::Execute)
    }
    fn validate(&self, value: &Value) -> Result<(), ToolError> {
        let args: CargoTestArg = serde_json::from_value(value.clone())
            .map_err(|error| ToolError::InvalidArguments(error.to_string()))?;
        validate_common(
            args.package.as_deref(),
            &args.features,
            args.target.as_deref(),
            args.timeout_seconds,
        )?;
        if let Some(test) = args.test.as_deref() {
            validate_name(test, "test target")?;
        }
        if args
            .test_filter
            .as_deref()
            .is_some_and(|value| value.starts_with('-'))
        {
            return Err(ToolError::InvalidArguments(
                "test_filter cannot start with '-'".to_owned(),
            ));
        }
        Ok(())
    }
    async fn execute(&self, context: &ToolContext, value: Value) -> Result<ToolResult, ToolError> {
        let args: CargoTestArg = serde_json::from_value(value)
            .map_err(|error| ToolError::InvalidArguments(error.to_string()))?;
        let mut command = vec!["test".to_owned()];
        add_common_args(
            &mut command,
            args.package,
            args.features,
            args.all_features,
            args.no_default_features,
            args.target,
        );
        if let Some(test) = args.test {
            command.extend(["--test".to_owned(), test]);
        }
        if let Some(filter) = args.test_filter {
            command.extend(["--".to_owned(), filter]);
        }
        run_cargo(
            context,
            command,
            CommandProfile::CargoTest,
            args.timeout_seconds,
            "cargo_test",
        )
        .await
    }
}

fn validate_common(
    package: Option<&str>,
    features: &[String],
    target: Option<&str>,
    timeout: Option<u64>,
) -> Result<(), ToolError> {
    if let Some(package) = package {
        validate_name(package, "package")?;
    }
    for feature in features {
        validate_name(feature, "feature")?;
    }
    if let Some(target) = target {
        validate_name(target, "target")?;
    }
    if timeout == Some(0) {
        return Err(ToolError::InvalidArguments(
            "timeout must be positive".to_owned(),
        ));
    }
    Ok(())
}

async fn run_cargo(
    context: &ToolContext,
    args: Vec<String>,
    profile: CommandProfile,
    requested_timeout: Option<u64>,
    kind: &str,
) -> Result<ToolResult, ToolError> {
    let mut limits = context.limits.command_limits(profile);
    limits.timeout_seconds = limits.bounded_timeout(requested_timeout);
    let mut result = run_process(
        context,
        "cargo",
        &args,
        context.workspace.root(),
        &BTreeMap::new(),
        &limits,
    )
    .await?;
    let stdout = result.content["stdout"].as_str().unwrap_or_default();
    let stderr = result.content["stderr"].as_str().unwrap_or_default();
    let diagnostics = parse_diagnostics(stdout);
    let test_failures = parse_test_failures(stdout, stderr);
    let outcome = result.metadata["outcome"].as_str().unwrap_or("completed");
    let success = result.metadata["success"].as_bool() == Some(true);
    let failure_kind = if outcome == "timeout" {
        Some("timeout")
    } else if diagnostics
        .iter()
        .any(|diagnostic| diagnostic.level == "error")
    {
        Some("compiler")
    } else if !test_failures.is_empty() {
        Some("test")
    } else if !success {
        Some("command")
    } else {
        None
    };
    let fingerprint =
        failure_kind.map(|failure_kind| fingerprint(failure_kind, &diagnostics, &test_failures));
    result.summary = if success {
        format!("{kind} succeeded")
    } else {
        format!("{kind} failed ({})", failure_kind.unwrap_or("command"))
    };
    result.metadata["kind"] = json!(kind);
    result.metadata["failure_kind"] = json!(failure_kind);
    result.metadata["failure_fingerprint"] = json!(fingerprint);
    result.metadata["diagnostics"] = json!(diagnostics);
    result.metadata["test_failures"] = json!(test_failures);
    result.content["diagnostics"] = json!(diagnostics);
    result.content["test_failures"] = json!(test_failures);
    Ok(result)
}

fn parse_diagnostics(output: &str) -> Vec<CargoDiagnostic> {
    output
        .lines()
        .filter_map(|line| {
            let value: Value = serde_json::from_str(line).ok()?;
            if value["reason"] != "compiler-message" {
                return None;
            }
            let message = &value["message"];
            let primary = message["spans"]
                .as_array()
                .and_then(|spans| spans.iter().find(|span| span["is_primary"] == true));
            Some(CargoDiagnostic {
                level: message["level"].as_str().unwrap_or("unknown").to_owned(),
                code: message["code"]["code"].as_str().map(str::to_owned),
                message: message["message"].as_str().unwrap_or_default().to_owned(),
                file: primary
                    .and_then(|span| span["file_name"].as_str())
                    .map(str::to_owned),
                line: primary
                    .and_then(|span| span["line_start"].as_u64())
                    .and_then(|line| usize::try_from(line).ok()),
                column: primary
                    .and_then(|span| span["column_start"].as_u64())
                    .and_then(|column| usize::try_from(column).ok()),
                rendered: message["rendered"].as_str().map(str::to_owned),
            })
        })
        .take(50)
        .collect()
}

fn parse_test_failures(stdout: &str, stderr: &str) -> Vec<String> {
    stdout
        .lines()
        .chain(stderr.lines())
        .filter_map(|line| {
            let trimmed = line.trim();
            if (trimmed.starts_with("test ") && trimmed.ends_with("FAILED"))
                || trimmed.contains("panicked at")
                || trimmed.starts_with("failures:")
            {
                Some(trimmed.to_owned())
            } else {
                None
            }
        })
        .take(50)
        .collect()
}

fn fingerprint(kind: &str, diagnostics: &[CargoDiagnostic], tests: &[String]) -> String {
    let normalized = json!({
        "kind":kind,
        "diagnostics":diagnostics.iter().map(|diagnostic| (&diagnostic.code, &diagnostic.message, &diagnostic.file, diagnostic.line)).collect::<Vec<_>>(),
        "tests":tests,
    });
    hex::encode(Sha256::digest(normalized.to_string().as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ExecutionLimits;
    use std::fs;
    use tokio_util::sync::CancellationToken;

    #[test]
    fn parses_compiler_and_test_failures() {
        let diagnostic = r#"{"reason":"compiler-message","message":{"level":"error","message":"mismatched types","code":{"code":"E0308"},"spans":[{"is_primary":true,"file_name":"src/lib.rs","line_start":4,"column_start":8}],"rendered":"error[E0308]"}}"#;
        let parsed = parse_diagnostics(diagnostic);
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].code.as_deref(), Some("E0308"));
        assert_eq!(
            parse_test_failures("test tests::broken ... FAILED", "").len(),
            1
        );
    }

    #[tokio::test]
    async fn classifies_real_rust_fixture() -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        fs::create_dir(root.path().join("src"))?;
        fs::write(
            root.path().join("Cargo.toml"),
            "[package]\nname = \"broken-fixture\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )?;
        fs::write(
            root.path().join("src/lib.rs"),
            "pub fn value() -> u32 { \"wrong\" }\n",
        )?;
        let context = ToolContext {
            session_id: agent_security::SessionId::new(),
            task_id: agent_security::TaskId::new(),
            call_id: agent_security::ToolCallId::new(),
            workspace: agent_security::WorkspaceGuard::new(root.path())?,
            cancellation: CancellationToken::new(),
            limits: ExecutionLimits::default(),
        };
        let build = CargoBuild.execute(&context, json!({})).await?;
        assert_eq!(build.metadata["failure_kind"], "compiler");
        assert!(
            build.metadata["diagnostics"]
                .as_array()
                .is_some_and(|items| !items.is_empty())
        );

        fs::write(
            root.path().join("src/lib.rs"),
            "pub fn value() -> u32 { 1 }\n#[cfg(test)] mod tests { #[test] fn broken() { assert_eq!(super::value(), 2); } }\n",
        )?;
        let test = CargoTest.execute(&context, json!({})).await?;
        assert_eq!(test.metadata["failure_kind"], "test");
        assert!(
            test.metadata["test_failures"]
                .as_array()
                .is_some_and(|items| !items.is_empty())
        );
        Ok(())
    }
}
