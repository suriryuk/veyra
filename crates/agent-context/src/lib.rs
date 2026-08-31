use agent_model::{Message, Role, ToolDefinition};
use globset::{Glob, GlobSetBuilder};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use thiserror::Error;
use walkdir::{DirEntry, WalkDir};

const MESSAGE_OVERHEAD: usize = 8;
const TOOL_OVERHEAD: usize = 12;
const DEFAULT_RANGE_RADIUS: usize = 20;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ContextProfileName {
    Default,
    Large,
    Legacy,
}

impl ContextProfileName {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Large => "large",
            Self::Legacy => "legacy",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenBudget {
    pub context_limit: usize,
    pub system: usize,
    pub task_plan: usize,
    pub source: usize,
    pub observation: usize,
    pub conversation: usize,
    pub output_reserve: usize,
}

impl TokenBudget {
    #[must_use]
    pub fn default_32k() -> Self {
        Self {
            context_limit: 32_768,
            system: 2_048,
            task_plan: 2_048,
            source: 16_384,
            observation: 6_144,
            conversation: 4_096,
            output_reserve: 2_048,
        }
    }

    #[must_use]
    pub fn large_65k() -> Self {
        Self {
            context_limit: 65_536,
            system: 4_096,
            task_plan: 4_096,
            source: 32_768,
            observation: 12_288,
            conversation: 8_192,
            output_reserve: 4_096,
        }
    }

    #[must_use]
    pub fn legacy(context_limit: usize) -> Self {
        let output_reserve = (context_limit / 16).clamp(256, 2_048);
        let input = context_limit.saturating_sub(output_reserve);
        Self {
            context_limit,
            system: input / 16,
            task_plan: input / 16,
            source: input / 2,
            observation: input * 3 / 16,
            conversation: input * 2 / 16,
            output_reserve,
        }
    }

    #[must_use]
    pub fn input_limit(&self) -> usize {
        self.context_limit.saturating_sub(self.output_reserve)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContextProfile {
    pub name: ContextProfileName,
    pub budget: TokenBudget,
}

impl ContextProfile {
    #[must_use]
    pub fn default_32k() -> Self {
        Self {
            name: ContextProfileName::Default,
            budget: TokenBudget::default_32k(),
        }
    }

    #[must_use]
    pub fn large_65k() -> Self {
        Self {
            name: ContextProfileName::Large,
            budget: TokenBudget::large_65k(),
        }
    }

    #[must_use]
    pub fn legacy(context_limit: usize) -> Self {
        Self {
            name: ContextProfileName::Legacy,
            budget: TokenBudget::legacy(context_limit),
        }
    }
}

pub trait TokenEstimator: Send + Sync {
    fn estimate_text(&self, text: &str) -> usize;

    fn estimate_message(&self, message: &Message) -> usize {
        MESSAGE_OVERHEAD
            + message
                .content
                .as_deref()
                .map_or(0, |content| self.estimate_text(content))
            + message.tool_calls.as_ref().map_or(0, |calls| {
                serde_json::to_string(calls).map_or(0, |value| self.estimate_text(&value))
            })
    }

    fn estimate_tools(&self, tools: &[ToolDefinition]) -> usize {
        serde_json::to_string(tools).map_or(TOOL_OVERHEAD, |value| {
            TOOL_OVERHEAD + self.estimate_text(&value)
        })
    }
}

#[derive(Debug, Default)]
pub struct ConservativeEstimator;

impl TokenEstimator for ConservativeEstimator {
    fn estimate_text(&self, text: &str) -> usize {
        text.len().div_ceil(3).max(1)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceSnippet {
    pub path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub content: String,
    pub score: usize,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RetrievalReport {
    pub backend: String,
    pub candidates: usize,
    pub selected: usize,
}

#[derive(Debug, Clone)]
pub struct RetrievalResult {
    pub snippets: Vec<SourceSnippet>,
    pub report: RetrievalReport,
}

#[derive(Debug, Error)]
pub enum ContextError {
    #[error("retrieval failed: {0}")]
    Retrieval(String),
    #[error("mandatory context exceeds the profile input limit")]
    MandatoryOverflow,
}

pub trait RepositoryRetriever: Send + Sync {
    fn retrieve(&self, task: &str, max_snippets: usize) -> Result<RetrievalResult, ContextError>;
}

#[derive(Debug, Clone)]
pub struct WorkspaceRetriever {
    root: PathBuf,
}

impl WorkspaceRetriever {
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn retrieve_with_rg(
        &self,
        task: &str,
        terms: &[String],
        max_snippets: usize,
    ) -> Result<Option<RetrievalResult>, ContextError> {
        let files = Command::new("rg")
            .args([
                "--files",
                "--hidden",
                "-g",
                "!.git/**",
                "-g",
                "!target/**",
                "-g",
                "!node_modules/**",
                "-g",
                "!models/**",
                "-g",
                "!logs/**",
            ])
            .current_dir(&self.root)
            .output();
        let Ok(files) = files else { return Ok(None) };
        if !files.status.success() {
            return Ok(None);
        }
        let file_names = String::from_utf8_lossy(&files.stdout);
        let mut candidates: BTreeSet<String> = file_names
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(str::to_owned)
            .collect();
        let pattern = terms
            .iter()
            .map(|term| regex::escape(term))
            .collect::<Vec<_>>()
            .join("|");
        let mut hits = glob_hits(&candidates, &task_globs(task));
        if !pattern.is_empty() {
            let output = Command::new("rg")
                .args([
                    "-n",
                    "--no-heading",
                    "--color",
                    "never",
                    "--hidden",
                    "-g",
                    "!.git/**",
                    "-g",
                    "!target/**",
                    "-g",
                    "!node_modules/**",
                    "-g",
                    "!models/**",
                    "-g",
                    "!logs/**",
                    "-e",
                    &pattern,
                    ".",
                ])
                .current_dir(&self.root)
                .output()
                .map_err(|error| ContextError::Retrieval(error.to_string()))?;
            if output.status.success() || output.status.code() == Some(1) {
                hits.extend(parse_rg_hits(&String::from_utf8_lossy(&output.stdout)));
            }
        }
        let candidate_count = candidates.len();
        let snippets = build_snippets(&self.root, terms, &mut candidates, hits, max_snippets);
        Ok(Some(RetrievalResult {
            report: RetrievalReport {
                backend: "ripgrep".to_owned(),
                candidates: candidate_count,
                selected: snippets.len(),
            },
            snippets,
        }))
    }

    fn retrieve_fallback(
        &self,
        task: &str,
        terms: &[String],
        max_snippets: usize,
    ) -> RetrievalResult {
        let ignored = read_ignore_patterns(&self.root);
        let mut candidates = BTreeSet::new();
        let mut hits = Vec::new();
        for entry in WalkDir::new(&self.root)
            .follow_links(false)
            .into_iter()
            .filter_entry(allowed_entry)
            .filter_map(Result::ok)
            .filter(|entry| entry.file_type().is_file())
        {
            let Ok(relative) = entry.path().strip_prefix(&self.root) else {
                continue;
            };
            let relative = relative.to_string_lossy().replace('\\', "/");
            if is_ignored(&relative, &ignored) {
                continue;
            }
            candidates.insert(relative.clone());
            let Ok(bytes) = std::fs::read(entry.path()) else {
                continue;
            };
            if bytes.len() > 1_048_576 || bytes.iter().take(8_192).any(|byte| *byte == 0) {
                continue;
            }
            let Ok(text) = std::str::from_utf8(&bytes) else {
                continue;
            };
            for (index, line) in text.lines().enumerate() {
                let lower = line.to_lowercase();
                let score = terms
                    .iter()
                    .filter(|term| lower.contains(term.as_str()))
                    .count();
                if score > 0 {
                    hits.push((relative.clone(), index + 1, score));
                }
            }
        }
        hits.extend(glob_hits(&candidates, &task_globs(task)));
        let candidate_count = candidates.len();
        let snippets = build_snippets(&self.root, terms, &mut candidates, hits, max_snippets);
        RetrievalResult {
            report: RetrievalReport {
                backend: "rust_fallback".to_owned(),
                candidates: candidate_count,
                selected: snippets.len(),
            },
            snippets,
        }
    }
}

fn read_ignore_patterns(root: &Path) -> Vec<String> {
    std::fs::read_to_string(root.join(".gitignore"))
        .ok()
        .into_iter()
        .flat_map(|contents| {
            contents
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty() && !line.starts_with('#') && !line.starts_with('!'))
                .map(|line| line.trim_start_matches('/').to_owned())
                .collect::<Vec<_>>()
        })
        .collect()
}

fn is_ignored(path: &str, patterns: &[String]) -> bool {
    patterns.iter().any(|pattern| {
        if let Some(suffix) = pattern.strip_prefix("*.") {
            return path
                .rsplit('.')
                .next()
                .is_some_and(|extension| extension == suffix);
        }
        let pattern = pattern.trim_end_matches('/');
        path == pattern || path.starts_with(&format!("{pattern}/"))
    })
}

impl RepositoryRetriever for WorkspaceRetriever {
    fn retrieve(&self, task: &str, max_snippets: usize) -> Result<RetrievalResult, ContextError> {
        let terms = task_terms(task);
        if let Some(result) = self.retrieve_with_rg(task, &terms, max_snippets)? {
            return Ok(result);
        }
        Ok(self.retrieve_fallback(task, &terms, max_snippets))
    }
}

fn allowed_entry(entry: &DirEntry) -> bool {
    if entry.depth() == 0 {
        return true;
    }
    !matches!(
        entry.file_name().to_string_lossy().as_ref(),
        ".git" | "target" | "node_modules" | "models" | "logs"
    )
}

fn parse_rg_hits(output: &str) -> Vec<(String, usize, usize)> {
    output
        .lines()
        .filter_map(|line| {
            let mut parts = line.splitn(3, ':');
            let path = parts.next()?.trim_start_matches("./").replace('\\', "/");
            let line = parts.next()?.parse().ok()?;
            Some((path, line, 1))
        })
        .collect()
}

fn task_globs(task: &str) -> Vec<String> {
    task.split_whitespace()
        .map(|value| value.trim_matches(['\'', '"', '`', ',', ';', ':', '(', ')']))
        .filter(|value| value.contains('*') || value.contains('?'))
        .map(|value| value.replace('\\', "/"))
        .take(8)
        .collect()
}

fn glob_hits(candidates: &BTreeSet<String>, patterns: &[String]) -> Vec<(String, usize, usize)> {
    let mut builder = GlobSetBuilder::new();
    let mut added = false;
    for pattern in patterns {
        if let Ok(glob) = Glob::new(pattern) {
            builder.add(glob);
            added = true;
        }
    }
    if !added {
        return Vec::new();
    }
    let Ok(globs) = builder.build() else {
        return Vec::new();
    };
    candidates
        .iter()
        .filter(|path| globs.is_match(path))
        .map(|path| (path.clone(), 1, 3))
        .collect()
}

fn build_snippets(
    root: &Path,
    terms: &[String],
    candidates: &mut BTreeSet<String>,
    hits: Vec<(String, usize, usize)>,
    max_snippets: usize,
) -> Vec<SourceSnippet> {
    let mut ranked = Vec::new();
    let mut seen = HashSet::new();
    for (path, line, hit_score) in hits {
        let key = (path.clone(), line / (DEFAULT_RANGE_RADIUS * 2 + 1));
        if !seen.insert(key) {
            continue;
        }
        let path_score = terms
            .iter()
            .filter(|term| path.to_lowercase().contains(term.as_str()))
            .count();
        ranked.push((hit_score * 10 + path_score * 5, path, line));
    }
    if ranked.is_empty() {
        for path in candidates.iter() {
            let score = terms
                .iter()
                .filter(|term| path.to_lowercase().contains(term.as_str()))
                .count();
            if score > 0 {
                ranked.push((score * 5, path.clone(), 1));
            }
        }
    }
    ranked.sort_by(|left, right| {
        right
            .0
            .cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.cmp(&right.2))
    });
    ranked
        .into_iter()
        .take(max_snippets)
        .filter_map(|(score, path, line)| read_snippet(root, path, line, score))
        .collect()
}

fn read_snippet(root: &Path, path: String, line: usize, score: usize) -> Option<SourceSnippet> {
    let bytes = std::fs::read(root.join(&path)).ok()?;
    if bytes.len() > 1_048_576 || bytes.iter().take(8_192).any(|byte| *byte == 0) {
        return None;
    }
    let text = std::str::from_utf8(&bytes).ok()?;
    let lines: Vec<_> = text.lines().collect();
    if lines.is_empty() {
        return None;
    }
    let start = line.saturating_sub(DEFAULT_RANGE_RADIUS).max(1);
    let end = (line + DEFAULT_RANGE_RADIUS).min(lines.len());
    let content = lines[start - 1..end]
        .iter()
        .enumerate()
        .map(|(index, value)| format!("{}: {value}", start + index))
        .collect::<Vec<_>>()
        .join("\n");
    Some(SourceSnippet {
        path,
        start_line: start,
        end_line: end,
        content,
        score,
        reason: format!("task term match near line {line}"),
    })
}

fn task_terms(task: &str) -> Vec<String> {
    let token = Regex::new(r"[\p{L}\p{N}_./\\*-]{3,}").ok();
    let stop = [
        "the",
        "and",
        "for",
        "with",
        "from",
        "this",
        "that",
        "into",
        "개발",
        "구현",
        "문서",
        "수정",
        "해주세요",
        "please",
        "implement",
    ];
    let mut terms = BTreeSet::new();
    if let Some(token) = token {
        for found in token.find_iter(task) {
            let value = found
                .as_str()
                .trim_matches(['.', '/', '\\', '*'])
                .to_lowercase();
            if value.len() >= 3 && !stop.contains(&value.as_str()) {
                terms.insert(value);
            }
        }
    }
    terms.into_iter().take(12).collect()
}

#[derive(Debug, Clone)]
pub struct ContextInput<'a> {
    pub system_prompt: &'a str,
    pub task: &'a str,
    pub plan: &'a [String],
    pub history: &'a [Message],
    pub sources: &'a [SourceSnippet],
    pub tools: &'a [ToolDefinition],
    pub aggressive: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContextUsage {
    pub context_limit: usize,
    pub input_limit: usize,
    pub system_tokens: usize,
    pub task_plan_tokens: usize,
    pub source_tokens: usize,
    pub observation_tokens: usize,
    pub conversation_tokens: usize,
    pub tool_tokens: usize,
    pub prompt_tokens: usize,
    pub output_reserve: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContextReport {
    pub profile: ContextProfileName,
    pub usage: ContextUsage,
    pub selected_sources: usize,
    pub dropped_sources: usize,
    pub selected_message_groups: usize,
    pub dropped_message_groups: usize,
    pub compressed_observations: usize,
    pub aggressive: bool,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ContextBuild {
    pub messages: Vec<Message>,
    pub max_output_tokens: u32,
    pub report: ContextReport,
}

pub struct ContextManager {
    profile: ContextProfile,
    estimator: Box<dyn TokenEstimator>,
}

impl ContextManager {
    #[must_use]
    pub fn new(profile: ContextProfile) -> Self {
        Self {
            profile,
            estimator: Box::<ConservativeEstimator>::default(),
        }
    }

    #[must_use]
    pub fn profile(&self) -> &ContextProfile {
        &self.profile
    }

    pub fn build(&self, input: ContextInput<'_>) -> Result<ContextBuild, ContextError> {
        let budget = &self.profile.budget;
        let input_limit = budget.input_limit();
        let tool_tokens = self.estimator.estimate_tools(input.tools);
        let system_text =
            truncate_to_tokens(input.system_prompt, budget.system, self.estimator.as_ref());
        let system = Message::system(system_text);
        let plan_text = if input.plan.is_empty() {
            input.task.to_owned()
        } else {
            format!(
                "Current task:\n{}\n\nCurrent plan:\n{}",
                input.task,
                input.plan.join("\n")
            )
        };
        let task = Message::user(truncate_to_tokens(
            &plan_text,
            budget.task_plan,
            self.estimator.as_ref(),
        ));
        let system_tokens = self.estimator.estimate_message(&system);
        let task_plan_tokens = self.estimator.estimate_message(&task);
        if tool_tokens + system_tokens + task_plan_tokens > input_limit {
            return Err(ContextError::MandatoryOverflow);
        }

        let mut messages = vec![system, task];
        let mut used = tool_tokens + system_tokens + task_plan_tokens;
        let source_cap = if input.aggressive {
            budget.source / 2
        } else {
            budget.source
        };
        let has_observations = input
            .history
            .iter()
            .any(|message| message.role == Role::Tool);
        let has_conversation = input.history.iter().any(|message| {
            !(message.role == Role::System
                && message.content.as_deref() == Some(input.system_prompt)
                || message.role == Role::User && message.content.as_deref() == Some(input.task))
        });
        let mut source_cap = source_cap;
        if !has_observations {
            source_cap += budget.observation;
        }
        if !has_conversation {
            source_cap += budget.conversation;
        }
        let mut observation_cap = if input.aggressive {
            budget.observation / 2
        } else {
            budget.observation
        };
        let mut conversation_cap = if input.aggressive {
            budget.conversation / 2
        } else {
            budget.conversation
        };
        if input.sources.is_empty() {
            observation_cap += budget.source / 2;
            conversation_cap += budget.source / 2;
        }
        if !has_observations {
            conversation_cap += budget.observation;
        }
        if !has_conversation {
            observation_cap += budget.conversation;
        }

        let mut source_tokens = 0;
        let mut selected_sources = 0;
        for source in input.sources {
            let content = format!(
                "Retrieved source: {}:{}-{}\nReason: {}\n{}",
                source.path, source.start_line, source.end_line, source.reason, source.content
            );
            let message = Message::system(content);
            let tokens = self.estimator.estimate_message(&message);
            if source_tokens + tokens <= source_cap && used + tokens <= input_limit {
                source_tokens += tokens;
                used += tokens;
                selected_sources += 1;
                messages.push(message);
            }
        }

        let keywords = task_terms(input.task);
        let groups = message_groups(input.history, input.task, input.system_prompt);
        let total_groups = groups.len();
        let mut ranked: Vec<_> = groups
            .into_iter()
            .enumerate()
            .map(|(index, group)| {
                let observation = group.iter().any(|message| message.role == Role::Tool);
                let body = group
                    .iter()
                    .filter_map(|message| message.content.as_deref())
                    .collect::<Vec<_>>()
                    .join("\n")
                    .to_lowercase();
                let relevance = keywords
                    .iter()
                    .filter(|word| body.contains(word.as_str()))
                    .count();
                let important = body.contains("failure")
                    || body.contains("fingerprint")
                    || body.contains("denied")
                    || body.contains("cargo_test")
                    || body.contains("cargo_build")
                    || body.contains("git_diff");
                (
                    important as usize * 1_000 + relevance * 100 + index,
                    index,
                    observation,
                    group,
                )
            })
            .collect();
        ranked.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));

        let mut observation_tokens = 0;
        let mut conversation_tokens = 0;
        let mut compressed_observations = 0;
        let mut chosen = Vec::new();
        for (_, index, observation, mut group) in ranked {
            if observation {
                let max_chars = if input.aggressive { 3_000 } else { 6_000 };
                for message in &mut group {
                    if message.role == Role::Tool {
                        if let Some(content) = message.content.as_mut() {
                            if content.len() > max_chars {
                                *content = compress_observation(content, max_chars);
                                compressed_observations += 1;
                            }
                        }
                    }
                }
            }
            let tokens: usize = group
                .iter()
                .map(|message| self.estimator.estimate_message(message))
                .sum();
            let category_used = if observation {
                observation_tokens
            } else {
                conversation_tokens
            };
            let category_cap = if observation {
                observation_cap
            } else {
                conversation_cap
            };
            if category_used + tokens <= category_cap && used + tokens <= input_limit {
                if observation {
                    observation_tokens += tokens;
                } else {
                    conversation_tokens += tokens;
                }
                used += tokens;
                chosen.push((index, group));
            }
        }
        chosen.sort_by_key(|(index, _)| *index);
        let selected_message_groups = chosen.len();
        for (_, group) in chosen {
            messages.extend(group);
        }

        let report = ContextReport {
            profile: self.profile.name,
            usage: ContextUsage {
                context_limit: budget.context_limit,
                input_limit,
                system_tokens,
                task_plan_tokens,
                source_tokens,
                observation_tokens,
                conversation_tokens,
                tool_tokens,
                prompt_tokens: used,
                output_reserve: budget.output_reserve,
            },
            selected_sources,
            dropped_sources: input.sources.len().saturating_sub(selected_sources),
            selected_message_groups,
            dropped_message_groups: total_groups.saturating_sub(selected_message_groups),
            compressed_observations,
            aggressive: input.aggressive,
            reasons: vec![
                "system and current task are mandatory".to_owned(),
                "sources are ranked by task-term matches".to_owned(),
                "recent relevant and important Tool groups are retained".to_owned(),
            ],
        };
        debug_assert!(report.usage.prompt_tokens <= input_limit);
        Ok(ContextBuild {
            messages,
            max_output_tokens: u32::try_from(budget.output_reserve).unwrap_or(u32::MAX),
            report,
        })
    }
}

fn message_groups(history: &[Message], task: &str, system_prompt: &str) -> Vec<Vec<Message>> {
    let mut groups = Vec::new();
    let mut index = 0;
    while index < history.len() {
        let message = &history[index];
        if message.role == Role::System && message.content.as_deref() == Some(system_prompt) {
            index += 1;
            continue;
        }
        if message.role == Role::User && message.content.as_deref() == Some(task) {
            index += 1;
            continue;
        }
        let mut group = vec![message.clone()];
        if message.role == Role::Assistant && message.tool_calls.is_some() {
            index += 1;
            while index < history.len() && history[index].role == Role::Tool {
                group.push(history[index].clone());
                index += 1;
            }
        } else {
            index += 1;
        }
        groups.push(group);
    }
    groups
}

fn truncate_to_tokens(text: &str, max_tokens: usize, estimator: &dyn TokenEstimator) -> String {
    if estimator.estimate_text(text) <= max_tokens {
        return text.to_owned();
    }
    let max_bytes = max_tokens.saturating_mul(3);
    let mut end = max_bytes.min(text.len());
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}\n...[truncated to context budget]", &text[..end])
}

fn compress_observation(content: &str, max_chars: usize) -> String {
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(content) {
        let compact = serde_json::json!({
            "summary": value.get("summary"),
            "truncated": true,
            "is_error": value.get("is_error"),
            "workflow_phase": value.get("workflow_phase"),
            "failure": value.get("failure"),
            "content": value.get("content").map(|inner| bounded_preview(&inner.to_string(), max_chars / 2)),
            "compression": "deterministic_head_tail"
        });
        return compact.to_string();
    }
    bounded_preview(content, max_chars)
}

fn bounded_preview(content: &str, max_chars: usize) -> String {
    if content.len() <= max_chars {
        return content.to_owned();
    }
    let half = max_chars / 2;
    let mut head = half;
    while head > 0 && !content.is_char_boundary(head) {
        head -= 1;
    }
    let mut tail = content.len().saturating_sub(half);
    while tail < content.len() && !content.is_char_boundary(tail) {
        tail += 1;
    }
    format!(
        "{}\n...[compressed]...\n{}",
        &content[..head],
        &content[tail..]
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_model::{RequestedToolCall, ToolDefinition};
    use serde_json::json;

    fn tool() -> ToolDefinition {
        ToolDefinition::function("read_file", "read", json!({"type":"object"}))
    }

    #[test]
    fn profiles_reserve_output() -> Result<(), Box<dyn std::error::Error>> {
        for profile in [ContextProfile::default_32k(), ContextProfile::large_65k()] {
            let manager = ContextManager::new(profile.clone());
            for size in [0, 1, 1_000, 100_000] {
                let task = "한".repeat(size);
                let build = manager.build(ContextInput {
                    system_prompt: "system",
                    task: &task,
                    plan: &[],
                    history: &[],
                    sources: &[],
                    tools: &[tool()],
                    aggressive: false,
                })?;
                assert!(build.report.usage.prompt_tokens <= profile.budget.input_limit());
                assert_eq!(
                    usize::try_from(build.max_output_tokens).ok(),
                    Some(profile.budget.output_reserve)
                );
            }
        }
        Ok(())
    }

    #[test]
    fn tool_call_and_result_stay_together() -> Result<(), Box<dyn std::error::Error>> {
        let call = RequestedToolCall {
            id: "call-1".to_owned(),
            name: "read_file".to_owned(),
            arguments: json!({"path":"src/lib.rs"}),
        };
        let history = vec![
            Message::system("system"),
            Message::user("task"),
            Message::assistant_response(None, &[call]),
            Message::tool("call-1", "result"),
        ];
        let build = ContextManager::new(ContextProfile::default_32k()).build(ContextInput {
            system_prompt: "system",
            task: "task",
            plan: &[],
            history: &history,
            sources: &[],
            tools: &[tool()],
            aggressive: false,
        })?;
        assert!(
            build
                .messages
                .iter()
                .any(|message| message.tool_calls.is_some())
        );
        assert!(
            build
                .messages
                .iter()
                .any(|message| message.role == Role::Tool)
        );
        Ok(())
    }

    #[test]
    fn important_observation_survives_compression() -> Result<(), Box<dyn std::error::Error>> {
        let payload = json!({
            "summary":"test failed",
            "content":{"denied":true,"output":"x".repeat(20_000)},
            "truncated":false,
            "is_error":true,
            "workflow_phase":"recovering",
            "failure":{"kind":"test","fingerprint":"abc123","occurrences":2,"replan_required":true}
        });
        let history = vec![Message::tool("call", payload.to_string())];
        let build = ContextManager::new(ContextProfile::default_32k()).build(ContextInput {
            system_prompt: "system",
            task: "fix test",
            plan: &[],
            history: &history,
            sources: &[],
            tools: &[tool()],
            aggressive: false,
        })?;
        let content = build
            .messages
            .iter()
            .find(|message| message.role == Role::Tool)
            .and_then(|message| message.content.as_deref())
            .unwrap_or_default();
        assert!(content.contains("abc123"));
        assert!(content.contains("test failed"));
        assert!(content.contains("denied"));
        Ok(())
    }

    #[test]
    fn retrieval_selects_ranges_not_entire_large_files() -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        std::fs::create_dir_all(temp.path().join("src"))?;
        std::fs::write(
            temp.path().join("src/lib.rs"),
            format!(
                "{}\nfn target_symbol() {{}}\n{}",
                "noise\n".repeat(200),
                "tail\n".repeat(200)
            ),
        )?;
        std::fs::write(temp.path().join("src/other.rs"), "fn unrelated() {}")?;
        let result = WorkspaceRetriever::new(temp.path()).retrieve("fix target_symbol", 4)?;
        assert_eq!(result.snippets.len(), 1);
        assert_eq!(result.snippets[0].path, "src/lib.rs");
        assert!(result.snippets[0].content.contains("target_symbol"));
        assert!(result.snippets[0].end_line - result.snippets[0].start_line <= 40);
        Ok(())
    }

    #[test]
    fn fallback_skips_ignored_and_binary_files() -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        std::fs::create_dir_all(temp.path().join("docs"))?;
        std::fs::create_dir_all(temp.path().join("src"))?;
        std::fs::write(temp.path().join(".gitignore"), "docs/\n*.bin\n")?;
        std::fs::write(temp.path().join("docs/hidden.rs"), "target_symbol")?;
        std::fs::write(temp.path().join("src/data.bin"), b"target_symbol\0")?;
        std::fs::write(temp.path().join("src/lib.rs"), "fn target_symbol() {}")?;
        let result = WorkspaceRetriever::new(temp.path()).retrieve_fallback(
            "target_symbol",
            &["target_symbol".to_owned()],
            8,
        );
        assert_eq!(result.report.backend, "rust_fallback");
        assert_eq!(result.snippets.len(), 1);
        assert_eq!(result.snippets[0].path, "src/lib.rs");
        Ok(())
    }

    #[test]
    fn large_fixture_selects_relevant_file_without_loading_repository()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        std::fs::create_dir_all(temp.path().join("src"))?;
        for index in 0..200 {
            std::fs::write(
                temp.path().join(format!("src/noise_{index}.rs")),
                format!("fn noise_{index}() {{}}\n{}", "irrelevant\n".repeat(200)),
            )?;
        }
        std::fs::write(
            temp.path().join("src/relevant.rs"),
            "pub fn unique_context_target() -> bool { true }\n",
        )?;
        let result =
            WorkspaceRetriever::new(temp.path()).retrieve("inspect unique_context_target", 8)?;
        assert_eq!(result.snippets.len(), 1);
        assert_eq!(result.snippets[0].path, "src/relevant.rs");
        assert!(result.report.candidates >= 201);
        assert!(result.snippets[0].content.len() < 1_000);
        Ok(())
    }

    #[test]
    fn task_glob_selects_matching_files() -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        std::fs::create_dir_all(temp.path().join("src"))?;
        std::fs::write(temp.path().join("src/lib.rs"), "pub fn library() {}")?;
        std::fs::write(temp.path().join("README.md"), "readme")?;
        let result = WorkspaceRetriever::new(temp.path()).retrieve("inspect src/*.rs", 8)?;
        assert!(
            result
                .snippets
                .iter()
                .any(|snippet| snippet.path == "src/lib.rs")
        );
        assert!(
            !result
                .snippets
                .iter()
                .any(|snippet| snippet.path == "README.md")
        );
        Ok(())
    }
}
