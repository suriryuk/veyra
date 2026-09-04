use clap::Parser;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Wrap};
use serde::Deserialize;
use serde_json::{Value, json};
use std::io;
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(name = "agent-tui", version, about = "Veyra terminal interface")]
struct Args {
    #[arg(long, default_value = "http://127.0.0.1:3000")]
    server_url: String,
    #[arg(long)]
    token: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct Session {
    id: String,
    status: String,
    recent_task: String,
}

struct App {
    client: reqwest::Client,
    base: String,
    token: Option<String>,
    sessions: Vec<Session>,
    selected: usize,
    detail: Value,
    input: String,
    status: String,
}

impl App {
    fn request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let value = self.client.request(
            method,
            format!("{}/api/v1{}", self.base.trim_end_matches('/'), path),
        );
        if let Some(token) = &self.token {
            value.bearer_auth(token)
        } else {
            value
        }
    }
    async fn refresh(&mut self) {
        let result = self
            .request(reqwest::Method::GET, "/sessions")
            .send()
            .await
            .and_then(reqwest::Response::error_for_status);
        match result {
            Ok(response) => match response.json::<Value>().await {
                Ok(value) => {
                    self.sessions =
                        serde_json::from_value(value["items"].clone()).unwrap_or_default();
                    if self.selected >= self.sessions.len() {
                        self.selected = self.sessions.len().saturating_sub(1);
                    }
                    self.status = "connected".into();
                }
                Err(error) => self.status = error.to_string(),
            },
            Err(error) => self.status = format!("reconnecting: {error}"),
        }
        if let Some(session) = self.sessions.get(self.selected) {
            if let Ok(response) = self
                .request(reqwest::Method::GET, &format!("/sessions/{}", session.id))
                .send()
                .await
                .and_then(reqwest::Response::error_for_status)
            {
                if let Ok(value) = response.json().await {
                    self.detail = value;
                }
            }
        }
    }
    async fn create_session(&mut self) {
        let request = self
            .request(reqwest::Method::POST, "/sessions")
            .json(&json!({"workspace":"."}));
        if let Err(error) = request
            .send()
            .await
            .and_then(reqwest::Response::error_for_status)
        {
            self.status = error.to_string();
        } else {
            self.refresh().await;
            self.selected = 0;
        }
    }
    async fn send(&mut self) {
        let Some(session) = self.sessions.get(self.selected) else {
            return;
        };
        if self.input.trim().is_empty() {
            return;
        }
        let message = std::mem::take(&mut self.input);
        let result = self
            .request(
                reqwest::Method::POST,
                &format!("/sessions/{}/messages", session.id),
            )
            .json(&json!({"message":message}))
            .send()
            .await
            .and_then(reqwest::Response::error_for_status);
        if let Err(error) = result {
            self.status = error.to_string();
        }
    }
    async fn approve(&mut self, allow: bool) {
        let Some(id) = pending_approval(&self.detail) else {
            return;
        };
        let action = if allow { "allow" } else { "deny" };
        let result = self
            .request(reqwest::Method::POST, &format!("/approvals/{id}/{action}"))
            .send()
            .await
            .and_then(reqwest::Response::error_for_status);
        if let Err(error) = result {
            self.status = error.to_string();
        } else {
            self.refresh().await;
        }
    }
}

#[tokio::main]
async fn main() -> io::Result<()> {
    let args = Args::parse();
    let mut app = App {
        client: reqwest::Client::new(),
        base: args.server_url,
        token: args
            .token
            .or_else(|| std::env::var("VEYRA_SERVER_TOKEN").ok()),
        sessions: Vec::new(),
        selected: 0,
        detail: json!({}),
        input: String::new(),
        status: "connecting".into(),
    };
    app.refresh().await;
    let mut terminal = ratatui::init();
    let result = run(&mut terminal, &mut app).await;
    ratatui::restore();
    result
}

async fn run(terminal: &mut ratatui::DefaultTerminal, app: &mut App) -> io::Result<()> {
    let mut refreshed = Instant::now();
    loop {
        terminal.draw(|frame| draw(frame, app))?;
        if refreshed.elapsed() >= Duration::from_secs(1) {
            app.refresh().await;
            refreshed = Instant::now();
        }
        if !event::poll(Duration::from_millis(100))? {
            continue;
        }
        let Event::Key(key) = event::read()? else {
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') if app.input.is_empty() => return Ok(()),
            KeyCode::Char('n') if app.input.is_empty() => app.create_session().await,
            KeyCode::Char('a') if app.input.is_empty() => app.approve(true).await,
            KeyCode::Char('d') if app.input.is_empty() => app.approve(false).await,
            KeyCode::Up if app.input.is_empty() => app.selected = app.selected.saturating_sub(1),
            KeyCode::Down if app.input.is_empty() => {
                app.selected = (app.selected + 1).min(app.sessions.len().saturating_sub(1))
            }
            KeyCode::Enter => app.send().await,
            KeyCode::Backspace => {
                app.input.pop();
            }
            KeyCode::Char(value) => app.input.push(value),
            _ => {}
        }
    }
}

fn draw(frame: &mut ratatui::Frame<'_>, app: &App) {
    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(12),
            Constraint::Length(3),
        ])
        .split(frame.area());
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                " VEYRA 0.9 ",
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(format!("  Local Agent  ·  {}", app.status)),
        ]))
        .block(Block::default().borders(Borders::ALL)),
        root[0],
    );
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(23),
            Constraint::Percentage(52),
            Constraint::Percentage(25),
        ])
        .split(root[1]);
    let sessions = app
        .sessions
        .iter()
        .enumerate()
        .map(|(index, value)| {
            ListItem::new(format!(
                "{} {}\n  {}",
                if index == app.selected { "›" } else { " " },
                if value.recent_task.is_empty() {
                    "새 세션"
                } else {
                    &value.recent_task
                },
                value.status
            ))
            .style(if index == app.selected {
                Style::default().fg(Color::Cyan)
            } else {
                Style::default()
            })
        })
        .collect::<Vec<_>>();
    frame.render_widget(
        List::new(sessions).block(
            Block::default()
                .title(" Sessions  [n] new ")
                .borders(Borders::ALL),
        ),
        columns[0],
    );
    let messages = app.detail["events"]
        .as_array()
        .map(|values| {
            values
                .iter()
                .rev()
                .filter_map(|event| {
                    let kind = event["type"].as_str()?;
                    matches!(kind, "token_delta" | "task_completed" | "task_failed").then(|| {
                        event["text"]
                            .as_str()
                            .or_else(|| event["answer"].as_str())
                            .or_else(|| event["error"].as_str())
                            .unwrap_or("")
                            .to_owned()
                    })
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default();
    frame.render_widget(
        Paragraph::new(if messages.is_empty() {
            "메시지를 입력해 작업을 시작하세요.".into()
        } else {
            messages
        })
        .wrap(Wrap { trim: false })
        .block(
            Block::default()
                .title(" Conversation ")
                .borders(Borders::ALL),
        ),
        columns[1],
    );
    let plan = app.detail["tasks"]
        .as_array()
        .and_then(|v| v.first())
        .and_then(|v| v["plan"].as_array())
        .map(|values| {
            values
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    format!(
                        "{}. {}",
                        index + 1,
                        value["description"].as_str().unwrap_or("")
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_else(|| "계획 대기 중".into());
    let approval = pending_approval(&app.detail).map_or(String::new(), |id| {
        format!("\n\n⚠ Approval\n{id}\n[a] allow  [d] deny")
    });
    frame.render_widget(
        Paragraph::new(format!("{plan}{approval}"))
            .wrap(Wrap { trim: true })
            .block(
                Block::default()
                    .title(" Plan / Context ")
                    .borders(Borders::ALL),
            ),
        columns[2],
    );
    frame.render_widget(
        Paragraph::new(format!("> {}", app.input)).block(
            Block::default()
                .title(" Enter send · q quit ")
                .borders(Borders::ALL),
        ),
        root[2],
    );
}

fn pending_approval(detail: &Value) -> Option<String> {
    detail["approvals"]
        .as_array()?
        .iter()
        .find(|item| item["status"] == "pending")?["approval_id"]
        .as_str()
        .map(ToOwned::to_owned)
}
