use std::process::Stdio;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::process::Command;

async fn read_request(socket: &mut TcpStream) -> std::io::Result<String> {
    let mut buffer = vec![0_u8; 16_384];
    let read = socket.read(&mut buffer).await?;
    Ok(String::from_utf8_lossy(&buffer[..read]).into_owned())
}

async fn respond(socket: &mut TcpStream, content_type: &str, body: &str) -> std::io::Result<()> {
    let response = format!(
        "HTTP/1.1 200 OK\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    );
    socket.write_all(response.as_bytes()).await
}

#[tokio::test]
async fn cli_run_checks_health_and_streams_mock_response() -> Result<(), Box<dyn std::error::Error>>
{
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let server = tokio::spawn(async move {
        let (mut initial_models, _) = listener.accept().await?;
        assert!(
            read_request(&mut initial_models)
                .await?
                .starts_with("GET /models")
        );
        respond(
            &mut initial_models,
            "application/json",
            "{\"data\":[{\"id\":\"mock\",\"status\":{\"value\":\"unloaded\"},\"architecture\":{\"input_modalities\":[\"text\"]}}]}",
        )
        .await?;

        let (mut load, _) = listener.accept().await?;
        assert!(
            read_request(&mut load)
                .await?
                .starts_with("POST /models/load")
        );
        respond(&mut load, "application/json", "{\"success\":true}").await?;

        let (mut models, _) = listener.accept().await?;
        assert!(read_request(&mut models).await?.starts_with("GET /models"));
        respond(
            &mut models,
            "application/json",
            "{\"data\":[{\"id\":\"mock\",\"status\":{\"value\":\"loaded\"},\"architecture\":{\"input_modalities\":[\"text\"]}}]}",
        )
        .await?;

        let (mut health, _) = listener.accept().await?;
        assert!(read_request(&mut health).await?.starts_with("GET /health"));
        respond(&mut health, "application/json", "{\"status\":\"ok\"}").await?;

        let (mut chat, _) = listener.accept().await?;
        assert!(
            read_request(&mut chat)
                .await?
                .starts_with("POST /v1/chat/completions")
        );
        let events = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"mock answer\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        );
        respond(&mut chat, "text/event-stream", events).await?;
        Ok::<(), std::io::Error>(())
    });

    let temp = tempfile::tempdir()?;
    let config_path = temp.path().join("agent.toml");
    std::fs::write(
        &config_path,
        format!(
            "[model]\nbase_url = \"http://{address}/v1\"\nmodel = \"mock\"\nrequest_timeout_seconds = 5\n\n[security]\nworkspace_root = {:?}\n\n[logging]\ndirectory = {:?}\n\n[storage]\ndatabase_path = {:?}\n",
            temp.path().join("work").display().to_string(),
            temp.path().join("logs").display().to_string(),
            temp.path().join("data/veyra.sqlite3").display().to_string()
        ),
    )?;
    let output = Command::new(env!("CARGO_BIN_EXE_veyra"))
        .arg("--config")
        .arg(config_path)
        .arg("run")
        .arg("say hello")
        .stdin(Stdio::null())
        .output()
        .await?;
    server.await??;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("mock answer"));
    Ok(())
}
