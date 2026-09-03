use std::io::{self, BufRead, Write};

fn field_value<'a>(line: &'a str, field: &str) -> Option<&'a str> {
    let start = line.find(field)? + field.len();
    let value = &line[start..];
    let end = value.find([',', '}']).unwrap_or(value.len());
    Some(value[..end].trim())
}

fn string_field<'a>(line: &'a str, field: &str) -> Option<&'a str> {
    let start = line.find(field)? + field.len();
    let value = &line[start..];
    let end = value.find('"')?;
    Some(&value[..end])
}

fn main() -> io::Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line?;
        let Some(id) = field_value(&line, "\"id\":") else {
            continue;
        };
        let response = if line.contains("\"method\":\"initialize\"") {
            let protocol = string_field(&line, "\"protocolVersion\":\"").unwrap_or("2025-11-25");
            format!(
                "{{\"jsonrpc\":\"2.0\",\"id\":{id},\"result\":{{\"protocolVersion\":\"{protocol}\",\"capabilities\":{{\"tools\":{{\"listChanged\":false}}}},\"serverInfo\":{{\"name\":\"veyra-test\",\"version\":\"0.6.0\"}}}}}}"
            )
        } else if line.contains("\"method\":\"tools/list\"")
            && line.contains("\"cursor\":\"page-2\"")
        {
            format!(
                "{{\"jsonrpc\":\"2.0\",\"id\":{id},\"result\":{{\"tools\":[{{\"name\":\"second\",\"description\":\"second page\",\"inputSchema\":{{\"type\":\"object\"}}}}]}}}}"
            )
        } else if line.contains("\"method\":\"tools/list\"") {
            format!(
                "{{\"jsonrpc\":\"2.0\",\"id\":{id},\"result\":{{\"tools\":[{{\"name\":\"echo\",\"description\":\"echo text\",\"inputSchema\":{{\"type\":\"object\",\"properties\":{{\"text\":{{\"type\":\"string\"}}}},\"required\":[\"text\"],\"additionalProperties\":false}}}}],\"nextCursor\":\"page-2\"}}}}"
            )
        } else if line.contains("\"method\":\"tools/call\"") {
            format!(
                "{{\"jsonrpc\":\"2.0\",\"id\":{id},\"result\":{{\"content\":[{{\"type\":\"text\",\"text\":\"fake stdio response\"}}],\"isError\":false}}}}"
            )
        } else {
            format!(
                "{{\"jsonrpc\":\"2.0\",\"id\":{id},\"error\":{{\"code\":-32601,\"message\":\"method not found\"}}}}"
            )
        };
        writeln!(stdout, "{response}")?;
        stdout.flush()?;
    }
    Ok(())
}
