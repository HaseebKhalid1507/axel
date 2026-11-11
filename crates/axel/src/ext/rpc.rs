//! Frame reader and writer, JSON-RPC types.
//!
//! The wire format is the host's (`CORE/docs/extensions/protocol.md`, "Message Format"):
//! `Content-Length: <n>\r\n\r\n<n bytes of JSON>`, no delimiter after the body, the same in
//! both directions. The writer puts out the bytes the old `extension.rs` wrote. The reader
//! never panics and never returns part of a body: a frame that is malformed, too large or
//! cut short is a `FrameError`, and the next call goes on with the next frame (section 7,
//! "Bad frame").

use std::io::{self, BufRead, Read, Write};

use serde::Serialize;
use serde_json::Value;

/// A body over this size is read and discarded (section 7, "Bad frame").
pub const MAX_BODY: usize = 64 * 1024 * 1024;
/// The size of the pieces an oversized body is discarded in.
pub const DISCARD_CHUNK: usize = 64 * 1024;
/// A header line keeps at most this many bytes (its tail); the rest is not a header.
const MAX_LINE: usize = 4096;

/// JSON-RPC 2.0: the method does not exist.
pub const METHOD_NOT_FOUND: i64 = -32601;

/// What went wrong with one frame. The reader has already moved past it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FrameError {
    /// `Content-Length` over `MAX_BODY`; the body was read and discarded.
    TooLarge(usize),
    /// Input ended inside the header block or the body.
    Truncated,
    /// `Content-Length: 0`.
    Empty,
    /// A `Content-Length` value that is not a number, or a header line that is not a header.
    BadHeader,
    /// A new `Content-Length` line where the blank line should have been. The frame is
    /// dropped; the next call reads the frame that line begins.
    MissingBlankLine,
    /// The reader failed (not end of input).
    Io(String),
}

/// Reads frames from the host's stdin.
pub struct FrameReader<R: BufRead> {
    inner: R,
    /// A `Content-Length` met where a blank line was expected: the next frame's header.
    pending: Option<Result<usize, FrameError>>,
}

impl<R: BufRead> FrameReader<R> {
    pub fn new(inner: R) -> Self {
        FrameReader { inner, pending: None }
    }

    /// The next frame's body. `None` at end of input. Lines that are not a
    /// `Content-Length` header between frames are skipped.
    pub fn next_frame(&mut self) -> Option<Result<Vec<u8>, FrameError>> {
        let len = match self.pending.take() {
            Some(len) => len,
            None => loop {
                let line = match self.read_line() {
                    Ok(Some(line)) => line,
                    Ok(None) => return None,
                    Err(e) => return Some(Err(e)),
                };
                if let Some(len) = content_length(&line) {
                    break len;
                }
            },
        };
        let len = match len {
            Ok(len) => len,
            Err(e) => return Some(Err(e)),
        };
        // The rest of the header block, up to the blank line.
        loop {
            let line = match self.read_line() {
                Ok(Some(line)) => line,
                Ok(None) => return Some(Err(FrameError::Truncated)),
                Err(e) => return Some(Err(e)),
            };
            if line.is_empty() {
                break;
            }
            if let Some(next) = content_length(&line) {
                self.pending = Some(next);
                return Some(Err(FrameError::MissingBlankLine));
            }
            if !is_header(&line) {
                return Some(Err(FrameError::BadHeader));
            }
        }
        if len == 0 {
            return Some(Err(FrameError::Empty));
        }
        if len > MAX_BODY {
            return Some(self.discard(len).and(Err(FrameError::TooLarge(len))));
        }
        let mut body = Vec::new();
        match (&mut self.inner).take(len as u64).read_to_end(&mut body) {
            Ok(n) if n == len => Some(Ok(body)),
            Ok(_) => Some(Err(FrameError::Truncated)),
            Err(e) => Some(Err(FrameError::Io(e.kind().to_string()))),
        }
    }

    fn discard(&mut self, mut left: usize) -> Result<(), FrameError> {
        let mut buf = vec![0u8; DISCARD_CHUNK];
        while left > 0 {
            let want = left.min(DISCARD_CHUNK);
            match self.inner.read(&mut buf[..want]) {
                Ok(0) => return Err(FrameError::Truncated),
                Ok(n) => left -= n,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(FrameError::Io(e.kind().to_string())),
            }
        }
        Ok(())
    }

    /// One line without its `\r\n` or `\n`; only the last `MAX_LINE` bytes of a long line
    /// are kept (a header that follows a body with no blank line is at the end). `None` at
    /// end of input with nothing read.
    fn read_line(&mut self) -> Result<Option<Vec<u8>>, FrameError> {
        let mut line = Vec::new();
        let mut any = false;
        loop {
            let buf = match self.inner.fill_buf() {
                Ok(buf) => buf,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(FrameError::Io(e.kind().to_string())),
            };
            if buf.is_empty() {
                return Ok(if any { Some(trim_eol(line)) } else { None });
            }
            any = true;
            let (used, found) = match buf.iter().position(|&b| b == b'\n') {
                Some(i) => {
                    line.extend_from_slice(&buf[..i]);
                    (i + 1, true)
                }
                None => {
                    line.extend_from_slice(buf);
                    (buf.len(), false)
                }
            };
            if line.len() > 2 * MAX_LINE {
                line.drain(..line.len() - MAX_LINE);
            }
            self.inner.consume(used);
            if found {
                if line.len() > MAX_LINE {
                    line.drain(..line.len() - MAX_LINE);
                }
                return Ok(Some(trim_eol(line)));
            }
        }
    }
}

fn trim_eol(mut line: Vec<u8>) -> Vec<u8> {
    while matches!(line.last(), Some(b'\r') | Some(b'\n')) {
        line.pop();
    }
    line
}

/// The value of a `Content-Length:` found anywhere in the line (case-insensitive; text before
/// it is garbage left by a broken frame). `None` when the line has none.
fn content_length(line: &[u8]) -> Option<Result<usize, FrameError>> {
    const KEY: &[u8] = b"content-length:";
    let at = line
        .windows(KEY.len())
        .rposition(|w| w.eq_ignore_ascii_case(KEY))?;
    let value = std::str::from_utf8(&line[at + KEY.len()..])
        .ok()
        .map(str::trim)
        .filter(|v| !v.is_empty() && v.bytes().all(|b| b.is_ascii_digit()))
        .and_then(|v| v.parse::<usize>().ok());
    Some(value.ok_or(FrameError::BadHeader))
}

/// `Name: value`, the name made of letters, digits and `-`.
fn is_header(line: &[u8]) -> bool {
    match line.iter().position(|&b| b == b':') {
        Some(0) | None => false,
        Some(i) => line[..i].iter().all(|b| b.is_ascii_alphanumeric() || *b == b'-'),
    }
}

/// One frame: the header, the blank line, the body, then a flush.
pub fn write_frame(w: &mut impl Write, body: &[u8]) -> io::Result<()> {
    w.write_all(format!("Content-Length: {}\r\n\r\n", body.len()).as_bytes())?;
    w.write_all(body)?;
    w.flush()
}

/// A message from the host. `id` is `None` for a notification (absent or `null`).
#[derive(Clone, Debug, PartialEq)]
pub struct Request {
    pub id: Option<Value>,
    pub method: String,
    pub params: Value,
}

/// Why a frame's body is not a request. Such a body is skipped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MessageError {
    NotJson,
    NotObject,
    NoMethod,
    /// An `id` that is neither a string nor a number.
    BadId,
}

/// Parses one frame's body. `params` absent is `null`.
pub fn parse_request(body: &[u8]) -> Result<Request, MessageError> {
    let v: Value = serde_json::from_slice(body).map_err(|_| MessageError::NotJson)?;
    let Value::Object(mut obj) = v else {
        return Err(MessageError::NotObject);
    };
    let method = match obj.remove("method") {
        Some(Value::String(m)) => m,
        _ => return Err(MessageError::NoMethod),
    };
    let id = match obj.remove("id") {
        None | Some(Value::Null) => None,
        Some(id @ (Value::String(_) | Value::Number(_))) => Some(id),
        Some(_) => return Err(MessageError::BadId),
    };
    let params = obj.remove("params").unwrap_or(Value::Null);
    Ok(Request { id, method, params })
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RpcError {
    pub code: i64,
    pub message: String,
}

/// An answer to a request with an `id`. Serialised with the keys in the old order:
/// `jsonrpc`, `id`, then `result` or `error`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Response {
    pub jsonrpc: &'static str,
    pub id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcError>,
}

impl Response {
    pub fn result(id: Value, result: Value) -> Response {
        Response { jsonrpc: "2.0", id, result: Some(result), error: None }
    }

    pub fn error(id: Value, code: i64, message: impl Into<String>) -> Response {
        Response {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(RpcError { code, message: message.into() }),
        }
    }

    /// `-32601` for a request whose method this binary does not know.
    pub fn method_not_found(id: Value, method: &str) -> Response {
        Response::error(id, METHOD_NOT_FOUND, format!("method not found: {method}"))
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        // Serialising a struct of `Value`s and strings cannot fail.
        serde_json::to_vec(self).unwrap_or_default()
    }
}

/// Writes one response as one frame.
pub fn write_response(w: &mut impl Write, response: &Response) -> io::Result<()> {
    write_frame(w, &response.to_bytes())
}
