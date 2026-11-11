//! Commit 12a: frames and config. Section 7, test 2 (the frame half; the `-32601` answer of
//! the loop is commit 12b's `serve`); section 9, test 3.

use std::io::{BufReader, Cursor, Read};
use std::path::{Path, PathBuf};

use axel::ext::config::{
    base_config_value, default_brain, EmbedderKind, ExtConfig, ForumMode, MAX_PROFILE_ENTRIES,
};
use axel::ext::rpc::{
    parse_request, write_frame, write_response, FrameError, FrameReader, MessageError, Response,
    INTERNAL_ERROR, MAX_BODY, MAX_OUT, METHOD_NOT_FOUND,
};
use serde_json::{json, Value};

fn frame(body: &str) -> Vec<u8> {
    let mut out = Vec::new();
    write_frame(&mut out, body.as_bytes()).unwrap();
    out
}

fn reader(bytes: Vec<u8>) -> FrameReader<BufReader<Cursor<Vec<u8>>>> {
    FrameReader::new(BufReader::new(Cursor::new(bytes)))
}

/// Every frame result until end of input.
fn all<R: std::io::BufRead>(mut r: FrameReader<R>) -> Vec<Result<Vec<u8>, FrameError>> {
    let mut out = Vec::new();
    while let Some(f) = r.next_frame() {
        out.push(f);
        assert!(out.len() < 100, "the reader does not end");
    }
    out
}

/// The requests a loop would act on: frames that parse, with an `id`.
fn answered(frames: &[Result<Vec<u8>, FrameError>]) -> Vec<Value> {
    frames
        .iter()
        .filter_map(|f| f.as_ref().ok())
        .filter_map(|b| parse_request(b).ok())
        .filter_map(|r| r.id)
        .collect()
}

#[test]
fn writer_puts_out_the_old_bytes() {
    let mut out = Vec::new();
    write_response(&mut out, &Response::result(json!("evt-001"), json!({"action": "continue"})))
        .unwrap();
    let body = r#"{"jsonrpc":"2.0","id":"evt-001","result":{"action":"continue"}}"#;
    assert_eq!(out, format!("Content-Length: {}\r\n\r\n{body}", body.len()).into_bytes());
}

#[test]
fn length_counts_bytes_not_chars() {
    let out = frame("\"é\"");
    assert!(out.starts_with(b"Content-Length: 4\r\n\r\n"));
    let got = all(reader(out));
    assert_eq!(got, vec![Ok("\"é\"".as_bytes().to_vec())]);
}

/// Section 7, test 2: garbage between frames, a body that is not JSON, `Content-Length: 0`,
/// a missing blank line, a 1 MiB body, a request without `id` → the next valid request is
/// read correctly.
#[test]
fn bad_frames_are_skipped_and_the_next_request_is_read() {
    let mut s = Vec::new();
    s.extend_from_slice(b"garbage line\r\nmore garbage\n\r\n");
    s.extend(frame(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#));
    s.extend_from_slice(b"\r\nnoise between frames\r\n");
    s.extend(frame("this is not json"));
    s.extend_from_slice(b"Content-Length: 0\r\n\r\n");
    // a missing blank line, directly followed by the next frame
    s.extend_from_slice(b"Content-Length: 30\r\n{\"jsonrpc\":\"2.0\",\"id\":9,\"method\":\"x\"}");
    let big = format!(r#"{{"jsonrpc":"2.0","method":"hook.handle","params":{{"pad":"{}"}}}}"#, "a".repeat(1 << 20));
    s.extend(frame(&big));
    s.extend(frame(r#"{"jsonrpc":"2.0","method":"hook.handle","params":{"kind":"x"}}"#));
    s.extend(frame(r#"{"jsonrpc":"2.0","id":2,"method":"hook.handle","params":{"kind":"before_message"}}"#));

    let got = all(reader(s));
    assert!(got.contains(&Err(FrameError::Empty)));
    assert!(got.contains(&Err(FrameError::MissingBlankLine)));
    assert!(got
        .iter()
        .any(|f| matches!(f, Ok(b) if parse_request(b) == Err(MessageError::NotJson))));
    let big_seen = got.iter().filter_map(|f| f.as_ref().ok()).any(|b| b.len() == big.len());
    assert!(big_seen, "the 1 MiB body is read whole");
    assert_eq!(answered(&got), vec![json!(1), json!(2)]);

    let last = got.last().unwrap().as_ref().unwrap();
    let req = parse_request(last).unwrap();
    assert_eq!(req.method, "hook.handle");
    assert_eq!(req.params, json!({"kind": "before_message"}));
}

#[test]
fn a_request_without_id_or_method() {
    let r = parse_request(br#"{"jsonrpc":"2.0","method":"hook.handle"}"#).unwrap();
    assert_eq!((r.id, r.params), (None, Value::Null));
    let r = parse_request(br#"{"jsonrpc":"2.0","id":null,"method":"m"}"#).unwrap();
    assert_eq!(r.id, None);
    assert_eq!(parse_request(br#"{"jsonrpc":"2.0","id":3}"#), Err(MessageError::NoMethod));
    assert_eq!(parse_request(br#"[1,2]"#), Err(MessageError::NotObject));
    assert_eq!(parse_request(br#"{"id":{},"method":"m"}"#), Err(MessageError::BadId));
}

/// Section 7, test 2, last sentence: an unknown method with an `id` → `-32601`.
#[test]
fn unknown_method_answer_is_32601() {
    let mut out = Vec::new();
    write_response(&mut out, &Response::method_not_found(json!(7), "no.such")).unwrap();
    let got = all(reader(out));
    let v: Value = serde_json::from_slice(got[0].as_ref().unwrap()).unwrap();
    assert_eq!(v["id"], json!(7));
    assert_eq!(v["error"]["code"], json!(METHOD_NOT_FOUND));
    assert_eq!(METHOD_NOT_FOUND, -32601);
    assert!(v.get("result").is_none());
}

#[test]
fn cut_short_is_an_error_never_a_partial_body() {
    let mut s = b"Content-Length: 100\r\n\r\n".to_vec();
    s.extend_from_slice(&[b'x'; 50]);
    assert_eq!(all(reader(s)), vec![Err(FrameError::Truncated)]);

    // input ends inside the header block
    assert_eq!(all(reader(b"Content-Length: 5\r\n".to_vec())), vec![Err(FrameError::Truncated)]);
    // nothing but garbage
    assert_eq!(all(reader(b"hello\r\nworld".to_vec())), vec![]);
    assert_eq!(all(reader(Vec::new())), vec![]);
}

#[test]
fn bad_length_values() {
    let mut s = b"Content-Length: -1\r\n\r\nContent-Length: abc\r\n\r\n".to_vec();
    s.extend(frame(r#"{"id":1,"method":"m"}"#));
    let got = all(reader(s));
    assert_eq!(got[0], Err(FrameError::BadHeader));
    assert_eq!(got[1], Err(FrameError::BadHeader));
    assert_eq!(answered(&got), vec![json!(1)]);
    // the header name in any case; another header is ignored
    let mut s = b"content-length: 2\r\nContent-Type: x\r\n\r\n{}".to_vec();
    s.extend_from_slice(b"Content-Length: 2\r\n{x\r\n\r\n{}");
    let got = all(reader(s));
    assert_eq!(got[0], Ok(b"{}".to_vec()));
    assert_eq!(got[1], Err(FrameError::BadHeader));
}

/// A stream of `n` bytes that holds nothing in memory.
fn filler(n: usize) -> impl Read {
    std::io::repeat(b' ').take(n as u64)
}

/// The limit: a body of `MAX_BODY - 1` and `MAX_BODY` bytes is read; `MAX_BODY + 1` is
/// discarded and the frame after it is read.
#[test]
fn size_limit_one_under_at_and_one_over() {
    assert_eq!(MAX_BODY, 64 * 1024 * 1024);
    for (n, ok) in [(MAX_BODY - 1, true), (MAX_BODY, true), (MAX_BODY + 1, false)] {
        let head = format!("Content-Length: {n}\r\n\r\n").into_bytes();
        let tail = frame(r#"{"id":5,"method":"m"}"#);
        let stream = Cursor::new(head).chain(filler(n)).chain(Cursor::new(tail));
        let mut r = FrameReader::new(BufReader::new(stream));
        match r.next_frame().unwrap() {
            Ok(body) => {
                assert!(ok, "{n} bytes must be refused");
                assert_eq!(body.len(), n);
            }
            Err(e) => {
                assert!(!ok, "{n} bytes must be read");
                assert_eq!(e, FrameError::TooLarge(n));
            }
        }
        let next = r.next_frame().unwrap().unwrap();
        assert_eq!(parse_request(&next).unwrap().id, Some(json!(5)));
        assert!(r.next_frame().is_none());
    }
}

#[test]
fn oversized_and_cut_short() {
    let n = MAX_BODY + 10;
    let stream = Cursor::new(format!("Content-Length: {n}\r\n\r\n").into_bytes()).chain(filler(1000));
    let mut r = FrameReader::new(BufReader::new(stream));
    assert_eq!(r.next_frame(), Some(Err(FrameError::Truncated)));
    assert_eq!(r.next_frame(), None);
}

/// The body of a response, read back from one written frame.
fn written_body(out: Vec<u8>) -> Vec<u8> {
    let got = all(reader(out));
    assert_eq!(got.len(), 1);
    got.into_iter().next().unwrap().unwrap()
}

/// RULING: wave2-c12a review finding 5: nothing over the host's 4 MiB leaves this side.
#[test]
fn responses_over_the_hosts_limit_become_an_error() {
    assert_eq!(MAX_OUT, 4 * 1024 * 1024);
    let id = json!("evt-9");
    let empty = Response::result(id.clone(), json!("")).to_bytes().len();
    let sized = |n: usize| Response::result(id.clone(), json!("a".repeat(n - empty)));

    // Exactly the limit: written as it is.
    let at = sized(MAX_OUT);
    assert_eq!(at.to_bytes().len(), MAX_OUT);
    let mut out = Vec::new();
    write_response(&mut out, &at).unwrap();
    assert_eq!(written_body(out), at.to_bytes());

    // One byte over: the error response with the same id, itself under the limit.
    let mut out = Vec::new();
    write_response(&mut out, &sized(MAX_OUT + 1)).unwrap();
    let body = written_body(out);
    assert!(body.len() <= MAX_OUT);
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["id"], id);
    assert_eq!(v["error"]["code"], json!(INTERNAL_ERROR));
    assert_eq!(INTERNAL_ERROR, -32603);
    assert_eq!(v["error"]["message"], json!("response too large"));
    assert_eq!(v["error"]["data"]["size"], json!(MAX_OUT + 1));
    assert!(v.get("result").is_none());
}

#[test]
fn write_frame_over_the_limit_writes_nothing() {
    let mut out = Vec::new();
    assert!(write_frame(&mut out, &vec![b' '; MAX_OUT + 1]).is_err());
    assert!(out.is_empty());
    write_frame(&mut out, &vec![b' '; MAX_OUT]).unwrap();
    assert_eq!(out.len(), format!("Content-Length: {MAX_OUT}\r\n\r\n").len() + MAX_OUT);
}

// ---- section 9, test 3 ----

fn home() -> PathBuf {
    PathBuf::from("/nonexistent-home-for-axel-tests")
}

fn cfg(config: Value) -> ExtConfig {
    ExtConfig::from_initialize(&json!({ "config": config }), &home())
}

#[test]
fn config_defaults() {
    for c in [ExtConfig::from_initialize(&json!({}), &home()), cfg(json!({}))] {
        let h = home();
        assert_eq!(c.brain, h.join(".local/share/axel/brain.r8"));
        assert_eq!(c.base, h.join(".synaps-cli"));
        assert_eq!(c.data_dir, h.join(".local/share/axel"));
        assert!(c.recall);
        assert_eq!(c.recall_max_items, 5);
        assert_eq!(c.recall_max_chars, 1600);
        assert_eq!(c.recall_budget_ms, 150);
        assert_eq!(c.recall_min_sem, 0.30);
        assert_eq!(c.recall_min_lex, 0.50);
        assert!(!c.recall_on_events);
        assert!(c.recall_exclude_paths.is_empty());
        assert!(c.capture);
        assert!(c.archives);
        assert_eq!(c.forum, ForumMode::Auto);
        assert!(!c.worker_scope_all);
        assert_eq!(c.forget_now_min_hidden_hours, 24);
        assert!(!c.handoff_on_start);
        assert_eq!(c.sources, None);
        assert!(c.archive_events.is_empty());
        assert_eq!(c.embedder, EmbedderKind::Onnx);
        assert_eq!(c.model_dir, h.join(".cache/velocirag/models/all-MiniLM-L6-v2"));
        assert!(c.log_queries);
        assert!(c.warnings.is_empty(), "{:?}", c.warnings);
    }
}

/// The manifest's defaults, as the host sends them (strings), give the same config.
#[test]
fn config_manifest_defaults_as_strings() {
    let c = cfg(json!({
        "brain": "", "synaps_base": "", "recall": "on", "recall_max_items": "5",
        "recall_max_chars": "1600", "recall_budget_ms": "150", "recall_min_sem": "0.30",
        "recall_min_lex": "0.50", "recall_on_events": "off", "recall_exclude_paths": "",
        "capture": "turns", "archives": "on", "forum": "auto", "worker_scope": "project",
        "forget_now_min_hidden_hours": "24", "handoff_on_start": "off", "sources": "",
        "archive_events": "", "embedder": "onnx", "model_dir": "", "log_queries": "on"
    }));
    assert!(c.warnings.is_empty(), "{:?}", c.warnings);
    assert_eq!(c.recall_max_items, 5);
    assert_eq!(c.forum, ForumMode::Auto);
    assert!(c.capture && c.recall && c.log_queries && !c.recall_on_events);
}

#[test]
fn config_given_values() {
    let c = cfg(json!({"recall_max_items": 3, "capture": "off", "forum": "false",
        "recall_min_sem": "0.4", "worker_scope": "all"}));
    assert_eq!(c.recall_max_items, 3);
    assert!(!c.capture);
    assert_eq!(c.forum, ForumMode::Off);
    assert_eq!(c.recall_min_sem, 0.4);
    assert!(c.worker_scope_all);
    assert!(c.warnings.is_empty(), "{:?}", c.warnings);
}

#[test]
fn config_unreadable_values_keep_defaults_and_warn() {
    let c = cfg(json!({"recall_max_items": "many", "capture": "all"}));
    assert_eq!(c.recall_max_items, 5);
    assert!(c.capture);
    assert_eq!(c.warnings.len(), 2, "{:?}", c.warnings);
    assert!(c.warnings.iter().any(|w| w.contains("recall_max_items")));
    assert!(c.warnings.iter().any(|w| w.contains("capture")));
    assert!(!c.warnings.iter().any(|w| w.contains("many")), "the value is not quoted");
}

#[test]
fn config_e2e_params() {
    let tmp = tempfile::tempdir().unwrap();
    let brain = tmp.path().join("brain.r8");
    let params = json!({
        "synaps_version": "0.1.0", "extension_protocol_version": 1, "plugin_id": "axel",
        "plugin_root": tmp.path().join("plugins/axel"),
        "config": {"brain": brain, "synaps_base": tmp.path(), "embedder": "hash"}
    });
    let c = ExtConfig::from_initialize(&params, &home());
    assert_eq!(c.brain, brain);
    assert_eq!(c.base, tmp.path());
    assert_eq!(c.data_dir, tmp.path());
    assert_eq!(c.embedder, EmbedderKind::Hash);
    assert!(c.warnings.is_empty(), "{:?}", c.warnings);
}

#[test]
fn config_switches_lists_and_bad_shapes() {
    let c = cfg(json!({"recall": false, "archives": "0", "log_queries": "TRUE",
        "recall_exclude_paths": " /a , ,/b/c ", "archive_events": "x,y",
        "recall_min_lex": 1.5, "embedder": [1], "forum": "1", "sources": "/s.toml"}));
    assert!(!c.recall && !c.archives && c.log_queries);
    assert_eq!(c.recall_exclude_paths, vec![PathBuf::from("/a"), PathBuf::from("/b/c")]);
    assert_eq!(c.archive_events, vec!["x".to_string(), "y".to_string()]);
    assert_eq!(c.recall_min_lex, 0.50);
    assert_eq!(c.embedder, EmbedderKind::Onnx);
    assert_eq!(c.forum, ForumMode::On);
    assert_eq!(c.sources.as_deref(), Some(Path::new("/s.toml")));
    assert_eq!(c.warnings.len(), 2, "{:?}", c.warnings);

    let c = ExtConfig::from_initialize(&json!({"config": "nope"}), &home());
    assert_eq!(c.recall_max_items, 5);
    assert_eq!(c.warnings.len(), 1);
    let c = ExtConfig::from_initialize(&json!(null), &home());
    assert!(c.warnings.is_empty());
}

/// RULING: wave2-c12a review finding 7: a path is taken as typed only when it is absolute
/// after `~` is expanded; `recall_budget_ms` is at most 2 s.
#[test]
fn paths_are_absolute_or_unreadable() {
    let h = home();
    let c = cfg(json!({"brain": "~/x/brain.r8", "model_dir": "~", "synaps_base": "~/base"}));
    assert!(c.warnings.is_empty(), "{:?}", c.warnings);
    assert_eq!(c.brain, h.join("x/brain.r8"));
    assert_eq!(c.model_dir, h);
    assert_eq!(c.base, h.join("base"));

    let c = cfg(json!({"brain": "x/brain.r8"}));
    assert_eq!(c.brain, default_brain(&h));
    assert_eq!(c.warnings.len(), 1, "{:?}", c.warnings);
    assert!(c.warnings[0].contains("brain"));

    // `~user` is not expanded and stays relative.
    let c = cfg(json!({"sources": "~other/s.toml", "model_dir": "./m"}));
    assert_eq!(c.sources, None);
    assert_eq!(c.model_dir, h.join(".cache/velocirag/models/all-MiniLM-L6-v2"));
    assert_eq!(c.warnings.len(), 2, "{:?}", c.warnings);

    // A list: the relative items are dropped with one warning naming the key.
    let c = cfg(json!({"recall_exclude_paths": "/a, rel, ~/b, also/rel"}));
    assert_eq!(c.recall_exclude_paths, vec![PathBuf::from("/a"), h.join("b")]);
    assert_eq!(c.warnings.len(), 1, "{:?}", c.warnings);
    assert!(c.warnings[0].contains("recall_exclude_paths"));

    let c = cfg(json!({"recall_budget_ms": "2000"}));
    assert_eq!(c.recall_budget_ms, 2000);
    assert!(c.warnings.is_empty());
    let c = cfg(json!({"recall_budget_ms": 2001}));
    assert_eq!(c.recall_budget_ms, 150);
    assert_eq!(c.warnings.len(), 1);
}

#[test]
fn the_base_config_reader() {
    let tmp = tempfile::tempdir().unwrap();
    let key = "memory.backend";
    assert_eq!(base_config_value(tmp.path(), key), None);
    let write = |text: &str| std::fs::write(tmp.path().join("config"), text).unwrap();
    write("# memory.backend = axel\nmodel = x\n");
    assert_eq!(base_config_value(tmp.path(), key), None);
    assert_eq!(base_config_value(tmp.path(), "model").as_deref(), Some("x"));
    write("memory.backend = axel\n");
    assert_eq!(base_config_value(tmp.path(), key).as_deref(), Some("axel"));
    write("memory.backend = \"files\" # comment\nmemory.backend.x = axel\n");
    assert_eq!(base_config_value(tmp.path(), key).as_deref(), Some("files"));
}

/// The extension and the command line must open the same brain when neither is told one.
#[test]
fn default_brain_is_the_command_lines_default() {
    // RULING: wave2-c12a review finding 2: the path is pinned as text, never through
    // `XDG_DATA_HOME`, so both sides name it whatever the shell sets.
    if let Some(real) = dirs::home_dir() {
        let pinned = PathBuf::from(format!("{}/.local/share/axel/brain.r8", real.display()));
        assert_eq!(default_brain(&real), pinned);
        assert_eq!(axel::config::AxelConfig::default().brain_path, pinned);
        assert_eq!(axel::config::data_dir_for(&real), real.join(".local/share"));
    }
    // For another home (a test's): under that home, never under the real one.
    assert_eq!(
        default_brain(&home()),
        home().join(".local/share/axel/brain.r8")
    );
    assert_eq!(cfg(json!({})).brain, home().join(".local/share/axel/brain.r8"));
    assert_eq!(cfg(json!({ "brain": "" })).brain, home().join(".local/share/axel/brain.r8"));
}

/// RULING: wave2-c12a review finding 1 (cap): a base folder too large to check whole turns
/// the forum off under `auto`, and `forum_decided_by` names the base itself.
#[test]
fn forum_off_when_the_base_has_too_many_entries() {
    assert_eq!(MAX_PROFILE_ENTRIES, 4096);
    let tmp = tempfile::tempdir().unwrap();
    let with = |mode: ForumMode| {
        let mut c = cfg(json!({}));
        c.base = tmp.path().to_path_buf();
        c.forum = mode;
        c
    };
    // Exactly the cap, no config anywhere: checked whole, on.
    for i in 0..MAX_PROFILE_ENTRIES {
        std::fs::write(tmp.path().join(format!("f{i:05}")), "").unwrap();
    }
    assert!(with(ForumMode::Auto).forum_on());
    assert_eq!(with(ForumMode::Auto).forum_decided_by(), None);
    // One more: too many to check, off.
    std::fs::write(tmp.path().join("f99999"), "").unwrap();
    assert!(!with(ForumMode::Auto).forum_on());
    assert_eq!(with(ForumMode::Auto).forum_decided_by(), Some(tmp.path().to_path_buf()));
    // `on` and `off` still do not look.
    assert!(with(ForumMode::On).forum_on());
    assert!(!with(ForumMode::Off).forum_on());
    assert_eq!(with(ForumMode::On).forum_decided_by(), None);
}

/// Section 6, "The clash": `auto` declares the forum unless the host's config says the host
/// still owns it; `on` and `off` do not look.
#[test]
fn forum_on_follows_the_hosts_backend_line() {
    let tmp = tempfile::tempdir().unwrap();
    let write = |text: &str| std::fs::write(tmp.path().join("config"), text).unwrap();
    let with = |mode: ForumMode| {
        let mut c = cfg(json!({}));
        c.base = tmp.path().to_path_buf();
        c.forum = mode;
        c.forum_on()
    };

    // No config file: the host's default is plain files, so the forum is ours.
    assert!(with(ForumMode::Auto));
    assert!(with(ForumMode::On));
    assert!(!with(ForumMode::Off));

    // The host still uses its Axel backend: its forum is live, ours is not declared.
    write("model = x\nmemory.backend = axel\n");
    assert!(!with(ForumMode::Auto));
    assert!(with(ForumMode::On), "on overrides the check");
    assert!(!with(ForumMode::Off));

    // Plain files, a commented-out line, a later line that wins.
    write("memory.backend = legacy\n");
    assert!(with(ForumMode::Auto));
    write("# memory.backend = axel\n");
    assert!(with(ForumMode::Auto));
    write("memory.backend = axel\nmemory.backend = legacy\n");
    assert!(with(ForumMode::Auto));
    write("memory.backend = legacy\n  memory.backend=axel\n");
    assert!(!with(ForumMode::Auto));

    // A base folder that does not exist reads as no config.
    let mut c = cfg(json!({}));
    c.base = PathBuf::from("/nonexistent/axel-tests/base");
    c.forum = ForumMode::Auto;
    assert!(c.forum_on());
    assert_eq!(c.forum_decided_by(), None);

    // RULING: wave2-c12a review finding 1: a profile's config is read too, since the host
    // may read it instead of the main one.
    let decided = |mode: ForumMode| {
        let mut c = cfg(json!({}));
        c.base = tmp.path().to_path_buf();
        c.forum = mode;
        c.forum_decided_by()
    };
    let profile = |name: &str, text: &str| {
        std::fs::create_dir_all(tmp.path().join(name)).unwrap();
        std::fs::write(tmp.path().join(name).join("config"), text).unwrap();
    };
    let sub = tmp.path().join("subconscious/config");

    // The main config says nothing, a profile says axel.
    write("model = x\n");
    profile("work", "memory.backend = legacy\n");
    profile("subconscious", "memory.backend = axel\n");
    assert!(!with(ForumMode::Auto));
    assert_eq!(decided(ForumMode::Auto), Some(sub.clone()));
    // On and off still do not look.
    assert!(with(ForumMode::On));
    assert!(!with(ForumMode::Off));
    assert_eq!(decided(ForumMode::On), None);
    assert_eq!(decided(ForumMode::Off), None);

    // The main config says legacy, a profile says axel.
    write("memory.backend = legacy\n");
    assert!(!with(ForumMode::Auto));
    assert_eq!(decided(ForumMode::Auto), Some(sub.clone()));

    // The main one decides first when both say axel.
    write("memory.backend = axel\n");
    assert_eq!(decided(ForumMode::Auto), Some(tmp.path().join("config")));

    // Main and two profiles legacy or commented out: on.
    write("memory.backend = legacy\n");
    profile("subconscious", "# memory.backend = axel\n");
    assert!(with(ForumMode::Auto));
    assert_eq!(decided(ForumMode::Auto), None);

    // A file named `config` two levels down is not read.
    profile("work/deeper", "memory.backend = axel\n");
    assert!(with(ForumMode::Auto));

    // A file (not a folder) under the base is not a profile.
    std::fs::write(tmp.path().join("notadir"), "memory.backend = axel\n").unwrap();
    assert!(with(ForumMode::Auto));

    // A directory that is a link is not followed.
    let elsewhere = tempfile::tempdir().unwrap();
    std::fs::write(elsewhere.path().join("config"), "memory.backend = axel\n").unwrap();
    std::os::unix::fs::symlink(elsewhere.path(), tmp.path().join("linked")).unwrap();
    assert!(with(ForumMode::Auto));
    assert_eq!(decided(ForumMode::Auto), None);
}
