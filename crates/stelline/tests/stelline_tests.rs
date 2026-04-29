//! Integration tests for the stelline crate.

use std::io::Write;

use axel_memkoshi::memory::{Memory, MemoryCategory};
use axel_stelline::context::{ContextManager, ContextMode, ContextTarget};
use axel_stelline::dedup;
use axel_stelline::extractor;
use axel_stelline::parser;
use axel_stelline::quality;
use tempfile::{NamedTempFile, TempDir};

fn write_session_file(lines: &[&str]) -> NamedTempFile {
    let mut f = NamedTempFile::new().unwrap();
    for line in lines {
        writeln!(f, "{line}").unwrap();
    }
    f
}

#[test]
fn parse_simple_jsonl_session() {
    let session = write_session_file(&[
        r#"{"type":"user","message":{"role":"user","content":"What is Rust?"}}"#,
        r#"{"type":"assistant","message":{"role":"assistant","content":"A systems language."}}"#,
        r#"{"type":"tool_use","name":"bash","input":{"command":"ls"}}"#,
    ]);
    let msgs = parser::parse_session(session.path()).unwrap();
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[0].role, "user");
    assert_eq!(msgs[0].content, "What is Rust?");
    assert_eq!(msgs[1].role, "assistant");
    assert_eq!(msgs[1].content, "A systems language.");
}

#[test]
fn format_transcript_produces_readable_output() {
    let msgs = vec![
        parser::TranscriptMessage {
            role: "user".into(),
            content: "Hello".into(),
            timestamp: None,
        },
        parser::TranscriptMessage {
            role: "assistant".into(),
            content: "Hi back".into(),
            timestamp: None,
        },
    ];
    let text = parser::format_transcript(&msgs);
    assert!(text.contains("USER: Hello"));
    assert!(text.contains("ASSISTANT: Hi back"));
}

#[test]
fn regex_extractor_finds_decisions() {
    let transcript =
        "USER: We decided to use PostgreSQL as the primary database backend for the project.";
    let mems = extractor::extract_regex(transcript);
    let event = mems
        .iter()
        .find(|m| m.category == MemoryCategory::Events)
        .expect("expected an Events memory");
    assert!(event.importance >= 0.7);
}

#[test]
fn regex_extractor_finds_preferences() {
    let transcript =
        "USER: I prefer TypeScript over JavaScript for any new project that we start.";
    let mems = extractor::extract_regex(transcript);
    assert!(
        mems.iter().any(|m| m.category == MemoryCategory::Preferences),
        "expected a Preferences memory, got {:?}",
        mems.iter().map(|m| m.category).collect::<Vec<_>>()
    );
}

#[test]
fn regex_extractor_skips_noise() {
    let transcript = "```rust\nlet x = 1;\nlet y = 2;\nlet z = x + y;\n```";
    let mems = extractor::extract_regex(transcript);
    assert!(
        mems.is_empty(),
        "code-fence noise should produce no memories, got {} memories",
        mems.len()
    );
}

#[test]
fn quality_gate_rejects_short_content() {
    let mut m = Memory::new(
        MemoryCategory::Events,
        "project-axel",
        "Reasonable Title",
        "ten chars.",
    );
    m.importance = 0.7;
    let result = quality::quality_gate(vec![m]);
    assert_eq!(result.accepted.len(), 0);
    assert_eq!(result.rejected.len(), 1);
    assert!(result.rejected[0].1.contains("Content too short"));
}

#[test]
fn quality_gate_accepts_valid_memory() {
    let mut m = Memory::new(
        MemoryCategory::Events,
        "project-axel",
        "Shipped the Stelline crate today",
        "Today we shipped the Stelline session intelligence crate as part of Axel.",
    );
    m.importance = 0.8;
    let result = quality::quality_gate(vec![m]);
    assert_eq!(result.accepted.len(), 1);
    assert_eq!(result.rejected.len(), 0);
}

#[test]
fn quality_gate_rejects_low_importance() {
    let mut m = Memory::new(
        MemoryCategory::Events,
        "project-axel",
        "Something Happened Today",
        "Some long enough content to pass the fifty character minimum easily.",
    );
    m.importance = 0.1;
    let result = quality::quality_gate(vec![m]);
    assert_eq!(result.rejected.len(), 1);
    assert!(result.rejected[0].1.contains("Importance below threshold"));
}

fn mem_with_title(title: &str) -> Memory {
    Memory {
        title: title.to_string(),
        ..Default::default()
    }
}

#[test]
fn dedup_detects_similar_titles() {
    let existing = vec![mem_with_title("Set up PostgreSQL database")];
    let candidate = mem_with_title("Setup PostgreSQL Database");
    assert!(
        dedup::is_duplicate(&candidate, &existing),
        "expected dedup to flag near-identical titles"
    );
}

#[test]
fn dedup_allows_different_titles() {
    let existing = vec![mem_with_title("Set up PostgreSQL")];
    let candidate = mem_with_title("Configure Redis cache");
    assert!(!dedup::is_duplicate(&candidate, &existing));
}

#[test]
fn context_manager_add_and_list() {
    let dir = TempDir::new().unwrap();
    let mut mgr = ContextManager::new();
    assert!(mgr.list_targets().is_empty());

    mgr.add_target(ContextTarget {
        name: "alpha".into(),
        path: dir.path().join("alpha.md").to_string_lossy().into_owned(),
        instruction: "alpha notes".into(),
        enabled: true,
        mode: ContextMode::Full,
    });
    mgr.add_target(ContextTarget {
        name: "beta".into(),
        path: dir.path().join("beta.md").to_string_lossy().into_owned(),
        instruction: "beta notes".into(),
        enabled: true,
        mode: ContextMode::Append,
    });

    let names: Vec<&str> = mgr.list_targets().iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, vec!["alpha", "beta"]);
}

#[test]
fn context_manager_remove() {
    let mut mgr = ContextManager::new();
    mgr.add_target(ContextTarget {
        name: "alpha".into(),
        path: "/tmp/alpha.md".into(),
        instruction: "x".into(),
        enabled: true,
        mode: ContextMode::Full,
    });
    assert_eq!(mgr.list_targets().len(), 1);

    let removed = mgr.remove_target("alpha");
    assert!(removed);
    assert!(mgr.list_targets().is_empty());

    // Removing again is a no-op.
    let removed = mgr.remove_target("alpha");
    assert!(!removed);
}
