//! Section 2 tests of commit 6, "Archive reader and units": 1 (listing), 2 (the reader
//! half), 3 to 6, 23 and 28 with its 4.1 cases. All pure: no database, no embedder.
//! None of them depends on what `screen::redact` blanks: where a key is expected, the test
//! makes it from `screen::redact` of the prompt, as the reader does.

use std::fs;
use std::path::{Path, PathBuf};

use axel::archive::{
    self, ArchiveFile, ArchivePolicy, ArchivedMessage, Block, ForgetMark, LiveArchive, Role,
    SkipReason, Start, Unit, UnitKind,
};
use axel::screen;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

/// Core's fixture directory: project key `p0123456789abcdef`.
const CORE_DIR: &str = "01a8f6fb3ec12e1d59d07bd6e3943c84c18415e3111dd5cba218b7ce9218b0c7";
const LIVE_ID: &str = "b436b6196e6ca0758c70d5faed628f29";
const WITHHELD: &str = "[archive: withheld]";

const REPORT: &str = "<event id=\"e1\" type=\"subagent_completion\" severity=\"low\" \
    source=\"worker-1\">REPORTMARK the worker finished the survey of the old tests</event>";
const HEARTBEAT: &str = "<event id=\"h1\" type=\"heartbeat\" severity=\"low\" \
    source=\"watcher\">BEATMARK the build machine is still busy with the other lane</event>";
const P4: &str = "P4MARK start on the reader now and tell me when the fixtures are in place";
const TAIL: &str = "TAIL I am copying the fixtures now and will run the listing right after that.";
const NEW: &str =
    "NEW the fixtures are in place and the listing prints the five files the reference reader prints.";
const P8: &str = "P8MARK good, now how many of them are live?";
const R9: &str = "REPLY9 four of the five are live and one is a tombstone, the one that was forgotten.";

const Q1: &str = "Q1MARK where does the indexer keep the size of a file it has read?";
const A1: &str = "A1MARK in the archive_index row of the file, beside its state.";
const Q2: &str = "Q2MARK forget that I told you about the offer from Acme, I have not accepted it";
const A2: &str = "A2MARK I will forget the offer now and say nothing more about it.";
const A2B: &str = "A2MORE it is forgotten.";
const Q3: &str = "Q3MARK and which test proves that a scan opens no file twice?";
const A3: &str = "A3MARK test 8 of section 2: the tree is made read-only and the scan passes.";

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/context-archives")
}

/// The hand-made fixture directory.
fn hand_dir() -> String {
    "c6".repeat(32)
}

fn policy() -> ArchivePolicy {
    ArchivePolicy {
        event_sources: Vec::new(),
        forget_tools: vec!["axel_forget".to_string(), "memory_forget".to_string()],
        keep_forget_turns: false,
    }
}

fn live(file: Result<ArchiveFile, SkipReason>) -> LiveArchive {
    match file {
        Ok(ArchiveFile::Live(a)) => a,
        other => panic!("not a live archive: {other:?}"),
    }
}

fn reason(file: Result<ArchiveFile, SkipReason>) -> String {
    match file {
        Err(SkipReason(reason)) => {
            assert!(!reason.is_empty());
            reason
        }
        Ok(file) => panic!("read, and it had to be refused: {file:?}"),
    }
}

/// `x8.0`: the kind's id letter and the position.
fn pos(u: &Unit) -> String {
    let letter = match u.kind {
        UnitKind::Exchange => 'x',
        UnitKind::Continued => 'c',
        UnitKind::Report => 'r',
        UnitKind::Note => 'n',
    };
    format!("{letter}{}.{}", u.source_index, u.block_index)
}

fn positions(units: &[Unit]) -> Vec<String> {
    units.iter().map(pos).collect()
}

fn units_of(a: &LiveArchive) -> Vec<Unit> {
    archive::extract_units(a, &policy())
}

fn holds(units: &[Unit], marker: &str) -> bool {
    units.iter().any(|u| u.body.contains(marker) || u.title.contains(marker) || u.aux.contains(marker))
}

fn key_of(prompt: &str) -> String {
    archive::turn_key(&screen::redact(prompt).text)
}

fn user(source_index: u32, text: &str) -> ArchivedMessage {
    ArchivedMessage { source_index, role: Role::User, blocks: vec![Block::Text(text.to_string())] }
}

fn asst(source_index: u32, text: &str) -> ArchivedMessage {
    ArchivedMessage {
        source_index,
        role: Role::Assistant,
        blocks: vec![Block::Text(text.to_string())],
    }
}

fn asst_tool(source_index: u32, text: &str, tool: &str) -> ArchivedMessage {
    ArchivedMessage {
        source_index,
        role: Role::Assistant,
        blocks: vec![
            Block::Text(text.to_string()),
            Block::ToolUse { name: tool.to_string(), input_head: "{}".to_string() },
        ],
    }
}

fn result(source_index: u32) -> ArchivedMessage {
    ArchivedMessage {
        source_index,
        role: Role::User,
        blocks: vec![Block::ToolResult { is_error: false }],
    }
}

fn arch(messages: Vec<ArchivedMessage>) -> LiveArchive {
    LiveArchive {
        id: "0".repeat(32),
        scope: hand_dir(),
        logical: "1".repeat(64),
        digest: "2".repeat(64),
        created_ms: Some(1_777_650_000_000),
        source_message_count: messages.last().map_or(0, |m| m.source_index + 1),
        note: String::new(),
        messages,
        file_size: 0,
        carried: None,
        start: Start::Unknown,
    }
}

fn pad(text: &str, chars: usize) -> String {
    let mut out = text.to_string();
    while out.chars().count() < chars {
        out.push('q');
    }
    out
}

// ------------------------------------------------------------------------------- files

fn id(n: u32) -> String {
    format!("{n:032x}")
}

fn live_body() -> Value {
    json!({
        "state": "live",
        "source_message_count": 2,
        "messages": [
            {"source_index": 0, "block_indices": [],
             "message": {"role": "user", "content": "GOODMARK what does the scan do with a file it cannot use?"}},
            {"source_index": 1, "block_indices": [],
             "message": {"role": "assistant", "content": "It skips the file with a reason and goes on with the next."}}
        ],
        "note": ""
    })
}

fn record(scope: &str, id: &str, body: Value) -> Value {
    json!({"version": 1, "scope": scope, "logical": "a".repeat(64), "id": id,
           "digest": "b".repeat(64), "body": body})
}

fn put(path: &Path, value: &Value) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
}

/// A temp folder with an empty project directory in it; the directory's name is the scope.
fn project() -> (tempfile::TempDir, PathBuf, String) {
    let tmp = tempfile::tempdir().unwrap();
    let scope = "ab".repeat(32);
    let dir = tmp.path().join(&scope);
    fs::create_dir_all(dir.join("s000001")).unwrap();
    (tmp, dir, scope)
}

/// Writes `value` as a shelf file named by `name_id` and reads it back.
fn read_shelf(value: &Value, name_id: &str) -> Result<ArchiveFile, SkipReason> {
    let (_tmp, dir, scope) = project();
    let path = dir.join("s000001").join(format!("{name_id}.json"));
    put(&path, value);
    archive::read_archive(&path, &scope, Some(name_id))
}

// ------------------------------------------------------------------------------ test 1

#[test]
fn t01_fixture_directory_lists_as_the_reference_reader_prints() {
    let dir = fixtures().join(CORE_DIR);
    let entries = archive::list_project(&dir);
    let listed: Vec<(&str, Option<&str>, u64)> =
        entries.iter().map(|e| (e.rel_name.as_str(), e.id_from_name.as_deref(), e.size)).collect();
    assert_eq!(
        listed,
        vec![
            ("000.json", None, 656),
            (
                "s000001/3fe5a51ef0d405efec79538b7717e255.json",
                Some("3fe5a51ef0d405efec79538b7717e255"),
                5135
            ),
            (
                "s000001/5ad68965a84888ab221823f5b4490804.json",
                Some("5ad68965a84888ab221823f5b4490804"),
                1455
            ),
            (
                "s000001/693c14d38641fb504a642f0e7cbfaac9.json",
                Some("693c14d38641fb504a642f0e7cbfaac9"),
                310
            ),
            ("s000001/b436b6196e6ca0758c70d5faed628f29.json", Some(LIVE_ID), 1696),
        ]
    );

    // The table of the reference reader: state, id, stored, source_message_count, created_ms.
    let mut table = Vec::new();
    let mut lives = Vec::new();
    for e in &entries {
        match archive::read_archive(&dir.join(&e.rel_name), CORE_DIR, e.id_from_name.as_deref()) {
            Ok(ArchiveFile::Live(a)) => {
                let created = a.created_ms.map_or("-".to_string(), |ms| ms.to_string());
                table.push(format!(
                    "live {} {} {} {created}",
                    a.id,
                    a.messages.len(),
                    a.source_message_count
                ));
                assert_eq!(a.file_size, e.size);
                assert_eq!(a.scope, CORE_DIR);
                assert_eq!(
                    a.logical,
                    "900c08a5903292975711f75ce85a7b6e30fab1b1223378ab1111cdbb7bc3d809"
                );
                assert_eq!((a.carried, a.start), (None, Start::Unknown));
                lives.push(a);
            }
            Ok(ArchiveFile::Tombstone { id, logical, digest }) => {
                assert_eq!(
                    logical,
                    "900c08a5903292975711f75ce85a7b6e30fab1b1223378ab1111cdbb7bc3d809"
                );
                assert_eq!(
                    digest,
                    "693c14d38641fb504a642f0e7cbfaac9336ef9061b19ba90ca4a6320b4e698f2"
                );
                table.push(format!("tombstone {id}"));
            }
            Err(reason) => panic!("{} was refused: {reason:?}", e.rel_name),
        }
    }
    assert_eq!(
        table,
        vec![
            "live 4c459668c1384294b09674c915b29c2d 2 2 -",
            "live 3fe5a51ef0d405efec79538b7717e255 1 1 1777640575090",
            "live 5ad68965a84888ab221823f5b4490804 1 1 1777642219347",
            "tombstone 693c14d38641fb504a642f0e7cbfaac9",
            "live b436b6196e6ca0758c70d5faed628f29 6 7 1777639267512",
        ]
    );

    // The reduced message: its text ends with core's marker and its input was left out.
    let reduced = &lives[1].messages[0];
    assert!(matches!(&reduced.blocks[0], Block::Text(t)
        if t.ends_with("[archive: 16773139 bytes omitted, over the per-archive limit]")));
    assert!(matches!(&reduced.blocks[1], Block::ToolUse { name, input_head }
        if name == "write" && input_head == "(omitted)"));
    // A tool input stored as its JSON text is that text, cut at 160 chars; the second input
    // of the message is stored as it was.
    let as_text = &lives[2].messages[0];
    assert!(matches!(&as_text.blocks[1], Block::ToolUse { name, input_head }
        if name == "evaluate" && input_head.starts_with("{\"of\":[{\"of\":[") && input_head.chars().count() == 160));
    assert!(matches!(&as_text.blocks[2], Block::ToolUse { name, input_head }
        if name == "read" && input_head == "{\"path\":\"src/eval.rs\"}"));
    // The slot file's id is inside the record.
    assert_eq!(lives[0].id, "4c459668c1384294b09674c915b29c2d");
    assert_eq!(lives[0].note, "Answered a question about the Makefile. Nothing pending.");
}

// ------------------------------------------------------------------------------ test 2

#[test]
fn t02_must_skip_directory_each_refused_with_a_reason_and_the_good_two_read() {
    use std::os::unix::fs::symlink;

    let (_tmp, dir, scope) = project();
    let shelf = dir.join("s000001");
    let good = shelf.join(format!("{}.json", id(1)));

    // The two good files.
    put(&dir.join("000.json"), &record(&scope, &id(100), live_body()));
    put(&good, &record(&scope, &id(1), live_body()));

    // A link to an archive, under an archive's name.
    symlink(&good, dir.join("001.json")).unwrap();
    // A file over 16 MiB (sparse).
    fs::File::create(dir.join("002.json")).unwrap().set_len(16 * 1024 * 1024 + 1).unwrap();
    // Names that are not archives.
    put(&dir.join("128.json"), &record(&scope, &id(101), live_body()));
    for bad_shelf in ["s1", "s000000", "s1000000"] {
        put(&dir.join(bad_shelf).join(format!("{}.json", id(7))), &record(&scope, &id(7), live_body()));
    }
    fs::write(dir.join("pending.0123456789abcdef.tmp"), b"{").unwrap();
    fs::write(dir.join("lock"), b"").unwrap();
    fs::write(shelf.join("pending.0123456789abcdef.tmp"), b"{").unwrap();
    fs::write(shelf.join("lock"), b"").unwrap();
    // A shelf that is a link to a shelf.
    symlink(&shelf, dir.join("s000002")).unwrap();
    // Records a reader cannot use.
    let mut v2 = record(&scope, &id(2), live_body());
    v2["version"] = json!(2);
    put(&shelf.join(format!("{}.json", id(2))), &v2);
    put(&shelf.join(format!("{}.json", id(3))), &record(&"cd".repeat(32), &id(3), live_body()));
    put(&shelf.join(format!("{}.json", id(4))), &record(&scope, &id(40), live_body()));
    let whole = serde_json::to_vec(&record(&scope, &id(5), live_body())).unwrap();
    fs::write(shelf.join(format!("{}.json", id(5))), &whole[..whole.len() / 2]).unwrap();
    put(&shelf.join(format!("{}.json", id(6))), &record(&scope, &id(6), json!({"state": "frozen"})));

    let entries = archive::list_project(&dir);
    let names: Vec<String> = entries.iter().map(|e| e.rel_name.clone()).collect();
    let in_shelf = |n: u32| format!("s000001/{}.json", id(n));
    assert_eq!(
        names,
        vec![
            "000.json".to_string(),
            "001.json".to_string(),
            "002.json".to_string(),
            in_shelf(1),
            in_shelf(2),
            in_shelf(3),
            in_shelf(4),
            in_shelf(5),
            in_shelf(6),
        ],
        "only archive names in the project directory and in shelves 1 to 999,999"
    );
    assert_eq!(entries[2].size, 16 * 1024 * 1024 + 1);
    assert_eq!(entries[0].id_from_name, None);
    assert_eq!(entries[3].id_from_name, Some(id(1)));

    // The scan finishes: every listed file is read or refused with its reason.
    let mut good_units = 0;
    let mut refused = Vec::new();
    for e in &entries {
        match archive::read_archive(&dir.join(&e.rel_name), &scope, e.id_from_name.as_deref()) {
            Ok(ArchiveFile::Live(a)) => {
                let units = units_of(&a);
                assert_eq!(positions(&units), vec!["x0.0"]);
                assert!(units[0].body.contains("GOODMARK"));
                good_units += 1;
            }
            Ok(other) => panic!("{}: {other:?}", e.rel_name),
            Err(SkipReason(reason)) => refused.push((e.rel_name.clone(), reason)),
        }
    }
    assert_eq!(good_units, 2);
    let expected = [
        ("001.json".to_string(), "not a regular file"),
        ("002.json".to_string(), "larger than 16 MiB"),
        (in_shelf(2), "version 2 is not 1"),
        (in_shelf(3), "a record of another project"),
        (in_shelf(4), "the id is not the file's name"),
        (in_shelf(5), "not valid JSON"),
        (in_shelf(6), "unknown state"),
    ];
    assert_eq!(refused.len(), expected.len());
    for ((name, reason), (want_name, want)) in refused.iter().zip(expected.iter()) {
        assert_eq!(name, want_name);
        assert_eq!(reason, want, "{name}");
    }

    // A name that is not an archive's is never opened, whoever asks.
    for (rel, name_id) in [
        ("128.json".to_string(), None),
        ("lock".to_string(), None),
        ("pending.0123456789abcdef.tmp".to_string(), None),
        (format!("s1/{}.json", id(7)), Some(id(7))),
        (format!("s000000/{}.json", id(7)), Some(id(7))),
        (format!("s1000000/{}.json", id(7)), Some(id(7))),
        ("s000001/lock".to_string(), None),
    ] {
        let why = reason(archive::read_archive(&dir.join(&rel), &scope, name_id.as_deref()));
        assert!(why.starts_with("not an archive name"), "{rel}: {why}");
    }
    // Nor is a file reached through a link, or under another directory's name.
    let through = dir.join("s000002").join(format!("{}.json", id(1)));
    assert_eq!(
        reason(archive::read_archive(&through, &scope, Some(&id(1)))),
        "the shelf is not a directory"
    );
    assert_eq!(
        reason(archive::read_archive(&good, &"cd".repeat(32), Some(&id(1)))),
        "not in the project directory"
    );
    // The event's id is checked against the record's, in a slot file too.
    assert_eq!(
        reason(archive::read_archive(&dir.join("000.json"), &scope, Some(&id(9)))),
        "the id is not the expected id"
    );
    assert!(archive::read_archive(&dir.join("000.json"), &scope, Some(&id(100))).is_ok());
    // A directory that is a link, a file or nothing lists nothing.
    assert!(archive::list_project(&dir.join("s000002")).is_empty());
    assert!(archive::list_project(&dir.join("lock")).is_empty());
    assert!(archive::list_project(&dir.join("nothing")).is_empty());
}

#[test]
fn t02_a_file_that_is_not_valid_is_refused_whole() {
    let scope = "ab".repeat(32);
    let tombstone = || record(&scope, &id(1), json!({"state": "tombstone"}));
    assert!(matches!(read_shelf(&tombstone(), &id(1)), Ok(ArchiveFile::Tombstone { .. })));

    // What a tombstone holds beside its state is in `f1_a_file_that_says_tombstone_is_a_forget`.

    let good = || record(&scope, &id(1), live_body());
    assert!(matches!(read_shelf(&good(), &id(1)), Ok(ArchiveFile::Live(_))));
    type Change = Box<dyn Fn(&mut Value)>;
    let broken: Vec<(&str, Change)> = vec![
        ("version as text", Box::new(|r| r["version"] = json!("1"))),
        ("version as a fraction", Box::new(|r| r["version"] = json!(1.0))),
        ("version 0", Box::new(|r| r["version"] = json!(0))),
        ("no version", Box::new(|r| {
            r.as_object_mut().unwrap().remove("version");
        })),
        ("created_ms null", Box::new(|r| r["created_ms"] = Value::Null)),
        ("created_ms negative", Box::new(|r| r["created_ms"] = json!(-1))),
        ("created_ms a fraction", Box::new(|r| r["created_ms"] = json!(1.5))),
        ("created_ms as text", Box::new(|r| r["created_ms"] = json!("1777650000000"))),
        ("an id in upper case", Box::new(|r| r["id"] = json!("A".repeat(32)))),
        ("no logical", Box::new(|r| {
            r.as_object_mut().unwrap().remove("logical");
        })),
        ("no digest", Box::new(|r| r["digest"] = json!(7))),
        ("body is text", Box::new(|r| r["body"] = json!("live"))),
        ("no state", Box::new(|r| {
            r["body"].as_object_mut().unwrap().remove("state");
        })),
        ("no note", Box::new(|r| {
            r["body"].as_object_mut().unwrap().remove("note");
        })),
        ("no messages", Box::new(|r| r["body"]["messages"] = json!({}))),
        ("a count below an index", Box::new(|r| r["body"]["source_message_count"] = json!(1))),
        ("a count over 4096", Box::new(|r| r["body"]["source_message_count"] = json!(4097))),
        ("indexes out of order", Box::new(|r| {
            r["body"]["messages"][1]["source_index"] = json!(0)
        })),
        ("a system message", Box::new(|r| {
            r["body"]["messages"][0]["message"]["role"] = json!("system")
        })),
        ("a message with no content", Box::new(|r| {
            r["body"]["messages"][0]["message"]["content"] = Value::Null
        })),
        ("an entry with no message", Box::new(|r| r["body"]["messages"][1] = json!(3))),
        ("no block_indices", Box::new(|r| {
            r["body"]["messages"][1]["block_indices"] = json!("0")
        })),
    ];
    for (what, change) in &broken {
        let mut r = good();
        change(&mut r);
        let why = reason(read_shelf(&r, &id(1)));
        assert!(!why.is_empty(), "{what}");
    }
    // Not an object at all, and not JSON at all.
    assert_eq!(reason(read_shelf(&json!([1]), &id(1))), "not a record");
    let (_tmp, dir, scope) = project();
    let path = dir.join("s000001").join(format!("{}.json", id(1)));
    fs::write(&path, b"\xff\xfe not json").unwrap();
    assert_eq!(reason(archive::read_archive(&path, &scope, Some(&id(1)))), "not valid JSON");
    // A file that is gone.
    fs::remove_file(&path).unwrap();
    assert!(reason(archive::read_archive(&path, &scope, Some(&id(1)))).starts_with("cannot stat"));
}

// ------------------------------------------------- wave2-c6-review, finding 1 (ruling)

#[test]
fn f1_a_file_that_says_tombstone_is_a_forget() {
    let scope = "ab".repeat(32);
    let tombstone = || record(&scope, &id(1), json!({"state": "tombstone"}));
    type Change = Box<dyn Fn(&mut Value)>;
    let extras: Vec<(&str, Change)> = vec![
        ("created_ms", Box::new(|t| t["created_ms"] = json!(1_777_650_000_000u64))),
        ("created_ms that is no number", Box::new(|t| t["created_ms"] = json!("soon"))),
        ("carried", Box::new(|t| t["carried"] = json!(0))),
        ("a later top-level field", Box::new(|t| t["some_later_field"] = json!({"x": 1}))),
        ("a note in the body", Box::new(|t| t["body"]["note"] = json!(""))),
        ("messages in the body", Box::new(|t| t["body"]["messages"] = json!([1, 2, 3]))),
        ("a count in the body", Box::new(|t| t["body"]["source_message_count"] = json!(9999))),
    ];
    for (what, change) in &extras {
        let mut t = tombstone();
        change(&mut t);
        match read_shelf(&t, &id(1)) {
            Ok(ArchiveFile::Tombstone { id: got, logical, digest }) => {
                assert_eq!(got, id(1), "{what}");
                assert_eq!(logical, "a".repeat(64), "{what}");
                assert_eq!(digest, "b".repeat(64), "{what}");
            }
            other => panic!("{what}: not a tombstone: {other:?}"),
        }
    }
    // In a slot file too, where the id is the record's.
    let (_tmp, dir, scope) = project();
    let mut t = record(&scope, &id(100), json!({"state": "tombstone", "note": "x"}));
    t["carried"] = json!(3);
    put(&dir.join("007.json"), &t);
    assert!(matches!(
        archive::read_archive(&dir.join("007.json"), &scope, None),
        Ok(ArchiveFile::Tombstone { id: got, .. }) if got == id(100)
    ));
    // A tombstone that lost its digest is still a forget of its id.
    let mut t = tombstone();
    t.as_object_mut().unwrap().remove("digest");
    t["logical"] = json!(7);
    assert!(matches!(
        read_shelf(&t, &id(1)),
        Ok(ArchiveFile::Tombstone { id: got, logical, digest })
            if got == id(1) && logical.is_empty() && digest.is_empty()
    ));

    // The version, scope and id checks still come first.
    let mut t = tombstone();
    t["version"] = json!(2);
    assert_eq!(reason(read_shelf(&t, &id(1))), "version 2 is not 1");
    let other = record(&"cd".repeat(32), &id(1), json!({"state": "tombstone"}));
    assert_eq!(reason(read_shelf(&other, &id(1))), "a record of another project");
    assert_eq!(reason(read_shelf(&tombstone(), &id(2))), "the id is not the file's name");
    // And a state that is not `tombstone` is not one.
    let odd = record(&scope, &id(1), json!({"state": "Tombstone"}));
    assert_eq!(reason(read_shelf(&odd, &id(1))), "unknown state");
}

// ------------------------------------------------------------------------------ test 3

#[test]
fn t03_created_ms_and_placeholder_inputs() {
    let scope = "ab".repeat(32);
    let body = json!({
        "state": "live",
        "source_message_count": 1,
        "messages": [{"source_index": 0, "block_indices": [0, 1, 2, 3, 4, 5, 6],
            "message": {"role": "assistant", "future_key": 1, "content": [
                {"type": "tool_use", "id": "t1", "name": "one", "input": {"_archive_withheld": true}},
                {"type": "tool_use", "id": "t2", "name": "two", "input": {"_archive_anything": "abc"}},
                {"type": "tool_use", "id": "t3", "name": "three", "input": {"_archive_omitted": "[archive: tool input of 48 bytes omitted, over the per-archive limit]"}},
                {"type": "tool_use", "id": "t4", "name": "four", "input": {"a": "x", "b": [1, 2]}},
                {"type": "tool_use", "id": "t5", "name": "five", "input": {"_archive_withheld": true, "other": 1}},
                {"type": "hologram", "data": "?"},
                {"type": "tool_use", "id": "t6", "name": "six", "input": {"text": "é".repeat(400)}}
            ]}}],
        "note": ""
    });
    let mut with = record(&scope, &id(1), body.clone());
    with["created_ms"] = json!(1_777_639_267_512u64);
    // An unknown record-level field of a live record is ignored.
    with["some_later_field"] = json!({"x": 1});
    let a = live(read_shelf(&with, &id(1)));
    assert_eq!(a.created_ms, Some(1_777_639_267_512));
    let heads: Vec<Option<(&str, &str)>> = a.messages[0]
        .blocks
        .iter()
        .map(|b| match b {
            Block::ToolUse { name, input_head } => Some((name.as_str(), input_head.as_str())),
            _ => None,
        })
        .collect();
    assert_eq!(heads[0], Some(("one", "(withheld)")));
    assert_eq!(heads[1], Some(("two", "abc")));
    assert_eq!(heads[2], Some(("three", "(omitted)")));
    assert_eq!(heads[3], Some(("four", "{\"a\":\"x\",\"b\":[1,2]}")));
    // Two keys: not a placeholder.
    assert_eq!(heads[4], Some(("five", "{\"_archive_withheld\":true,\"other\":1}")));
    // An unknown block type is ignored and keeps its place.
    assert!(matches!(a.messages[0].blocks[5], Block::Other));
    // 160 chars, not bytes.
    let (_, long) = heads[6].unwrap();
    assert_eq!(long.chars().count(), 160);
    assert!(long.starts_with("{\"text\":\"éé"));

    // Without the field there is no time: the indexer takes the file's.
    let a = live(read_shelf(&record(&scope, &id(1), body), &id(1)));
    assert_eq!(a.created_ms, None);
    assert_eq!((a.carried, a.start), (None, Start::Unknown));
}

// ------------------------------------------------------------------------------ test 4

fn assert_the_retry_exchange(units: &[Unit]) {
    assert_eq!(positions(units), vec!["x1.0", "n0.0"]);
    let x = &units[0];
    assert_eq!(x.kind, UnitKind::Exchange);
    assert!(x.body.starts_with("User: Where is the retry limit set?\nAssistant: "), "{}", x.body);
    assert!(x.body.contains("MAX_RETRIES is 5"));
    assert_eq!(x.title, "Where is the retry limit set?");
    assert_eq!(x.turn_key, Some(key_of("Where is the retry limit set?")));
    let lines: Vec<&str> = x.aux.lines().collect();
    assert_eq!(
        lines,
        vec![
            "read: {\"path\":\"src/net/retries.rs\"}",
            "read: {\"path\":\"src/net/retry.rs\"}",
            "context_checkpoint: (withheld)",
        ]
    );
    // Tool results are not stored, and the withheld last message produces nothing.
    assert!(!holds(units, "No such file"));
    assert!(!holds(units, "archive: withheld"));
    // The note: one unit (the indexer stores it as `conversation_summary`).
    let n = &units[1];
    assert_eq!(n.kind, UnitKind::Note);
    assert_eq!(n.title, "Working note at rollover");
    assert_eq!(
        n.body,
        "Goal: find where the retry limit is set. Done: MAX_RETRIES is 5, in src/net/retry.rs. Next: nothing."
    );
    assert_eq!(n.turn_key, None);
}

#[test]
fn t04_extract_units_on_the_6_of_7_fixture() {
    let path = fixtures().join(CORE_DIR).join("s000001").join(format!("{LIVE_ID}.json"));
    let mut a = live(archive::read_archive(&path, CORE_DIR, Some(LIVE_ID)));
    assert_eq!(a.messages[0].source_index, 1, "index 0 was a system message");

    // Read whole (an event named it a later part, or the record says `carried` 0), the
    // fixture gives what the spec's test 4 lists.
    a.start = Start::Whole;
    assert_eq!(archive::carried_head(&a), None);
    assert_the_retry_exchange(&units_of(&a));

    // The hand-made copy whose record has `carried: 0` is read whole with no event at all.
    let copy = fixtures().join(hand_dir()).join("s000001").join(format!("{LIVE_ID}.json"));
    let marked = live(archive::read_archive(&copy, &hand_dir(), Some(LIVE_ID)));
    assert_eq!((marked.carried, marked.start), (Some(0), Start::Unknown));
    assert_eq!(archive::carried_head(&marked), None);
    assert_the_retry_exchange(&units_of(&marked));

    // As the scan finds core's file, with neither: its first stored index is 1, so rule (c)
    // takes it for a successor ("what the reader cannot know"): too little is indexed, and
    // the prompt is never paired.
    a.start = Start::Unknown;
    assert_eq!(archive::carried_head(&a), Some(1));
    assert_eq!(positions(&units_of(&a)), vec!["n0.0"]);
}

// ------------------------------------------------------------------------------ test 5

#[test]
fn t05_classification() {
    // A `subagent_completion` event is a report.
    let units = units_of(&arch(vec![user(0, REPORT)]));
    assert_eq!(positions(&units), vec!["r0.0"]);
    assert_eq!(units[0].kind, UnitKind::Report);
    assert_eq!(units[0].body, "REPORTMARK the worker finished the survey of the old tests");
    assert_eq!(units[0].turn_key, None);

    // A heartbeat event is nothing by default, a unit when its source is in `archive_events`.
    let beat = arch(vec![user(0, HEARTBEAT)]);
    assert!(units_of(&beat).is_empty());
    let mut with_source = policy();
    with_source.event_sources = vec!["watcher".to_string()];
    let units = archive::extract_units(&beat, &with_source);
    assert_eq!(units.len(), 1);
    assert!(units[0].body.contains("BEATMARK"));
    assert_eq!((units[0].source_index, units[0].block_index), (0, 0));
    with_source.event_sources = vec!["someone-else".to_string()];
    assert!(archive::extract_units(&beat, &with_source).is_empty());

    // `[Request interrupted by user]`, `[Context continuation …` and the host's advisory
    // are nothing: no prompt, no unit, and they do not end the exchange around them.
    let prompt = "PROMPT5 how do I run only the archive reader tests on the build machine?";
    let units = units_of(&arch(vec![
        user(0, prompt),
        asst(1, "R5A use the test flag with the name of the test file."),
        user(2, "[Request interrupted by user]"),
        asst(3, "R5B and add locked so the lock file is not touched."),
        user(4, "[Context continuation: the window before this one was archived]"),
        asst(5, "R5C that is all."),
        user(6, "[Host context advisory: 80% of the window is used]"),
    ]));
    assert_eq!(positions(&units), vec!["x0.0"]);
    for marker in ["PROMPT5", "R5A", "R5B", "R5C"] {
        assert!(units[0].body.contains(marker), "{marker}");
    }
    assert!(!holds(&units, "Request interrupted"));
    assert!(!holds(&units, "Context continuation"));
    assert!(!holds(&units, "Host context advisory"));

    // A 40-char exchange is nothing; 60 chars is the least that is stored.
    let twenty = "a question, 20 chars";
    assert_eq!(twenty.chars().count(), 20);
    assert!(units_of(&arch(vec![user(0, twenty), asst(1, "an answer, 20 chars.")])).is_empty());
    let thirty = pad("thirty chars of a question ", 30);
    let short = arch(vec![user(0, &thirty), asst(1, &pad("an answer ", 29))]);
    assert!(units_of(&short).is_empty(), "59 chars");
    let enough = arch(vec![user(0, &thirty), asst(1, &pad("an answer ", 30))]);
    assert_eq!(positions(&units_of(&enough)), vec!["x0.0"], "60 chars");

    // An archive whose first message is assistant text: one `Continued` unit.
    let text = "CONTMARK the second half of an answer that began in the part before this one.";
    let units = units_of(&arch(vec![asst(0, text)]));
    assert_eq!(positions(&units), vec!["c0.0"]);
    assert_eq!(units[0].kind, UnitKind::Continued);
    assert_eq!(units[0].body, format!("Assistant (continued): {text}"));
    assert_eq!(units[0].turn_key, None);
    // The same minimum.
    assert!(units_of(&arch(vec![asst(0, "too short to keep")])).is_empty());

    // A prompt with tool calls and no assistant text is nothing (ruling 1a).
    let silent = arch(vec![user(0, prompt), asst_tool(1, "  ", "bash"), result(2)]);
    assert!(units_of(&silent).is_empty());

    // The body is cut at 4,096 bytes on a char boundary; the title is 12 words.
    let long = "é".repeat(5000);
    let units = units_of(&arch(vec![user(0, "one two three four five six seven eight nine ten eleven twelve thirteen"), asst(1, &long)]));
    assert!(units[0].body.len() <= 4096 && units[0].body.len() >= 4094);
    assert_eq!(units[0].title, "one two three four five six seven eight nine ten eleven twelve");
}

// ------------------------------------------------------------------------------ test 6

#[test]
fn t06_same_prompt_twice_is_two_rows() {
    let first = "FIRSTREPLY carrying on with the reader: the listing is done and tested.";
    let second = "SECONDREPLY carrying on again: the units and the marks are done as well.";
    let a = arch(vec![user(0, "continue"), asst(1, first), user(2, "continue"), asst(3, second)]);
    let units = units_of(&a);
    assert_eq!(positions(&units), vec!["x0.0", "x2.0"]);
    assert!(units[0].body.contains("FIRSTREPLY") && !units[0].body.contains("SECONDREPLY"));
    assert!(units[1].body.contains("SECONDREPLY") && !units[1].body.contains("FIRSTREPLY"));
    assert_eq!(units[0].turn_key, units[1].turn_key);
    assert_eq!(units[0].turn_key, Some(key_of("continue")));

    let ids: Vec<String> = units.iter().map(|u| archive::unit_id(&a.scope, &a.id, u)).collect();
    assert_ne!(ids[0], ids[1]);
    // The same file again gives the same ids.
    let again: Vec<String> =
        units_of(&a).iter().map(|u| archive::unit_id(&a.scope, &a.id, u)).collect();
    assert_eq!(ids, again);
    // Another directory, another archive: other ids.
    assert_ne!(archive::unit_id(&"cd".repeat(32), &a.id, &units[0]), ids[0]);
    assert_ne!(archive::unit_id(&a.scope, &"f".repeat(32), &units[0]), ids[0]);

    // The id is the spec's formula.
    let mut hasher = Sha256::new();
    hasher.update(b"axel/unit/1\0");
    hasher.update(a.scope.as_bytes());
    hasher.update(b"\0");
    hasher.update(a.id.as_bytes());
    hasher.update(b"\0");
    hasher.update(b"x2.0");
    assert_eq!(ids[1], format!("mem_{}", &hex::encode(hasher.finalize())[..16]));
    // A kind is part of the id.
    let mut as_continued = units[1].clone();
    as_continued.kind = UnitKind::Continued;
    assert_ne!(archive::unit_id(&a.scope, &a.id, &as_continued), ids[1]);
}

#[test]
fn turn_key_and_normalise() {
    assert_eq!(archive::normalise("  a \n\t b  c "), "a b c");
    assert_eq!(archive::normalise(" \n "), "");
    assert_eq!(archive::turn_key("a  b"), archive::turn_key(" a b\n"));
    assert_ne!(archive::turn_key("a b"), archive::turn_key("a c"));
    let mut hasher = Sha256::new();
    hasher.update(b"axel/turn/3\0a b");
    assert_eq!(archive::turn_key(" a\tb "), hex::encode(hasher.finalize())[..16]);
}

// ----------------------------------------------------------------------------- test 23

/// Three exchanges; the second holds a call of `tool`.
fn three(second_prompt: &str, tool: &str) -> LiveArchive {
    arch(vec![
        user(0, Q1),
        asst(1, A1),
        user(2, second_prompt),
        asst_tool(3, A2, tool),
        result(4),
        asst(5, A2B),
        user(6, Q3),
        asst(7, A3),
    ])
}

fn mark(turn_key: Option<String>, head: bool, tail: bool) -> ForgetMark {
    ForgetMark { turn_key, head, tail }
}

#[test]
fn t23_the_forget_turn_by_position() {
    let p = policy();

    // The second exchange holds the call: units for the first and the third only.
    let a = three(Q2, "axel_forget");
    let units = units_of(&a);
    assert_eq!(positions(&units), vec!["x0.0", "x6.0"]);
    for marker in ["A2MARK", "A2MORE", "Q2MARK", "axel_forget"] {
        assert!(!holds(&units, marker), "{marker}");
    }
    assert!(units[0].body.contains("A1MARK") && units[1].body.contains("A3MARK"));
    assert_eq!(archive::forgotten_keys(&a, &p), vec![mark(Some(key_of(Q2)), false, false)]);

    // (a) the second prompt holds `sk-` + 40 characters; (b) it is 5,000 bytes long.
    let with_token = format!("Q2MARK my key is sk-{} and you must forget it", "a".repeat(40));
    let long = format!("Q2MARK {}", "forget this ".repeat(420));
    assert!(long.len() >= 5000);
    for prompt in [&with_token, &long] {
        let a = three(prompt, "axel_forget");
        let units = units_of(&a);
        assert_eq!(positions(&units), vec!["x0.0", "x6.0"]);
        assert!(!holds(&units, "A2MARK") && !holds(&units, "Q2MARK"));
        // The key of the second prompt as stored: the whole of it, before any cut.
        assert_eq!(archive::forgotten_keys(&a, &p), vec![mark(Some(key_of(prompt)), false, false)]);
    }
    let cut_at = long.char_indices().map(|(i, _)| i).take_while(|i| *i <= 4096).last().unwrap();
    assert_ne!(key_of(&long), key_of(&long[..cut_at]));

    // (c) it is stored as `[archive: withheld]`. The placeholder may be a worker's report in
    // the middle of the turn (wave2-c6-review, finding 2), so the exchange before it goes
    // as well, and the mark carries that prompt's key. The third stays.
    let a = three(WITHHELD, "axel_forget");
    let units = units_of(&a);
    assert_eq!(positions(&units), vec!["x6.0"]);
    for marker in ["Q1MARK", "A1MARK", "A2MARK", "A2MORE"] {
        assert!(!holds(&units, marker), "{marker}");
    }
    assert_eq!(archive::forgotten_keys(&a, &p), vec![mark(Some(key_of(Q1)), false, false)]);

    // The leading span, before the first prompt, holds the call: no `Continued` unit.
    let leading_text = "LEADMARK the rest of an answer that began before this file, long enough to keep.";
    let a = arch(vec![asst_tool(0, leading_text, "axel_forget"), result(1), user(2, Q3), asst(3, A3)]);
    let units = units_of(&a);
    assert_eq!(positions(&units), vec!["x2.0"]);
    assert!(!holds(&units, "LEADMARK"));
    assert_eq!(archive::forgotten_keys(&a, &p), vec![mark(None, true, false)]);
    // Without the call it is one.
    let a = arch(vec![asst_tool(0, leading_text, "bash"), result(1), user(2, Q3), asst(3, A3)]);
    assert_eq!(positions(&units_of(&a)), vec!["c0.0", "x2.0"]);
    assert!(archive::forgotten_keys(&a, &p).is_empty());

    // Names (B3-5.2, R3-2c).
    for name in ["axel_forget", "memory_forget", "axel_forget_2", "memory_forget_13"] {
        assert!(p.is_forget_tool(name), "{name}");
        let a = three(Q2, name);
        assert_eq!(positions(&units_of(&a)), vec!["x0.0", "x6.0"], "{name}");
        assert_eq!(archive::forgotten_keys(&a, &p).len(), 1, "{name}");
    }
    for name in ["forget", "axel_forget_x", "axel_forgetful", "axel_forget_", "axel_forget_2x", "xaxel_forget", ""] {
        assert!(!p.is_forget_tool(name), "{name}");
    }
    for name in ["forget", "axel_forget_x", "axel_forgetful"] {
        let a = three(Q2, name);
        let units = units_of(&a);
        assert_eq!(positions(&units), vec!["x0.0", "x2.0", "x6.0"], "{name}");
        assert!(units[1].body.contains("A2MARK") && units[1].body.contains("A2MORE"));
        assert!(archive::forgotten_keys(&a, &p).is_empty(), "{name}");
    }

    // A withheld prompt with no call after it: a `Continued` unit with the reply, and the
    // exchange before it does not contain that reply.
    let a = three(WITHHELD, "bash");
    let units = units_of(&a);
    assert_eq!(positions(&units), vec!["x0.0", "c3.0", "x6.0"]);
    assert!(units[1].body.contains("A2MARK") && units[1].body.contains("A2MORE"));
    assert!(!units[0].body.contains("A2MARK") && !units[0].body.contains("A2MORE"));
    assert_eq!(units[1].turn_key, None);
    assert!(archive::forgotten_keys(&a, &p).is_empty());

    // The call in the last exchange: `tail`.
    let a = arch(vec![user(0, Q1), asst(1, A1), user(2, Q2), asst_tool(3, A2, "axel_forget")]);
    assert_eq!(positions(&units_of(&a)), vec!["x0.0"]);
    assert_eq!(archive::forgotten_keys(&a, &p), vec![mark(Some(key_of(Q2)), false, true)]);

    // A worker's report inside the span that holds the call goes with it.
    let a = arch(vec![user(0, Q2), asst_tool(1, A2, "axel_forget"), user(2, REPORT), user(3, Q3), asst(4, A3)]);
    assert_eq!(positions(&units_of(&a)), vec!["x3.0"]);

    // `keep_forget_turns`: all three exchanges are units and there is no mark.
    let mut keep = policy();
    keep.keep_forget_turns = true;
    let a = three(Q2, "axel_forget");
    let units = archive::extract_units(&a, &keep);
    assert_eq!(positions(&units), vec!["x0.0", "x2.0", "x6.0"]);
    assert!(units[1].body.contains("A2MARK"));
    assert!(archive::forgotten_keys(&a, &keep).is_empty());
    // The rows that were there keep their positions, so their ids.
    let before = units_of(&a);
    assert_eq!(archive::unit_id(&a.scope, &a.id, &before[1]), archive::unit_id(&a.scope, &a.id, &units[2]));
}

// ------------------------------------------------- wave2-c6-review, finding 2 (ruling)

#[test]
fn f2_a_withheld_text_inside_a_forget_turn_does_not_shield_the_prompt() {
    let p = policy();
    let mid = "MIDMARK I am looking for what you told me about the offer, one moment please.";

    // Prompt, reply, withheld, forget call, in one file: no unit holds the prompt or the
    // reply, and the mark carries the prompt's key. The exchange before the turn stays.
    let a = arch(vec![
        user(0, Q1),
        asst(1, A1),
        user(2, Q2),
        asst(3, A2),
        user(4, WITHHELD),
        asst_tool(5, A2B, "axel_forget"),
        result(6),
        user(7, Q3),
        asst(8, A3),
    ]);
    let units = units_of(&a);
    assert_eq!(positions(&units), vec!["x0.0", "x7.0"]);
    for marker in ["Q2MARK", "A2MARK", "A2MORE", "Acme"] {
        assert!(!holds(&units, marker), "{marker}");
    }
    assert_eq!(archive::forgotten_keys(&a, &p), vec![mark(Some(key_of(Q2)), false, false)]);
    // The same file with another tool: the exchange is there, and the reply after the
    // placeholder is a unit of its own when it is long enough.
    let mut plain = a.clone();
    plain.messages[5] = asst_tool(5, mid, "bash");
    let units = units_of(&plain);
    assert_eq!(positions(&units), vec!["x0.0", "x2.0", "c5.0", "x7.0"]);
    assert!(units[1].body.contains("Q2MARK") && units[1].body.contains("A2MARK"));
    assert!(archive::forgotten_keys(&plain, &p).is_empty());
    // `keep_forget_turns` brings the turn back whole.
    let mut keep = policy();
    keep.keep_forget_turns = true;
    assert_eq!(positions(&archive::extract_units(&a, &keep)), vec!["x0.0", "x2.0", "x7.0"]);
    assert!(archive::forgotten_keys(&a, &keep).is_empty());

    // The same across two files: the first ends with the prompt and its reply, and the
    // placeholder and the call open the second. The second file's mark has `head`.
    let first = arch(vec![user(0, Q1), asst(1, A1), user(2, Q2), asst(3, A2)]);
    assert_eq!(positions(&units_of(&first)), vec!["x0.0", "x2.0"]);
    assert!(archive::forgotten_keys(&first, &p).is_empty());
    // As a later part, read whole: no prompt of the turn is in this file, so no key.
    let mut second = arch(vec![
        user(0, WITHHELD),
        asst_tool(1, mid, "axel_forget"),
        result(2),
        user(3, Q3),
        asst(4, A3),
    ]);
    second.start = Start::Whole;
    let units = units_of(&second);
    assert_eq!(positions(&units), vec!["x3.0"]);
    assert!(!holds(&units, "MIDMARK"));
    assert_eq!(archive::forgotten_keys(&second, &p), vec![mark(None, true, false)]);
    // As a successor: the prompt is carried, then the tail, then the placeholder and the
    // call. The key is the carried prompt's, as this file stores it.
    let second = arch(vec![
        user(1, Q2),
        asst(2, A2),
        user(3, WITHHELD),
        asst_tool(4, mid, "axel_forget"),
        result(5),
        user(6, Q3),
        asst(7, A3),
    ]);
    assert_eq!(archive::carried_head(&second), Some(1));
    let units = units_of(&second);
    assert_eq!(positions(&units), vec!["x6.0"]);
    assert!(!holds(&units, "MIDMARK") && !holds(&units, "Q2MARK"));
    assert_eq!(archive::forgotten_keys(&second, &p), vec![mark(Some(key_of(Q2)), true, false)]);

    // Two withheld texts in a row before the call: the walk passes both.
    let a = arch(vec![
        user(0, Q1),
        asst(1, A1),
        user(2, Q2),
        asst(3, A2),
        user(4, WITHHELD),
        asst(5, mid),
        user(6, WITHHELD),
        asst_tool(7, A2B, "axel_forget"),
        user(8, Q3),
        asst(9, A3),
    ]);
    let units = units_of(&a);
    assert_eq!(positions(&units), vec!["x0.0", "x8.0"]);
    for marker in ["Q2MARK", "A2MARK", "MIDMARK", "A2MORE"] {
        assert!(!holds(&units, marker), "{marker}");
    }
    assert_eq!(archive::forgotten_keys(&a, &p), vec![mark(Some(key_of(Q2)), false, false)]);

    // A withheld text AFTER the call does not clear `tail`: only a human prompt shows that
    // the turn ended here. What follows the placeholder is still read in this file.
    let after = "AFTERMARK it is done, and I have nothing more to say about that in this window.";
    let a = arch(vec![
        user(0, Q1),
        asst(1, A1),
        user(2, Q2),
        asst_tool(3, A2, "axel_forget"),
        user(4, WITHHELD),
        asst(5, after),
    ]);
    assert_eq!(positions(&units_of(&a)), vec!["x0.0", "c5.0"]);
    assert_eq!(archive::forgotten_keys(&a, &p), vec![mark(Some(key_of(Q2)), false, true)]);
    // With a human prompt after it, `tail` is clear.
    let mut messages = a.messages.clone();
    messages.push(user(6, Q3));
    messages.push(asst(7, A3));
    assert_eq!(
        archive::forgotten_keys(&arch(messages), &p),
        vec![mark(Some(key_of(Q2)), false, false)]
    );

    // Two forget spans in one file give two marks, in file order.
    let a = arch(vec![
        user(0, Q2),
        asst_tool(1, A2, "axel_forget"),
        user(2, Q1),
        asst(3, A1),
        user(4, Q3),
        asst_tool(5, A3, "memory_forget"),
    ]);
    assert_eq!(positions(&units_of(&a)), vec!["x2.0"]);
    assert_eq!(
        archive::forgotten_keys(&a, &p),
        vec![mark(Some(key_of(Q2)), false, false), mark(Some(key_of(Q3)), false, true)]
    );
    // A second call behind a placeholder in the same turn: two marks with the one key.
    let a = arch(vec![
        user(0, Q2),
        asst_tool(1, A2, "axel_forget"),
        user(2, WITHHELD),
        asst_tool(3, A2B, "axel_forget"),
    ]);
    assert!(units_of(&a).is_empty());
    assert_eq!(
        archive::forgotten_keys(&a, &p),
        vec![mark(Some(key_of(Q2)), false, true), mark(Some(key_of(Q2)), false, true)]
    );
}

// ----------------------------------------------------------------------------- test 28

/// The nine stored messages of a window as `successor_from_parts` builds one, the first at
/// `first`: (1) the chat's first request, 80 characters; (2) an earlier request, 70
/// characters; (3) a `subagent_completion` event; (4) the request that started the turn in
/// progress; (5) the assistant tail, with the text `TAIL` and a `tool_use`; (6) its
/// `tool_result`; (7) an assistant message with the text `NEW`; (8) a new request; (9) its
/// reply.
fn nine(first: u32, tail_tool: &str) -> Vec<ArchivedMessage> {
    let p1 = pad("P1MARK the chat's first request ", 80);
    let p2 = pad("P2MARK an earlier request ", 70);
    assert_eq!((p1.chars().count(), p2.chars().count()), (80, 70));
    vec![
        user(first, &p1),
        user(first + 1, &p2),
        user(first + 2, REPORT),
        user(first + 3, P4),
        asst_tool(first + 4, TAIL, tail_tool),
        result(first + 5),
        asst(first + 6, NEW),
        user(first + 7, P8),
        asst(first + 8, R9),
    ]
}

const CARRIED_MARKERS: [&str; 4] = ["P1MARK", "P2MARK", "REPORTMARK", "P4MARK"];

fn assert_read_as_a_successor(a: &LiveArchive) {
    let units = units_of(a);
    assert_eq!(positions(&units), vec!["c7.0", "x8.0"]);
    assert_eq!(units[0].kind, UnitKind::Continued);
    assert!(units[0].body.contains("NEW") && !units[0].body.contains("TAIL"));
    assert!(units[1].body.starts_with("User: P8MARK") && units[1].body.contains("REPLY9"));
    for marker in CARRIED_MARKERS {
        assert!(!holds(&units, marker), "{marker}");
    }
    assert!(!holds(&units, "TAIL"));
    assert_eq!(archive::carried_head(a), Some(4));
}

fn assert_read_whole(a: &LiveArchive, first: u32) {
    let units = units_of(a);
    let want = vec![format!("r{}.0", first + 2), format!("x{}.0", first + 3), format!("x{}.0", first + 7)];
    assert_eq!(positions(&units), want);
    assert!(units[0].body.contains("REPORTMARK"));
    assert!(units[1].body.starts_with("User: P4MARK"));
    assert!(units[1].body.contains("TAIL") && units[1].body.contains("NEW"));
    assert!(units[2].body.contains("P8MARK") && units[2].body.contains("REPLY9"));
    // The two first prompts yield nothing, because they have no reply (rule 1a).
    assert!(!holds(&units, "P1MARK") && !holds(&units, "P2MARK"));
    assert_eq!(archive::carried_head(a), None);
}

#[test]
fn t28_the_carried_head() {
    // Stored without the continuation message at index 0: `source_index` 1 to 9.
    let a = arch(nine(1, "bash"));
    assert_eq!(a.source_message_count, 10);
    assert_read_as_a_successor(&a);

    // The layout that pairs wrongly without the rule: the tail answered a large request
    // core did not keep. Nothing pairs request 2 with the tail.
    let a = arch(vec![
        user(1, &pad("P1MARK the chat's first request ", 80)),
        user(2, "P2MARK a small old request"),
        asst(3, TAIL),
        user(4, P8),
        asst(5, R9),
    ]);
    let units = units_of(&a);
    assert_eq!(positions(&units), vec!["x4.0"]);
    assert!(!holds(&units, "P2MARK") && !holds(&units, "TAIL") && !holds(&units, "P1MARK"));
    assert_eq!(archive::carried_head(&a), Some(2));

    // The same nine messages with `source_index` 0 to 8 (a first window, or a later part)
    // are read as before.
    assert_read_whole(&arch(nine(0, "bash")), 0);

    // A first window that ends with a prompt and no reply yields no unit for that prompt.
    let a = arch(vec![user(0, Q1), asst(1, A1), user(2, P8)]);
    let units = units_of(&a);
    assert_eq!(positions(&units), vec!["x0.0"]);
    assert!(!holds(&units, "P8MARK"));

    // A first stored message that is a user text starting `[Context continuation` marks a
    // successor too.
    let a = arch(vec![
        user(0, "[Context continuation: the conversation so far was archived as ctx-1]"),
        user(1, P4),
        asst(2, TAIL),
        asst(3, NEW),
        user(4, P8),
        asst(5, R9),
    ]);
    assert_eq!(archive::carried_head(&a), Some(2));
    let units = units_of(&a);
    assert_eq!(positions(&units), vec!["c3.0", "x4.0"]);
    assert!(!holds(&units, "P4MARK") && !holds(&units, "TAIL"));
    // A successor with no assistant message yields no unit.
    let a = arch(vec![user(1, P4), user(2, REPORT), user(3, P8)]);
    assert!(units_of(&a).is_empty());
    assert_eq!(archive::carried_head(&a), Some(2));
    // An archive with no message has no head.
    assert_eq!(archive::carried_head(&arch(Vec::new())), None);

    // A later part that begins above 0 (4.1, risk 1): what the event said spares it.
    assert_eq!(Start::from_event(Some(5), Some(2)), Start::Whole);
    assert_eq!(Start::from_event(Some(3), Some(1)), Start::Carried);
    assert_eq!(Start::from_event(Some(2), Some(1)), Start::Unknown);
    assert_eq!(Start::from_event(None, Some(2)), Start::Unknown);
    assert_eq!(Start::from_event(Some(3), None), Start::Unknown);
    assert_eq!(Start::from_event(None, None), Start::Unknown);
    assert_eq!(Start::from_event(Some(1), Some(1)), Start::Unknown);
    let mut a = arch(nine(1, "bash"));
    a.start = Start::from_event(Some(5), Some(2));
    assert_read_whole(&a, 1);
    // With `Unknown` the same nine messages are read as a successor (the first case).
    a.start = Start::from_event(Some(2), Some(1));
    assert_read_as_a_successor(&a);
    // The event confirms a successor whose first index it cannot see.
    let mut a = arch(nine(0, "bash"));
    a.start = Start::Carried;
    assert_eq!(positions(&units_of(&a)), vec!["c6.0", "x7.0"]);
    assert_eq!(archive::carried_head(&a), Some(4));

    // The record's mark wins (4.1, ruling 5a).
    let mut a = arch(nine(1, "bash"));
    a.carried = Some(0);
    a.start = Start::Carried;
    assert_read_whole(&a, 1);
    let mut a = arch(nine(0, "bash"));
    a.carried = Some(5);
    a.start = Start::Whole;
    let units = units_of(&a);
    assert_eq!(positions(&units), vec!["c6.0", "x7.0"]);
    assert!(units[0].body.contains("NEW") && !units[0].body.contains("TAIL"));
    assert_eq!(archive::carried_head(&a), Some(4));
    // No assistant message among the copies: the fourth stands in for the tail, and the
    // unit that follows is a `Continued` unit holding `TAIL` and `NEW`.
    a.carried = Some(4);
    assert_eq!(archive::carried_head(&a), Some(3));
    let units = units_of(&a);
    assert_eq!(positions(&units), vec!["c4.0", "x7.0"]);
    assert!(units[0].body.contains("TAIL") && units[0].body.contains("NEW"));
    for marker in CARRIED_MARKERS {
        assert!(!holds(&units, marker), "{marker}");
    }
    // A mark larger than the message count is read as none: the event and the index decide.
    let mut a = arch(nine(1, "bash"));
    a.carried = Some(10);
    assert_read_as_a_successor(&a);
    let mut a = arch(nine(0, "bash"));
    a.carried = Some(10);
    assert_read_whole(&a, 0);

    // With the tail's `tool_use` named `axel_forget`: no `c7.0` either, and one mark with
    // `head` true and the key of message 4 as stored.
    let a = arch(nine(1, "axel_forget"));
    let units = units_of(&a);
    assert_eq!(positions(&units), vec!["x8.0"]);
    assert!(!holds(&units, "NEW"));
    assert_eq!(archive::forgotten_keys(&a, &policy()), vec![mark(Some(key_of(P4)), true, false)]);
    // A withheld prompt nearer to the call than any human one: no key.
    let mut messages = nine(1, "axel_forget");
    messages[3] = user(4, WITHHELD);
    assert_eq!(archive::forgotten_keys(&arch(messages), &policy()), vec![mark(None, true, false)]);
    // Read whole, the same call is in the exchange of message 4, which is no leading span.
    let mut a = arch(nine(1, "axel_forget"));
    a.start = Start::Whole;
    assert_eq!(positions(&units_of(&a)), vec!["r3.0", "x8.0"]);
    assert_eq!(archive::forgotten_keys(&a, &policy()), vec![mark(Some(key_of(P4)), false, false)]);
}

#[test]
fn t28_the_carried_head_from_files() {
    let shelf = fixtures().join(hand_dir()).join("s000001");
    let entries = archive::list_project(&fixtures().join(hand_dir()));
    assert_eq!(entries.len(), 3);

    // The hand-made successor, as core stores one today: no mark in the record.
    let name = "28000000000000000000000000000001";
    let mut a = live(archive::read_archive(&shelf.join(format!("{name}.json")), &hand_dir(), Some(name)));
    assert_eq!((a.carried, a.start), (None, Start::Unknown));
    assert!(a.messages[0].source_index > 0);
    assert_eq!(a.source_message_count, 10);
    assert_read_as_a_successor(&a);
    assert_eq!(units_of(&a)[0].aux, "", "the tail's call is not the new span's");
    // Named by an event as part 2, the same file is read whole.
    a.start = Start::from_event(Some(5), Some(2));
    assert_read_whole(&a, 1);

    // The same messages with `carried: 5` in the record: a successor whatever an event said.
    let name = "28000000000000000000000000000002";
    let mut a = live(archive::read_archive(&shelf.join(format!("{name}.json")), &hand_dir(), Some(name)));
    assert_eq!(a.carried, Some(5));
    a.start = Start::Whole;
    assert_read_as_a_successor(&a);

    // A record whose `carried` is not a non-negative integer, or is larger than its message
    // count, is read as having none.
    let text = fs::read_to_string(shelf.join(format!("{name}.json"))).unwrap();
    let base: Value = serde_json::from_str(&text).unwrap();
    for (value, want) in [
        (json!(-1), None),
        (json!("5"), None),
        (json!(1.5), None),
        (Value::Null, None),
        (json!(10), None),
        (json!(9), Some(9)),
        (json!(0), Some(0)),
    ] {
        let (_tmp, dir, scope) = project();
        let mut r = base.clone();
        r["scope"] = json!(scope);
        r["carried"] = value.clone();
        let path = dir.join("s000001").join(format!("{name}.json"));
        put(&path, &r);
        let a = live(archive::read_archive(&path, &scope, Some(name)));
        assert_eq!(a.carried, want, "{value}");
        match want {
            None => assert_read_as_a_successor(&a),
            Some(0) => assert_read_whole(&a, 1),
            // All nine are copies; the tail is the first assistant message among them, and
            // what follows it is read (what followed the tail in the old window is stored
            // after it and cannot be told from new messages).
            Some(_) => assert_read_as_a_successor(&a),
        }
    }
}
