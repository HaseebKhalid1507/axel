//! Commit 7: the notes reader (spec 2.6 steps 1 and 2; section 2, tests 20 and 21, the
//! parsing halves: what `notes::read_file` returns). The fixture is the contract test of the
//! file format, read from core's `store.rs`.

use std::fs;
use std::path::{Path, PathBuf};

use axel::notes::{self, NoteRecord, NoteSensitivity, NotesError};
use tempfile::TempDir;

const KEY: &str = "p1111111111111111";

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/notes")
        .join(format!("project-{KEY}.jsonl"))
}

fn line(id: &str, ts: u64, content: &str) -> String {
    format!(
        r#"{{"namespace":"project-{KEY}","timestamp_ms":{ts},"content":"{content}","tags":[],"id":"{id}","project":"{KEY}"}}"#
    )
}

fn three() -> String {
    format!(
        "{}\n{}\n{}\n",
        line("mem-a", 1, "alpha"),
        line("mem-b", 2, "beta"),
        line("mem-c", 3, "gamma")
    )
}

fn temp_file(tmp: &TempDir, body: &[u8]) -> PathBuf {
    let path = tmp.path().join(format!("project-{KEY}.jsonl"));
    fs::write(&path, body).unwrap();
    path
}

fn ids(live: &[NoteRecord]) -> Vec<&str> {
    live.iter().map(|n| n.id.as_str()).collect()
}

/// Section 2, test 20, the parsing half.
#[test]
fn test_20_fixture_gives_three_normal_and_one_secret() {
    let path = fixture();
    let got = notes::read_file(&path, KEY).unwrap();
    assert_eq!(got.size, fs::metadata(&path).unwrap().len());
    assert!(got.mtime_ns > 0);
    assert_eq!(ids(&got.live), ["mem-one", "mem-two", "mem-three", "mem-secret"]);

    let one = &got.live[0];
    assert_eq!(one.content, "later text of one", "the later line wins");
    assert_eq!(one.tags, ["a"]);
    assert_eq!(one.timestamp_ms, 1_700_000_000_700);
    assert_eq!(one.sensitivity, NoteSensitivity::Normal);
    assert_eq!(one.max_age_days, None);

    assert_eq!(got.live[1].sensitivity, NoteSensitivity::Sensitive);

    let three = &got.live[2];
    assert!(three.tags.is_empty(), "\"tags\": null is no tags");
    assert_eq!(three.max_age_days, Some(30));
    assert_eq!(three.sensitivity, NoteSensitivity::Normal);

    let secret = &got.live[3];
    assert_eq!(secret.sensitivity, NoteSensitivity::Secret);
    assert_eq!(secret.content, "SECRET-BODY-SENTINEL");

    let normal = got
        .live
        .iter()
        .filter(|n| n.sensitivity != NoteSensitivity::Secret)
        .count();
    assert_eq!(normal, 3);
}

#[test]
fn the_other_project_reads_only_its_own_record() {
    let got = notes::read_file(&fixture(), "p2222222222222222").unwrap();
    assert_eq!(ids(&got.live), ["mem-foreign"]);
}

#[test]
fn an_id_on_two_live_lines_gives_the_later_text() {
    let tmp = TempDir::new().unwrap();
    let body = format!(
        "{}\n{}\n{}\n",
        line("mem-a", 1, "old"),
        line("mem-b", 2, "beta"),
        line("mem-a", 3, "new")
    );
    let got = notes::read_file(&temp_file(&tmp, body.as_bytes()), KEY).unwrap();
    assert_eq!(ids(&got.live), ["mem-a", "mem-b"]);
    assert_eq!(got.live[0].content, "new");
    assert_eq!(got.live[0].timestamp_ms, 3);
}

#[test]
fn a_tombstone_wins_in_either_order() {
    let tomb = r#"{"tombstone":"mem-b","timestamp_ms":9}"#;
    let tmp = TempDir::new().unwrap();
    let after = format!("{}{tomb}\n", three());
    let got = notes::read_file(&temp_file(&tmp, after.as_bytes()), KEY).unwrap();
    assert_eq!(ids(&got.live), ["mem-a", "mem-c"]);

    let before = format!("{tomb}\n{}", three());
    let got = notes::read_file(&temp_file(&tmp, before.as_bytes()), KEY).unwrap();
    assert_eq!(ids(&got.live), ["mem-a", "mem-c"]);

    // A record written again after its tombstone is still forgotten.
    let again = format!("{}{tomb}\n{}\n", three(), line("mem-b", 4, "back"));
    let got = notes::read_file(&temp_file(&tmp, again.as_bytes()), KEY).unwrap();
    assert_eq!(ids(&got.live), ["mem-a", "mem-c"]);
}

#[test]
fn a_torn_last_line_is_ignored_also_when_not_utf8() {
    let tmp = TempDir::new().unwrap();
    let mut body = three().into_bytes();
    body.extend_from_slice(b"{\"content\":\"caf\xc3");
    let got = notes::read_file(&temp_file(&tmp, &body), KEY).unwrap();
    assert_eq!(ids(&got.live), ["mem-a", "mem-b", "mem-c"]);

    // A last line with no break that parses is a record.
    let whole = format!("{}{}", three(), line("mem-d", 4, "delta"));
    let got = notes::read_file(&temp_file(&tmp, whole.as_bytes()), KEY).unwrap();
    assert_eq!(ids(&got.live), ["mem-a", "mem-b", "mem-c", "mem-d"]);
}

/// Section 2, test 21, the parsing half: a bad read is an error, never a smaller set.
#[test]
fn test_21_a_bad_read_is_an_error_and_an_empty_file_is_empty() {
    let tmp = TempDir::new().unwrap();
    let path = temp_file(&tmp, three().as_bytes());
    assert_eq!(notes::read_file(&path, KEY).unwrap().live.len(), 3);

    // A broken line in the middle.
    let broken = format!(
        "{}\nthis is not json\n{}\n{}\n",
        line("mem-a", 1, "alpha"),
        line("mem-b", 2, "beta"),
        line("mem-c", 3, "gamma")
    );
    fs::write(&path, broken).unwrap();
    match notes::read_file(&path, KEY) {
        Err(NotesError::BadLine { line, reason }) => {
            assert_eq!(line, 2);
            assert!(!reason.is_empty());
        }
        other => panic!("expected BadLine, got {other:?}"),
    }

    // Cut in the middle of a line that is not the last.
    let a = line("mem-a", 1, "alpha");
    let cut = format!(
        "{}\n{}\n{}\n",
        &a[..a.len() / 2],
        line("mem-b", 2, "beta"),
        line("mem-c", 3, "gamma")
    );
    fs::write(&path, cut).unwrap();
    assert!(matches!(
        notes::read_file(&path, KEY),
        Err(NotesError::BadLine { line: 1, .. })
    ));

    // A line that is not UTF-8 in the middle, and JSON that is not an object.
    let mut bytes = format!("{a}\n").into_bytes();
    bytes.extend_from_slice(b"{\"content\":\"caf\xc3\"}\n");
    bytes.extend_from_slice(format!("{}\n", line("mem-c", 3, "gamma")).as_bytes());
    fs::write(&path, bytes).unwrap();
    assert!(matches!(
        notes::read_file(&path, KEY),
        Err(NotesError::BadLine { line: 2, .. })
    ));
    fs::write(&path, format!("{a}\n[1,2]\n")).unwrap();
    assert!(matches!(
        notes::read_file(&path, KEY),
        Err(NotesError::BadLine { line: 2, .. })
    ));

    // An empty file parses in full: no notes.
    fs::write(&path, "").unwrap();
    let empty = notes::read_file(&path, KEY).unwrap();
    assert_eq!(empty.size, 0);
    assert!(empty.live.is_empty());

    // The full file back: the three again.
    fs::write(&path, three()).unwrap();
    assert_eq!(
        ids(&notes::read_file(&path, KEY).unwrap().live),
        ["mem-a", "mem-b", "mem-c"]
    );
}

#[test]
fn objects_that_are_not_records_are_skipped_like_core() {
    let tmp = TempDir::new().unwrap();
    let body = format!(
        "{}\n{{\"namespace\":\"x\"}}\n{{\"tombstone\":\"mem-a\"}}\n   \n{}\n",
        line("mem-a", 1, "alpha"),
        line("mem-b", 2, "beta")
    );
    // A tombstone without its timestamp is not core's tombstone line: mem-a stays.
    let got = notes::read_file(&temp_file(&tmp, body.as_bytes()), KEY).unwrap();
    assert_eq!(ids(&got.live), ["mem-a", "mem-b"]);
}

#[test]
fn a_missing_file_or_a_link_is_an_io_error() {
    let tmp = TempDir::new().unwrap();
    let missing = tmp.path().join("project-p1111111111111111.jsonl");
    assert!(matches!(
        notes::read_file(&missing, KEY),
        Err(NotesError::Io(_))
    ));
    let link = tmp.path().join("link.jsonl");
    std::os::unix::fs::symlink(fixture(), &link).unwrap();
    assert!(matches!(notes::read_file(&link, KEY), Err(NotesError::Io(_))));
}
