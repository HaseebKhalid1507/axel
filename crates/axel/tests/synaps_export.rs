//! T-AX1 to T-AX4 of the cut spec (section A2): `axel synaps-export`.
//!
//! Every brain here is a fixture built in a temp folder. The table
//! definitions are the ones the writers use, copied statement for statement:
//!
//! * `memories`: `crates/axel/src/r8.rs:235-251`, plus the columns that
//!   `crates/axel/src/project_memory.rs:234-275` adds.
//! * `memory_tombstones`: `crates/axel/src/project_memory.rs:287-291`.
//! * `synaps_history`, `synaps_captures`, `synaps_fingerprints`,
//!   `synaps_migrations`: these are NOT in this repository. They are created
//!   by the Synaps helper, `sidecars/axel-memory-service/src/database.rs:23-37`
//!   in Synaps core; `synaps_scope_members` by `scope.rs:32` there.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;

use axel::synaps_export::{export, projection_digest, scope_of, ExportError, CREATED_MS_FIELD};
use rusqlite::{params, Connection};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS memories (
    id              TEXT PRIMARY KEY,
    category        TEXT NOT NULL,
    topic           TEXT NOT NULL,
    title           TEXT NOT NULL,
    abstract_text   TEXT,
    content         TEXT NOT NULL,
    confidence      TEXT DEFAULT 'medium',
    importance      REAL DEFAULT 0.5,
    source_sessions TEXT DEFAULT '[]',
    tags            TEXT DEFAULT '[]',
    related_topics  TEXT DEFAULT '[]',
    created         TEXT NOT NULL,
    updated         TEXT,
    trust_level     REAL DEFAULT 1.0,
    signature       TEXT
);
ALTER TABLE memories ADD COLUMN project_key TEXT;
ALTER TABLE memories ADD COLUMN sensitivity TEXT NOT NULL DEFAULT 'normal';
ALTER TABLE memories ADD COLUMN retention TEXT NOT NULL DEFAULT 'standard';
ALTER TABLE memories ADD COLUMN provenance TEXT NOT NULL DEFAULT '';
ALTER TABLE memories ADD COLUMN expires_at TEXT;
ALTER TABLE memories ADD COLUMN superseded_by TEXT;
ALTER TABLE memories ADD COLUMN supersedes TEXT NOT NULL DEFAULT '[]';
ALTER TABLE memories ADD COLUMN contradicts TEXT NOT NULL DEFAULT '[]';
ALTER TABLE memories ADD COLUMN confirmed_by TEXT NOT NULL DEFAULT '[]';
ALTER TABLE memories ADD COLUMN invalidated_at TEXT;
CREATE TABLE IF NOT EXISTS memory_tombstones (
    id          TEXT PRIMARY KEY,
    project_key TEXT NOT NULL,
    deleted_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS synaps_history(
  project_key TEXT NOT NULL,id TEXT PRIMARY KEY,logical TEXT NOT NULL,digest TEXT NOT NULL,
  fingerprint TEXT,source_count INTEGER NOT NULL,message_count INTEGER NOT NULL,
  messages TEXT NOT NULL,note TEXT NOT NULL,created_ms INTEGER NOT NULL,tombstoned INTEGER NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS synaps_history_fp ON synaps_history(project_key,fingerprint,tombstoned);
CREATE TABLE IF NOT EXISTS synaps_captures(
  project_key TEXT NOT NULL,capture_id TEXT PRIMARY KEY,note_id TEXT NOT NULL UNIQUE,
  payload_digest TEXT NOT NULL,source_digest TEXT NOT NULL,evidence TEXT NOT NULL,
  withheld INTEGER NOT NULL,tombstoned INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS synaps_fingerprints(
  project_key TEXT NOT NULL,kind TEXT NOT NULL,digest TEXT NOT NULL,
  PRIMARY KEY(project_key,kind,digest));
CREATE TABLE IF NOT EXISTS synaps_migrations(
  project_key TEXT NOT NULL,migration_id TEXT PRIMARY KEY,manifest_digest TEXT NOT NULL,
  committed INTEGER NOT NULL DEFAULT 0,counts TEXT,apply_digest TEXT);
";
const SCOPE_SCHEMA: &str = "CREATE TABLE IF NOT EXISTS synaps_scope_members(member TEXT PRIMARY KEY,canonical TEXT NOT NULL); CREATE INDEX IF NOT EXISTS synaps_scope_group ON synaps_scope_members(canonical);";

const K1: &str = "p1111111111111111";
const K2: &str = "p2222222222222222";
const K3: &str = "p3333333333333333";

struct Fixture {
    _dir: tempfile::TempDir,
    brain: PathBuf,
    out: PathBuf,
    conn: Option<Connection>,
}

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let brain = dir.path().join("brain.r8");
        let conn = Connection::open(&brain).unwrap();
        // The real file is in WAL mode (F11).
        let mode: String = conn.query_row("PRAGMA journal_mode=WAL", [], |r| r.get(0)).unwrap();
        assert_eq!(mode, "wal");
        conn.execute_batch(SCHEMA).unwrap();
        Self { out: dir.path().join("out"), _dir: dir, brain, conn: Some(conn) }
    }
    fn conn(&self) -> &Connection {
        self.conn.as_ref().unwrap()
    }
    /// Close the writer: SQLite checkpoints and removes `-wal` and `-shm`.
    fn close(&mut self) {
        self.conn.take().unwrap().close().unwrap();
        assert!(!sidecar(&self.brain, "-wal").exists());
        assert!(!sidecar(&self.brain, "-shm").exists());
    }
    #[allow(clippy::too_many_arguments)]
    fn history(&self, key: &str, id: &str, logical: &str, digest: &str, fingerprint: Option<&str>,
               source: i64, count: i64, messages: &str, note: &str, created: i64, tombstoned: bool) {
        self.conn().execute(
            "INSERT INTO synaps_history(project_key,id,logical,digest,fingerprint,source_count,message_count,messages,note,created_ms,tombstoned) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)",
            params![key, id, logical, digest, fingerprint, source, count, messages, note, created, tombstoned],
        ).unwrap();
    }
    fn fingerprint(&self, key: &str, kind: &str, value: &str) {
        self.conn().execute("INSERT INTO synaps_fingerprints VALUES(?1,?2,?3)", params![key, kind, value]).unwrap();
    }
    fn tombstone(&self, id: &str, key: &str, deleted_at: &str) {
        self.conn().execute(
            "INSERT INTO memory_tombstones(id,project_key,deleted_at) VALUES(?1,?2,?3)",
            params![id, key, deleted_at],
        ).unwrap();
    }
    /// A row of `memories` as the helper's `insert_record` writes it
    /// (`database.rs:196-197`): `record` is the record JSON with an empty
    /// `content`, the text is in the `content` column.
    fn record(&self, key: &str, id: &str, record: &str, content: &str) {
        let provenance = format!(
            r#"{{"schema":"synaps-axel/1","record":{record},"content_bytes":{}}}"#,
            content.len()
        );
        self.memory(Some(key), id, &provenance, content);
    }
    fn memory(&self, key: Option<&str>, id: &str, provenance: &str, content: &str) {
        self.conn().execute(
            "INSERT INTO memories(id,category,topic,title,abstract_text,content,confidence,importance,source_sessions,tags,related_topics,created,updated,trust_level,signature,project_key,sensitivity,retention,provenance,expires_at,superseded_by,supersedes,contradicts,confirmed_by,invalidated_at)
             VALUES(?1,'events','synaps','','',?2,'medium',0.5,'[]','[]','[]','1970-01-01T00:00:00+00:00',NULL,1.0,NULL,?3,'normal','standard',?4,NULL,NULL,'[]','[]','[]',NULL)",
            params![id, content, key, provenance],
        ).unwrap();
    }
}

fn sidecar(brain: &Path, suffix: &str) -> PathBuf {
    let mut name = brain.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

fn sha_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// Independent of the exporter: SHA-256 over `be64(len) ‖ part`.
fn parts(parts: &[&[u8]]) -> String {
    let mut h = Sha256::new();
    for p in parts {
        h.update((p.len() as u64).to_be_bytes());
        h.update(p);
    }
    hex::encode(h.finalize())
}

fn note_record(key: &str, id: &str, ts: u64, content: &str, meta: &str) -> String {
    format!(
        r#"{{"namespace":"project-{key}","timestamp_ms":{ts},"content":{},"tags":["t"],"meta":{meta},"id":"{id}","project":"{key}","provenance":{{"source":"test"}},"sensitivity":"normal","retention":"standard"}}"#,
        serde_json::to_string(content).unwrap()
    )
}

/// A note with a session, as an extension stores it: only such a note has a
/// `note` fingerprint (`sidecar database.rs:71-82`).
fn session_note(key: &str, id: &str, ts: u64, session: &str) -> String {
    format!(
        r#"{{"namespace":"project-{key}","timestamp_ms":{ts},"content":"","tags":[],"id":"{id}","project":"{key}","provenance":{{"source":"extension","session":"{session}"}},"sensitivity":"normal","retention":"standard"}}"#
    )
}

fn forum_record(key: &str, id: &str) -> String {
    format!(
        r#"{{"namespace":"forum","timestamp_ms":7,"content":"","tags":[],"meta":{{"_synaps_forum":{{"schema":1}}}},"id":"{id}","project":"{key}","provenance":{{"source":"forum","session":"g"}},"sensitivity":"normal","retention":"standard"}}"#
    )
}

fn all_output(out: &Path) -> String {
    let mut text = String::new();
    for entry in walk(out) {
        text.push_str(&fs::read_to_string(entry).unwrap());
    }
    text
}

fn walk(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            files.extend(walk(&path));
        } else {
            files.push(path);
        }
    }
    files
}

const MSG_A: &str = r#"[{"source_index":0,"block_indices":[0],"message":{"content":"hello archive","role":"user"}}]"#;
const MSG_C: &str = r#"[{"source_index":0,"block_indices":[0],"message":{"content":"suppressed archive text","role":"user"}},{"source_index":1,"block_indices":[0],"message":{"content":"more","role":"assistant"}}]"#;

/// T-AX1 mapping: the exported lines and the manifest are exactly the
/// expected JSON; the live line has `created_ms`, the tombstone lines do not.
#[test]
fn t_ax1_mapping_lines_and_manifest_are_exact() {
    let mut fx = Fixture::new();
    let la = "1a".repeat(32);
    let lc = "2c".repeat(32);
    let (id_a, id_b, id_c) = ("a".repeat(32), "b".repeat(32), "c".repeat(32));
    let dig_a = parts(&[la.as_bytes(), &3u64.to_be_bytes(), MSG_A.as_bytes()]);
    let dig_b = "db".repeat(32);
    let dig_c = parts(&[lc.as_bytes(), &9u64.to_be_bytes(), MSG_C.as_bytes()]);
    let fp_c = "fc".repeat(32);
    let orphan = "ff".repeat(32);

    // A live row; a tombstoned row, as the helper leaves it; a row that is
    // suppressed only through synaps_fingerprints (by its fingerprint).
    fx.history(K1, &id_a, &la, &dig_a, Some(&"fa".repeat(32)), 3, 1, MSG_A, "a note", 1000, false);
    fx.history(K1, &id_b, &la, &dig_b, None, 5, 2, "[]", "", 500, true);
    fx.history(K1, &id_c, &lc, &dig_c, Some(&fp_c), 9, 2, MSG_C, "secret note of c", 2000, false);
    fx.fingerprint(K1, "history", &dig_b);
    fx.fingerprint(K1, "history", &fp_c);
    // A fingerprint row with no history row.
    fx.fingerprint(K1, "history", &orphan);

    // A note; a forum post; a note tombstone; a note row that also has one.
    let n1 = format!("mem-{}", "1".repeat(32));
    let n3 = format!("mem-{}", "3".repeat(32));
    let n4 = format!("mem-{}", "4".repeat(32));
    let post = format!("msg-{}", "2".repeat(64));
    let meta = r#"{"z":1.50,"_synaps_repository":{"source_worktree_root":"/synthetic/root"},"a":1e2}"#;
    fx.record(K1, &n1, &note_record(K1, &n1, 10, "", meta), "note one");
    fx.record(K1, &post, &forum_record(K1, &post), "a forum post body");
    fx.tombstone(&n3, K1, "1970-01-01T00:00:05+00:00");
    fx.record(K1, &n4, &note_record(K1, &n4, 20, "", "{}"), "forgotten note text");
    fx.tombstone(&n4, K1, "1970-01-01T00:00:06.000789+00:00");
    fx.close();

    let manifest = export(&fx.brain, &fx.out).unwrap();

    let scope = parts(&[K1.as_bytes()]);
    assert_eq!(scope, scope_of(K1));
    assert_eq!(CREATED_MS_FIELD, "created_ms");
    let tomb = |logical: &str, id: &str, digest: &str| {
        format!(r#"{{"version":1,"scope":"{scope}","logical":"{logical}","id":"{id}","digest":"{digest}","body":{{"state":"tombstone"}}}}"#)
    };
    let live = format!(
        r#"{{"version":1,"scope":"{scope}","logical":"{la}","id":"{id_a}","digest":"{dig_a}","{CREATED_MS_FIELD}":1000,"body":{{"state":"live","source_message_count":3,"messages":{MSG_A},"note":"a note"}}}}"#
    );
    // Oldest first: b (500), a (1000), c (2000).
    let archives = format!("{}\n{live}\n{}\n", tomb(&la, &id_b, &dig_b), tomb(&lc, &id_c, &dig_c));
    let notes = format!(
        "{}\n{{\"tombstone\":\"{n3}\",\"timestamp_ms\":5000}}\n{{\"tombstone\":\"{n4}\",\"timestamp_ms\":6000}}\n",
        note_record(K1, &n1, 10, "note one", meta)
    );
    let dir = fx.out.join(K1);
    assert_eq!(fs::read_to_string(dir.join("archives.jsonl")).unwrap(), archives);
    assert_eq!(fs::read_to_string(dir.join("notes.jsonl")).unwrap(), notes);

    // Every line is JSON; only the live one has created_ms.
    for line in archives.lines() {
        let v: Value = serde_json::from_str(line).unwrap();
        assert_eq!(v.get(CREATED_MS_FIELD).is_some(), v["body"]["state"] == "live", "{line}");
    }
    // Nothing forgotten or left out reached a plain file.
    let written = all_output(&fx.out);
    for hidden in ["forgotten note text", "suppressed archive text", "secret note of c", "a forum post body"] {
        assert!(!written.contains(hidden), "{hidden}");
    }

    let source = fs::read(&fx.brain).unwrap();
    let expected = json!({
        "source": {"path": fx.brain.to_str().unwrap(), "bytes": source.len(), "sha256": sha_bytes(&source)},
        "projects": {
            K1: {
                "archives": 3, "live": 1, "tombstones": 2, "messages": 1,
                "notes": 1, "note_tombstones": 2, "forum_posts": 1,
                "forum_tombstones": 0, "captures": 0, "foreign_memories": 0,
                "sha256": {
                    "archives.jsonl": sha_bytes(archives.as_bytes()),
                    "notes.jsonl": sha_bytes(notes.as_bytes()),
                },
                "logical": [la, lc],
                "scope_group": null,
                "source_worktree_roots": ["/synthetic/root"],
            }
        },
        "forum_posts": 1,
        "forum_tombstones": 0,
        "captures": 0,
        "foreign_memories": {"total": 0, "list": []},
        "orphan_fingerprints": [{"project_key": K1, "value": orphan}],
        "fingerprints_by_kind": {"history": 3, "note": 0, "capture": 0},
        "digest_mismatch": [],
        "notes_suppressed": 1,
        "check_failed": [],
    });
    assert_eq!(serde_json::to_value(&manifest).unwrap(), expected);
    let file = fs::read_to_string(fx.out.join("manifest.json")).unwrap();
    assert_eq!(serde_json::from_str::<Value>(&file).unwrap(), expected);
    assert!(manifest.clean());
    println!("MANIFEST-EXAMPLE-BEGIN\n{file}MANIFEST-EXAMPLE-END");
}

/// T-AX2 bytes: a `messages` text with unsorted keys and a float survives
/// byte for byte, and the digest is checked over those bytes.
#[test]
fn t_ax2_messages_survive_byte_for_byte() {
    let mut fx = Fixture::new();
    // Keys not sorted, floats that a parse and re-serialise would rewrite
    // (1.50 -> 1.5, 1e2 -> 100.0), an escape that would be normalised.
    let messages = r#"[{"source_index":0,"block_indices":[0],"message":{"role":"user","content":"z \u00e9","b":1.50,"a":1e2}}]"#;
    assert_ne!(serde_json::from_str::<Value>(messages).unwrap().to_string(), messages);
    let logical = "ab".repeat(32);
    let good = "1".repeat(32);
    let bad = "2".repeat(32);
    let digest = parts(&[logical.as_bytes(), &4u64.to_be_bytes(), messages.as_bytes()]);
    assert_eq!(digest, projection_digest(&logical, 4, messages));
    fx.history(K1, &good, &logical, &digest, None, 4, 1, messages, "n", 1, false);
    // The same bytes under a digest that is not theirs: written, and listed.
    fx.history(K1, &bad, &logical, &"0".repeat(64), None, 4, 1, messages, "n", 2, false);
    fx.close();

    let manifest = export(&fx.brain, &fx.out).unwrap();
    let text = fs::read_to_string(fx.out.join(K1).join("archives.jsonl")).unwrap();
    let needle = format!(r#""source_message_count":4,"messages":{messages},"note":"n"}}}}"#);
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 2);
    assert!(lines.iter().all(|line| line.ends_with(&needle)), "{text}");
    assert_eq!(manifest.digest_mismatch, vec![bad]);
    assert!(manifest.check_failed.is_empty());
    assert!(!manifest.clean());
}

/// T-AX3 read-only: hash and mtime unchanged, no `-shm` or `-wal` appears,
/// a non-empty `-wal` (or `-journal`) beside the source is exit code 3.
#[test]
fn t_ax3_source_is_never_written_and_a_wal_is_refused() {
    let mut fx = Fixture::new();
    let logical = "ab".repeat(32);
    let digest = parts(&[logical.as_bytes(), &3u64.to_be_bytes(), MSG_A.as_bytes()]);
    fx.history(K1, &"a".repeat(32), &logical, &digest, None, 3, 1, MSG_A, "private note", 1, false);
    fx.close();
    // The header says WAL (bytes 18 and 19 are 02 02), as the real file's does.
    assert_eq!(fs::read(&fx.brain).unwrap()[18..20], [2, 2]);

    let listing = |dir: &Path| {
        let mut names: Vec<_> = fs::read_dir(dir).unwrap().map(|e| e.unwrap().file_name()).collect();
        names.sort();
        names
    };
    let folder = fx.brain.parent().unwrap().to_owned();
    let before = (sha_bytes(&fs::read(&fx.brain).unwrap()), fs::metadata(&fx.brain).unwrap().modified().unwrap());
    let names = listing(&folder);

    // Through the library, then through the binary.
    export(&fx.brain, &fx.out).unwrap();
    let out2 = folder.join("out2");
    let run = |out: &Path| {
        Command::new(env!("CARGO_BIN_EXE_axel"))
            .env_remove("AXEL_BRAIN")
            .arg("--brain").arg(&fx.brain)
            .arg("synaps-export").arg("--out").arg(out)
            .output().unwrap()
    };
    let done = run(&out2);
    assert_eq!(done.status.code(), Some(0), "{}", String::from_utf8_lossy(&done.stderr));
    let printed = format!("{}{}", String::from_utf8_lossy(&done.stdout), String::from_utf8_lossy(&done.stderr));
    assert!(printed.contains(K1) && printed.contains("archives 1"), "{printed}");
    assert!(!printed.contains("hello archive") && !printed.contains("private note"), "{printed}");

    let after = (sha_bytes(&fs::read(&fx.brain).unwrap()), fs::metadata(&fx.brain).unwrap().modified().unwrap());
    assert_eq!(before, after);
    assert!(!sidecar(&fx.brain, "-shm").exists());
    assert!(!sidecar(&fx.brain, "-wal").exists());
    assert!(!sidecar(&fx.brain, "-journal").exists());
    let mut expected = names.clone();
    expected.extend(["out".into(), "out2".into()]);
    expected.sort();
    assert_eq!(listing(&folder), expected);

    // 0600 files in 0700 directories.
    let mode = |p: &Path| fs::metadata(p).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode(&fx.out), 0o700);
    assert_eq!(mode(&fx.out.join(K1)), 0o700);
    for file in walk(&fx.out) {
        assert_eq!(mode(&file), 0o600, "{}", file.display());
    }

    // --out that is not empty: exit code 2, nothing added to it.
    let again = export(&fx.brain, &fx.out).unwrap_err();
    assert!(matches!(again, ExportError::Out(_)));
    assert_eq!(again.exit_code(), 2);

    // An empty -wal holds nothing: allowed.
    fs::write(sidecar(&fx.brain, "-wal"), b"").unwrap();
    export(&fx.brain, &folder.join("out3")).unwrap();

    // A non-empty one is refused before any query and before --out is made.
    for suffix in ["-wal", "-journal"] {
        fs::remove_file(sidecar(&fx.brain, "-wal")).ok();
        fs::write(sidecar(&fx.brain, suffix), b"not empty").unwrap();
        let out = folder.join(format!("refused{suffix}"));
        let error = export(&fx.brain, &out).unwrap_err();
        assert!(matches!(error, ExportError::Sidecar { .. }), "{error}");
        assert_eq!(error.exit_code(), 3);
        assert!(error.to_string().contains(&format!("brain.r8{suffix} is not empty")), "{error}");
        let refused = run(&out);
        assert_eq!(refused.status.code(), Some(3));
        assert!(String::from_utf8_lossy(&refused.stderr).contains("is not empty"));
        assert!(!out.exists());
        fs::remove_file(sidecar(&fx.brain, suffix)).unwrap();
    }
    assert_eq!(sha_bytes(&fs::read(&fx.brain).unwrap()), before.0);
    assert!(!sidecar(&fx.brain, "-shm").exists());
}

/// T-AX4: the manifest names every key, and a forget is not an orphan.
#[test]
fn t_ax4_manifest_names_every_key_and_a_forget_is_not_an_orphan() {
    let build = |with_history_fingerprint: bool| {
        let mut fx = Fixture::new();
        // K2 has only forum posts. K3 has only a foreign memory.
        for n in ["7", "8"] {
            let post = format!("msg-{}", n.repeat(64));
            fx.record(K2, &post, &forum_record(K2, &post), "post body");
        }
        fx.memory(Some(K3), "mem_abc12345", "", "made with axel remember");
        // One forgotten note and one forgotten forum post, as the helper
        // makes them: a `note` fingerprint, a tombstone with the id, no row.
        fx.fingerprint(K2, "note", &"e1".repeat(32));
        fx.tombstone(&format!("mem-{}", "5".repeat(32)), K2, "2026-11-10T12:00:00.123456789+00:00");
        fx.fingerprint(K2, "note", &"e2".repeat(32));
        fx.tombstone(&format!("msg-{}", "6".repeat(64)), K2, "2026-11-10T12:00:01+00:00");
        if with_history_fingerprint {
            fx.fingerprint(K2, "history", &"e3".repeat(32));
        }
        fx.close();
        let manifest = export(&fx.brain, &fx.out).unwrap();
        (fx, serde_json::to_value(&manifest).unwrap())
    };

    let (fx, manifest) = build(true);
    let empty = sha_bytes(b"");
    let tombstone = format!("{{\"tombstone\":\"mem-{}\",\"timestamp_ms\":1794312000123}}\n", "5".repeat(32));
    assert_eq!(fs::read_to_string(fx.out.join(K2).join("notes.jsonl")).unwrap(), tombstone);
    assert_eq!(
        manifest["projects"],
        json!({
            K2: {
                "archives": 0, "live": 0, "tombstones": 0, "messages": 0,
                "notes": 0, "note_tombstones": 1, "forum_posts": 2,
                "forum_tombstones": 1, "captures": 0, "foreign_memories": 0,
                "sha256": {"archives.jsonl": empty, "notes.jsonl": sha_bytes(tombstone.as_bytes())},
                "logical": [], "scope_group": null, "source_worktree_roots": [],
            },
            K3: {
                "archives": 0, "live": 0, "tombstones": 0, "messages": 0,
                "notes": 0, "note_tombstones": 0, "forum_posts": 0,
                "forum_tombstones": 0, "captures": 0, "foreign_memories": 1,
                "sha256": {"archives.jsonl": empty, "notes.jsonl": empty},
                "logical": [], "scope_group": null, "source_worktree_roots": [],
            },
        })
    );
    assert_eq!(manifest["forum_posts"], json!(2));
    // The forgotten forum post is in no file, and it is counted.
    assert_eq!(manifest["forum_tombstones"], json!(1));
    assert_eq!(manifest["captures"], json!(0));
    assert_eq!(manifest["fingerprints_by_kind"], json!({"history": 1, "note": 2, "capture": 0}));
    assert_eq!(manifest["orphan_fingerprints"], json!([{"project_key": K2, "value": "e3".repeat(32)}]));
    assert_eq!(manifest["foreign_memories"], json!({"total": 1, "list": [{"project_key": K3, "id": "mem_abc12345"}]}));
    assert_eq!(manifest["digest_mismatch"], json!([]));
    assert_eq!(manifest["check_failed"], json!([]));
    let written = all_output(&fx.out);
    assert!(!written.contains("post body") && !written.contains("made with axel remember"));

    // Without the `history` row nothing is an orphan; the two `note`
    // fingerprints are still counted and still stop nothing.
    let (_fx, manifest) = build(false);
    assert_eq!(manifest["orphan_fingerprints"], json!([]));
    assert_eq!(manifest["fingerprints_by_kind"], json!({"history": 0, "note": 2, "capture": 0}));
}

/// Not in the spec's list: "any project of the row's scope group". A forget
/// recorded under one member hides the archive and the note of another.
#[test]
fn scope_group_carries_a_forget_across_aliases() {
    let mut fx = Fixture::new();
    fx.conn().execute_batch(SCOPE_SCHEMA).unwrap();
    for (member, canonical) in [(K1, K1), (K2, K1), (K3, K3)] {
        fx.conn().execute("INSERT INTO synaps_scope_members VALUES(?1,?2)", params![member, canonical]).unwrap();
    }
    let logical = "ab".repeat(32);
    let digest = parts(&[logical.as_bytes(), &3u64.to_be_bytes(), MSG_A.as_bytes()]);
    // The same content in K1 (alias group with K2) and in K3 (alone).
    fx.history(K1, &"a".repeat(32), &logical, &digest, None, 3, 1, MSG_A, "", 1, false);
    fx.history(K3, &"c".repeat(32), &logical, &digest, None, 3, 1, MSG_A, "", 1, false);
    fx.fingerprint(K2, "history", &digest);
    let note = format!("mem-{}", "1".repeat(32));
    fx.record(K1, &note, &note_record(K1, &note, 10, "", "{}"), "forgotten through the alias");
    fx.tombstone(&note, K2, "1970-01-01T00:00:01+00:00");
    fx.close();

    let manifest = serde_json::to_value(export(&fx.brain, &fx.out).unwrap()).unwrap();
    let group = json!({"canonical": K1, "members": [K1, K2]});
    assert_eq!(manifest["projects"][K1]["tombstones"], json!(1));
    assert_eq!(manifest["projects"][K1]["live"], json!(0));
    assert_eq!(manifest["projects"][K1]["scope_group"], group);
    assert_eq!(manifest["projects"][K2]["scope_group"], group);
    assert_eq!(manifest["projects"][K2]["note_tombstones"], json!(1));
    assert_eq!(manifest["projects"][K3]["live"], json!(1));
    assert_eq!(manifest["projects"][K3]["scope_group"], Value::Null);
    assert_eq!(manifest["notes_suppressed"], json!(1));
    assert_eq!(manifest["orphan_fingerprints"], json!([]));
    assert!(!all_output(&fx.out).contains("forgotten through the alias"));
}

/// Review finding 1: one of two identical notes of one session is forgotten.
/// The forget leaves a tombstone and a `note` fingerprint; the twin row is
/// still in `memories`, and the helper's reads hide it. Neither's text may
/// reach a file.
#[test]
fn a_note_fingerprint_hides_the_twin_of_a_forgotten_note() {
    let mut fx = Fixture::new();
    let text = "said twice in one session";
    let forgotten = format!("mem-{}", "1".repeat(32));
    let twin = format!("mem-{}", "2".repeat(32));
    let other = format!("mem-{}", "3".repeat(32));
    let namespace = format!("project-{K1}");
    // Independent of the exporter: namespace, source, session, content.
    let fp = parts(&[namespace.as_bytes(), b"extension", b"s1", text.as_bytes()]);
    fx.tombstone(&forgotten, K1, "1970-01-01T00:00:05+00:00");
    fx.fingerprint(K1, "note", &fp);
    fx.record(K1, &twin, &session_note(K1, &twin, 10, "s1"), text);
    // Same session, other text: not hidden.
    fx.record(K1, &other, &session_note(K1, &other, 20, "s1"), "said once");
    fx.close();

    let manifest = serde_json::to_value(export(&fx.brain, &fx.out).unwrap()).unwrap();
    let written = all_output(&fx.out);
    assert!(!written.contains(text), "{written}");
    assert!(!written.contains(&twin), "{written}");
    assert!(written.contains("said once"));
    assert_eq!(manifest["notes_suppressed"], json!(1));
    assert_eq!(manifest["projects"][K1]["notes"], json!(1));
    assert_eq!(manifest["projects"][K1]["note_tombstones"], json!(1));
    assert_eq!(manifest["fingerprints_by_kind"]["note"], json!(1));
    assert_eq!(manifest["orphan_fingerprints"], json!([]));
    assert_eq!(manifest["check_failed"], json!([]));
}

/// The namespace is in the hash and the fingerprint belongs to a scope group:
/// a forum post's `note` fingerprint, a fingerprint of another group, and a
/// fingerprint of another session hide no project note.
#[test]
fn a_forum_note_fingerprint_does_not_hide_a_project_note() {
    let mut fx = Fixture::new();
    let text = "the same words as a forgotten forum post";
    let note = format!("mem-{}", "1".repeat(32));
    let namespace = format!("project-{K1}");
    fx.record(K1, &note, &session_note(K1, &note, 10, "s1"), text);
    fx.fingerprint(K1, "note", &parts(&[b"forum", b"extension", b"s1", text.as_bytes()]));
    fx.fingerprint(K1, "note", &parts(&[namespace.as_bytes(), b"extension", b"s2", text.as_bytes()]));
    // The exact fingerprint, but under a key of another scope group.
    fx.fingerprint(K3, "note", &parts(&[namespace.as_bytes(), b"extension", b"s1", text.as_bytes()]));
    fx.close();

    let manifest = serde_json::to_value(export(&fx.brain, &fx.out).unwrap()).unwrap();
    let notes = fs::read_to_string(fx.out.join(K1).join("notes.jsonl")).unwrap();
    assert_eq!(notes.lines().count(), 1);
    let line: Value = serde_json::from_str(notes.trim_end()).unwrap();
    assert_eq!(line["content"], json!(text));
    assert_eq!(line["id"], json!(note));
    assert_eq!(manifest["notes_suppressed"], json!(0));
    assert_eq!(manifest["projects"][K1]["notes"], json!(1));
    assert_eq!(manifest["fingerprints_by_kind"]["note"], json!(3));
    assert_eq!(manifest["check_failed"], json!([]));
}

/// Review finding 7: `--brain` must be on the command line. Without it, or
/// with only `AXEL_BRAIN`, the command exits 2 and creates nothing.
#[test]
fn the_binary_needs_brain_on_the_command_line() {
    let mut fx = Fixture::new();
    let logical = "ab".repeat(32);
    let digest = parts(&[logical.as_bytes(), &3u64.to_be_bytes(), MSG_A.as_bytes()]);
    fx.history(K1, &"a".repeat(32), &logical, &digest, None, 3, 1, MSG_A, "", 1, false);
    fx.close();
    let folder = fx.brain.parent().unwrap().to_owned();
    let mut listing: Vec<_> = fs::read_dir(&folder).unwrap().map(|e| e.unwrap().file_name()).collect();
    listing.sort();
    let axel = || {
        let mut command = Command::new(env!("CARGO_BIN_EXE_axel"));
        command.env_remove("AXEL_BRAIN");
        command
    };
    let refused = |done: std::process::Output| {
        assert_eq!(done.status.code(), Some(2));
        assert_eq!(String::from_utf8_lossy(&done.stderr).trim(), "synaps-export needs --brain <brain.r8>");
        assert!(done.stdout.is_empty());
        assert!(!fx.out.exists());
        let mut now: Vec<_> = fs::read_dir(&folder).unwrap().map(|e| e.unwrap().file_name()).collect();
        now.sort();
        assert_eq!(now, listing);
    };

    refused(axel().arg("synaps-export").arg("--out").arg(&fx.out).output().unwrap());
    // The variable names a brain that would export cleanly: still refused.
    refused(axel().env("AXEL_BRAIN", &fx.brain).arg("synaps-export").arg("--out").arg(&fx.out).output().unwrap());

    // The flag after the subcommand is the command line too, and it wins
    // over the variable.
    let done = axel()
        .env("AXEL_BRAIN", folder.join("not-there.r8"))
        .arg("synaps-export").arg("--brain").arg(&fx.brain).arg("--out").arg(&fx.out)
        .output().unwrap();
    assert_eq!(done.status.code(), Some(0), "{}", String::from_utf8_lossy(&done.stderr));
    assert!(fx.out.join("manifest.json").is_file());
}

/// Review finding 9: a key whose only row is a forgotten forum post is
/// listed, with the count.
#[test]
fn a_forum_tombstone_is_counted_for_its_key() {
    let mut fx = Fixture::new();
    fx.tombstone(&format!("msg-{}", "6".repeat(64)), K2, "2026-11-10T12:00:01+00:00");
    fx.tombstone(&format!("msg-{}", "7".repeat(64)), K2, "2026-11-10T12:00:02+00:00");
    fx.tombstone(&format!("msg-{}", "8".repeat(64)), K1, "not a time");
    fx.close();
    let manifest = serde_json::to_value(export(&fx.brain, &fx.out).unwrap()).unwrap();
    assert_eq!(manifest["forum_tombstones"], json!(3));
    assert_eq!(manifest["projects"][K2]["forum_tombstones"], json!(2));
    assert_eq!(manifest["projects"][K1]["forum_tombstones"], json!(1));
    assert_eq!(manifest["projects"][K2]["note_tombstones"], json!(0));
    assert_eq!(manifest["check_failed"], json!([]));
    assert_eq!(fs::read_to_string(fx.out.join(K2).join("notes.jsonl")).unwrap(), "");
}

/// Review finding 9: captures are counted, and a capture's note is not
/// written and is named as `capture_note`.
#[test]
fn captures_are_counted_and_their_notes_are_named() {
    let mut fx = Fixture::new();
    let capture = |key: &str, id: &str, note: &str, withheld: bool| {
        fx.conn().execute(
            "INSERT INTO synaps_captures(project_key,capture_id,note_id,payload_digest,source_digest,evidence,withheld,tombstoned) VALUES(?1,?2,?3,?4,?5,?6,?7,0)",
            params![key, id, note, "d1".repeat(32), "d2".repeat(32), "evidence text of a capture", withheld],
        ).unwrap();
    };
    let record = |key: &str, id: &str| {
        format!(
            r#"{{"namespace":"captures","timestamp_ms":5,"content":"","tags":[],"id":"{id}","project":"{key}","provenance":{{"source":"capture"}},"sensitivity":"normal","retention":"standard"}}"#
        )
    };
    let n1 = format!("mem-{}", "1".repeat(32));
    let n2 = format!("mem-{}", "2".repeat(32));
    let n3 = format!("mem-{}", "3".repeat(32));
    let n4 = format!("mem-{}", "4".repeat(32));
    capture(K1, "cap-1", &n1, false);
    fx.record(K1, &n1, &record(K1, &n1), "body of capture one");
    capture(K1, "cap-2", &n2, true);
    fx.record(K1, &n2, &record(K1, &n2), "body of the withheld capture");
    // A capture whose note row is gone: counted, nothing to name.
    capture(K2, "cap-3", &n3, false);
    // An ordinary note beside them is still exported.
    fx.record(K1, &n4, &note_record(K1, &n4, 10, "", "{}"), "an ordinary note");
    fx.close();

    let exported = export(&fx.brain, &fx.out).unwrap();
    assert!(!exported.clean());
    let manifest = serde_json::to_value(&exported).unwrap();
    assert_eq!(manifest["captures"], json!(3));
    assert_eq!(manifest["projects"][K1]["captures"], json!(2));
    assert_eq!(manifest["projects"][K2]["captures"], json!(1));
    assert_eq!(manifest["projects"][K1]["notes"], json!(1));
    assert_eq!(
        manifest["check_failed"],
        json!([
            {"project_key": K1, "id": n1, "check": "capture_note"},
            {"project_key": K1, "id": n2, "check": "capture_note"},
        ])
    );
    let written = all_output(&fx.out);
    for hidden in ["body of capture one", "body of the withheld capture", "evidence text of a capture"] {
        assert!(!written.contains(hidden), "{hidden}");
    }
    assert!(written.contains("an ordinary note"));
}

/// Review finding 9: a `history` fingerprint is an orphan unless a row of
/// its own scope group has the value. The same content under an unrelated
/// key does not count, and that key's archive stays live.
#[test]
fn an_orphan_is_judged_within_its_scope_group() {
    let mut fx = Fixture::new();
    fx.conn().execute_batch(SCOPE_SCHEMA).unwrap();
    for (member, canonical) in [(K1, K1), (K2, K1), (K3, K3)] {
        fx.conn().execute("INSERT INTO synaps_scope_members VALUES(?1,?2)", params![member, canonical]).unwrap();
    }
    let logical = "ab".repeat(32);
    let digest = parts(&[logical.as_bytes(), &3u64.to_be_bytes(), MSG_A.as_bytes()]);
    let fp = "fa".repeat(32);
    fx.history(K1, &"a".repeat(32), &logical, &digest, Some(&fp), 3, 1, MSG_A, "", 1, false);
    // K3 is alone: the digest and the fingerprint of K1's row are not its own.
    fx.fingerprint(K3, "history", &digest);
    fx.fingerprint(K3, "history", &fp);
    fx.close();

    let manifest = serde_json::to_value(export(&fx.brain, &fx.out).unwrap()).unwrap();
    // Sorted by value, as the manifest lists them.
    let mut orphans = [digest.clone(), fp.clone()];
    orphans.sort();
    assert_eq!(
        manifest["orphan_fingerprints"],
        json!([{"project_key": K3, "value": orphans[0]}, {"project_key": K3, "value": orphans[1]}])
    );
    assert_eq!(manifest["projects"][K1]["live"], json!(1));
    assert_eq!(manifest["projects"][K1]["tombstones"], json!(0));
}

/// Review finding 11: the source is opened and checked before `--out` is
/// created, so a refusal leaves no directory; and a path through a symbolic
/// link is refused with a message that says so.
#[test]
fn a_refused_source_leaves_no_out_directory() {
    let mut fx = Fixture::new();
    let logical = "ab".repeat(32);
    let digest = parts(&[logical.as_bytes(), &3u64.to_be_bytes(), MSG_A.as_bytes()]);
    fx.history(K1, &"a".repeat(32), &logical, &digest, None, 3, 1, MSG_A, "", 1, false);
    fx.close();
    let folder = fx.brain.parent().unwrap().to_owned();

    // A database that is not a Synaps brain; a file that is not a database;
    // a path that is not there.
    let plain = folder.join("plain.r8");
    Connection::open(&plain).unwrap().execute_batch("CREATE TABLE memories(id TEXT PRIMARY KEY);").unwrap();
    let garbage = folder.join("garbage.r8");
    fs::write(&garbage, vec![b'x'; 8192]).unwrap();
    for source in [plain, garbage, folder.join("missing.r8")] {
        let error = export(&source, &fx.out).unwrap_err();
        assert!(matches!(error, ExportError::Source(_)), "{error}");
        assert_eq!(error.exit_code(), 1);
        assert!(!fx.out.exists(), "{}", source.display());
    }

    // A link as the last component is refused before anything is opened.
    let real = folder.join("real");
    fs::create_dir(&real).unwrap();
    fs::copy(&fx.brain, real.join("brain.r8")).unwrap();
    let file_link = folder.join("linked.r8");
    std::os::unix::fs::symlink(&fx.brain, &file_link).unwrap();
    let error = export(&file_link, &fx.out).unwrap_err();
    assert!(error.to_string().contains("is not a regular file"), "{error}");
    assert!(!fx.out.exists());

    // Through the real path it exports.
    let by_real = export(&real.join("brain.r8"), &fx.out).unwrap();

    // The same brain reached through a linked directory (macOS `/var` is one) exports the
    // same, and a non-empty `-wal` beside the real file is still found through the link.
    let link = folder.join("link");
    std::os::unix::fs::symlink(&real, &link).unwrap();
    let out2 = folder.join("out-through-link");
    let by_link = export(&link.join("brain.r8"), &out2).unwrap();
    assert_eq!(by_link.source.path, link.join("brain.r8").to_string_lossy());
    assert_eq!((by_link.source.bytes, &by_link.source.sha256), (by_real.source.bytes, &by_real.source.sha256));
    assert_eq!(by_link.projects, by_real.projects);
    fs::write(real.join("brain.r8-wal"), b"x").unwrap();
    let out3 = folder.join("out-wal");
    let error = export(&link.join("brain.r8"), &out3).unwrap_err();
    assert_eq!(error.exit_code(), 3, "{error}");
    assert!(!out3.exists());
}
