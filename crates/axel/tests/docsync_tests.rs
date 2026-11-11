//! Commit 22, the owner's note folders (section 2.8): section 2's test 22, and the rules the
//! task names: no row is deleted because a file or a folder is missing or unreadable, and no
//! file outside the named folders is read. Temp folders, temp brain, the hash embedder.

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use axel::brain_v2::{self, OpenOpts};
use axel::docsync::{self, SyncReport};
use axel::embed::EmbedService;
use axel::ext::config::ExtConfig;
use axel::ext::types::{Boot, IndexerCtx, Job, Mode, Status, TestHooks};
use axel::indexer_start;
use axel::vindex::{VecDelta, VecKey, VecScope};
use rusqlite::Connection;
use serde_json::json;
use tempfile::TempDir;
use velocirag::embedder::Embedder;

const BODY: &str = "A note of the owner, long enough to be indexed by the folder sync of Axel.";

/// (id, content, hidden, source_mtime_ms, has embedding).
type Row = (i64, String, Option<String>, Option<i64>, bool);

struct World {
    tmp: TempDir,
    ctx: IndexerCtx,
    vec_rx: Receiver<VecDelta>,
    _jobs_rx: Receiver<Job>,
}

fn now_ms() -> i64 {
    SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis() as i64
}

/// A brain recorded for `brain_model`, an indexer context on it, and `<tmp>/notes`.
fn world_with(embedder: Option<Embedder>, brain_model: &str, test: TestHooks) -> World {
    let tmp = TempDir::new().unwrap();
    for dir in ["base", "model", "brain", "notes"] {
        std::fs::create_dir_all(tmp.path().join(dir)).unwrap();
    }
    let brain = tmp.path().join("brain").join("brain.r8");
    let opts = OpenOpts { agent: None, model: brain_model, dim: 384, upgrade_v1: false };
    drop(brain_v2::open_v2(&brain, &opts).unwrap());
    let config = json!({"config": {
        "brain": brain.to_string_lossy(),
        "synaps_base": tmp.path().join("base").to_string_lossy(),
        "model_dir": tmp.path().join("model").to_string_lossy(),
        "embedder": "hash",
    }});
    let cfg = ExtConfig::from_initialize(&config, tmp.path());
    let status = Arc::new(Status::default());
    let boot = Boot { embedder, clock: Arc::new(now_ms), test: Arc::new(test), status };
    let (mut ctx, jobs_rx, vec_rx) = indexer_start::open_ctx(cfg, boot);
    assert_eq!(ctx.mode.load(Ordering::SeqCst), Mode::Ready as u8);
    ctx.cfg.sources = Some(tmp.path().join("sources.toml"));
    let w = World { tmp, ctx, vec_rx, _jobs_rx: jobs_rx };
    w.sources(&[("notes", "notes")]);
    w
}

fn world() -> World {
    world_with(Some(Embedder::hashing()), "hash-384", TestHooks::default())
}

impl World {
    fn path(&self, rel: &str) -> PathBuf {
        self.tmp.path().join(rel)
    }
    /// `sources.toml` with these (name, folder under tmp).
    fn sources(&self, list: &[(&str, &str)]) {
        let mut toml = String::new();
        for (name, rel) in list {
            toml.push_str(&format!(
                "[[source]]\nname = \"{name}\"\npath = \"{}\"\npriority = \"high\"\n\n",
                self.path(rel).display()
            ));
        }
        std::fs::write(self.path("sources.toml"), toml).unwrap();
    }
    fn file(&self, rel: &str, body: &str) -> PathBuf {
        let p = self.path(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, body).unwrap();
        p
    }
    fn sync(&mut self) -> SyncReport {
        docsync::sync(&mut self.ctx).unwrap()
    }
    fn conn(&self) -> &Connection {
        self.ctx.conn.as_ref().unwrap()
    }
    fn deltas(&self) -> Vec<VecDelta> {
        self.vec_rx.try_iter().collect()
    }
    fn count(&self) -> i64 {
        self.conn().query_row("SELECT COUNT(*) FROM documents", [], |r| r.get(0)).unwrap()
    }
    /// (id, content, hidden, source_mtime_ms, has embedding) of the row for `doc_id`.
    fn row(&self, doc_id: &str) -> Option<Row> {
        self.conn()
            .query_row(
                "SELECT id, content, hidden, source_mtime_ms, embedding IS NOT NULL FROM documents WHERE doc_id = ?1",
                [doc_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
            )
            .ok()
    }
    /// From here on, an `embed_wait` call ends the embed thread (`embed_called` says so).
    fn trap_embed(&mut self) {
        let test = Arc::new(TestHooks { panic_embed: true, ..TestHooks::default() });
        self.ctx.embed = EmbedService::start(Some(Embedder::hashing()), self.ctx.jobs.clone(), test);
    }
    fn embed_called(&self) -> bool {
        std::thread::sleep(Duration::from_millis(300));
        self.ctx.embed.stopped()
    }
    fn doc_ids(&self) -> Vec<String> {
        let mut stmt = self.conn().prepare("SELECT doc_id FROM documents ORDER BY doc_id").unwrap();
        stmt.query_map([], |r| r.get(0)).unwrap().map(|r| r.unwrap()).collect()
    }
}

fn canon(p: &Path) -> String {
    std::fs::canonicalize(p).unwrap().to_string_lossy().into_owned()
}

/// Sets a file's modification time `secs` seconds ahead of now.
fn touch_ahead(p: &Path, secs: u64) {
    let f = std::fs::File::options().write(true).open(p).unwrap();
    f.set_modified(SystemTime::now() + Duration::from_secs(secs)).unwrap();
}

fn upserted(deltas: &[VecDelta]) -> Vec<i64> {
    let mut ids = Vec::new();
    for d in deltas {
        if let VecDelta::Upsert(items) = d {
            for item in items {
                assert_eq!(item.scope, VecScope::Documents);
                assert_eq!(item.v.len(), 384);
                if let VecKey::Doc(id) = item.key {
                    ids.push(id);
                }
            }
        }
    }
    ids
}

fn removed(deltas: &[VecDelta]) -> Vec<i64> {
    let mut ids = Vec::new();
    for d in deltas {
        if let VecDelta::Remove(keys) = d {
            for key in keys {
                if let VecKey::Doc(id) = key {
                    ids.push(*id);
                }
            }
        }
    }
    ids
}

// ── Test 22: upsert_document ────────────────────────────────────────────────

#[test]
fn t22_upsert_document_on_an_existing_doc_id_keeps_the_row_and_its_marks() {
    let w = world();
    let conn = w.conn();
    let v1 = vec![0.5f32; 384];
    let v2 = vec![0.25f32; 384];
    let id = velocirag::db::upsert_document(conn, "notes::a", "first text of a", &json!({}), &v1, Some("/x/a.md")).unwrap();
    conn.execute(
        "UPDATE documents SET hidden = 'missing_source', access_count = 7, excitability = 0.9,
                source_mtime_ms = 123, last_accessed = '2020-01-01 00:00:00' WHERE id = ?1",
        [id],
    )
    .unwrap();

    let again = velocirag::db::upsert_document(conn, "notes::a", "second text of a", &json!({"k": 1}), &v2, Some("/y/a.md")).unwrap();
    assert_eq!(again, id, "the row id is kept");
    fn col<T: rusqlite::types::FromSql>(conn: &Connection, id: i64, name: &str) -> T {
        conn.query_row(&format!("SELECT {name} FROM documents WHERE id = ?1"), [id], |r| r.get(0)).unwrap()
    }
    let content: String = col(conn, id, "content");
    let meta: String = col(conn, id, "metadata");
    let path: String = col(conn, id, "file_path");
    let hidden: Option<String> = col(conn, id, "hidden");
    let access: i64 = col(conn, id, "access_count");
    let exc: f64 = col(conn, id, "excitability");
    let mtime: Option<i64> = col(conn, id, "source_mtime_ms");
    let last: Option<String> = col(conn, id, "last_accessed");
    let blob: Vec<u8> = col(conn, id, "embedding");
    assert_eq!(content, "second text of a");
    assert_eq!(meta, r#"{"k":1}"#);
    assert_eq!(path, "/y/a.md");
    assert_eq!(hidden.as_deref(), Some("missing_source"));
    assert_eq!(access, 7);
    assert!((exc - 0.9).abs() < 1e-9);
    assert_eq!(mtime, Some(123));
    assert_eq!(last.as_deref(), Some("2020-01-01 00:00:00"));
    assert_eq!(&blob[..4], &0.25f32.to_le_bytes());
    assert_eq!(w.count(), 1);

    // The word index has one entry for the doc_id, with the new text.
    let fts: Vec<String> = conn
        .prepare("SELECT content FROM chunks_fts WHERE doc_id = 'notes::a'")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .map(|r| r.unwrap())
        .collect();
    assert_eq!(fts, vec!["second text of a".to_string()]);

    assert!(velocirag::db::upsert_document(conn, "bad", "x", &json!({}), &[0.0; 3], None).is_err());
}

/// `Database::insert_document` is `upsert_document` (spec 2642, wave 4 ruling R-22.4): the
/// command line's re-index of a changed file works and keeps the row id; the vector index
/// takes the key again.
#[test]
fn t22_the_command_line_still_reindexes_a_changed_document() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("cli.r8");
    axel::r8::Brain::create_with_model(&path, Some("test"), velocirag::embedder::HASH_MODEL_ID, 384).unwrap();
    let mut search = axel::search::BrainSearch::open_with_embedder(&path, Embedder::hashing()).unwrap();
    search.index_document("notes::a", "the first text of the note", None, Some("/n/a.md")).unwrap();
    let first = search.db().get_document("notes::a").unwrap().unwrap();
    search.index_document("notes::a", "the second text of the note", None, Some("/n/a.md")).unwrap();
    let doc = search.db().get_document("notes::a").unwrap().unwrap();
    assert_eq!(doc.content, "the second text of the note");
    assert_eq!(doc.id, first.id, "the row id is kept");
    let hits = search.search("the second text of the note", 5).unwrap();
    assert_eq!(hits.results.iter().filter(|r| r.doc_id == "notes::a").count(), 1);
}

// ── Test 22: the sync ───────────────────────────────────────────────────────

#[test]
fn t22_a_new_file_gets_its_row_and_its_vector_in_one_tick() {
    let mut w = world();
    let p = w.file("notes/new.md", BODY);
    let report = w.sync();
    assert_eq!(report.indexed, 1);
    assert_eq!(report.waiting, 0);
    let (id, content, hidden, mtime, has_vec) = w.row("notes::new").expect("a row");
    assert_eq!(content, BODY, "stored whole, as the file is");
    assert_eq!(hidden, None);
    assert!(has_vec);
    let file_ms = std::fs::metadata(&p)
        .unwrap()
        .modified()
        .unwrap()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64;
    assert_eq!(mtime, Some(file_ms));
    let path: String = w.conn().query_row("SELECT file_path FROM documents WHERE id = ?1", [id], |r| r.get(0)).unwrap();
    assert_eq!(path, canon(&p));
    assert_eq!(upserted(&w.deltas()), vec![id]);

    // Unchanged: nothing more.
    let report = w.sync();
    assert_eq!(report, SyncReport::default());
    assert!(w.deltas().is_empty());
}

#[test]
fn t22_with_no_embedder_a_new_file_gets_no_row_and_waits() {
    let mut w = world_with(None, "hash-384", TestHooks::default());
    w.file("notes/new.md", BODY);
    let report = w.sync();
    assert_eq!(report.indexed, 0);
    assert_eq!(report.waiting, 1);
    assert_eq!(w.count(), 0);
    assert!(w.deltas().is_empty());

    // The scheduled tick puts that count where `status` reads it, and takes it back to 0
    // when nothing waits any more.
    assert_eq!(w.ctx.status.docs_waiting.load(Ordering::Relaxed), 0);
    docsync::tick(&mut w.ctx).unwrap();
    assert_eq!(w.ctx.status.docs_waiting.load(Ordering::Relaxed), 1);
    std::fs::remove_file(w.path("notes/new.md")).unwrap();
    docsync::tick(&mut w.ctx).unwrap();
    assert_eq!(w.ctx.status.docs_waiting.load(Ordering::Relaxed), 0);
}

#[test]
fn t22_with_model_loaded_false_embed_wait_is_not_called() {
    // The embedder runs but the brain was made for another model. Every batch would sleep
    // 3 s: a sync that called `embed_wait` would take that long.
    let test = TestHooks { embed_sleep_ms: 3_000, ..TestHooks::default() };
    let mut w = world_with(Some(Embedder::hashing()), "another-model", test);
    assert!(!w.ctx.status.model_loaded.load(Ordering::Relaxed));
    w.file("notes/new.md", BODY);
    let start = Instant::now();
    let report = w.sync();
    assert!(start.elapsed() < Duration::from_millis(1_500), "{:?}", start.elapsed());
    assert_eq!(report.waiting, 1);
    assert_eq!(w.count(), 0);
}

#[test]
fn t22_a_nested_second_source_gives_each_file_one_row_and_no_link_is_followed() {
    let mut w = world();
    w.sources(&[("outer", "notes"), ("inner", "notes/inner")]);
    let a = w.file("notes/a.md", BODY);
    let b = w.file("notes/inner/b.md", BODY);
    let c = w.file("notes/inner/deep/c.txt", BODY);
    w.file("notes/short.md", "too short");
    w.file("notes/paper.pdf", BODY);
    let secret = w.file("outside/secret.md", "SECRET-OUTSIDE the named folders, long enough to be indexed.");
    w.file("outside/dir/other.md", "SECRET-OUTSIDE in a folder a link points at, long enough.");
    std::os::unix::fs::symlink(&secret, w.path("notes/link.md")).unwrap();
    std::os::unix::fs::symlink(w.path("outside/dir"), w.path("notes/linkdir")).unwrap();

    let report = w.sync();
    assert_eq!(report.indexed, 3);
    assert_eq!(w.doc_ids(), vec!["inner::b", "inner::deep::c", "outer::a"]);
    for p in [&a, &b, &c] {
        let n: i64 = w
            .conn()
            .query_row("SELECT COUNT(*) FROM documents WHERE file_path = ?1", [canon(p)], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 1, "{}", p.display());
    }
    let leaked: i64 = w
        .conn()
        .query_row("SELECT COUNT(*) FROM documents WHERE content LIKE '%SECRET-OUTSIDE%'", [], |r| r.get(0))
        .unwrap();
    assert_eq!(leaked, 0, "a link out of a folder is not followed");
    assert_eq!(upserted(&w.deltas()).len(), 3);
    assert_eq!(w.sync(), SyncReport::default());
}

#[test]
fn t22_a_deleted_file_is_hidden_and_kept_and_shown_again_when_it_returns() {
    let mut w = world();
    let a = w.file("notes/a.md", BODY);
    w.file("notes/b.md", BODY);
    w.sync();
    let (id_a, ..) = w.row("notes::a").unwrap();
    let (id_b, ..) = w.row("notes::b").unwrap();
    w.deltas();

    let a_time = std::fs::metadata(&a).unwrap().modified().unwrap();
    std::fs::remove_file(&a).unwrap();
    let report = w.sync();
    assert_eq!(report.hidden, 1);
    assert_eq!(w.count(), 2, "the row is kept");
    let (id, content, hidden, ..) = w.row("notes::a").unwrap();
    assert_eq!((id, content.as_str(), hidden.as_deref()), (id_a, BODY, Some("missing_source")));
    assert_eq!(w.row("notes::b").unwrap().2, None);
    assert_eq!(removed(&w.deltas()), vec![id_a]);
    // Again: no second hide.
    assert_eq!(w.sync().hidden, 0);

    // The file comes back unchanged: the mark is cleared and the stored vector goes back.
    w.file("notes/a.md", BODY);
    std::fs::File::options().write(true).open(&a).unwrap().set_modified(a_time).unwrap();
    let report = w.sync();
    assert_eq!((report.restored, report.indexed), (1, 0));
    assert_eq!(w.row("notes::a").unwrap().2, None);
    assert_eq!(w.row("notes::a").unwrap().0, id_a);
    assert_eq!(upserted(&w.deltas()), vec![id_a]);

    // A changed file is updated in place: same id, new text, new time.
    let b = w.path("notes/b.md");
    std::fs::write(&b, format!("{BODY} And a second sentence.")).unwrap();
    touch_ahead(&b, 5);
    let report = w.sync();
    assert_eq!(report.indexed, 1);
    let (id, content, _, mtime, _) = w.row("notes::b").unwrap();
    assert_eq!(id, id_b);
    assert!(content.ends_with("And a second sentence."));
    assert!(mtime.unwrap() > now_ms());
    assert_eq!(w.count(), 2);

    // Deleted and then changed while away: the return writes the new text and clears the mark.
    std::fs::remove_file(&b).unwrap();
    w.sync();
    assert_eq!(w.row("notes::b").unwrap().2.as_deref(), Some("missing_source"));
    std::fs::write(&b, format!("{BODY} A third version.")).unwrap();
    touch_ahead(&b, 60);
    w.sync();
    let (id, content, hidden, ..) = w.row("notes::b").unwrap();
    assert_eq!((id, hidden), (id_b, None));
    assert!(content.ends_with("A third version."));
}

#[test]
fn t22_a_missing_or_empty_folder_loses_nothing() {
    let mut w = world();
    w.file("notes/a.md", BODY);
    w.file("notes/sub/b.md", BODY);
    w.sync();
    assert_eq!(w.count(), 2);

    // The mount point itself is gone: nothing is hidden, nothing deleted.
    std::fs::rename(w.path("notes"), w.path("away")).unwrap();
    let report = w.sync();
    assert_eq!(report, SyncReport::default());
    let hidden: i64 = w.conn().query_row("SELECT COUNT(*) FROM documents WHERE hidden IS NOT NULL", [], |r| r.get(0)).unwrap();
    assert_eq!(hidden, 0);

    // The mount point is there and empty: the rows are hidden, never deleted.
    std::fs::create_dir(w.path("notes")).unwrap();
    assert_eq!(w.sync().hidden, 2);
    assert_eq!(w.count(), 2);

    // Mounted again: all of it is back.
    std::fs::remove_dir(w.path("notes")).unwrap();
    std::fs::rename(w.path("away"), w.path("notes")).unwrap();
    assert_eq!(w.sync().restored, 2);
    let hidden: i64 = w.conn().query_row("SELECT COUNT(*) FROM documents WHERE hidden IS NOT NULL", [], |r| r.get(0)).unwrap();
    assert_eq!(hidden, 0);
}

#[test]
fn t22_an_unreadable_file_or_folder_keeps_its_row() {
    use std::os::unix::fs::PermissionsExt;
    let mut w = world();
    let a = w.file("notes/a.md", BODY);
    w.file("notes/locked/b.md", BODY);
    w.sync();
    assert_eq!(w.count(), 2);

    // A changed file that cannot be opened: the row stays as it was, visible.
    std::fs::write(&a, format!("{BODY} changed")).unwrap();
    touch_ahead(&a, 5);
    std::fs::set_permissions(&a, std::fs::Permissions::from_mode(0o000)).unwrap();
    // A folder that cannot be listed or entered.
    std::fs::set_permissions(w.path("notes/locked"), std::fs::Permissions::from_mode(0o000)).unwrap();
    let as_root = std::fs::read(&a).is_ok();
    let report = w.sync();
    std::fs::set_permissions(w.path("notes/locked"), std::fs::Permissions::from_mode(0o755)).unwrap();
    std::fs::set_permissions(&a, std::fs::Permissions::from_mode(0o644)).unwrap();
    if as_root {
        eprintln!("running as root: permissions do not stop a read; skipped");
        return;
    }
    assert_eq!(report.hidden, 0);
    assert_eq!(report.indexed, 0);
    assert_eq!(report.skipped, 1);
    assert_eq!(w.count(), 2);
    assert_eq!(w.row("notes::a").unwrap().1, BODY);
    assert_eq!(w.row("notes::a").unwrap().2, None);
    assert_eq!(w.row("notes::locked::b").unwrap().2, None);

    // Readable again: the change arrives.
    assert_eq!(w.sync().indexed, 1);
    assert!(w.row("notes::a").unwrap().1.ends_with("changed"));
}

#[test]
fn t22_the_sync_stops_when_the_mode_is_not_ready() {
    let mut w = world();
    let a = w.file("notes/a.md", BODY);
    w.sync();
    std::fs::remove_file(&a).unwrap();
    // Another text: the same one would take over the row of `a` (R-22.2).
    w.file("notes/b.md", &format!("{BODY} Another text."));

    w.ctx.mode.store(Mode::Starting as u8, Ordering::SeqCst);
    assert_eq!(w.sync(), SyncReport::default());
    w.ctx.mode.store(Mode::Degraded as u8, Ordering::SeqCst);
    assert_eq!(w.sync(), SyncReport::default());
    assert_eq!(w.row("notes::a").unwrap().2, None);
    assert!(w.row("notes::b").is_none());

    w.ctx.mode.store(Mode::Ready as u8, Ordering::SeqCst);
    let report = w.sync();
    assert_eq!((report.hidden, report.indexed), (1, 1));
}

#[test]
fn t22_at_most_eight_files_are_embedded_per_tick() {
    let mut w = world();
    for i in 0..10 {
        w.file(&format!("notes/n{i:02}.md"), BODY);
    }
    let report = w.sync();
    assert_eq!((report.indexed, report.waiting), (8, 2));
    assert_eq!(w.count(), 8);
    let report = w.sync();
    assert_eq!((report.indexed, report.waiting), (2, 0));
    assert_eq!(w.count(), 10);
}

#[test]
fn t22_rows_the_sync_does_not_own_are_left_alone() {
    let mut w = world();
    let a = w.file("notes/a.md", BODY);
    w.sync();
    // Hidden by someone else: neither restored, rewritten nor re-marked.
    w.conn().execute("UPDATE documents SET hidden = 'memory_hidden' WHERE doc_id = 'notes::a'", []).unwrap();
    std::fs::write(&a, format!("{BODY} changed")).unwrap();
    touch_ahead(&a, 5);
    assert_eq!(w.sync(), SyncReport::default());
    assert_eq!(w.row("notes::a").unwrap().1, BODY);
    std::fs::remove_file(&a).unwrap();
    assert_eq!(w.sync(), SyncReport::default());
    assert_eq!(w.row("notes::a").unwrap().2.as_deref(), Some("memory_hidden"));

    // A row with no file, or under no named folder, is not touched.
    let v = vec![0.1f32; 384];
    velocirag::db::upsert_document(w.conn(), "mem-1", "a memory shadow", &json!({"type": "memory"}), &v, None).unwrap();
    velocirag::db::upsert_document(w.conn(), "x::gone", "elsewhere", &json!({}), &v, Some("/nonexistent/x/gone.md")).unwrap();
    assert_eq!(w.sync(), SyncReport::default());
    let hidden: i64 = w
        .conn()
        .query_row("SELECT COUNT(*) FROM documents WHERE hidden IS NOT NULL AND doc_id <> 'notes::a'", [], |r| r.get(0))
        .unwrap();
    assert_eq!(hidden, 0);
}

#[test]
fn t22_the_tick_does_nothing_without_a_brain() {
    let mut w = world();
    w.file("notes/a.md", BODY);
    w.ctx.conn = None;
    assert!(docsync::tick(&mut w.ctx).is_ok());
    assert_eq!(docsync::sync(&mut w.ctx).unwrap(), SyncReport::default());
}

// ── Wave 4 rulings R-22 ─────────────────────────────────────────────────────

/// R-22.1: a file over 2 MiB is never opened and counts as skipped, every tick; a row it
/// had keeps its text and stays visible.
#[test]
fn r22_1_a_file_over_two_mib_is_skipped_and_keeps_its_row() {
    let big = format!("{BODY}\n{}", "x".repeat(2 * 1024 * 1024));
    let mut w = world();
    let a = w.file("notes/a.md", BODY);
    w.file("notes/huge.txt", &big);
    let report = w.sync();
    assert_eq!((report.indexed, report.skipped, report.waiting), (1, 1, 0));
    assert!(w.row("notes::huge").is_none());

    // A noted file that grows past the bound.
    std::fs::write(&a, &big).unwrap();
    touch_ahead(&a, 5);
    let report = w.sync();
    assert_eq!((report.indexed, report.skipped, report.hidden), (0, 2, 0));
    let (_, content, hidden, ..) = w.row("notes::a").unwrap();
    assert_eq!((content.as_str(), hidden), (BODY, None));

    // With no model it is still skipped, not left waiting for one.
    let mut w = world_with(None, "hash-384", TestHooks::default());
    w.file("notes/huge.md", &big);
    let report = w.sync();
    assert_eq!((report.skipped, report.waiting), (1, 0));
}

/// R-22.2: a file whose time changed and whose text did not gets its time only: no
/// `embed_wait`, no slot, no vector sent.
#[test]
fn r22_2_a_touched_file_is_not_embedded_again() {
    let mut w = world();
    let a = w.file("notes/a.md", BODY);
    w.sync();
    let (id, ..) = w.row("notes::a").unwrap();
    w.deltas();
    w.trap_embed();
    touch_ahead(&a, 5);
    let report = w.sync();
    assert_eq!((report.retimed, report.indexed, report.waiting), (1, 0, 0));
    let (id2, content, hidden, mtime, _) = w.row("notes::a").unwrap();
    assert_eq!((id2, content.as_str(), hidden), (id, BODY, None));
    assert!(mtime.unwrap() > now_ms());
    assert!(w.deltas().is_empty());
    assert_eq!(w.sync(), SyncReport::default());
    assert!(!w.embed_called(), "embed_wait was called");
}

/// R-22.2: a renamed folder: one tick, every row visible again under its new path with its
/// old row id, its stored vector sent, no `embed_wait`.
#[test]
fn r22_2_a_renamed_folder_keeps_its_rows() {
    let mut w = world();
    let texts = ["first", "second", "third"];
    for (i, t) in texts.iter().enumerate() {
        w.file(&format!("notes/old/n{i}.md"), &format!("{BODY} The {t} note."));
    }
    assert_eq!(w.sync().indexed, 3);
    let ids: Vec<i64> = (0..3).map(|i| w.row(&format!("notes::old::n{i}")).unwrap().0).collect();
    w.deltas();
    w.trap_embed();

    std::fs::rename(w.path("notes/old"), w.path("notes/new")).unwrap();
    let report = w.sync();
    assert_eq!((report.hidden, report.moved, report.indexed, report.waiting), (3, 3, 0, 0));
    assert_eq!(w.count(), 3);
    for (i, t) in texts.iter().enumerate() {
        let (id, content, hidden, mtime, _) = w.row(&format!("notes::new::n{i}")).expect("the row moved");
        assert_eq!((id, hidden), (ids[i], None));
        assert!(content.ends_with(&format!("The {t} note.")));
        let p = w.path(&format!("notes/new/n{i}.md"));
        let path: String = w.conn().query_row("SELECT file_path FROM documents WHERE id = ?1", [id], |r| r.get(0)).unwrap();
        assert_eq!(path, canon(&p));
        let file_ms = std::fs::metadata(&p).unwrap().modified().unwrap().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis() as i64;
        assert_eq!(mtime, Some(file_ms));
        let meta: String = w.conn().query_row("SELECT metadata FROM documents WHERE id = ?1", [id], |r| r.get(0)).unwrap();
        assert!(!meta.contains("missing_since"), "{meta}");
        let fts: String = w
            .conn()
            .query_row("SELECT doc_id FROM chunks_fts WHERE content = ?1", [&content], |r| r.get(0))
            .unwrap();
        assert_eq!(fts, format!("notes::new::n{i}"));
    }
    let deltas = w.deltas();
    let mut sent = upserted(&deltas);
    sent.sort();
    assert_eq!(sent, ids);
    assert_eq!(w.sync(), SyncReport::default());
    assert!(!w.embed_called(), "embed_wait was called");
}

/// R-22.2: of two hidden rows with the text, the one hidden last is taken; a row hidden for
/// another reason is never taken.
#[test]
fn r22_2_the_row_hidden_last_is_taken_and_no_other_mark() {
    let mut w = world();
    let b = w.file("notes/b.md", BODY);
    w.sync();
    let a = w.file("notes/a.md", BODY);
    w.sync();
    let (id_a, ..) = w.row("notes::a").unwrap();
    let (id_b, ..) = w.row("notes::b").unwrap();
    assert!(id_a > id_b);
    std::fs::remove_file(&a).unwrap();
    assert_eq!(w.sync().hidden, 1);
    std::thread::sleep(Duration::from_millis(20));
    std::fs::remove_file(&b).unwrap();
    assert_eq!(w.sync().hidden, 1);

    w.file("notes/c.md", BODY);
    assert_eq!(w.sync().moved, 1);
    assert_eq!(w.row("notes::c").unwrap().0, id_b, "the row hidden last");
    assert_eq!(w.row("notes::a").unwrap().2.as_deref(), Some("missing_source"));

    // Hidden by someone else: not taken; the new file gets its own row.
    w.conn().execute("UPDATE documents SET hidden = 'memory_hidden' WHERE id = ?1", [id_a]).unwrap();
    w.file("notes/d.md", BODY);
    let report = w.sync();
    assert_eq!((report.moved, report.indexed), (0, 1));
    assert_ne!(w.row("notes::d").unwrap().0, id_a);
    assert_eq!(w.row("notes::a").unwrap().2.as_deref(), Some("memory_hidden"));
}

/// R-22.3: a restored row's stored vector is sent even when the model is not loaded.
#[test]
fn r22_3_a_restored_row_gets_its_vector_without_a_model() {
    let mut w = world();
    let a = w.file("notes/a.md", BODY);
    w.sync();
    let (id, ..) = w.row("notes::a").unwrap();
    let time = std::fs::metadata(&a).unwrap().modified().unwrap();
    std::fs::remove_file(&a).unwrap();
    w.sync();
    w.deltas();
    w.ctx.status.model_loaded.store(false, Ordering::Relaxed);
    w.file("notes/a.md", BODY);
    std::fs::File::options().write(true).open(&a).unwrap().set_modified(time).unwrap();
    assert_eq!(w.sync().restored, 1);
    assert_eq!(upserted(&w.deltas()), vec![id]);
}

/// R-22.3: a file replaced by an older version (a backup) is seen.
#[test]
fn r22_3_a_file_replaced_by_an_older_one_is_indexed() {
    let mut w = world();
    let a = w.file("notes/a.md", &format!("{BODY} The newer text."));
    w.sync();
    std::fs::write(&a, format!("{BODY} The older text, from a backup.")).unwrap();
    let f = std::fs::File::options().write(true).open(&a).unwrap();
    f.set_modified(SystemTime::now() - Duration::from_secs(86_400)).unwrap();
    drop(f);
    assert_eq!(w.sync().indexed, 1);
    assert!(w.row("notes::a").unwrap().1.ends_with("from a backup."));
}
