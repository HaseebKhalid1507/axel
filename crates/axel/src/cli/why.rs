//! `why [--last <n>] [--session <id>] [--json]`: what a recall picked and why (section 3,
//! "Seeing why"). Reads `recall_log`; titles are read from the rows the picks name, never
//! from the log, so a purged row leaves no title behind. Everything that came from a row
//! passes `screen::show`.

use std::process::ExitCode;

use chrono::{TimeZone, Utc};
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{json, Value};

use crate::brain_v2::{self, Role};
use crate::error::Result;
use crate::ext::types::BrainArg;
use crate::screen;

#[derive(clap::Args)]
pub struct Args {
    /// The last N recalls
    #[arg(long, value_name = "N")]
    pub last: Option<usize>,
    /// Only the recalls of this chat
    #[arg(long, value_name = "ID")]
    pub session: Option<String>,
    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

const FIELD: usize = 200;
const TITLE: usize = 160;

// SPEC: 3281 does not give the default of `--last`: the last recall alone.
const DEFAULT_LAST: usize = 1;

pub fn run(args: &Args, brain: &BrainArg) -> Result<ExitCode> {
    // Section 1, "which subcommand opens how": `connect(Cli)`, reads only.
    let conn = brain_v2::connect(&brain.path, Role::Cli)?;
    // RULING: v2 real-binaries round 2, item 3: titles are read from rows.
    crate::cli::show::forget_pass(&conn)?;
    let recalls = read(&conn, args.last.unwrap_or(DEFAULT_LAST), args.session.as_deref())?;
    if recalls.is_empty() && !args.json {
        println!("no recall is logged");
    }
    for r in &recalls {
        if args.json {
            // SPEC: 3281 says only `--json`: one object per recall, one per line, oldest
            // first, so `why --last 1 --json` is one object with `outcome` and `total_ms`.
            println!("{r}");
        } else {
            for line in lines(r) {
                println!("{line}");
            }
        }
    }
    Ok(ExitCode::SUCCESS)
}

fn show(s: &str) -> String {
    screen::show(s, FIELD)
}

/// Every string in `v` through `screen::show`.
fn shown(v: Value) -> Value {
    match v {
        Value::String(s) => Value::String(show(&s)),
        Value::Array(a) => Value::Array(a.into_iter().map(shown).collect()),
        Value::Object(m) => Value::Object(m.into_iter().map(|(k, v)| (show(&k), shown(v))).collect()),
        other => other,
    }
}

/// The newest `last` rows (of one chat with `session`), oldest first, each with its
/// picks' titles read from the rows. Every string has passed `screen::show`.
pub fn read(conn: &Connection, last: usize, session: Option<&str>) -> Result<Vec<Value>> {
    let mut stmt = conn.prepare(
        "SELECT id, ts_ms, session_id, logical, query_preview, outcome, total_ms, embed_ms,
                candidates, selected, detail
           FROM recall_log WHERE ?1 IS NULL OR session_id = ?1
          ORDER BY id DESC LIMIT ?2",
    )?;
    let limit = i64::try_from(last).unwrap_or(i64::MAX);
    let rows = stmt.query_map(params![session, limit], |r| {
        Ok((
            r.get::<_, i64>(0)?,
            r.get::<_, i64>(1)?,
            r.get::<_, Option<String>>(2)?,
            r.get::<_, Option<String>>(3)?,
            r.get::<_, Option<String>>(4)?,
            r.get::<_, String>(5)?,
            r.get::<_, f64>(6)?,
            r.get::<_, Option<f64>>(7)?,
            r.get::<_, i64>(8)?,
            r.get::<_, i64>(9)?,
            r.get::<_, String>(10)?,
        ))
    })?;
    let mut out = Vec::new();
    for row in rows {
        let (id, ts_ms, session_id, logical, preview, outcome, total_ms, embed_ms, candidates, selected, detail) = row?;
        let detail: Value = serde_json::from_str::<Value>(&detail).ok().filter(Value::is_object).unwrap_or_else(|| json!({}));
        let mut picks = Vec::new();
        for pick in detail["picks"].as_array().into_iter().flatten() {
            let mut p = pick.clone();
            let (title, state) = title_of(conn, pick["ref"].as_str().unwrap_or(""))?;
            p["title"] = json!(title.map(|t| screen::show(&t, TITLE)));
            p["row"] = json!(state);
            picks.push(p);
        }
        out.push(shown(json!({
            "id": id,
            "ts_ms": ts_ms,
            "session_id": session_id,
            "logical": logical,
            "query_preview": preview,
            "outcome": outcome,
            "total_ms": total_ms,
            "embed_ms": embed_ms,
            "candidates": candidates,
            "selected": selected,
            "query_tokens": detail["query_tokens"],
            "embed": detail["embed"],
            "thresholds": detail["thresholds"],
            "generators": detail["generators"],
            "picks": picks,
            "dropped": detail["dropped"],
        })));
    }
    out.reverse();
    Ok(out)
}

/// The title of the row a pick names, and whether it is `visible`, `hidden` or `gone`.
fn title_of(conn: &Connection, r#ref: &str) -> Result<(Option<String>, &'static str)> {
    let found: Option<(String, bool)> = if let Some(id) = r#ref.strip_prefix("m:") {
        conn.query_row(
            "SELECT m.title, COALESCE(e.hidden_ms IS NOT NULL, 0) OR m.superseded_by IS NOT NULL
               FROM memories m LEFT JOIN memory_ext e ON e.mem_id = m.id WHERE m.id = ?1",
            [id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()?
    } else if let Some(doc_id) = r#ref.strip_prefix("doc:") {
        conn.query_row(
            "SELECT COALESCE(CASE WHEN json_valid(metadata) THEN json_extract(metadata, '$.title') END,
                             file_path, doc_id),
                    hidden IS NOT NULL
               FROM documents WHERE doc_id = ?1",
            [doc_id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()?
    } else {
        None
    };
    Ok(match found {
        Some((title, true)) => (Some(title), "hidden"),
        Some((title, false)) => (Some(title), "visible"),
        None => (None, "gone"),
    })
}

/// Who wrote a pick, from its `origin`.
fn who(origin: &str) -> String {
    match origin {
        "tool" => "saved by the assistant".to_string(),
        "note_file" => "Synaps note saved by the assistant".to_string(),
        "archive" => "earlier chat".to_string(),
        "capture" => "earlier chat, not yet archived".to_string(),
        "cli" => "saved with the axel command".to_string(),
        "legacy" => "saved before v2".to_string(),
        "document" => "file".to_string(),
        other => other.to_string(),
    }
}

/// At most two decimals, no trailing zeros: 0.67, 3.93, 0.
fn num(v: &Value) -> String {
    let x = v.as_f64().unwrap_or(0.0);
    let s = format!("{x:.2}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s.is_empty() || s == "-0" { "0".to_string() } else { s.to_string() }
}

fn date(ms: i64) -> String {
    Utc.timestamp_millis_opt(ms)
        .single()
        .map(|d| d.format("%Y-%m-%d %H:%M:%S UTC").to_string())
        .unwrap_or_else(|| ms.to_string())
}

fn text(v: &Value) -> String {
    v.as_str().map(str::to_string).unwrap_or_default()
}

// The weights of `retrieval.rs` (JR's ranker: words 2.0, meaning 2.5, same chat 1.0, kind
// 0.75, recent 0.5), as section 3 prints them.
const W_WORDS: &str = "2.0";
const W_MEANING: &str = "2.5";
const W_KIND: &str = "0.75";
const W_RECENT: &str = "0.5";

/// The text of one recall: the outcome and timings, then per pick
/// `n. [ref] who · date · total = words … · reasons` and its title, then the dropped items.
pub fn lines(r: &Value) -> Vec<String> {
    let mut l = Vec::new();
    let embed = match (r["embed"]["used"].as_bool(), r["embed_ms"].as_f64()) {
        (Some(true), Some(ms)) => format!("embedding {} ms", num(&json!(ms))),
        _ => match r["embed"]["fallback"].as_str() {
            Some(why) => format!("words only ({why})"),
            None => "words only".to_string(),
        },
    };
    l.push(format!(
        "recall {} · {} · {} · {} ms, {} · {} candidates, {} selected",
        r["id"],
        r["ts_ms"].as_i64().map(date).unwrap_or_default(),
        text(&r["outcome"]),
        num(&r["total_ms"]),
        embed,
        r["candidates"],
        r["selected"],
    ));
    if let Some(s) = r["session_id"].as_str() {
        l.push(format!("  chat {s}"));
    }
    if let Some(q) = r["query_preview"].as_str().filter(|q| !q.is_empty()) {
        l.push(format!("  prompt: {q}"));
    }
    let g = &r["generators"];
    if g.is_object() {
        l.push(format!(
            "  found: {} by words in memories, {} in documents, {} by meaning; {} past the floor, {} after dedupe",
            g["mem_words"], g["doc_words"], g["by_meaning"], g["past_floor"], g["after_dedupe"]
        ));
    }
    for p in r["picks"].as_array().into_iter().flatten() {
        let parts = &p["parts"];
        let reasons: Vec<String> = p["reasons"].as_array().into_iter().flatten().map(text).collect();
        l.push(format!(
            "  {}. [{}] {} · {} · total {} = words {}×{} + meaning {}×{} + same chat {} + kind {}×{} + recent {}×{} · {}",
            p["n"],
            text(&p["ref"]),
            who(&text(&p["origin"])),
            p["created"].as_str().unwrap_or("date unknown"),
            num(&parts["total"]),
            num(&parts["lexical"]),
            W_WORDS,
            num(&parts["semantic"]),
            W_MEANING,
            num(&parts["session"]),
            num(&parts["class"]),
            W_KIND,
            num(&parts["recency"]),
            W_RECENT,
            reasons.join(", "),
        ));
        match (p["title"].as_str(), p["row"].as_str()) {
            (Some(t), Some("hidden")) => l.push(format!("     {t} (hidden now)")),
            (Some(t), _) => l.push(format!("     {t}")),
            (None, _) => l.push("     (no longer in the brain)".to_string()),
        }
    }
    for d in r["dropped"].as_array().into_iter().flatten() {
        l.push(format!("  dropped [{}] {}", text(&d["ref"]), text(&d["why"])));
    }
    l
}
