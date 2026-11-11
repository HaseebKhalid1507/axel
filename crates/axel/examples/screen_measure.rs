//! `screen_measure [--names] <path>...`
//!
//! How much does `screen::redact` blank on real text, and what does the blanked text look
//! like? (Wave 2 review of commit 8, finding A1: measure before the rules change.)
//!
//! Each path is a file or a folder (walked; links are not followed). A `.json` file is
//! parsed and every string value in it, at any depth, is one input; a `.jsonl` file the same
//! per line; any other file under 4 MiB is one input if it is UTF-8. With `--names`, every
//! file name (without its folder) is one more input.
//!
//! The report never holds a blanked token, a part of one, the text around it, or a path. A
//! token is reported as a SHAPE: its length, and its first 60 characters with every
//! uppercase letter written `A`, every lowercase letter `a`, every digit `9`, every other
//! letter `x`, a line break `⏎`, another control character `·`, and anything else as it is.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;

use axel::screen;
use serde_json::Value;

const MAX_PLAIN: u64 = 4 << 20;
const SHAPE_CHARS: usize = 60;
const TOP: usize = 40;

/// The shape of a blanked token: never the token.
fn shape(text: &str) -> String {
    text.chars()
        .take(SHAPE_CHARS)
        .map(|c| {
            if c.is_uppercase() {
                'A'
            } else if c.is_lowercase() {
                'a'
            } else if c.is_numeric() {
                '9'
            } else if c.is_alphabetic() {
                'x'
            } else if c == '\n' {
                '⏎'
            } else if c.is_control() {
                '·'
            } else {
                c
            }
        })
        .collect()
}

#[derive(Default)]
struct Rule {
    hits: u64,
    in_names: u64,
    /// "len <characters>  <shape>", for a named value with the name ending in front.
    shapes: HashMap<String, u64>,
    /// `named_value` only: the matched name ending.
    endings: BTreeMap<String, u64>,
    /// `long_token` only.
    with_separator: u64,
    /// `key_block` only: the bytes blanked by each hit.
    bytes: Vec<usize>,
}

#[derive(Default)]
struct Stats {
    files: u64,
    files_skipped: u64,
    lines_skipped: u64,
    inputs: u64,
    inputs_hit: u64,
    name_inputs: u64,
    name_inputs_hit: u64,
    input_bytes: u64,
    hits: u64,
    rules: BTreeMap<String, Rule>,
}

impl Stats {
    /// One input through `redact`.
    fn take(&mut self, text: &str, is_name: bool) {
        self.inputs += 1;
        self.name_inputs += u64::from(is_name);
        self.input_bytes += text.len() as u64;
        let spans = screen::redact_spans(text);
        if spans.is_empty() {
            return;
        }
        self.inputs_hit += 1;
        self.name_inputs_hit += u64::from(is_name);
        for (range, hit) in spans {
            self.hits += 1;
            let blanked = text.get(range.clone()).unwrap_or("");
            // The fixed shapes are told apart by their prefix, which is a constant of
            // screen.rs, not a part of what the owner wrote.
            let key = match (hit.rule, hit.name.as_deref()) {
                ("prefix", Some(prefix)) => format!("prefix {prefix}"),
                (rule, _) => rule.to_string(),
            };
            let rule = self.rules.entry(key).or_default();
            rule.hits += 1;
            rule.in_names += u64::from(is_name);
            let mut line = String::new();
            if hit.rule == "named_value" {
                let ending = hit.name.as_deref().unwrap_or("?");
                *rule.endings.entry(ending.to_string()).or_default() += 1;
                let _ = write!(line, "{ending:<14} ");
            }
            let _ = write!(line, "len {:>5}  {}", blanked.chars().count(), shape(blanked));
            *rule.shapes.entry(line).or_default() += 1;
            if hit.rule == "long_token" {
                rule.with_separator += u64::from(blanked.contains(['-', '_']));
            }
            if hit.rule == "key_block" {
                rule.bytes.push(range.len());
            }
        }
    }

    /// Every string value of a JSON document, at any depth.
    fn value(&mut self, value: &Value) {
        match value {
            Value::String(text) => self.take(text, false),
            Value::Array(items) => items.iter().for_each(|item| self.value(item)),
            Value::Object(map) => map.values().for_each(|item| self.value(item)),
            _ => {}
        }
    }

    fn file(&mut self, path: &Path, names: bool) {
        self.files += 1;
        if names {
            if let Some(name) = path.file_name().and_then(|name| name.to_str()) {
                self.take(name, true);
            }
        }
        let extension = path.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase);
        match extension.as_deref() {
            Some("json") => match fs::read(path).ok().and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok()) {
                Some(value) => self.value(&value),
                None => self.files_skipped += 1,
            },
            Some("jsonl") => {
                let Ok(file) = fs::File::open(path) else {
                    self.files_skipped += 1;
                    return;
                };
                let mut reader = BufReader::new(file);
                let mut line = Vec::new();
                loop {
                    line.clear();
                    match reader.read_until(b'\n', &mut line) {
                        Ok(0) => break,
                        Ok(_) if line.iter().all(u8::is_ascii_whitespace) => {}
                        Ok(_) => match serde_json::from_slice::<Value>(&line) {
                            Ok(value) => self.value(&value),
                            Err(_) => self.lines_skipped += 1,
                        },
                        Err(_) => {
                            self.files_skipped += 1;
                            break;
                        }
                    }
                }
            }
            _ => {
                let text = fs::metadata(path)
                    .ok()
                    .filter(|meta| meta.len() < MAX_PLAIN)
                    .and_then(|_| fs::read(path).ok())
                    .and_then(|bytes| String::from_utf8(bytes).ok());
                match text {
                    Some(text) => self.take(&text, false),
                    None => self.files_skipped += 1,
                }
            }
        }
    }

    /// A file, or a folder with everything under it. Links are not followed.
    fn path(&mut self, root: &Path, names: bool) {
        for entry in walkdir::WalkDir::new(root).follow_links(false).follow_root_links(false) {
            match entry {
                Ok(entry) if entry.file_type().is_file() => self.file(entry.path(), names),
                Ok(_) => {}
                Err(_) => self.files_skipped += 1,
            }
        }
    }

    fn report(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "screen_measure: what screen::redact blanks (shapes only, never the text)");
        let _ = writeln!(out);
        let _ = writeln!(out, "files read:                {}", self.files);
        let _ = writeln!(out, "files or folders skipped:  {} (unreadable, not UTF-8, 4 MiB or more, or not JSON)", self.files_skipped);
        let _ = writeln!(out, "jsonl lines skipped:       {} (not JSON)", self.lines_skipped);
        let _ = writeln!(out, "inputs read:               {} (file names among them: {})", self.inputs, self.name_inputs);
        let _ = writeln!(out, "bytes of input:            {}", self.input_bytes);
        let _ = writeln!(out, "inputs with a hit:         {} (file names among them: {})", self.inputs_hit, self.name_inputs_hit);
        let _ = writeln!(out, "hits:                      {}", self.hits);
        let _ = writeln!(out);
        let _ = writeln!(out, "hits per rule (in file names):");
        for (key, rule) in &self.rules {
            let _ = writeln!(out, "  {key:<20} {:>8} ({})", rule.hits, rule.in_names);
        }
        for (key, rule) in &self.rules {
            let _ = writeln!(out);
            let _ = writeln!(out, "== {key}: {} hits", rule.hits);
            if key == "long_token" {
                let _ = writeln!(out, "  hits whose shape contains - or _:                           {}", rule.with_separator);
            }
            if !rule.bytes.is_empty() {
                let mut bytes = rule.bytes.clone();
                bytes.sort_unstable();
                let _ = writeln!(
                    out,
                    "  bytes blanked per hit: min {}, median {}, max {}",
                    bytes[0],
                    bytes[bytes.len() / 2],
                    bytes[bytes.len() - 1]
                );
            }
            if !rule.endings.is_empty() {
                let _ = writeln!(out, "  matched name endings:");
                for (ending, count) in &rule.endings {
                    let _ = writeln!(out, "    {count:>8}  {ending}");
                }
            }
            let mut shapes: Vec<(&String, &u64)> = rule.shapes.iter().collect();
            shapes.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
            let _ = writeln!(
                out,
                "  the {} most frequent of {} shapes (count, {}length in characters, first {SHAPE_CHARS} characters):",
                TOP.min(shapes.len()),
                shapes.len(),
                if key == "named_value" { "name ending, " } else { "" }
            );
            for (line, count) in shapes.into_iter().take(TOP) {
                let _ = writeln!(out, "    {count:>8}  {line}");
            }
        }
        out
    }
}

fn main() {
    let mut names = false;
    let mut paths = Vec::new();
    for arg in std::env::args_os().skip(1) {
        if arg == "--names" {
            names = true;
        } else {
            paths.push(arg);
        }
    }
    if paths.is_empty() {
        eprintln!("usage: screen_measure [--names] <path>...");
        std::process::exit(2);
    }
    let mut stats = Stats::default();
    for path in &paths {
        stats.path(Path::new(path), names);
    }
    print!("{}", stats.report());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shape_is_not_the_text() {
        assert_eq!(shape("2025-11-03-Axel-V2-Notes"), "9999-99-99-Aaaa-A9-Aaaaa");
        assert_eq!(shape("aB3+/=_-.~ é É ß 中 ٣"), "aA9+/=_-.~ a A a x 9");
        assert_eq!(shape("one\ntwo\r\u{1b}"), "aaa⏎aaa··");
        assert_eq!(shape(""), "");
        let long = "Ab1-".repeat(40);
        assert_eq!(shape(&long), "Aa9-".repeat(15));
        assert_eq!(shape(&"é".repeat(100)).chars().count(), SHAPE_CHARS);
    }

    /// (case, the text with a made-up token, the secret part of it, the rule). Joined from
    /// parts so that no key-shaped literal is in the repository. No secret part holds `A`,
    /// `a`, `9` or `x`: a shape can then never equal a piece of one.
    fn tokens() -> Vec<(&'static str, String, String, &'static str)> {
        let join = |parts: &[&str]| parts.concat();
        let prefix = join(&["sk", "-", "Zq7Lm2Rt8Wp4", "Yk6Hn3Vb5Gd1"]);
        let value = join(&["Hj5Kd8", "Wm2Qz7"]);
        let long = join(&["Zq7Lm2Rt8Wp4Yk6H", "-", "n3Vb5Gd1Fs0Jc7Ue2Io4"]);
        let body = join(&["MIIEowIBQEFu", "\n", "q1w2e3r4t5y6"]);
        let password = join(&["hunter", "2uvw"]);
        vec![
            ("prefix", prefix.clone(), prefix, "prefix sk-"),
            ("named value", format!("token={value}"), value, "named_value"),
            ("long token", long.clone(), long, "long_token"),
            (
                "key block",
                format!("-----BEGIN EC PRIVATE{}\n{body}\n-----END EC PRIVATE{}", " KEY-----", " KEY-----"),
                body,
                "key_block",
            ),
            ("url", format!("postgres://bob:{password}@host/db"), password, "url_password"),
        ]
    }

    #[test]
    fn the_report_never_holds_a_token_or_the_text_around_it() {
        let mut stats = Stats::default();
        for (_, text, _, _) in tokens() {
            stats.take(&format!("QUOKKO zeppelin {text} gondola WOMBIT"), false);
        }
        stats.take(&["Wk3Tn8Pd5Lg2", "-", "Rc6Mf1Hs4Bv7Zj5Qy0Wn", ".md"].concat(), true);
        stats.take("nothing here", false);
        let report = stats.report();

        assert_eq!((stats.inputs, stats.inputs_hit, stats.hits), (7, 6, 6));
        assert_eq!((stats.name_inputs, stats.name_inputs_hit), (1, 1));
        for (case, _, _, rule) in tokens() {
            let hits = stats.rules.get(rule).map_or(0, |r| r.hits);
            assert!(hits >= 1, "case {case}: no hit under {rule}");
            assert!(report.contains(&format!("== {rule}: ")), "case {case}: {rule} is not in the report");
        }
        let long = &stats.rules["long_token"];
        assert_eq!((long.hits, long.in_names, long.with_separator), (2, 1, 2));
        assert_eq!(stats.rules["named_value"].endings.get("token"), Some(&1));
        assert_eq!(stats.rules["key_block"].bytes.len(), 1);
        assert!(report.contains("bytes blanked per hit: min "));
        assert!(report.contains("Aa9Aa9Aa9Aa9-Aa9Aa9Aa9Aa9Aa9Aa9Aa\n"));
        assert!(!report.contains(".aa"), "the report holds what stood after a token");

        for word in ["QUOKKO", "WOMBIT", "zeppelin", "gondola", "postgres", "host/db", "Wk3T", "Qy0W"] {
            assert!(!report.contains(word), "the report holds the text around a token: {word}");
        }
        // No four characters in a row of a token that hold a letter or a digit.
        for (case, _, secret, _) in tokens() {
            let chars: Vec<char> = secret.chars().collect();
            assert!(!secret.contains(['A', 'a', '9', 'x']), "case {case}: the made-up token holds a shape letter");
            for (at, window) in chars.windows(4).enumerate() {
                if !window.iter().any(|c| c.is_alphanumeric()) {
                    continue;
                }
                let piece: String = window.iter().collect();
                assert!(!report.contains(&piece), "case {case}: the report holds characters {at}..{} of the token", at + 4);
            }
        }
    }
}
