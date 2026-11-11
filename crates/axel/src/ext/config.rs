//! `ExtConfig` from `initialize.params`.
//!
//! The host resolves the manifest's `extension.config` entries (environment, user config,
//! manifest `default`) and passes the object as `params.config` (`CORE/docs/extensions/
//! protocol.md`, "Extension config"; spec section 11, "Tests and the embedder"). Values arrive
//! as the JSON of the manifest's `default` (strings) or as the string the user wrote. Numbers
//! are read from a string or a number, switches from `on|off|true|false|1|0`; anything
//! unreadable keeps the default and adds a line to `warnings` (section 9).

use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ForumMode {
    Auto,
    On,
    Off,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EmbedderKind {
    Onnx,
    Off,
    Hash,
}

#[derive(Clone, Debug)]
pub struct ExtConfig {
    pub brain: PathBuf,
    pub base: PathBuf,
    pub data_dir: PathBuf,
    pub recall: bool,
    pub recall_max_items: usize,
    pub recall_max_chars: usize,
    pub recall_budget_ms: u64,
    pub recall_min_sem: f64,
    pub recall_min_lex: f64,
    pub recall_on_events: bool,
    pub recall_exclude_paths: Vec<PathBuf>,
    pub capture: bool,
    pub archives: bool,
    pub forum: ForumMode,
    pub worker_scope_all: bool,
    pub forget_now_min_hidden_hours: u32,
    pub handoff_on_start: bool,
    pub sources: Option<PathBuf>,
    pub archive_events: Vec<String>,
    pub embedder: EmbedderKind,
    pub model_dir: PathBuf,
    pub log_queries: bool,
    pub warnings: Vec<String>,
}

/// The default brain under `home` (section 1: `~/.local/share/axel/brain.r8`).
pub fn default_brain(home: &Path) -> PathBuf {
    home.join(".local/share/axel/brain.r8")
}

/// The default model folder under `home` (section 11, "The model").
pub fn default_model_dir(home: &Path) -> PathBuf {
    home.join(".cache/velocirag/models/all-MiniLM-L6-v2")
}

struct Reader<'a> {
    cfg: Option<&'a Map<String, Value>>,
    warnings: Vec<String>,
}

impl Reader<'_> {
    /// The value as text; `None` when absent, `null` or empty (the default applies, no
    /// warning). Numbers and booleans are given as their JSON text; an array or an object
    /// is unreadable (the default, with a warning).
    fn text(&mut self, key: &str) -> Option<String> {
        let v = self.cfg?.get(key)?;
        let s = match v {
            Value::Null => return None,
            Value::String(s) => s.trim().to_string(),
            Value::Number(n) => n.to_string(),
            Value::Bool(b) => b.to_string(),
            Value::Array(_) | Value::Object(_) => {
                self.warnings
                    .push(format!("config {key}: unreadable value, using the default"));
                return None;
            }
        };
        (!s.is_empty()).then_some(s)
    }

    fn warn<T: std::fmt::Display>(&mut self, key: &str, default: T) -> T {
        // The value itself is not quoted: `status` shows this line.
        self.warnings
            .push(format!("config {key}: unreadable value, using the default ({default})"));
        default
    }

    /// One of `choices` (lowercase, exact); else the default with a warning.
    fn choice<T: Copy + std::fmt::Display>(&mut self, key: &str, default: T, choices: &[(&str, T)]) -> T {
        let Some(s) = self.text(key) else { return default };
        let s = s.to_ascii_lowercase();
        match choices.iter().find(|(name, _)| *name == s) {
            Some((_, v)) => *v,
            None => self.warn(key, default),
        }
    }

    fn switch(&mut self, key: &str, default: bool) -> bool {
        self.choice(key, default, SWITCH)
    }

    /// A number in `min..=max`; out of range is unreadable.
    fn number<T>(&mut self, key: &str, default: T, min: T, max: T) -> T
    where
        T: std::str::FromStr + PartialOrd + Copy + std::fmt::Display,
    {
        let Some(s) = self.text(key) else { return default };
        match s.parse::<T>() {
            Ok(v) if v >= min && v <= max => v,
            _ => self.warn(key, default),
        }
    }

    fn fraction(&mut self, key: &str, default: f64) -> f64 {
        let Some(s) = self.text(key) else { return default };
        match s.parse::<f64>() {
            Ok(v) if (0.0..=1.0).contains(&v) => v,
            _ => self.warn(key, default),
        }
    }

    fn path(&mut self, key: &str) -> Option<PathBuf> {
        self.text(key).map(PathBuf::from)
    }

    /// Comma-separated, each item trimmed, empty items dropped.
    fn list(&mut self, key: &str) -> Vec<String> {
        self.text(key)
            .map(|s| {
                s.split(',')
                    .map(str::trim)
                    .filter(|p| !p.is_empty())
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default()
    }
}

const SWITCH: &[(&str, bool)] = &[
    ("on", true),
    ("true", true),
    ("1", true),
    ("off", false),
    ("false", false),
    ("0", false),
];

impl std::fmt::Display for ForumMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ForumMode::Auto => "auto",
            ForumMode::On => "on",
            ForumMode::Off => "off",
        })
    }
}

impl std::fmt::Display for EmbedderKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            EmbedderKind::Onnx => "onnx",
            EmbedderKind::Off => "off",
            EmbedderKind::Hash => "hash",
        })
    }
}

/// `capture`: `turns` or `off` (section 9); the switch words are read too.
#[derive(Clone, Copy)]
struct Capture(bool);

impl std::fmt::Display for Capture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.0 { "turns" } else { "off" })
    }
}

/// `worker_scope`: `project` or `all`.
#[derive(Clone, Copy)]
struct Scope(bool);

impl std::fmt::Display for Scope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.0 { "all" } else { "project" })
    }
}

impl ExtConfig {
    /// Never fails. `params` is the whole `initialize.params`; the entries are read from its
    /// `config` object and `plugin_root` feeds the base rule (`project::synaps_base`).
    pub fn from_initialize(params: &Value, home: &Path) -> ExtConfig {
        let mut warnings = Vec::new();
        let cfg = match params.get("config") {
            None | Some(Value::Null) => None,
            Some(Value::Object(m)) => Some(m),
            Some(_) => {
                warnings.push("config: not an object, using every default".to_string());
                None
            }
        };
        let mut r = Reader { cfg, warnings };

        let brain = r.path("brain").unwrap_or_else(|| default_brain(home));
        let plugin_root = params
            .get("plugin_root")
            .and_then(Value::as_str)
            .map(PathBuf::from)
            .unwrap_or_default();
        // The config value when set is the rule's first answer; it is taken here so the
        // value never depends on how `project::synaps_base` reads its other two inputs.
        let base = match r.path("synaps_base") {
            Some(base) => base,
            None => crate::project::synaps_base(None, &plugin_root, home),
        };
        // SPEC: 519/3894: `data_dir` (where `extension.log` goes) is defined nowhere; it is
        // the brain's folder, so a sandbox brain keeps its log in the sandbox.
        let data_dir = brain
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));

        let recall = r.switch("recall", true);
        // SPEC: 4374/4377: a number outside the range the description gives (1-8, 0-1) is
        // read as unreadable: the default and a warning, not a clamp.
        let recall_max_items = r.number("recall_max_items", 5usize, 1, 8);
        let recall_max_chars = r.number("recall_max_chars", 1600usize, 1, usize::MAX);
        let recall_budget_ms = r.number("recall_budget_ms", 150u64, 1, u64::MAX);
        let recall_min_sem = r.fraction("recall_min_sem", 0.30);
        let recall_min_lex = r.fraction("recall_min_lex", 0.50);
        let recall_on_events = r.switch("recall_on_events", false);
        let recall_exclude_paths = r.list("recall_exclude_paths").into_iter().map(PathBuf::from).collect();
        let capture = r
            .choice(
                "capture",
                Capture(true),
                &[
                    ("turns", Capture(true)),
                    ("on", Capture(true)),
                    ("true", Capture(true)),
                    ("1", Capture(true)),
                    ("off", Capture(false)),
                    ("false", Capture(false)),
                    ("0", Capture(false)),
                ],
            )
            .0;
        let archives = r.switch("archives", true);
        let forum = r.choice(
            "forum",
            ForumMode::Auto,
            &[
                ("auto", ForumMode::Auto),
                ("on", ForumMode::On),
                ("true", ForumMode::On),
                ("1", ForumMode::On),
                ("off", ForumMode::Off),
                ("false", ForumMode::Off),
                ("0", ForumMode::Off),
            ],
        );
        let worker_scope_all = r
            .choice("worker_scope", Scope(false), &[("project", Scope(false)), ("all", Scope(true))])
            .0;
        let forget_now_min_hidden_hours = r.number("forget_now_min_hidden_hours", 24u32, 0, u32::MAX);
        let handoff_on_start = r.switch("handoff_on_start", false);
        let sources = r.path("sources");
        let archive_events = r.list("archive_events");
        let embedder = r.choice(
            "embedder",
            EmbedderKind::Onnx,
            &[
                ("onnx", EmbedderKind::Onnx),
                ("off", EmbedderKind::Off),
                ("hash", EmbedderKind::Hash),
            ],
        );
        let model_dir = r.path("model_dir").unwrap_or_else(|| default_model_dir(home));
        let log_queries = r.switch("log_queries", true);

        ExtConfig {
            brain,
            base,
            data_dir,
            recall,
            recall_max_items,
            recall_max_chars,
            recall_budget_ms,
            recall_min_sem,
            recall_min_lex,
            recall_on_events,
            recall_exclude_paths,
            capture,
            archives,
            forum,
            worker_scope_all,
            forget_now_min_hidden_hours,
            handoff_on_start,
            sources,
            archive_events,
            embedder,
            model_dir,
            log_queries,
            warnings: r.warnings,
        }
    }

    /// Auto: reads `<base>/config` once (section 6).
    // SPEC: 5047 and tests/skeleton.rs:284 fix this stub's answer at `false`, while section 6
    // ("The clash") makes `auto` with an unreadable config mean on. 12a's row does not name
    // this function, so the stub stays (forum off: fails closed). The body section 6 wants is
    // `On => true, Off => false, Auto => base_config_value(&self.base, "memory.backend")
    // .as_deref() != Some("axel")`, once the skeleton test's line is changed.
    pub fn forum_on(&self) -> bool {
        false
    }
}

/// The value of `key` in `<base>/config` (`key = value` lines; `#` begins a comment; the
/// last line for the key wins; quotes around the value are dropped). `None` when the file
/// cannot be read or has no such line. The reader that `forum = auto` uses (section 6).
pub fn base_config_value(base: &Path, key: &str) -> Option<String> {
    let text = std::fs::read_to_string(base.join("config")).ok()?;
    let mut found = None;
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        let Some((k, v)) = line.split_once('=') else { continue };
        if k.trim() == key {
            let v = v.trim();
            let v = v
                .strip_prefix('"')
                .and_then(|v| v.strip_suffix('"'))
                .unwrap_or(v);
            found = Some(v.to_string());
        }
    }
    found
}
