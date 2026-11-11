//! `plugin install --plugins-dir <dir> [--force] [--print]` (section 1, "The binary";
//! section 9, "The manifest").

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use crate::error::{AxelError, Result};
use crate::ext::types::BrainArg;

#[derive(clap::Args)]
pub struct Args {
    #[command(subcommand)]
    pub cmd: Cmd,
}

#[derive(clap::Subcommand)]
pub enum Cmd {
    /// Write the plugin manifest into the Synaps plugins folder
    Install {
        #[arg(long, value_name = "DIR")]
        plugins_dir: Option<PathBuf>,
        /// Replace an existing manifest, keeping a copy
        #[arg(long)]
        force: bool,
        /// Print the manifest and write nothing
        #[arg(long)]
        print: bool,
    },
}

/// The `command` of `MANIFEST`; `install` replaces it with the running binary's path.
pub const MANIFEST_COMMAND: &str = "/home/haseeb/.local/bin/axel-v2";

/// The one manifest (section 9). `plugin/plugin.json` is a copy, held equal by a test.
pub const MANIFEST: &str = r#"{
  "name": "axel",
  "version": "2.0.0",
  "description": "Axel: one local brain for Synaps. Recall before each message, search and remember tools, learning from chats, the project forum. Reads Synaps's archive and note files and never writes them. Nothing leaves this machine.",
  "author": "Haseeb Khalid",
  "license": "MIT",
  "extension": {
    "protocol_version": 1,
    "runtime": "process",
    "command": "/home/haseeb/.local/bin/axel-v2",
    "args": ["extension"],
    "permissions": ["privacy.llm_content", "session.lifecycle", "tools.register"],
    "hooks": [
      {"hook": "before_message"},
      {"hook": "on_message_complete"},
      {"hook": "on_session_start"},
      {"hook": "on_session_end"},
      {"hook": "on_context_archived"},
      {"hook": "on_memory_forgotten"}
    ],
    "config": [
      {"key": "brain",            "description": "Path of the brain file. Empty: ~/.local/share/axel/brain.r8", "default": ""},
      {"key": "synaps_base",      "description": "Synaps base directory to read archives and notes from. Empty: derived from the plugin directory.", "default": ""},
      {"key": "recall",           "description": "Automatic recall before each message: on or off.", "default": "on"},
      {"key": "recall_max_items", "description": "Most items injected per message (1-8).", "default": "5"},
      {"key": "recall_max_chars", "description": "Most characters injected per message.", "default": "1600"},
      {"key": "recall_budget_ms", "description": "Time allowed for one recall, in ms. After it the message goes on without recall.", "default": "150"},
      {"key": "recall_min_sem",   "description": "Least similarity by meaning for an item to be injected (0-1).", "default": "0.30"},
      {"key": "recall_min_lex",   "description": "Least share of the prompt's words an item must contain, when meaning does not pass (0-1).", "default": "0.50"},
      {"key": "recall_on_events", "description": "Also recall on host event messages: on or off.", "default": "off"},
      {"key": "recall_exclude_paths", "description": "Comma-separated folders whose files are never recalled automatically or searched.", "default": ""},
      {"key": "capture",          "description": "Learning from chats as they happen: turns, or off.", "default": "turns"},
      {"key": "archives",         "description": "Index Synaps's context archives: on or off.", "default": "on"},
      {"key": "forum",            "description": "Provide the project forum tools: auto (on unless Synaps still uses its Axel backend), on, or off.", "default": "auto"},
      {"key": "worker_scope",     "description": "What subagent workers can search: project (this project's chat memory only) or all (also global memories and note files).", "default": "project"},
      {"key": "forget_now_min_hidden_hours", "description": "Hours a memory must have been hidden before 'axel-v2 forget --now' will delete it. 0: at once.", "default": "24"},
      {"key": "handoff_on_start", "description": "Inject the handoff written with 'axel handoff set' at session start: on or off.", "default": "off"},
      {"key": "sources",          "description": "Path of sources.toml listing note folders. Empty: ~/.config/axel/sources.toml, else the built-in list.", "default": ""},
      {"key": "archive_events",   "description": "Comma-separated event sources to index from archives besides worker reports.", "default": ""},
      {"key": "embedder",         "description": "onnx (the local model), off (words only), or hash (tests only).", "default": "onnx"},
      {"key": "model_dir",        "description": "Folder with the model and tokenizer.json. Empty: ~/.cache/velocirag/models/all-MiniLM-L6-v2", "default": ""},
      {"key": "log_queries",      "description": "Keep the first 120 characters of a prompt in the recall log, for turns that are captured anyway: on or off.", "default": "on"}
    ]
  }
}
"#;

/// The manifest with `command` set to `binary`.
pub fn manifest_for(binary: &Path) -> Result<String> {
    let path = binary
        .to_str()
        .ok_or_else(|| AxelError::Other(format!("the binary's path is not UTF-8: {}", binary.display())))?;
    let quoted = serde_json::to_string(path)?;
    Ok(MANIFEST.replacen(&format!("\"{MANIFEST_COMMAND}\""), &quoted, 1))
}

/// The absolute, canonical path of the running binary.
fn running_binary() -> Result<PathBuf> {
    Ok(std::fs::canonicalize(std::env::current_exe()?)?)
}

pub fn run(args: &Args, _brain: &BrainArg) -> Result<ExitCode> {
    let Cmd::Install { plugins_dir, force, print } = &args.cmd;
    // `--plugins-dir` has no default (section 1): the real plugins folder is never guessed.
    let Some(dir) = plugins_dir else {
        eprintln!("error: plugin install needs --plugins-dir <DIR>");
        return Ok(ExitCode::from(2));
    };
    let text = manifest_for(&running_binary()?)?;
    if *print {
        print!("{text}");
        return Ok(ExitCode::SUCCESS);
    }
    let folder = dir.join("axel").join(".synaps-plugin");
    let target = folder.join("plugin.json");
    std::fs::create_dir_all(&folder)?;
    if target.symlink_metadata().is_ok() {
        if !*force {
            eprintln!(
                "{} exists; nothing was written. --force replaces it and keeps a copy.",
                target.display()
            );
            return Ok(ExitCode::from(1));
        }
        let backup = backup_path(&folder);
        std::fs::copy(&target, &backup)?;
        println!("kept the old manifest as {}", backup.display());
    }
    let tmp = folder.join("plugin.json.tmp");
    std::fs::write(&tmp, &text)?;
    std::fs::rename(&tmp, &target)?;
    println!("installed {}", target.display());
    Ok(ExitCode::SUCCESS)
}

/// `plugin.json.bak-<stamp>`, a name not taken yet.
fn backup_path(folder: &Path) -> PathBuf {
    let stamp = chrono::Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
    let mut path = folder.join(format!("plugin.json.bak-{stamp}"));
    let mut n = 1;
    while path.symlink_metadata().is_ok() {
        path = folder.join(format!("plugin.json.bak-{stamp}-{n}"));
        n += 1;
    }
    path
}
