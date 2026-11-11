//! `axel synaps-export --brain <brain.r8> --out <DIR>`.
//!
//! `--brain` is the SOURCE here: a Synaps `brain.r8`, opened immutable by
//! [`crate::synaps_export`]. Nothing of Axel's own brain handling runs.
//! It must be written on the command line: the default brain and the
//! `AXEL_BRAIN` environment variable are refused, so that no file is exported
//! that the caller did not name.
//! Prints ids and counts only, never content.

use std::path::PathBuf;
use std::process::ExitCode;

use crate::error::Result;
use crate::ext::types::BrainArg;
use crate::synaps_export::{export, MANIFEST_FILE};

#[derive(clap::Args, Debug, Clone)]
pub struct Args {
    /// Output directory; must not exist or must be empty
    #[arg(long)]
    pub out: PathBuf,
}

/// Exit codes: 0 clean; 1 written but with a digest mismatch or a failed
/// check, or any other failure; 2 `--out` is not usable; 3 a non-empty
/// `-wal` or `-journal` sits beside the source, or `--brain` was not given
/// on the command line.
///
/// `brain.on_command_line` is `ArgMatches::value_source("brain") ==
/// Some(CommandLine)` of the top-level matches. Anything but the command line
/// is refused before `brain.path` is touched and before `--out` is created.
pub fn run(args: &Args, brain: &BrainArg) -> Result<ExitCode> {
    if !brain.on_command_line {
        eprintln!("synaps-export needs --brain <brain.r8>");
        return Ok(ExitCode::from(2));
    }
    let manifest = match export(&brain.path, &args.out) {
        Ok(manifest) => manifest,
        Err(e) => {
            eprintln!("synaps-export: {e}");
            return Ok(ExitCode::from(e.exit_code()));
        }
    };
    println!(
        "synaps-export: {} project key(s) from {} ({} bytes, sha256 {})",
        manifest.projects.len(),
        manifest.source.path,
        manifest.source.bytes,
        manifest.source.sha256
    );
    for (key, p) in &manifest.projects {
        println!(
            "  {key}  archives {} (live {}, tombstones {}, messages {})  notes {}  note_tombstones {}  forum_posts {}  foreign_memories {}",
            p.archives, p.live, p.tombstones, p.messages, p.notes, p.note_tombstones, p.forum_posts, p.foreign_memories
        );
    }
    let f = &manifest.fingerprints_by_kind;
    println!(
        "left out: forum_posts {}, forum_tombstones {}, captures {}, foreign_memories {}, orphan_fingerprints {}",
        manifest.forum_posts,
        manifest.forum_tombstones,
        manifest.captures,
        manifest.foreign_memories.total,
        manifest.orphan_fingerprints.len()
    );
    println!(
        "fingerprints: history {}, note {}, capture {}; notes_suppressed {}",
        f.history, f.note, f.capture, manifest.notes_suppressed
    );
    for orphan in &manifest.orphan_fingerprints {
        println!("orphan fingerprint: {} {}", orphan.project_key, orphan.value);
    }
    for id in &manifest.digest_mismatch {
        eprintln!("digest mismatch: {id}");
    }
    for failed in &manifest.check_failed {
        eprintln!(
            "check failed, not written: {} {} ({})",
            failed.project_key.as_deref().unwrap_or("-"),
            failed.id,
            failed.check
        );
    }
    println!("manifest: {}", args.out.join(MANIFEST_FILE).display());
    Ok(if manifest.clean() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    })
}
