//! Commit 20: the manifest and `plugin install`. Section 9, tests 1 and 2.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use axel::cli::plugin::{manifest_for, MANIFEST, MANIFEST_COMMAND};
use serde_json::Value;

fn bin() -> PathBuf {
    std::fs::canonicalize(env!("CARGO_BIN_EXE_axel")).unwrap()
}

/// Runs the binary with a brain path inside `tmp`, so nothing outside it is named.
fn axel(tmp: &Path, args: &[&str]) -> Output {
    Command::new(bin())
        .arg("--brain")
        .arg(tmp.join("unused-brain.r8"))
        .args(args)
        .env_remove("AXEL_BRAIN")
        .output()
        .unwrap()
}

fn names(list: &Value, field: Option<&str>) -> Vec<String> {
    list.as_array()
        .unwrap()
        .iter()
        .map(|v| match field {
            Some(f) => v[f].as_str().unwrap().to_string(),
            None => v.as_str().unwrap().to_string(),
        })
        .collect()
}

/// Test 1: the constant parses as JSON; its hooks are exactly the six; its permissions the
/// three; `plugin/plugin.json` equals it.
#[test]
fn manifest_constant() {
    let m: Value = serde_json::from_str(MANIFEST).unwrap();
    let ext = &m["extension"];
    assert_eq!(m["name"], "axel");
    assert_eq!(m["version"], "0.1.0");
    assert_eq!(ext["protocol_version"], 1);
    assert_eq!(ext["runtime"], "process");
    assert_eq!(ext["command"], MANIFEST_COMMAND);
    assert_eq!(ext["args"], serde_json::json!(["extension"]));
    assert_eq!(
        names(&ext["hooks"], Some("hook")),
        [
            "before_message",
            "on_message_complete",
            "on_session_start",
            "on_session_end",
            "on_context_archived",
            "on_memory_forgotten"
        ]
    );
    for h in ext["hooks"].as_array().unwrap() {
        assert_eq!(h.as_object().unwrap().len(), 1, "a hook names nothing but its kind");
    }
    assert_eq!(
        names(&ext["permissions"], None),
        ["privacy.llm_content", "session.lifecycle", "tools.register"]
    );
    assert!(ext.get("deferred").is_none() && m.get("deferred").is_none());
    let config = ext["config"].as_array().unwrap();
    assert_eq!(config.len(), 21);
    for entry in config {
        assert!(entry.get("type").is_none(), "no type on config entries");
        assert!(entry["default"].is_string());
    }

    let copy = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../plugin/plugin.json"))
        .unwrap();
    assert_eq!(copy, MANIFEST, "plugin/plugin.json equals the constant");
}

#[test]
fn manifest_for_sets_only_the_command() {
    let text = manifest_for(Path::new("/opt/a \"b\"/axel")).unwrap();
    let mut got: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(got["extension"]["command"], "/opt/a \"b\"/axel");
    got["extension"]["command"] = MANIFEST_COMMAND.into();
    assert_eq!(got, serde_json::from_str::<Value>(MANIFEST).unwrap());
}

/// Test 2.
#[test]
fn install_into_a_temp_plugins_dir() {
    let tmp = tempfile::tempdir().unwrap();
    let plugins = tmp.path().join("plugins");
    let dir = plugins.to_str().unwrap();
    let target = plugins.join("axel/.synaps-plugin/plugin.json");

    // --print writes nothing
    let out = axel(tmp.path(), &["plugin", "install", "--plugins-dir", dir, "--print"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let printed: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(printed["extension"]["command"], bin().to_str().unwrap());
    assert!(!plugins.exists(), "--print touched the plugins folder");

    // first install: <tmp>/axel/.synaps-plugin/plugin.json, command = the canonical binary
    let out = axel(tmp.path(), &["plugin", "install", "--plugins-dir", dir]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let written = std::fs::read_to_string(&target).unwrap();
    let m: Value = serde_json::from_str(&written).unwrap();
    assert_eq!(m["extension"]["command"], bin().to_str().unwrap());
    assert_eq!(written, manifest_for(&bin()).unwrap());

    // a second run is refused and changes nothing
    std::fs::write(&target, "old manifest").unwrap();
    let out = axel(tmp.path(), &["plugin", "install", "--plugins-dir", dir]);
    assert!(!out.status.success());
    assert_eq!(std::fs::read_to_string(&target).unwrap(), "old manifest");

    // --force keeps the old file at plugin.json.bak-*
    let out = axel(tmp.path(), &["plugin", "install", "--plugins-dir", dir, "--force"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(std::fs::read_to_string(&target).unwrap(), written);
    let baks: Vec<PathBuf> = std::fs::read_dir(target.parent().unwrap())
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.file_name().unwrap().to_str().unwrap().starts_with("plugin.json.bak-"))
        .collect();
    assert_eq!(baks.len(), 1);
    assert_eq!(std::fs::read_to_string(&baks[0]).unwrap(), "old manifest");
    let names: Vec<_> = std::fs::read_dir(target.parent().unwrap()).unwrap().collect();
    assert_eq!(names.len(), 2, "no temporary file is left");
}

/// `--command` replaces the running binary's path, with `--print` and with a real install.
#[test]
fn install_command_override() {
    let tmp = tempfile::tempdir().unwrap();
    let plugins = tmp.path().join("plugins");
    let dir = plugins.to_str().unwrap();
    let fake = tmp.path().join("pkg-axel");
    std::fs::write(&fake, "").unwrap();
    let fake_s = fake.to_str().unwrap();

    let out = axel(tmp.path(), &["plugin", "install", "--plugins-dir", dir, "--print", "--command", fake_s]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let printed: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(printed["extension"]["command"], fake_s);

    let out = axel(tmp.path(), &["plugin", "install", "--plugins-dir", dir, "--command", fake_s]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let written = std::fs::read_to_string(plugins.join("axel/.synaps-plugin/plugin.json")).unwrap();
    assert_eq!(written, manifest_for(&fake).unwrap());
}

/// A relative or missing `--command` is refused and nothing is written.
#[test]
fn install_command_override_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let plugins = tmp.path().join("plugins");
    let dir = plugins.to_str().unwrap();
    let missing = tmp.path().join("no-such-axel");
    for (cmd, word) in [("bin/axel", "absolute"), (missing.to_str().unwrap(), "no such file")] {
        let out = axel(tmp.path(), &["plugin", "install", "--plugins-dir", dir, "--command", cmd]);
        assert_eq!(out.status.code(), Some(2), "{cmd}");
        assert!(out.stdout.is_empty());
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(err.contains(word) && err.contains("nothing was written"), "{err}");
    }
    assert!(!plugins.exists());
}

/// No personal path is compiled in: no `/home/` literal anywhere under `src/`.
#[test]
fn no_home_literal_in_src() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut stack = vec![src];
    let mut seen = 0;
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else if let Ok(text) = std::fs::read_to_string(&path) {
                seen += 1;
                assert!(!text.contains("/home/"), "{} has a /home/ literal", path.display());
            }
        }
    }
    assert!(seen > 10);
}

/// Test 2, last clause: without `--plugins-dir` it is a usage error and writes nothing.
#[test]
fn install_without_plugins_dir_is_a_usage_error() {
    let tmp = tempfile::tempdir().unwrap();
    for extra in [&[][..], &["--print"][..], &["--force"][..]] {
        let mut args = vec!["plugin", "install"];
        args.extend_from_slice(extra);
        let out = axel(tmp.path(), &args);
        assert_eq!(out.status.code(), Some(2), "{args:?}");
        assert!(out.stdout.is_empty());
        assert!(String::from_utf8_lossy(&out.stderr).contains("--plugins-dir"));
    }
    assert_eq!(std::fs::read_dir(tmp.path()).unwrap().count(), 0);
}
