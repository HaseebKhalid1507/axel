//! Commit 5: project keys (spec 2.1, section 2 test 19), with core's own key tests ported
//! (`agent-core/tests/memory_project_scope.rs`: `project_scope_is_canonical_over_path_spellings`,
//! `project_root_discovery_is_stable_across_nested_cwds`; `context_archive.rs`'s
//! `digest_parts` as the directory and logical names).

use std::fs;
use std::path::Path;

use axel::project::{self, ProjectRef};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

/// Core's `ProjectScope::for_canonical_root` key, written out independently.
fn core_key(root: &Path) -> String {
    let canonical = root.canonicalize().unwrap();
    let digest = Sha256::digest(canonical.as_os_str().as_encoded_bytes());
    let mut key = String::from("p");
    for byte in digest.iter().take(8) {
        key.push_str(&format!("{byte:02x}"));
    }
    key
}

#[test]
fn key_of_root_is_the_fixed_vector() {
    // sha256("/") = 8a5edab282632443…; "/" has no parent, so it is its own root.
    assert_eq!(
        project::derive_key(Path::new("/")).as_deref(),
        Some("p8a5edab282632443")
    );
}

#[test]
fn project_dir_and_logical_are_length_prefixed_sha256() {
    assert_eq!(
        project::project_dir("p8a5edab282632443"),
        "57ae36a2fd52081ab8fddf6a95a240b67d8929c8de495e2b63d5bb4368f33ae5"
    );
    assert_eq!(
        project::logical("session-1"),
        "c11dd7dbfb3f174e135a26ada48d677dfcafb2ddc2c38d68ec737c3dbdaa22d3"
    );
    assert_eq!(
        project::logical(""),
        "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc"
    );
}

/// Core: `project_scope_is_canonical_over_path_spellings`.
#[test]
fn project_scope_is_canonical_over_path_spellings() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("proj-a");
    fs::create_dir_all(root.join("sub")).unwrap();
    fs::write(root.join(".synaps-project"), "").unwrap();
    let direct = project::derive_key(&root).unwrap();
    let dotted = project::derive_key(&root.join("sub").join("..")).unwrap();
    assert_eq!(direct, dotted);
    assert_eq!(direct, core_key(&root));
}

/// Core: `project_root_discovery_is_stable_across_nested_cwds` (the override case is not
/// ported: `derive_key` is the rule without the env override).
#[test]
fn project_root_discovery_is_stable_across_nested_cwds() {
    let tmp = TempDir::new().unwrap();

    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).unwrap();
    fs::create_dir_all(repo.join("src/deep/nested")).unwrap();
    let from_root = project::derive_key(&repo).unwrap();
    let from_nested = project::derive_key(&repo.join("src/deep/nested")).unwrap();
    assert_eq!(from_root, from_nested);
    assert_eq!(from_root, core_key(&repo));

    let worktree = tmp.path().join("wt");
    fs::create_dir_all(worktree.join("crates/x")).unwrap();
    fs::write(worktree.join(".git"), "gitdir: /elsewhere/.git/worktrees/wt\n").unwrap();
    let wt_root = project::derive_key(&worktree).unwrap();
    let wt_nested = project::derive_key(&worktree.join("crates/x")).unwrap();
    assert_eq!(wt_root, wt_nested);
    assert_eq!(wt_root, core_key(&worktree));
    assert_ne!(wt_root, from_root);

    let plain = tmp.path().join("plain");
    fs::create_dir_all(plain.join("sub")).unwrap();
    fs::write(plain.join(".synaps-project"), "").unwrap();
    assert_eq!(
        project::derive_key(&plain.join("sub")).unwrap(),
        core_key(&plain)
    );
}

#[test]
fn no_marker_gives_the_folder_itself() {
    // The temp folder could sit below a marker on the build machine: then the walk finds
    // that marker, and the folder's key is the marker folder's, as for tmp itself.
    let tmp = TempDir::new().unwrap();
    let bare = tmp.path().join("bare/somewhere");
    fs::create_dir_all(&bare).unwrap();
    let got = project::derive_key(&bare).unwrap();
    let above = project::derive_key(tmp.path()).unwrap();
    if above == core_key(tmp.path()) {
        assert_eq!(got, core_key(&bare));
    } else {
        assert_eq!(got, above);
    }
}

/// Section 2, test 19.
#[test]
fn test_19_keys() {
    let tmp = TempDir::new().unwrap();
    let folder = tmp.path().join("work");
    fs::create_dir_all(&folder).unwrap();
    fs::write(folder.join(".git"), "gitdir: /elsewhere/.git/worktrees/work\n").unwrap();
    let key = project::derive_key(&folder).unwrap();
    assert_eq!(key, core_key(&folder));

    let identity = folder.join("synaps-memory-identity.json");
    fs::write(&identity, r#"{"key":"p1111111111111111"}"#).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&identity, fs::Permissions::from_mode(0o000)).unwrap();
    }
    assert_eq!(project::derive_key(&folder).unwrap(), key);
    let resolved = project::resolve(folder.to_str()).unwrap();
    assert_eq!(
        resolved,
        ProjectRef {
            key: key.clone(),
            scope_dir: project::project_dir(&key),
        }
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&identity, fs::Permissions::from_mode(0o600)).unwrap();
    }

    assert!(project::resolve(None).is_none());
    assert!(project::resolve(Some(tmp.path().join("missing").to_str().unwrap())).is_none());

    let home = tmp.path().join("home");
    let plugin_root = tmp.path().join(".synaps/plugins/axel");
    fs::create_dir_all(&plugin_root).unwrap();
    assert_eq!(
        project::synaps_base(None, &plugin_root, &home),
        home.join(".synaps-cli")
    );
}

#[test]
fn synaps_base_takes_config_then_a_real_base_then_home() {
    let tmp = TempDir::new().unwrap();
    let home = tmp.path().join("home");
    let x = tmp.path().join("base");
    let plugin_root = x.join("plugins/axel");
    fs::create_dir_all(&plugin_root).unwrap();
    let cfg = tmp.path().join("cfg");

    assert_eq!(project::synaps_base(Some(&cfg), &plugin_root, &home), cfg);
    assert_eq!(
        project::synaps_base(None, &plugin_root, &home),
        home.join(".synaps-cli")
    );

    fs::create_dir_all(x.join("context-archives")).unwrap();
    assert_eq!(project::synaps_base(None, &plugin_root, &home), x);
    fs::remove_dir(x.join("context-archives")).unwrap();

    fs::write(x.join("config"), "").unwrap();
    assert_eq!(project::synaps_base(None, &plugin_root, &home), x);

    // A `config` directory is not a file; a parent not named `plugins` is not a base.
    let y = tmp.path().join("other");
    fs::create_dir_all(y.join("config")).unwrap();
    fs::create_dir_all(y.join("plugins/axel")).unwrap();
    assert_eq!(
        project::synaps_base(None, &y.join("plugins/axel"), &home),
        home.join(".synaps-cli")
    );
    fs::create_dir_all(x.join("ext/axel")).unwrap();
    assert_eq!(
        project::synaps_base(None, &x.join("ext/axel"), &home),
        home.join(".synaps-cli")
    );
}
