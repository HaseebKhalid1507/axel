//! Commit 10: the forum contract copied from core (spec section 6; its test 1). The four
//! contract tests of core travel inside `src/forum.rs`; these call the same code from outside
//! the crate, as `forum_store` and `tools_forum` will.

use axel::forum::{
    is_digest, valid_id, valid_project, Author, Cursor, Entry, Envelope, Post, Read, Status,
    MAX_BODY, MAX_LIMIT, SNIPPET_BYTES,
};

const PROJECT: &str = "p1111111111111111";

fn post() -> Post {
    Post {
        request_key: "finding-1".into(),
        thread_id: None,
        reply_to: None,
        title: "Finding".into(),
        body: "Synthetic useful note".into(),
        retention_days: 30,
    }
}

fn author() -> Author {
    Author {
        actor: format!("actor-{}", "a".repeat(32)),
        group: format!("group-{}", "b".repeat(32)),
        parent: None,
    }
}

/// Section 6, test 1.
#[test]
fn test_1_exact_digest_vector() {
    let e = Envelope::new(PROJECT, author(), &post()).unwrap();
    assert_eq!(
        e.digest,
        "79f1294388dea6ade99737365b1f0fb55e8afe447ad9c25f5c568e03476a7ccf"
    );
    assert_eq!(e.id(), format!("msg-{}", e.digest));
    assert!(e.is_root());
    assert_eq!(e.thread_id, e.id());
    assert_eq!(e.source(), format!("forum:{}", e.id()));
}

#[test]
fn fresh_and_child_authors_validate_and_differ() {
    let a = Author::fresh();
    a.validate().unwrap();
    let b = Author::default();
    b.validate().unwrap();
    assert_ne!(a.actor, b.actor);
    assert_ne!(a.group, b.group);
    let c = a.child();
    c.validate().unwrap();
    assert_eq!(c.group, a.group);
    assert_eq!(c.parent.as_deref(), Some(a.actor.as_str()));
    let mut own_parent = a.clone();
    own_parent.parent = Some(a.actor.clone());
    assert!(own_parent.validate().is_err());
    let upper = Author {
        actor: format!("actor-{}", "A".repeat(32)),
        ..author()
    };
    assert!(upper.validate().is_err());
}

#[test]
fn ids_projects_and_digests() {
    assert!(valid_id(&format!("msg-{}", "0".repeat(64))));
    assert!(!valid_id(&format!("msg-{}", "0".repeat(63))));
    assert!(!valid_id("../../bad"));
    assert!(valid_project(PROJECT));
    assert!(!valid_project("p0000000000000000"));
    assert!(!valid_project("p111111111111111"));
    assert!(is_digest(&"f".repeat(64)));
    assert!(!is_digest(&"F".repeat(64)));
    assert!(Envelope::new("p0000000000000000", author(), &post()).is_err());
}

#[test]
fn entries_validate_whole_and_as_snippets() {
    let p = Post {
        body: "é".repeat(MAX_BODY / 2),
        ..post()
    };
    let e = Envelope::new(PROJECT, author(), &p).unwrap();
    let whole = Entry {
        id: e.id(),
        timestamp_ms: 5,
        envelope: e.clone(),
        body: p.body.clone(),
        truncated: false,
    };
    whole.validate().unwrap();
    let snippet = Entry {
        body: "é".repeat(SNIPPET_BYTES / 2),
        truncated: true,
        ..whole.clone()
    };
    snippet.validate().unwrap();
    let changed = Entry {
        body: "tampered".into(),
        ..whole.clone()
    };
    assert!(changed.validate().is_err());
    assert_eq!(
        whole.cursor(),
        Cursor {
            timestamp_ms: 5,
            id: e.id()
        }
    );
}

#[test]
fn reads_and_cursors_are_bounded() {
    Read::default().validate().unwrap();
    assert_eq!(Read::default().limit, 8);
    for limit in [0, MAX_LIMIT + 1] {
        assert!(Read {
            limit,
            ..Read::default()
        }
        .validate()
        .is_err());
    }
    let bad_cursor = Cursor {
        timestamp_ms: i64::MAX as u64 + 1,
        id: format!("msg-{}", "0".repeat(64)),
    };
    assert!(bad_cursor.validate().is_err());
    let r: Read = serde_json::from_str(r#"{"limit":3,"extra":1}"#).unwrap_or_default();
    assert_eq!(r.limit, 8, "unknown fields are refused");
    assert_eq!(
        serde_json::to_string(&Status::Tombstoned).unwrap(),
        "\"tombstoned\""
    );
}
