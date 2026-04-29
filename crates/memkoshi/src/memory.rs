//! Core memory types.
//!
//! Defines the [`Memory`] record, its categorical metadata, and the
//! [`StagedMemory`] envelope used while a memory awaits review.

use chrono::{DateTime, Utc};
use rand::RngCore;
use serde::{Deserialize, Serialize};

/// High-level taxonomy a [`Memory`] belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryCategory {
    /// User or agent preferences.
    Preferences,
    /// Named entities — people, projects, systems.
    Entities,
    /// Discrete events that occurred at a point in time.
    Events,
    /// Case studies; long-form narrative observations.
    Cases,
    /// Recurring behavioural or structural patterns.
    Patterns,
}

impl MemoryCategory {
    /// Lowercase string form, matching the serde representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Preferences => "preferences",
            Self::Entities => "entities",
            Self::Events => "events",
            Self::Cases => "cases",
            Self::Patterns => "patterns",
        }
    }

    /// Parse a lowercase tag back into a category.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "preferences" => Some(Self::Preferences),
            "entities" => Some(Self::Entities),
            "events" => Some(Self::Events),
            "cases" => Some(Self::Cases),
            "patterns" => Some(Self::Patterns),
            _ => None,
        }
    }
}

/// Subjective confidence the agent has in a memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Confidence {
    /// Strong evidence; treat as reliable.
    High,
    /// Reasonable but not certain.
    Medium,
    /// Tentative; flag for verification.
    Low,
}

impl Confidence {
    /// Lowercase tag.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::High => "high",
            Self::Medium => "medium",
            Self::Low => "low",
        }
    }

    /// Parse a lowercase tag back into a confidence level.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "high" => Some(Self::High),
            "medium" => Some(Self::Medium),
            "low" => Some(Self::Low),
            _ => None,
        }
    }
}

/// Review state for a memory waiting in the staging queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReviewStatus {
    /// Awaiting review.
    Pending,
    /// Approved and promoted (or eligible for promotion).
    Approved,
    /// Rejected; will not be promoted.
    Rejected,
}

impl ReviewStatus {
    /// Lowercase tag.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Approved => "approved",
            Self::Rejected => "rejected",
        }
    }

    /// Parse a lowercase tag back into a status.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "pending" => Some(Self::Pending),
            "approved" => Some(Self::Approved),
            "rejected" => Some(Self::Rejected),
            _ => None,
        }
    }
}

/// A single memory record.
///
/// `abstract_text` corresponds to the `abstract` field in the Python
/// reference and JSON form (renamed here because `abstract` is a reserved
/// keyword in Rust).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    /// Identifier of the form `mem_XXXXXXXX` (8 lowercase hex chars).
    pub id: String,
    /// Coarse taxonomy.
    pub category: MemoryCategory,
    /// Topic label (≤ 128 chars by convention).
    pub topic: String,
    /// Short title (≤ 256 chars by convention).
    pub title: String,
    /// One- or two-sentence abstract (≤ 512 chars by convention).
    #[serde(rename = "abstract")]
    pub abstract_text: String,
    /// Full memory body.
    pub content: String,
    /// Subjective confidence.
    pub confidence: Confidence,
    /// Importance in `[0.0, 1.0]`. Default `0.5`.
    pub importance: f64,
    /// Sessions that contributed to this memory.
    pub source_sessions: Vec<String>,
    /// Free-form tags.
    pub tags: Vec<String>,
    /// Other topics this memory references.
    pub related_topics: Vec<String>,
    /// Creation timestamp (UTC).
    pub created: DateTime<Utc>,
    /// Last update timestamp (UTC), if any.
    pub updated: Option<DateTime<Utc>>,
    /// Trust level in `[0.0, 1.0]`. Default `1.0`.
    pub trust_level: f64,
    /// HMAC-SHA256 signature (hex), if signed.
    pub signature: Option<String>,
}

impl Memory {
    /// Create a new memory with default importance/trust and no signature.
    pub fn new(
        category: MemoryCategory,
        topic: impl Into<String>,
        title: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            id: Self::generate_id(),
            category,
            topic: topic.into(),
            title: title.into(),
            abstract_text: String::new(),
            content: content.into(),
            confidence: Confidence::Medium,
            importance: 0.5,
            source_sessions: Vec::new(),
            tags: Vec::new(),
            related_topics: Vec::new(),
            created: Utc::now(),
            updated: None,
            trust_level: 1.0,
            signature: None,
        }
    }

    /// Generate a fresh `mem_XXXXXXXX` identifier with 8 hex characters.
    pub fn generate_id() -> String {
        let mut bytes = [0u8; 4];
        rand::rng().fill_bytes(&mut bytes);
        format!("mem_{}", hex::encode(bytes))
    }
}

impl Default for Memory {
    fn default() -> Self {
        Self::new(MemoryCategory::Events, "default", "Default Memory", "default content")
    }
}

/// A [`Memory`] together with review metadata while it sits in staging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StagedMemory {
    /// The wrapped memory.
    pub memory: Memory,
    /// Current review state.
    pub review_status: ReviewStatus,
    /// Optional reviewer notes (e.g. rejection reason).
    pub reviewer_notes: Option<String>,
    /// Timestamp the memory entered staging.
    pub staged_at: DateTime<Utc>,
}

impl StagedMemory {
    /// Wrap a memory as a fresh `Pending` staging entry.
    pub fn pending(memory: Memory) -> Self {
        Self {
            memory,
            review_status: ReviewStatus::Pending,
            reviewer_notes: None,
            staged_at: Utc::now(),
        }
    }
}
