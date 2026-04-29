//! HMAC-SHA256 signing and verification for [`Memory`] records.

use hmac::{Hmac, Mac};
use sha2::Sha256;

use crate::memory::Memory;

type HmacSha256 = Hmac<Sha256>;

/// Signs and verifies [`Memory`] records using HMAC-SHA256.
///
/// The signed payload is `"{id}:{category}:{title}:{content}"`. This binds
/// the identity and tampering-relevant fields without including timestamps
/// (which may be rewritten during migration).
#[derive(Clone)]
pub struct MemorySigner {
    key: Vec<u8>,
}

impl MemorySigner {
    /// Construct a new signer from a raw secret key.
    pub fn new(key: &[u8]) -> Self {
        Self { key: key.to_vec() }
    }

    fn payload(memory: &Memory) -> String {
        format!(
            "{}:{}:{}:{}",
            memory.id,
            memory.category.as_str(),
            memory.title,
            memory.content
        )
    }

    /// Compute the hex-encoded HMAC-SHA256 signature for `memory`.
    pub fn sign(&self, memory: &Memory) -> String {
        let mut mac = HmacSha256::new_from_slice(&self.key)
            .expect("HMAC-SHA256 accepts keys of any length");
        mac.update(Self::payload(memory).as_bytes());
        hex::encode(mac.finalize().into_bytes())
    }

    /// Verify the signature stored on `memory`. Returns `false` if the
    /// memory is unsigned or the signature does not match.
    pub fn verify(&self, memory: &Memory) -> bool {
        let Some(sig_hex) = memory.signature.as_deref() else {
            return false;
        };
        let Ok(sig) = hex::decode(sig_hex) else {
            return false;
        };
        let mut mac = HmacSha256::new_from_slice(&self.key)
            .expect("HMAC-SHA256 accepts keys of any length");
        mac.update(Self::payload(memory).as_bytes());
        mac.verify_slice(&sig).is_ok()
    }
}
