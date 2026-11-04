//! Stable, versioned byte encoding for descriptors and metadata that cross process boundaries.
//!
//! Every message is a UTF-8 JSON object `{"hkv": <version>, "type": "<kind>", "body": {...}}`.
//! JSON keeps the format language-neutral (llm-d sidecars and vLLM `kv_transfer_params` are JSON)
//! and inspectable. Decoders reject unknown versions and kinds instead of guessing.

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Current wire format version. Bump on any incompatible change to an encoded type.
pub const WIRE_VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
struct Envelope<T> {
    hkv: u32,
    #[serde(rename = "type")]
    kind: String,
    body: T,
}

/// Encode `body` as a versioned message of type `kind`.
pub fn encode<T: Serialize>(kind: &str, body: &T) -> Vec<u8> {
    let env = Envelope {
        hkv: WIRE_VERSION,
        kind: kind.to_owned(),
        body,
    };
    // Serializing plain data structs to JSON cannot fail.
    serde_json::to_vec(&env).expect("wire types serialize")
}

/// Decode a message produced by [`encode`], checking version and kind.
pub fn decode<T: DeserializeOwned>(kind: &str, bytes: &[u8]) -> Result<T> {
    let env: Envelope<serde_json::Value> = serde_json::from_slice(bytes)
        .map_err(|e| Error::Invalid(format!("malformed hetero-kv message: {e}")))?;
    if env.hkv != WIRE_VERSION {
        return Err(Error::Unsupported(format!(
            "hetero-kv wire version {} (this build reads {WIRE_VERSION})",
            env.hkv
        )));
    }
    if env.kind != kind {
        return Err(Error::Invalid(format!(
            "expected a {kind} message, got {}",
            env.kind
        )));
    }
    serde_json::from_value(env.body)
        .map_err(|e| Error::Invalid(format!("malformed {kind} message: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_and_rejections() {
        let bytes = encode("thing", &vec![1u32, 2]);
        assert_eq!(decode::<Vec<u32>>("thing", &bytes).unwrap(), vec![1, 2]);
        assert!(matches!(
            decode::<Vec<u32>>("other", &bytes),
            Err(Error::Invalid(_))
        ));
        let future = br#"{"hkv": 99, "type": "thing", "body": []}"#;
        assert!(matches!(
            decode::<Vec<u32>>("thing", future),
            Err(Error::Unsupported(_))
        ));
        assert!(decode::<Vec<u32>>("thing", b"not json").is_err());
    }
}
