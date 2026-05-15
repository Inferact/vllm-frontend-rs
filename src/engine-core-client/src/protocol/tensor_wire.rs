use rmpv::Value;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_tuple::{Deserialize_tuple, Serialize_tuple};

/// Tensors and ndarrays are encoded with this extension type in Python.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/v1/serial_utils.py#L42>
const CUSTOM_TYPE_RAW_VIEW: i8 = 3;

/// Python ndarray/tensor wire tuple encoded as `(dtype, shape, data)`.
///
/// This matches the custom msgpack representation built by Python
/// `serial_utils.encode_ndarray` / `encode_tensor`.
///
/// Original Python definitions:
/// <https://github.com/vllm-project/vllm/blob/f22d6e026798a74e6542a52ef776c054f2de572a/vllm/v1/serial_utils.py#L237-L273>
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple)]
pub struct WireNdArray {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data: WireArrayData,
}

/// Python tensor wire tuple encoded as `(dtype, shape, data)`.
///
/// This is the same wire shape as [`WireNdArray`]; multimodal request payloads
/// use it for `torch.Tensor` values.
pub type WireTensor = WireNdArray;

/// Python array/tensor payload reference inside [`WireNdArray`].
///
/// The data can be either an inline msgpack raw-view extension or an index into
/// the multipart aux-frame list carried alongside the primary msgpack frame.
///
/// Original Python definitions:
/// <https://github.com/vllm-project/vllm/blob/f22d6e026798a74e6542a52ef776c054f2de572a/vllm/v1/serial_utils.py#L237-L273>
#[derive(Debug, Clone, PartialEq)]
pub enum WireArrayData {
    /// The index of the aux frame where the raw bytes of this array/tensor are
    /// stored.
    AuxIndex(usize),
    /// The raw bytes of this array/tensor.
    RawView(Vec<u8>),
}

impl<'de> Deserialize<'de> for WireArrayData {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        match value {
            Value::Ext(tag, bytes) if tag == CUSTOM_TYPE_RAW_VIEW => Ok(Self::RawView(bytes)),
            Value::Ext(tag, _) => Err(serde::de::Error::custom(format!(
                "unsupported extension type code {tag}"
            ))),
            Value::Integer(index) => {
                index.as_u64().map(|index| Self::AuxIndex(index as usize)).ok_or_else(|| {
                    serde::de::Error::custom("aux frame index must be a non-negative integer")
                })
            }
            other => Err(serde::de::Error::custom(format!(
                "expected raw-view ext or aux frame index, got {other:?}"
            ))),
        }
    }
}

impl Serialize for WireArrayData {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // TODO: outbound request serialization currently only supports inline
        // raw-view bytes. Emitting aux frames needs transport-level plumbing;
        // serializing `AuxIndex` here only preserves an already-built reference.
        match self {
            Self::AuxIndex(index) => serializer.serialize_u64(*index as u64),
            Self::RawView(bytes) => {
                Value::Ext(CUSTOM_TYPE_RAW_VIEW, bytes.clone()).serialize(serializer)
            }
        }
    }
}
