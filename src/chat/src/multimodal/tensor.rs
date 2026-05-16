use std::collections::HashMap;

use llm_multimodal::{ModelSpecificValue, PreprocessedImages};
use vllm_engine_core_client::protocol::multimodal::NestedTensorValue;
use vllm_engine_core_client::protocol::tensor_wire::WireTensor;

use crate::error::{Error, Result};

/// Owned tensor/scalar representation used while splitting fields per image.
#[derive(Clone)]
pub(super) enum TensorValue {
    /// Float tensor with row-major flat data and shape.
    F32 { data: Vec<f32>, shape: Vec<usize> },
    /// Signed integer tensor with row-major flat data and shape.
    I64 { data: Vec<i64>, shape: Vec<usize> },
    /// Unsigned integer tensor with row-major flat data and shape.
    U32 { data: Vec<u32>, shape: Vec<usize> },
    /// Non-tensor nested value that is shared or copied as-is.
    Scalar(NestedTensorValue),
}

/// Collect `pixel_values` and model-specific outputs into one tensor map.
pub(super) fn collect_tensors(preprocessed: &PreprocessedImages) -> HashMap<String, TensorValue> {
    let mut tensors = HashMap::from([(
        "pixel_values".to_string(),
        TensorValue::F32 {
            data: preprocessed.pixel_values.iter().copied().collect(),
            shape: preprocessed.pixel_values.shape().to_vec(),
        },
    )]);
    for (key, value) in &preprocessed.model_specific {
        tensors.insert(key.clone(), tensor_value_from_model_specific(value));
    }
    tensors
}

/// Convert one `llm-multimodal` model-specific value into wire-builder input.
fn tensor_value_from_model_specific(value: &ModelSpecificValue) -> TensorValue {
    match value {
        ModelSpecificValue::Tensor { data, shape } => TensorValue::F32 {
            data: data.clone(),
            shape: shape.clone(),
        },
        ModelSpecificValue::IntTensor { data, shape } => TensorValue::I64 {
            data: data.clone(),
            shape: shape.clone(),
        },
        ModelSpecificValue::UintTensor { data, shape } => TensorValue::U32 {
            data: data.clone(),
            shape: shape.clone(),
        },
        ModelSpecificValue::Int(value) => TensorValue::Scalar(NestedTensorValue::Int(*value)),
        ModelSpecificValue::Float(value) => TensorValue::Scalar(NestedTensorValue::Float(*value)),
        ModelSpecificValue::IntVec(values) => TensorValue::Scalar(NestedTensorValue::List(
            values.iter().map(|value| NestedTensorValue::Int(*value)).collect(),
        )),
        ModelSpecificValue::UintVec(values) => TensorValue::Scalar(NestedTensorValue::List(
            values.iter().map(|value| NestedTensorValue::Int(*value as i64)).collect(),
        )),
        ModelSpecificValue::FloatVec(values) => TensorValue::Scalar(NestedTensorValue::List(
            values.iter().map(|value| NestedTensorValue::Float(*value as f64)).collect(),
        )),
        ModelSpecificValue::TupleVec(values) => TensorValue::Scalar(NestedTensorValue::List(
            values
                .iter()
                .map(|(height, width)| {
                    NestedTensorValue::List(vec![
                        NestedTensorValue::Int(*height as i64),
                        NestedTensorValue::Int(*width as i64),
                    ])
                })
                .collect(),
        )),
        ModelSpecificValue::Bool(value) => {
            TensorValue::Scalar(NestedTensorValue::Int(i64::from(*value)))
        }
    }
}

/// Extract one image from a batched tensor field.
///
/// Batched fields use their first axis as image index and drop that axis in the
/// per-feature value, matching vLLM's batched-field semantics.
pub(super) fn slice_batched_tensor_value(
    tensor: &TensorValue,
    index: usize,
) -> Result<NestedTensorValue> {
    match tensor {
        TensorValue::F32 { data, shape } => {
            let (shape, data) = slice_first_axis_range(shape, data, index, index + 1, true)?;
            Ok(NestedTensorValue::Tensor(
                WireTensor::from_f32(shape, data).map_err(Error::Multimodal)?,
            ))
        }
        TensorValue::I64 { data, shape } => {
            let (shape, data) = slice_first_axis_range(shape, data, index, index + 1, true)?;
            Ok(NestedTensorValue::Tensor(
                WireTensor::from_i64(shape, data).map_err(Error::Multimodal)?,
            ))
        }
        TensorValue::U32 { data, shape } => {
            let (shape, data) = slice_first_axis_range(shape, data, index, index + 1, true)?;
            Ok(NestedTensorValue::Tensor(
                WireTensor::from_u32(shape, data).map_err(Error::Multimodal)?,
            ))
        }
        TensorValue::Scalar(value) => Ok(value.clone()),
    }
}

/// Extract one image's variable-length range from a flat tensor field.
///
/// Flat fields keep the first axis as the sliced length for this image.
pub(super) fn slice_flat_tensor_value(
    tensor: &TensorValue,
    start: usize,
    end: usize,
) -> Result<NestedTensorValue> {
    match tensor {
        TensorValue::F32 { data, shape } => {
            let (shape, data) = slice_first_axis_range(shape, data, start, end, false)?;
            Ok(NestedTensorValue::Tensor(
                WireTensor::from_f32(shape, data).map_err(Error::Multimodal)?,
            ))
        }
        TensorValue::I64 { data, shape } => {
            let (shape, data) = slice_first_axis_range(shape, data, start, end, false)?;
            Ok(NestedTensorValue::Tensor(
                WireTensor::from_i64(shape, data).map_err(Error::Multimodal)?,
            ))
        }
        TensorValue::U32 { data, shape } => {
            let (shape, data) = slice_first_axis_range(shape, data, start, end, false)?;
            Ok(NestedTensorValue::Tensor(
                WireTensor::from_u32(shape, data).map_err(Error::Multimodal)?,
            ))
        }
        TensorValue::Scalar(value) => Ok(value.clone()),
    }
}

/// Convert a tensor or scalar value without per-image slicing.
pub(super) fn full_tensor_value(tensor: &TensorValue) -> Result<NestedTensorValue> {
    match tensor {
        TensorValue::F32 { data, shape } => Ok(NestedTensorValue::Tensor(
            WireTensor::from_f32(shape.clone(), data.clone()).map_err(Error::Multimodal)?,
        )),
        TensorValue::I64 { data, shape } => Ok(NestedTensorValue::Tensor(
            WireTensor::from_i64(shape.clone(), data.clone()).map_err(Error::Multimodal)?,
        )),
        TensorValue::U32 { data, shape } => Ok(NestedTensorValue::Tensor(
            WireTensor::from_u32(shape.clone(), data.clone()).map_err(Error::Multimodal)?,
        )),
        TensorValue::Scalar(value) => Ok(value.clone()),
    }
}

/// Compute the first-axis range for one image in a flat tensor.
///
/// `sizes_key` names a companion tensor whose entries are cumulative slice
/// sizes per image.
pub(super) fn flat_range_for_index(
    tensors: &HashMap<String, TensorValue>,
    sizes_key: &str,
    index: usize,
) -> Result<(usize, usize)> {
    let sizes = tensors.get(sizes_key).ok_or_else(|| {
        Error::Multimodal(format!("flat tensor sizes key `{sizes_key}` is missing"))
    })?;
    let sizes = tensor_as_usize_vec(sizes)?;
    let size = *sizes.get(index).ok_or_else(|| {
        Error::Multimodal(format!(
            "flat tensor sizes key `{sizes_key}` has no entry for image {index}"
        ))
    })?;
    let start = sizes[..index].iter().sum::<usize>();
    Ok((start, start + size))
}

/// Read a tensor value as per-image sizes for flat slicing.
fn tensor_as_usize_vec(tensor: &TensorValue) -> Result<Vec<usize>> {
    match tensor {
        TensorValue::I64 { data, .. } => data
            .iter()
            .map(|value| {
                usize::try_from(*value)
                    .map_err(|_| Error::Multimodal(format!("negative flat tensor size `{value}`")))
            })
            .collect(),
        TensorValue::U32 { data, .. } => Ok(data.iter().map(|value| *value as usize).collect()),
        _ => Err(Error::Multimodal(
            "flat tensor sizes must be int64 or uint32".to_string(),
        )),
    }
}

/// Slice a flat row-major tensor along its first axis.
fn slice_first_axis_range<T: Clone>(
    shape: &[usize],
    data: &[T],
    start: usize,
    end: usize,
    drop_axis: bool,
) -> Result<(Vec<usize>, Vec<T>)> {
    let first_dim = *shape
        .first()
        .ok_or_else(|| Error::Multimodal("tensor has no first dimension".to_string()))?;
    if start > end || end > first_dim {
        return Err(Error::Multimodal(format!(
            "invalid tensor slice {start}..{end} for first dimension {first_dim}"
        )));
    }
    let expected_len = shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim).ok_or_else(|| {
            Error::Multimodal(format!("tensor shape {shape:?} has too many elements"))
        })
    })?;
    if expected_len != data.len() {
        return Err(Error::Multimodal(format!(
            "tensor shape {shape:?} expects {expected_len} elements, got {}",
            data.len()
        )));
    }
    let stride = shape[1..].iter().product::<usize>();
    let data_start = start * stride;
    let data_end = end * stride;
    let out_shape = if drop_axis {
        shape[1..].to_vec()
    } else {
        let mut shape = shape.to_vec();
        shape[0] = end - start;
        shape
    };
    Ok((out_shape, data[data_start..data_end].to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_first_axis_range_errors_on_shape_data_mismatch() {
        let error = slice_first_axis_range(&[2, 2], &[1.0_f32, 2.0, 3.0], 0, 1, true).unwrap_err();

        assert!(
            matches!(error, Error::Multimodal(message) if message.contains("expects 4 elements"))
        );
    }
}
