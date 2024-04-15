use ndarray::{Array, Dimension, ErrorKind, ShapeBuilder, ShapeError};
use ndarray_rand::{
    rand_distr::{Normal, Uniform},
    RandomExt,
};

use crate::Activations;

/// Create a tensor with values drawn from a uniform distribution over the interval [a, b]
pub fn uniform<D: Dimension>(shape: impl ShapeBuilder<Dim = D>, a: f64, b: f64) -> Array<f64, D> {
    let dist = Uniform::new_inclusive(a, b);
    Array::random(shape, dist)
}

/// Create a tensor with values drawn from a normal distribution with the provided mean and standard deviation
pub fn normal<D: Dimension>(
    shape: impl ShapeBuilder<Dim = D>,
    mean: f64,
    sd: f64,
) -> Array<f64, D> {
    let dist = Normal::new(mean, sd).unwrap();
    Array::random(shape, dist)
}

// fn shape_to_vec<D: Dimension>(shape: impl ShapeBuilder<Dim = D>) -> Vec<usize> {
//   Array::<u8, D>::zeros(shape).shape().to_owned()
// }

fn calculate_fans<D: Dimension>(
    shape: impl ShapeBuilder<Dim = D>,
) -> Result<(f64, f64), ShapeError> {
    let temp = Array::<u8, D>::zeros(shape);
    let dims = temp.shape();

    let [fan_out, fan_in, rest @ ..] = dims else {
        return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
    };
    let receptive_field_size = rest.into_iter().product::<usize>();

    Ok((
        (fan_in * receptive_field_size) as f64,
        (fan_out * receptive_field_size) as f64,
    ))
}

/// Create a tensor with values drawn from a Xavier uniform distribution
///
/// Recommended for use with the sigmoid activation function
pub fn xavier_uniform<D: Dimension>(
    shape: impl ShapeBuilder<Dim = D> + Copy,
    gain: f64,
) -> Result<Array<f64, D>, ShapeError> {
    let (fan_in, fan_out) = calculate_fans(shape)?;
    let a = gain * f64::sqrt(6.0 / (fan_in + fan_out));
    let dist = Uniform::new_inclusive(-a, a);
    Ok(Array::random(shape, dist))
}

/// Create a tensor with values drawn from a Xavier normal distribution
///
/// Recommended for use with the sigmoid activation function
pub fn xavier_normal<D: Dimension>(
    shape: impl ShapeBuilder<Dim = D> + Copy,
    gain: f64,
) -> Result<Array<f64, D>, ShapeError> {
    let (fan_in, fan_out) = calculate_fans(shape)?;
    let sd = gain * f64::sqrt(2.0 / (fan_in + fan_out));
    let dist = Normal::new(0.0, sd).unwrap();
    Ok(Array::random(shape, dist))
}

fn calculate_gain(activation: Activations) -> f64 {
    match activation {
        Activations::Identity => 1.,
        Activations::Relu => 2f64.sqrt(),
    }
}

/// Create a tensor with values drawn from a Kaiming uniform distribution
///
/// Recommended for use with ReLU or Leaky ReLU activation functions.
pub fn kaiming_uniform<D: Dimension>(
    shape: impl ShapeBuilder<Dim = D> + Copy,
    activation: Activations,
) -> Result<Array<f64, D>, ShapeError> {
    let (fan_in, _) = calculate_fans(shape)?;
    let gain = calculate_gain(activation);
    let a = gain * f64::sqrt(6.0 / fan_in);
    let dist = Uniform::new_inclusive(-a, a);
    Ok(Array::random(shape, dist))
}

/// Create a tensor with values drawn from a Kaiming uniform distribution
///
/// Recommended for use with ReLU or Leaky ReLU activation functions.
pub fn kaiming_normal<D: Dimension>(
    shape: impl ShapeBuilder<Dim = D> + Copy,
    activation: Activations,
) -> Result<Array<f64, D>, ShapeError> {
    let (fan_in, _) = calculate_fans(shape)?;
    let gain = calculate_gain(activation);
    let sd = gain * f64::sqrt(2.0 / fan_in);
    let dist = Normal::new(0.0, sd).unwrap();
    Ok(Array::random(shape, dist))
}
