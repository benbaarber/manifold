use crate::{f, Activations};
use ndarray::{s, Array, Array1, Array4, Ix4};
use serde::{Deserialize, Serialize};

use super::types::IsolatedLayer;

/// 2D Convolutional Layer
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Conv2D {
    w: Array4<f64>,
    b: Array1<f64>,
    x: Option<Array4<f64>>,
    activation: Activations,
}

impl Conv2D {
    pub fn new(c_in: usize, c_out: usize, kernel_shape: (usize, usize), activation: Activations) -> Self {
        let (kh, kw) = kernel_shape;
        let weight_shape = (c_out, c_in, kh, kw);
        let bias_shape = c_out;
        let bias_bound = 1.0 / f64::sqrt((c_in * kh * kw) as f64);
        Self {
            w: f::kaiming_uniform(weight_shape, activation).unwrap(),
            b: f::uniform(bias_shape, -bias_bound, bias_bound),
            x: None,
            activation,
        }
    }
}

impl IsolatedLayer<Ix4> for Conv2D {
    fn forward(&mut self, x: Array4<f64>) -> Array4<f64> {
        let (batch_size, _, in_h, in_w) = x.dim();
        let (c_out, _, kh, kw) = self.w.dim();
        let (out_h, out_w) = (in_h - kh + 1, in_w - kw + 1);
        let mut out = Array::zeros((batch_size, c_out, out_h, out_w));

        for n in 0..batch_size {
            for c in 0..c_out {
                let kernel = self.w.slice(s![c, .., .., ..]);
                for i in 0..out_h {
                    for j in 0..out_w {
                        let window = x.slice(s![n, .., i..i+kh, j..j+kw]);
                        let dot = &window * &kernel;
                        out[[n, c, i, j]] = dot.sum() + self.b[c];
                    }
                }
            }
        }

        out
    }

    fn backward(&mut self, grad_output: Array4<f64>) -> Array4<f64> {
        todo!()
    }

    fn gradients(&self) -> (Array4<f64>, Array1<f64>) {
        todo!()
    }
}
