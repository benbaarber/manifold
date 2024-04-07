use manifold::nn::fc3_iso::{GradientRetention, Manifold};
use manifold::optimizers::mbgd3_iso::MiniBatchGradientDescent;
use manifold::Losses3;

use manifold::Activations;

use rand::{prelude::*, thread_rng};

fn gen_training_data() -> (Vec<f64>, Vec<f64>) {
    let mut rng = thread_rng();

    let classes: Vec<(Vec<f64>, Vec<f64>)> = vec![
        (vec![0., 1.], vec![1., 0.]),
        (vec![1., 1.], vec![0., 1.]),
        (vec![1., 0.], vec![1., 0.]),
        (vec![0., 0.], vec![0., 1.]),
    ];

    let data = classes.choose(&mut rng).unwrap();
    (data.0.clone(), data.1.clone())
}

fn main() {
    let (mut x, mut y) = (vec![], vec![]);
    for _ in 0..5000 {
        let (_x, _y) = gen_training_data();
        x.push(_x);
        y.push(_y);
    }

    let (mut tx, mut ty) = (vec![], vec![]);
    for _ in 0..50 {
        let (_x, _y) = gen_training_data();
        tx.push(_x);
        ty.push(_y);
    }

    let mut nn = Manifold::new(2, 2, vec![4]);
    nn.set_hidden_activation(Activations::Relu)
        .set_loss(Losses3::SoftmaxCrossEntropy)
        .weave();

    let threes = MiniBatchGradientDescent::prepare(x, y);

    let mut trainer = MiniBatchGradientDescent::new(&mut nn);
    trainer
        .set_learning_rate(0.01)
        .set_decay(0.999)
        .set_epochs(1)
        .set_sample_size(4)
        .verbose()
        .train(&threes.0, &threes.1);
}
