package edu.nju.software.neuralnetwork.bpnetwork.simpleBP;

import edu.nju.software.neuralnetwork.utils.DMatrix;

public class HiddenLayer extends Layer{
    public HiddenLayer(int size){
        this.size = size;
    }

    @Override
    protected void init() {
        weights = DMatrix.randn(size, prev.size, 0.1D);
        biases = DMatrix.randn(size, 1);

        nabla_weights_sum = DMatrix.zeros(size, prev.size);
        nabla_biases_sum = DMatrix.zeros(size, 1);
    }

    @Override
    public DMatrix feedforword(DMatrix input) {
        in = input;
        z = DMatrix.add(DMatrix.multiple(weights, input), biases);
        try {
            out = DMatrix.function(z, activation, activation.getClass().getMethod("compute", Double.TYPE));
        } catch (NoSuchMethodException e) {
            e.printStackTrace();
        }
        return next.feedforword(out);
    }

    @Override
    public void backpropagation(DMatrix error) {
        DMatrix delta = null;
        try {
            delta = DMatrix.function(z, activation, activation.getClass().getMethod("derivative", Double.TYPE));
        } catch (NoSuchMethodException e) {
            e.printStackTrace();
        }
        error = DMatrix.hadamard(error, delta);

        DMatrix nabla_weights = DMatrix.multiple(error, DMatrix.transpose(in));
        DMatrix nabla_biases = error;

        nabla_weights_sum = DMatrix.add(nabla_weights_sum, nabla_weights);
        nabla_biases_sum = DMatrix.add(nabla_biases_sum, nabla_biases);

        DMatrix prev_error = DMatrix.multiple(DMatrix.transpose(weights), error);
        prev.backpropagation(prev_error);
    }

    @Override
    public void update(double lr, int batch_size, int n, double lambda) {
        DMatrix nabla_weights = DMatrix.multiple(nabla_weights_sum, (lr / batch_size));
        DMatrix nabla_biases = DMatrix.multiple(nabla_biases_sum, (lr / batch_size));

        nabla_weights_sum = DMatrix.zeros(size, prev.size);
        nabla_biases_sum = DMatrix.zeros(size, 1);

        weights = DMatrix.multiple(weights, (1 - lr * (lambda / n)));
        weights = DMatrix.minus(weights, nabla_weights);

        biases = DMatrix.minus(biases, nabla_biases);

        next.update(lr, batch_size, n, lambda);
    }
}
