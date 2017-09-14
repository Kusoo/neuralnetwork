package edu.nju.software.neuralnetwork.bpnetwork.simpleBP;

import edu.nju.software.neuralnetwork.bpnetwork.CostFunction;
import edu.nju.software.neuralnetwork.utils.DMatrix;

public class OutputLayer extends Layer{
    private CostFunction costFunction;

    public OutputLayer(int size, CostFunction costFunction) {
        this.size = size;
        this.costFunction = costFunction;
    }

    @Override
    protected void init() {
        weights = DMatrix.randn(size, prev.size, 0.1D);
        biases = DMatrix.randn(size, 1);

        nabla_weights_sum = DMatrix.zeros(size, prev.size);
        nabla_biases_sum = DMatrix.zeros(size, 1);
    }

    public DMatrix feedforword(DMatrix input) {
        in = input;
        z = DMatrix.add(DMatrix.multiple(weights, input), biases);
        try {
            out = DMatrix.function(z, activation, activation.getClass().getMethod("compute", Double.TYPE));
        } catch (NoSuchMethodException e) {
            e.printStackTrace();
        }
        return out;
    }

    public void backpropagation(DMatrix y) {
        DMatrix delta = null;
        try {
            delta = DMatrix.function(z, activation, activation.getClass().getMethod("derivative", Double.TYPE));
        } catch (NoSuchMethodException e) {
            e.printStackTrace();
        }
        DMatrix error = costFunction.cost(delta, out, y);

        DMatrix nabla_weights = DMatrix.multiple(error, DMatrix.transpose(in));
        DMatrix nabla_biases = error;

        nabla_weights_sum = DMatrix.add(nabla_weights_sum, nabla_weights);
        nabla_biases_sum = DMatrix.add(nabla_biases_sum, nabla_biases);

        DMatrix prev_error = DMatrix.multiple(DMatrix.transpose(weights), error);
        prev.backpropagation(prev_error);
    }

    @Override
    public void update(double lr, int epoch_size, int n, double lambda) {
        DMatrix nabla_weights = DMatrix.multiple(nabla_weights_sum, (lr / epoch_size));
        DMatrix nabla_biases = DMatrix.multiple(nabla_biases_sum, (lr / epoch_size));

        nabla_weights_sum = DMatrix.zeros(size, prev.size);
        nabla_biases_sum = DMatrix.zeros(size, 1);

        weights = DMatrix.multiple(weights, (1 - lr * (lambda / n)));
        weights = DMatrix.minus(weights, nabla_weights);

        biases = DMatrix.minus(biases, nabla_biases);
    }
}
