package edu.nju.software.neuralnetwork.bpnetwork.convolutionalBP;

import edu.nju.software.neuralnetwork.utils.DMatrix;

public class FullConLayer extends Layer {
    private int prev_num;
    private int num;

    private DMatrix weights;
    private DMatrix biases;

    private DMatrix weights_error_sum;
    private DMatrix biases_error_sum;

    private DMatrix in;
    private DMatrix z;
    private DMatrix out;

    public FullConLayer(int prev_num, int num) {
        this.prev_num = prev_num;
        this.num = num;

        weights = DMatrix.randn(num, prev_num, 0.1);
        weights_error_sum = DMatrix.zeros(num, prev_num);

        biases = DMatrix.randn(num, 1);
        biases_error_sum = DMatrix.zeros(num, 1);
    }

    @Override
    protected void initMap() {
        //do nothing
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
    public DMatrix feedforword(DMatrix[] inputs) {
        DMatrix input = DMatrix.dimdown(inputs, new int[]{prev_num, 1});
        return feedforword(input);
    }

    @Override
    public void backpropagation(DMatrix[] error) {

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

        DMatrix weights_error = DMatrix.multiple(error, DMatrix.transpose(in));
        DMatrix biases_error = error;

        weights_error_sum = DMatrix.add(weights_error_sum, weights_error);
        biases_error_sum = DMatrix.add(biases_error_sum, biases_error);

        DMatrix prev_error = DMatrix.multiple(DMatrix.transpose(weights), error);
        prev.backpropagation(prev_error);
    }

    @Override
    public void update(double lr, int size, int n, double lambda) {
        DMatrix nabla_weights = DMatrix.multiple(weights_error_sum, (lr / size));
        DMatrix nabla_biases = DMatrix.multiple(biases_error_sum, (lr / size));

        weights_error_sum = DMatrix.zeros(num, prev_num);
        biases_error_sum = DMatrix.zeros(num, 1);

        weights = DMatrix.multiple(weights, (1 - lr * (lambda / n)));
        weights = DMatrix.minus(weights, nabla_weights);

        biases = DMatrix.minus(biases, nabla_biases);

        next.update(lr, size, n, lambda);
    }
}
