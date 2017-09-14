package edu.nju.software.neuralnetwork.bpnetwork.simpleBP;

import edu.nju.software.neuralnetwork.utils.DMatrix;

public class InputLayer extends Layer {
    public InputLayer(int size) {
        this.size = size;
    }

    @Override
    protected void init() {
        //do nothing
    }

    public DMatrix feedforword(DMatrix input) {
        return next.feedforword(input);
    }

    public void backpropagation(DMatrix cost) {
        //do nothing
    }

    public void update(double lr, int batch_size, int n, double lambda) {
        next.update(lr, batch_size, n, lambda);
    }
}
