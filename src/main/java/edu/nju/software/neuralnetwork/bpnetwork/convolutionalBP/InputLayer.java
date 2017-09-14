package edu.nju.software.neuralnetwork.bpnetwork.convolutionalBP;

import edu.nju.software.neuralnetwork.utils.DMatrix;

public class InputLayer extends Layer {

    public InputLayer(int[] map_size) {
        this.map_size = map_size;
    }

    @Override
    protected void initMap() {
        //do nothing
    }

    @Override
    public DMatrix feedforword(DMatrix input) {
        return feedforword(new DMatrix[]{input});
    }

    @Override
    public DMatrix feedforword(DMatrix[] inputs) {
        return next.feedforword(inputs);
    }

    @Override
    public void backpropagation(DMatrix[] error) {
        //do nothing
    }

    @Override
    public void backpropagation(DMatrix error) {
        //do nothing
    }

    @Override
    public void update(double lr, int size, int n, double lambda) {
        //do nothing
        next.update(lr, size, n, lambda);
    }
}
