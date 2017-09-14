package edu.nju.software.neuralnetwork.bpnetwork.convolutionalBP;

import edu.nju.software.neuralnetwork.bpnetwork.ActivationFunction;
import edu.nju.software.neuralnetwork.utils.DMatrix;

public abstract class Layer {
    protected Layer prev;
    protected Layer next;
    protected ActivationFunction activation;

    protected int[] map_size;

    public Layer getPrev() {
        return prev;
    }

    public void setPrev(Layer prev) {
        this.prev = prev;
        initMap();
    }

    protected abstract void initMap();

    public Layer getNext() {
        return next;
    }

    public void setNext(Layer next) {
        this.next = next;
    }

    public ActivationFunction getActivation() {
        return activation;
    }

    public void setActivation(ActivationFunction activation) {
        this.activation = activation;
    }

    public abstract DMatrix feedforword(DMatrix input);

    public abstract DMatrix feedforword(DMatrix[] inputs);

    public abstract void backpropagation(DMatrix[] error);

    public abstract void backpropagation(DMatrix error);

    public abstract void update(double lr, int size, int n, double lambda);
}
