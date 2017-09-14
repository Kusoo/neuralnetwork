package edu.nju.software.neuralnetwork.bpnetwork.simpleBP;

import edu.nju.software.neuralnetwork.bpnetwork.ActivationFunction;
import edu.nju.software.neuralnetwork.utils.DMatrix;

public abstract class Layer {
    protected Layer prev;
    protected Layer next;

    protected int size;

    protected ActivationFunction activation;
    protected DMatrix weights;
    protected DMatrix biases;
    protected DMatrix in;
    protected DMatrix z;
    protected DMatrix out;
    protected DMatrix nabla_weights_sum;
    protected DMatrix nabla_biases_sum;

    public Layer() {
    }

    public Layer getPrev() {
        return prev;
    }

    public void setPrev(Layer prev) {
        this.prev = prev;
        init();
    }

    protected abstract void init();

    public Layer getNext() {
        return next;
    }

    public void setNext(Layer next) {
        this.next = next;
    }

    public void setActivation(ActivationFunction activation) {
        this.activation = activation;
    }

    public abstract DMatrix feedforword(DMatrix input);

    public abstract void backpropagation(DMatrix y);

    public abstract void update(double lr, int batch_size, int n, double lambda);
}
