package edu.nju.software.neuralnetwork.bpnetwork;

public class Sigmoid implements ActivationFunction {

    public double compute(double in) {
        return 1.0 / (1.0 + Math.exp(-in));
    }

    public double derivative(double in) {
        return compute(in) * (1 - compute(in));
    }
}
