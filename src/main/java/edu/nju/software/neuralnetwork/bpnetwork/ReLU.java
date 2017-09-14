package edu.nju.software.neuralnetwork.bpnetwork;

/**
 * Created by LShuai on 2016/3/10.
 */
public class ReLU implements ActivationFunction{
    public double compute(double in) {
        return Math.max(0.0, in);
    }

    public double derivative(double in) {
        return 1;
    }
}
