package edu.nju.software.neuralnetwork.bpnetwork;

public interface ActivationFunction {
    double compute(double in);

    double derivative(double in);
}
