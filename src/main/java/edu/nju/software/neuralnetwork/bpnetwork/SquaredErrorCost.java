package edu.nju.software.neuralnetwork.bpnetwork;

import edu.nju.software.neuralnetwork.utils.DMatrix;

public class SquaredErrorCost implements CostFunction {

    public DMatrix cost(DMatrix delta, DMatrix a, DMatrix y) {
        DMatrix cost = DMatrix.hadamard(DMatrix.minus(a, y), delta);
        return cost;
    }
}