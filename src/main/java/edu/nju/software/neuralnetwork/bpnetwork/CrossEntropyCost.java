package edu.nju.software.neuralnetwork.bpnetwork;

import edu.nju.software.neuralnetwork.utils.DMatrix;

public class CrossEntropyCost implements CostFunction {

    public DMatrix cost(DMatrix delta, DMatrix a, DMatrix y) {
        DMatrix cost = DMatrix.minus(a, y);
        return cost;
    }
}
