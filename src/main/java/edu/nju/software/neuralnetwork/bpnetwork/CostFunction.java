package edu.nju.software.neuralnetwork.bpnetwork;

import edu.nju.software.neuralnetwork.utils.DMatrix;

public interface CostFunction {
    DMatrix cost(DMatrix delta, DMatrix a, DMatrix y);
}
