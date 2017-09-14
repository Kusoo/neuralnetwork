package edu.nju.software.neuralnetwork.bpnetwork;

import edu.nju.software.neuralnetwork.utils.DMatrix;

import java.util.ArrayList;

public abstract class Network {

    protected CostFunction costFunction;
    protected ActivationFunction activationFunction;

    protected int[] dm2int(DMatrix dm) {
        return dm.argmax(4, 10);
    }

    public abstract void SGD(ArrayList<DMatrix[]> trainSet, ArrayList<DMatrix[]> testSet, int epochs, double lr, int batch_size, double lmbda);
}
