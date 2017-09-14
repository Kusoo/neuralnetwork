package edu.nju.software.neuralnetwork.bpnetwork.simpleBP;

import edu.nju.software.neuralnetwork.bpnetwork.*;
import edu.nju.software.neuralnetwork.utils.DMatrix;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;

public class SimpNetwork extends Network {
    private int[] size;
    private Layer inputLayer;
    private Layer outputLayer;

    public SimpNetwork(int[] size, CostFunction costFunction, ActivationFunction activationFunction) {
        this.size = size;
        this.costFunction = costFunction;
        this.activationFunction = activationFunction;
        init();
    }

    private void init() {
        Layer remb = null;

        for (int i = 0; i < size.length; i++) {
            if (i == 0) {
                inputLayer = new InputLayer(size[i]);
                remb = inputLayer;
            } else {
                DMatrix weights;
                DMatrix biases;
                if (i == size.length - 1) {
                    outputLayer = new OutputLayer(size[i], costFunction);
                    outputLayer.setActivation(activationFunction);
                    remb.setNext(outputLayer);
                    outputLayer.setPrev(remb);
                } else {
                    HiddenLayer simpleLayer = new HiddenLayer(size[i]);
                    simpleLayer.setActivation(activationFunction);
                    remb.setNext(simpleLayer);
                    simpleLayer.setPrev(remb);
                    remb = simpleLayer;
                }
            }
        }
    }

    public void accuracy(ArrayList<DMatrix[]> data) {
        int totalNum = data.size();
        int accurateNum = 0;
        Iterator it = data.iterator();

        while(it.hasNext()) {
            DMatrix[] pair = (DMatrix[])it.next();
            DMatrix input = pair[0];
            DMatrix y = pair[1];
            DMatrix output = inputLayer.feedforword(input);
            int[] o_result = dm2int(output);
            int[] y_result = dm2int(y);
            if(Arrays.equals(o_result, y_result)) {
                ++accurateNum;
            }
        }

        System.out.println("accuracy: " + accurateNum + "/" + totalNum);
    }

    public int[] recognize(DMatrix input) {
        DMatrix output = this.inputLayer.feedforword(input);
        return dm2int(output);
    }

    public void error(ArrayList<DMatrix[]> data) {
        Iterator it = data.iterator();

        while(it.hasNext()) {
            DMatrix[] pair = (DMatrix[])it.next();
            DMatrix input = pair[0];
            DMatrix y = pair[1];
            DMatrix output = inputLayer.feedforword(input);
            int[] o_result = dm2int(output);
            int[] y_result = dm2int(y);
            if(!Arrays.equals(o_result, y_result)) {
                for(int i = 0; i < 4; ++i) {
                    System.out.print(o_result[i] + ",");
                }

                System.out.print("       ");

                for(int i = 0; i < 4; ++i) {
                    System.out.print(y_result[i] + ",");
                }

                System.out.println();
            }
        }
    }

    public void SGD(ArrayList<DMatrix[]> trainSet, ArrayList<DMatrix[]> testSet, int epochs, double lr, int batch_size, double lmbda) {
        int len = trainSet.size();

        for(int i = 0; i < epochs; ++i) {
            Collections.shuffle(trainSet);
            int m = 0;

            for(int n = batch_size; n < len; n += batch_size) {
                for(int j = m; j < n; ++j) {
                    DMatrix input = trainSet.get(j)[0];
                    DMatrix y = trainSet.get(j)[1];
                    inputLayer.feedforword(input);
                    outputLayer.backpropagation(y);
                }

                inputLayer.update(lr, batch_size, len, lmbda);
                m += batch_size;
            }

            System.out.print("number " + (i + 1) + " epoch:      ");
            accuracy(trainSet);
        }
        accuracy(testSet);
    }
}
