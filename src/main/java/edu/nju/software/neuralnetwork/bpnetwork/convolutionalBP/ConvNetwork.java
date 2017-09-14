package edu.nju.software.neuralnetwork.bpnetwork.convolutionalBP;

import edu.nju.software.neuralnetwork.bpnetwork.ActivationFunction;
import edu.nju.software.neuralnetwork.bpnetwork.CostFunction;
import edu.nju.software.neuralnetwork.bpnetwork.Network;
import edu.nju.software.neuralnetwork.utils.DMatrix;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;

public class ConvNetwork extends Network{
    private Layer inputLayer;
    private Layer outputLayer;

    public ConvNetwork(CostFunction costFunction, ActivationFunction activationFunction){
        this.costFunction = costFunction;
        this.activationFunction = activationFunction;
        init();
    }

    private void init(){
        inputLayer = new InputLayer(new int[]{25, 60});
        Layer c1 = new ConvLayer(5, new int[]{4, 5}, 1);
        c1.setActivation(activationFunction);
        inputLayer.setNext(c1);
        c1.setPrev(inputLayer);
        Layer s1 = new MaxPoolLayer(5, new int[]{2, 2});
        c1.setNext(s1);
        s1.setPrev(c1);
        Layer c2 = new ConvLayer(10, new int[]{4, 5}, 5);
        c2.setActivation(activationFunction);
        s1.setNext(c2);
        c2.setPrev(s1);
        Layer s2 = new MaxPoolLayer(10, new int[]{2, 2});
        c2.setNext(s2);
        s2.setPrev(c2);
        Layer f1 = new FullConLayer(4 * 12 * 10, 40);
        f1.setActivation(activationFunction);
        s2.setNext(f1);
        f1.setPrev(s2);
        outputLayer = new OutputLayer(40, 40, costFunction);
        outputLayer.setActivation(activationFunction);
        f1.setNext(outputLayer);
        outputLayer.setPrev(f1);
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

    @Override
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
        //error(trainSet);
        accuracy(testSet);
    }
}
