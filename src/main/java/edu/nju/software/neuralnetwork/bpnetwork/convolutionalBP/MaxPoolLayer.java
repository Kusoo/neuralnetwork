package edu.nju.software.neuralnetwork.bpnetwork.convolutionalBP;

import edu.nju.software.neuralnetwork.utils.DMatrix;
import edu.nju.software.neuralnetwork.utils.concurrent.ConTask;

public class MaxPoolLayer extends Layer {
    private int pool_num;
    private int[] pool_size;

    private DMatrix[] maps;
    private DMatrix[] prev_maps;

    public MaxPoolLayer(int pool_num, int[] pool_size) {
        this.pool_num = pool_num;
        this.pool_size = pool_size;

        maps = new DMatrix[pool_num];
    }

    @Override
    protected void initMap() {
        //init map_size
        map_size = new int[2];
        map_size[0] = prev.map_size[0] / pool_size[0];
        map_size[1] = prev.map_size[1] / pool_size[1];
    }

    @Override
    public DMatrix feedforword(DMatrix input) {
        //do nothing
        return null;
    }

    @Override
    public DMatrix feedforword(final DMatrix[] inputs) {
        prev_maps = inputs;
        //pooling operation
        /*
        for (int i = 0; i < pool_num; i++) {
            DMatrix pool = DMatrix.maxpool(inputs[i], pool_size);
            maps[i] = pool;
        }
        */
        new ConTask(pool_num){
            @Override
            public void process(int start, int end) {
                for (int i = start; i < end; i++) {
                    DMatrix pool = DMatrix.maxpool(inputs[i], pool_size);
                    maps[i] = pool;
                }
            }
        }.start();
        return next.feedforword(maps);
    }

    @Override
    public void backpropagation(final DMatrix[] errors) {
        final DMatrix[] prev_errors = new DMatrix[pool_num];
        /*
        for (int i = 0; i < pool_num; i++) {
            DMatrix error = errors[i];
            DMatrix ones = DMatrix.ones(pool_size[0], pool_size[1]);
            DMatrix kron = DMatrix.kron(error, ones);
            DMatrix location = DMatrix.maxlocation(prev_maps[i], pool_size);
            DMatrix prev_error = DMatrix.hadamard(kron, location);
            prev_errors[i] = prev_error;
        }
        */
        new ConTask(pool_num){
            @Override
            public void process(int start, int end) {
                for (int i = start; i < end; i++) {
                    DMatrix error = errors[i];
                    DMatrix ones = DMatrix.ones(pool_size[0], pool_size[1]);
                    DMatrix kron = DMatrix.kron(error, ones);
                    DMatrix location = DMatrix.maxlocation(prev_maps[i], pool_size);
                    DMatrix prev_error = DMatrix.hadamard(kron, location);
                    prev_errors[i] = prev_error;
                }
            }
        }.start();
        prev.backpropagation(prev_errors);
    }

    @Override
    public void backpropagation(DMatrix error) {
        DMatrix[] errors = DMatrix.dimup(error, pool_num, map_size[0], map_size[1]);
        backpropagation(errors);
    }

    @Override
    public void update(double lr, int size, int n, double lambda) {
        //do nothing
        next.update(lr, size, n, lambda);
    }
}
