package edu.nju.software.neuralnetwork.bpnetwork.convolutionalBP;

import edu.nju.software.neuralnetwork.utils.DMatrix;
import edu.nju.software.neuralnetwork.utils.concurrent.ConTask;

import java.util.Random;

public class SConvLayer extends Layer {
    private int filter_num;         //number of filter
    private int[] filter_size;      //size of filter

    private int last_filter_num;

    private DMatrix[] maps;

    private DMatrix[][] weights;    //weights of filters
    private double[] biases;        //biases of filters

    private DMatrix[][] weights_error_sum;
    private double[] biases_error_sum;

    private DMatrix[] prev_maps;
    private DMatrix[] z;

    public SConvLayer(int filter_num, int[] filter_size, int last_filter_num) {
        this.filter_num = filter_num;
        this.filter_size = filter_size;
        this.last_filter_num = last_filter_num;

        maps = new DMatrix[filter_num];

        //init weights of filters and sum of weight's error
        weights = new DMatrix[filter_num][last_filter_num];
        weights_error_sum = new DMatrix[filter_num][last_filter_num];

        biases = new double[filter_num];
        biases_error_sum = new double[filter_num];

        Random random = new Random();
        for (int i = 0; i < filter_num; i++) {
            for (int j = 0; j < last_filter_num; j++) {
                weights[i][j] = DMatrix.randn(filter_size[0], filter_size[1], 0.1);
                weights_error_sum[i][j] = DMatrix.zeros(filter_size[0], filter_size[1]);
            }
            biases[i] = random.nextGaussian();
            biases_error_sum[i] = 0.0;
        }
    }

    @Override
    protected void initMap() {
        //init map_size
        map_size = new int[2];
        map_size[0] = prev.map_size[0] - filter_size[0] + 1;
        map_size[1] = prev.map_size[1] - filter_size[1] + 1;
    }

    @Override
    public DMatrix feedforword(DMatrix input) {
        //do nothing
        return feedforword(new DMatrix[]{input});
    }

    @Override
    public DMatrix feedforword(final DMatrix[] inputs) {
        prev_maps = inputs;
        z = new DMatrix[filter_num];
        //conv operation
        final int inputs_len = inputs.length;
        new ConTask(filter_num){
            @Override
            public void process(int start, int end) {
                for (int i = start; i < end; i++) {
                    DMatrix sum = null;
                    for (int j = 0; j < inputs_len; j++) {
                        DMatrix conv = DMatrix.conv(inputs[j], weights[i][j]);
                        sum = (sum == null ? conv : DMatrix.add(sum, conv));

                    }
                    z[i] = DMatrix.add(sum, biases[i]);
                    try {
                        maps[i] = DMatrix.function(z[i], activation, activation.getClass().getMethod("compute", Double.TYPE));
                    } catch (NoSuchMethodException e) {
                        e.printStackTrace();
                    }
                }
            }
        }.start();
        return next.feedforword(maps);
    }

    @Override
    public void backpropagation(final DMatrix[] errors) {
        for (int i = 0; i < filter_num; i++) {
            DMatrix delta = null;
            try {
                delta = DMatrix.function(z[i], activation, activation.getClass().getMethod("derivative", Double.TYPE));
            } catch (NoSuchMethodException e) {
                e.printStackTrace();
            }
            errors[i] = DMatrix.hadamard(errors[i], delta);
            biases_error_sum[i] += errors[i].sum();
        }

        final DMatrix[] prev_errors = new DMatrix[last_filter_num];
        new ConTask(last_filter_num){

            @Override
            public void process(int start, int end) {
                for (int i = start; i < end; i++) {
                    DMatrix prev_error = null;
                    for (int j = 0; j < filter_num; j++) {
                        DMatrix conv = DMatrix.conv(prev_maps[i], errors[j]);
                        weights_error_sum[j][i] = DMatrix.add(weights_error_sum[j][i], conv);

                        DMatrix padding = DMatrix.padding(errors[j], filter_size[0] - 1, filter_size[1] - 1);
                        DMatrix conv2 = DMatrix.conv(padding, DMatrix.rot180(weights[j][i]));

                        prev_error = (prev_error == null ? conv2 : DMatrix.add(prev_error, conv2));
                    }
                    prev_errors[i] = prev_error;
                }
            }
        }.start();
        prev.backpropagation(prev_errors);
    }

    @Override
    public void backpropagation(DMatrix error) {
        //do nothing
    }

    @Override
    public void update(final double lr, final int size, final int n, final double lambda) {
        new ConTask(filter_num){
            @Override
            public void process(int start, int end) {
                for (int i = start; i < end; i++) {
                    for (int j = 0; j < last_filter_num; j++) {
                        DMatrix weights_error = DMatrix.multiple(weights_error_sum[i][j], (lr / size));
                        weights_error_sum[i][j] = DMatrix.zeros(filter_size[0], filter_size[1]);
                        weights[i][j] = DMatrix.multiple(weights[i][j], (1 - lr * (lambda / n)));
                        weights[i][j] = DMatrix.minus(weights[i][j], weights_error);
                    }

                    biases[i] = biases_error_sum[i] * (1 - (lr / size));
                    biases_error_sum[i] = 0.0;
                }
            }
        }.start();
        next.update(lr, size, n, lambda);
    }
}
