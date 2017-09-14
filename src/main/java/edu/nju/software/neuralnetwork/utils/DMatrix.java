package edu.nju.software.neuralnetwork.utils;

import org.jblas.NativeBlas;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.Random;

public class DMatrix {

    private int row;
    private int col;

    private int length;

    private double[] data;

    public DMatrix(int row, int col, double[] data) {
        this.row = row;
        this.col = col;
        length = row * col;

        this.data = data;
    }

    public DMatrix(int row, int col) {
        this(row, col, new double[row * col]);
    }

    public DMatrix(double[] data) {
        this(data.length, 1, data);
    }

    public DMatrix(double[][] data) {
        this(data.length, data[0].length);

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                put(i, j, data[i][j]);
            }
        }
    }

    private int index(int i, int j) {
        return j * row + i;
    }

    public double get(int i, int j) {
        return data[index(i, j)];
    }

    public void put(int i, int j, double value) {
        data[index(i, j)] = value;
    }

    public void reshape(int row, int col) {
        this.row = row;
        this.col = col;
    }

    public int[] argmax(int row, int col) {
        int[] result = new int[row];
        for (int i = 0; i < row; i++) {
            int index = 0;
            for (int j = 0; j < col; j++) {
                if (data[i * col + j] > data[i * col + index]) {
                    index = j;
                }
            }
            result[i] = index;
        }
        return result;
    }

    public DMatrix copy() {
        DMatrix result = new DMatrix(row, col);
        System.arraycopy(data, 0, result.data, 0, length);
        return result;
    }

    public double sum() {
        double sum = 0.0;
        for (int i = 0; i < length; i++) {
            sum += data[i];
        }
        return sum;
    }

    public static DMatrix zeros(int row, int col) {
        DMatrix result = new DMatrix(row, col);
        Arrays.fill(result.data, 0.0);
        return result;
    }

    public static DMatrix ones(int row, int col) {
        DMatrix result = new DMatrix(row, col);
        Arrays.fill(result.data, 1.0);
        return result;
    }

    public static DMatrix means(int row, int col) {
        int len = row * col;
        double value = 1.0 / len;
        DMatrix result = new DMatrix(row, col);
        Arrays.fill(result.data, value);
        return result;
    }

    public static DMatrix randn(int row, int col) {
        DMatrix result = new DMatrix(row, col);
        Random random = new Random();
        for (int i = 0; i < result.length; i++) {
            result.data[i] = random.nextGaussian();
        }
        return result;
    }

    public static DMatrix randn(int row, int col, double alpha) {
        DMatrix result = new DMatrix(row, col);
        Random random = new Random();
        for (int i = 0; i < result.length; i++) {
            result.data[i] = random.nextGaussian() * alpha;
        }
        return result;
    }

    public static DMatrix[] dimup(DMatrix dm, int num, int row, int col) {
        DMatrix[] results = new DMatrix[num];
        int index = 0;
        for (int n = 0; n < num; n++) {
            DMatrix result = new DMatrix(row, col);
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < col; j++) {
                    result.put(i, j, dm.data[index]);
                    index++;
                }
            }
            results[n] = result;
        }
        return results;
    }

    public static DMatrix dimdown(DMatrix[] dms, int[] size) {
        DMatrix result = new DMatrix(size[0], size[1]);
        int index = 0;
        for (DMatrix dm : dms) {
            for (int i = 0; i < dm.row; i++) {
                for (int j = 0; j < dm.col; j++) {
                    result.data[index] = dm.get(i, j);
                    index++;
                }
            }
        }
        return result;
    }

    public static DMatrix add(DMatrix dm1, DMatrix dm2) {
        DMatrix result = new DMatrix(dm1.row, dm1.col);
        for (int i = 0; i < result.length; i++) {
            result.data[i] = dm1.data[i] + dm2.data[i];
        }
        return result;
    }

    public static DMatrix add(DMatrix dm, double value) {
        DMatrix result = new DMatrix(dm.row, dm.col);
        for (int i = 0; i < result.length; i++) {
            result.data[i] = dm.data[i] + value;
        }
        return result;
    }

    public static DMatrix minus(DMatrix dm1, DMatrix dm2) {
        DMatrix result = new DMatrix(dm1.row, dm1.col);
        for (int i = 0; i < result.length; i++) {
            result.data[i] = dm1.data[i] - dm2.data[i];
        }
        return result;
    }

    public static DMatrix multiple(DMatrix dm1, DMatrix dm2) {
        DMatrix result = new DMatrix(dm1.row, dm2.col);
        if (dm2.col == 1) {
            for (int j = 0; j < dm1.col; j++) {
                double xj = dm2.data[j];
                if (xj != 0.0D) {
                    for (int i = 0; i < dm1.row; ++i) {
                        result.data[i] += dm1.get(i, j) * xj;
                    }
                }
            }
        } else {
            NativeBlas.dgemm('N', 'N', result.row, result.col, dm1.col, 1.0D, dm1.data, 0,
                    dm1.row, dm2.data, 0, dm2.row, 0.0D, result.data, 0, result.row);
        }
        return result;
    }

    public static DMatrix multiple(DMatrix dm, double value) {
        DMatrix result = new DMatrix(dm.row, dm.col);
        for (int i = 0; i < result.length; i++) {
            result.data[i] = dm.data[i] * value;
        }
        return result;
    }

    //矩阵的阿达马乘积
    public static DMatrix hadamard(DMatrix dm1, DMatrix dm2) {
        DMatrix result = new DMatrix(dm1.row, dm1.col);
        for (int i = 0; i < result.length; i++) {
            result.data[i] = dm1.data[i] * dm2.data[i];
        }
        return result;
    }

    //矩阵的kronecker乘积
    public static DMatrix kron(DMatrix dm1, DMatrix dm2) {
        int row = dm1.row * dm2.row;
        int col = dm1.col * dm2.col;
        DMatrix result = new DMatrix(row, col);
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                double temp = dm1.get(i / dm2.row, j / dm2.col) * dm2.get(i % dm2.row, j % dm2.col);
                result.put(i, j, temp);
            }
        }
        return result;
    }

    public static DMatrix function(DMatrix dm, Object obj, Method method) {
        DMatrix result = new DMatrix(dm.row, dm.col);
        for (int i = 0; i < result.length; i++) {
            try {
                result.data[i] = (Double) method.invoke(obj, dm.data[i]);
            } catch (IllegalAccessException e) {
                e.printStackTrace();
            } catch (InvocationTargetException e) {
                e.printStackTrace();
            }
        }
        return result;
    }

    public static DMatrix transpose(DMatrix dm) {
        DMatrix result = new DMatrix(dm.col, dm.row);
        for (int i = 0; i < dm.col; i++) {
            for (int j = 0; j < dm.row; j++) {
                result.put(i, j, dm.get(j, i));
            }
        }
        return result;
    }

    //对矩阵四周填充len长度的0
    public static DMatrix padding(DMatrix dm, int row, int col) {
        DMatrix result = DMatrix.zeros(dm.row + 2 * row, dm.col + 2 * col);
        for (int i = 0; i < dm.row; i++) {
            for (int j = 0; j < dm.col; j++) {
                result.put(i + row, j + col, dm.get(i, j));
            }
        }
        return result;
    }


    public static DMatrix maxpool(DMatrix dm, int[] pool_size) {
        int row = dm.row / pool_size[0];
        int col = dm.col / pool_size[1];
        DMatrix result = new DMatrix(row, col);
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                int start_x = pool_size[0] * i;
                int start_y = pool_size[1] * j;
                double max = dm.get(start_x, start_y);
                for (int m = start_x; m < start_x + pool_size[0]; m++) {
                    for (int n = start_y; n < start_y + pool_size[1]; n++) {
                        max = (dm.get(m, n) > max) ? dm.get(m, n) : max;
                    }
                }
                result.put(i, j, max);
            }
        }
        return result;
    }

    public static DMatrix maxlocation(DMatrix dm, int[] pool_size) {
        DMatrix result = DMatrix.zeros(dm.row, dm.col);
        int row = dm.row / pool_size[0];
        int col = dm.col / pool_size[1];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                int start_x = pool_size[0] * i;
                int start_y = pool_size[1] * j;
                double max = dm.get(start_x, start_y);
                int max_row = start_x;
                int max_col = start_y;
                for (int m = start_x; m < start_x + pool_size[0]; m++) {
                    for (int n = start_y; n < start_y + pool_size[1]; n++) {
                        if (dm.get(m, n) > max) {
                            max_row = m;
                            max_col = n;
                            max = dm.get(m, n);
                        }
                    }
                }
                result.put(max_row, max_col, 1.0);
            }
        }
        return result;
    }

    public static DMatrix meanpool(DMatrix dm, DMatrix filter) {
        int row = dm.row / filter.row;
        int col = dm.col / filter.col;
        DMatrix result = new DMatrix(row, col);
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                int start_x = filter.row * i;
                int start_y = filter.col * j;
                double sum = 0.0;
                for (int m = start_x; m < start_x + filter.row; m++) {
                    for (int n = start_y; n < start_y + filter.col; n++) {
                        sum += dm.get(m, n) * filter.get(m - start_x, n - start_y);
                    }
                }
                result.put(i, j, sum);
            }
        }
        return result;
    }

    public static DMatrix conv(DMatrix src, DMatrix filter) {
        int row = src.row - filter.row + 1;
        int col = src.col - filter.col + 1;
        DMatrix result = new DMatrix(row, col);
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                double sum = 0.0;
                for (int m = 0; m < filter.row; m++) {
                    for (int n = 0; n < filter.col; n++) {
                        sum += src.get(i + m, j + n) * filter.get(m, n);
                    }
                }
                result.put(i, j, sum);
            }
        }
        return result;
    }

    public static DMatrix[] conv(DMatrix[] inputs, DMatrix[][] filters){
        int input_num = inputs.length;
        int out_num = filters.length;

        int input_x = inputs[0].row;
        int input_y = inputs[0].col;

        int filter_x = filters[0][0].row;
        int filter_y = filters[0][0].col;
        int filter_size = filter_x * filter_y;

        int row = input_x - filter_x + 1;
        int col = input_y - filter_y + 1;
        int size = row * col;

        int temp_x = row * col;
        int temp_y = filter_x * filter_y;
        int temp_size = temp_x * temp_y;
        DMatrix input_m = new DMatrix(row * col, filter_x * filter_y * input_num);
        for(int i = 0; i < input_num; i++){
            DMatrix input = inputs[i];
            DMatrix temp = im2row(input, new int[]{filter_x, filter_y});
            System.arraycopy(temp.data, 0, input_m.data, temp_size*i, temp_size);
        }

        DMatrix filter_m = new DMatrix(input_num * filter_size, out_num);
        for(int i = 0; i < out_num; i++){
            for(int j = 0; j < input_num; j++){
                int index = (input_num * i + j) * filter_size;
                System.arraycopy(filters[i][j].data, 0, filter_m.data, index, filter_size);
            }
        }

        DMatrix result = multiple(input_m, filter_m);

        DMatrix[] results = new DMatrix[out_num];

        for(int i = 0; i < out_num; i++){
            int index = i * size;
            results[i] = new DMatrix(row, col);
            System.arraycopy(result.data, index, results[i].data, 0, size);
        }
        return results;
    }

    //matlab中的im2col的变形且为sliding模式
    public static DMatrix im2row(DMatrix dm, int[] filter){
        int row = dm.row - filter[0] + 1;
        int col = dm.col -filter[1] + 1;
        DMatrix result = new DMatrix(row * col, filter[0] * filter[1]);
        for (int i = 0; i < col; i++) {
            for (int j = 0; j < row; j++) {
                for (int m = 0; m < filter[1]; m++) {
                    for (int n = 0; n < filter[0]; n++) {
                        result.put(i * row + j, m * filter[0] + n, dm.get(n + j, m + i));
                    }
                }
            }
        }
        return result;
    }

    //将矩阵进行rot180操作
    public static DMatrix rot180(DMatrix dm) {
        int row = dm.row;
        int col = dm.col;
        DMatrix result = new DMatrix(row, col);
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                result.put(i, j, dm.get(row - 1 - i, col - 1 - j));
            }
        }
        return result;
    }

    public void print() {
        System.out.println("[");
        for (int i = 0; i < row; i++) {
            System.out.print("[");
            for (int j = 0; j < col; j++) {
                System.out.print((int) get(i, j));
                System.out.print(",");
            }
            System.out.print("]");
            System.out.println();
        }
        System.out.println("]");
    }

}
