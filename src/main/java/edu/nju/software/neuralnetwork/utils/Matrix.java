package edu.nju.software.neuralnetwork.utils;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Random;

public class Matrix {
    private int row;
    private int col;

    private double[][] data;

    public Matrix(int row, int col) {
        this.row = row;
        this.col = col;
        data = new double[row][col];
    }

    public Matrix(double[] data) {
        row = 1;
        col = data.length;
        this.data = new double[row][col];
        this.data[0] = data;
    }

    public Matrix(double[][] data) {
        this.data = data;
        row = data.length;
        col = data[0].length;
    }

    public void init_biases() {
        Random random = new Random();
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                data[i][j] = random.nextGaussian();
            }
        }
    }

    public void init_weights() {
        Random random = new Random();
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                data[i][j] = random.nextGaussian() / 10.0;
            }
        }
    }

    public void reshape(int row, int col) {
        int size = this.row * this.col;
        double[] temp = new double[size];
        int index = 0;
        for (int i = 0; i < this.row; i++) {
            for (int j = 0; j < this.col; j++) {
                temp[index] = data[i][j];
                index++;
            }
        }
        index = 0;
        data = new double[row][col];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                data[i][j] = temp[index];
                index++;
            }
        }

        this.row = row;
        this.col = col;
    }

    public int[] argmax() {
        int[] result = new int[row];
        for (int i = 0; i < row; i++) {
            int index = 0;
            for (int j = 0; j < col; j++) {
                if (data[i][j] > data[i][index]) {
                    index = j;
                }
            }
            result[i] = index;
        }
        return result;
    }

    public double sum() {
        double result = 0.0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                result += data[i][j];
            }
        }
        return result;
    }

    public static Matrix[] dimup(Matrix dm, int num, int row, int col) {
        Matrix[] results = new Matrix[num];
        double[] temp = new double[num * row * col];
        int index = 0;
        for (int i = 0; i < dm.row; i++) {
            for (int j = 0; j < dm.col; j++) {
                temp[index] = dm.data[i][j];
                index++;
            }
        }
        index = 0;
        for (int n = 0; n < num; n++) {
            Matrix result = new Matrix(row, col);
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < col; j++) {
                    result.data[i][j] = temp[index];
                    index++;
                }
            }
            results[n] = result;
        }
        return results;
    }

    public static Matrix dimdown(Matrix[] dms, int[] size) {
        double[] temp = new double[size[0] * size[1]];
        int index = 0;
        for (Matrix dm : dms) {
            for (int i = 0; i < dm.row; i++) {
                for (int j = 0; j < dm.col; j++) {
                    temp[index] = dm.data[i][j];
                    index++;
                }
            }
        }
        Matrix result = new Matrix(size[0], size[1]);
        index = 0;
        for (int i = 0; i < size[0]; i++) {
            for (int j = 0; j < size[1]; j++) {
                result.data[i][j] = temp[index];
                index++;
            }
        }
        return result;
    }

    public static Matrix zeros(Matrix dm) {
        Matrix result = new Matrix(dm.row, dm.col);
        for (int i = 0; i < dm.row; i++) {
            for (int j = 0; j < dm.col; j++) {
                result.data[i][j] = 0.0;
            }
        }
        return result;
    }

    public static Matrix zeros(int row, int col) {
        Matrix result = new Matrix(row, col);
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                result.data[i][j] = 0.0;
            }
        }
        return result;
    }

    public static Matrix ones(int row, int col) {
        Matrix result = new Matrix(row, col);
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                result.data[i][j] = 1.0;
            }
        }
        return result;
    }

    public static Matrix means(int row, int col) {
        Matrix result = new Matrix(row, col);
        int num = row * col;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                result.data[i][j] = 1.0 / num;
            }
        }
        return result;
    }

    public static Matrix copy(Matrix dm) {
        Matrix result = new Matrix(dm.row, dm.col);
        for (int i = 0; i < dm.row; i++) {
            for (int j = 0; j < dm.col; j++) {
                result.data[i][j] = dm.data[i][j];
            }
        }
        return result;
    }

    public static Matrix add(Matrix dm1, Matrix dm2) {
        Matrix result = new Matrix(dm1.row, dm1.col);
        for (int i = 0; i < dm1.row; i++) {
            for (int j = 0; j < dm1.col; j++) {
                result.data[i][j] = dm1.data[i][j] + dm2.data[i][j];
            }
        }
        return result;
    }

    public static Matrix add(Matrix dm1, double num) {
        Matrix result = new Matrix(dm1.row, dm1.col);
        for (int i = 0; i < dm1.row; i++) {
            for (int j = 0; j < dm1.col; j++) {
                result.data[i][j] = dm1.data[i][j] + num;
            }
        }
        return result;
    }

    public static Matrix minus(Matrix dm1, Matrix dm2) {
        Matrix result = new Matrix(dm1.row, dm1.col);
        for (int i = 0; i < dm1.row; i++) {
            for (int j = 0; j < dm1.col; j++) {
                result.data[i][j] = dm1.data[i][j] - dm2.data[i][j];
            }
        }
        return result;
    }

    public static Matrix multiple(Matrix dm1, Matrix dm2) {
        Matrix result = new Matrix(dm1.row, dm2.col);
        for (int i = 0; i < dm1.row; i++) {
            for (int j = 0; j < dm2.col; j++) {
                result.data[i][j] = 0.0;
                for (int k = 0; k < dm1.col; k++) {
                    result.data[i][j] += dm1.data[i][k] * dm2.data[k][j];
                }
            }
        }
        return result;
    }

    public static Matrix multiple(Matrix dm, double num) {
        Matrix result = new Matrix(dm.row, dm.col);
        for (int i = 0; i < dm.row; i++) {
            for (int j = 0; j < dm.col; j++) {
                result.data[i][j] = dm.data[i][j] * num;
            }
        }
        return result;
    }

    //矩阵的阿达马乘积
    public static Matrix hadamard(Matrix dm1, Matrix dm2) {
        Matrix result = new Matrix(dm1.row, dm1.col);
        for (int i = 0; i < dm1.row; i++) {
            for (int j = 0; j < dm1.col; j++) {
                result.data[i][j] = dm1.data[i][j] * dm2.data[i][j];
            }
        }
        return result;
    }

    //矩阵的kronecker乘积
    public static Matrix kron(Matrix dm1, Matrix dm2) {
        int row = dm1.row * dm2.row;
        int col = dm1.col * dm2.col;
        Matrix result = new Matrix(row, col);
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                result.data[i][j] = dm1.data[i / dm2.row][j / dm2.col] * dm2.data[i % dm2.row][j % dm2.col];
            }
        }
        return result;
    }

    public static Matrix function(Matrix dm, Object obj, Method method) {
        Matrix result = new Matrix(dm.row, dm.col);
        for (int i = 0; i < dm.row; i++) {
            for (int j = 0; j < dm.col; j++) {
                try {
                    result.data[i][j] = (Double) method.invoke(obj, dm.data[i][j]);
                } catch (IllegalAccessException e) {
                    e.printStackTrace();
                } catch (InvocationTargetException e) {
                    e.printStackTrace();
                }
            }
        }
        return result;
    }

    public static Matrix transpose(Matrix dm) {
        Matrix out = new Matrix(dm.col, dm.row);
        for (int i = 0; i < dm.col; i++) {
            for (int j = 0; j < dm.row; j++) {
                out.data[i][j] = dm.data[j][i];
            }
        }
        return out;
    }

    //对矩阵四周填充len长度的0
    public static Matrix padding(Matrix dm, int row, int col) {
        Matrix result = Matrix.zeros(dm.row + 2 * row, dm.col + 2 * col);
        for (int i = 0; i < dm.row; i++) {
            for (int j = 0; j < dm.col; j++) {
                result.data[i + row][j + col] = dm.data[i][j];
            }
        }
        return result;
    }

    public static Matrix maxpool(Matrix dm, int[] pool_size) {
        int row = dm.row / pool_size[0];
        int col = dm.col / pool_size[1];
        Matrix result = new Matrix(row, col);
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                int start_x = pool_size[0] * i;
                int start_y = pool_size[1] * j;
                double max = dm.data[start_x][start_y];
                for (int m = start_x; m < start_x + pool_size[0]; m++) {
                    for (int n = start_y; n < start_y + pool_size[1]; n++) {
                        max = (dm.data[m][n] > max) ? dm.data[m][n] : max;
                    }
                }
                result.data[i][j] = max;
            }
        }
        return result;
    }

    public static Matrix maxlocation(Matrix dm, int[] pool_size) {
        Matrix result = Matrix.zeros(dm.row, dm.col);
        int row = dm.row / pool_size[0];
        int col = dm.col / pool_size[1];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                int start_x = pool_size[0] * i;
                int start_y = pool_size[1] * j;
                double max = dm.data[start_x][start_y];
                int max_row = start_x;
                int max_col = start_y;
                for (int m = start_x; m < start_x + pool_size[0]; m++) {
                    for (int n = start_y; n < start_y + pool_size[1]; n++) {
                        if (dm.data[m][n] > max) {
                            max_row = m;
                            max_col = n;
                            max = dm.data[m][n];
                        }
                    }
                }
                result.data[max_row][max_col] = 1.0;
            }
        }
        return result;
    }

    public static Matrix meanpool(Matrix dm, Matrix filter) {
        int row = dm.row / filter.row;
        int col = dm.col / filter.col;
        Matrix result = new Matrix(row, col);
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                int start_x = filter.row * i;
                int start_y = filter.col * j;
                double sum = 0.0;
                for (int m = start_x; m < start_x + filter.row; m++) {
                    for (int n = start_y; n < start_y + filter.col; n++) {
                        sum += dm.data[m][n] * filter.data[m - start_x][n - start_y];
                    }
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }

    public static Matrix conv(Matrix src, Matrix filter) {
        int row = src.row - filter.row + 1;
        int col = src.col - filter.col + 1;
        Matrix result = new Matrix(row, col);
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                double sum = 0.0;
                for (int m = 0; m < filter.row; m++) {
                    for (int n = 0; n < filter.col; n++) {
                        sum += src.data[i + m][j + n] * filter.data[m][n];
                    }
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }

    //将filter进行rot180后再卷积
    public static Matrix conv2(Matrix src, Matrix filter) {
        Matrix new_filter = rot180(filter);
        return conv(src, new_filter);
    }

    public static Matrix rot180(Matrix dm) {
        int row = dm.row;
        int col = dm.col;
        Matrix result = new Matrix(row, col);
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                result.data[i][j] = dm.data[row - 1 - i][col - 1 - j];
            }
        }
        return result;
    }

    public void print() {
        System.out.println("[");
        for (int i = 0; i < row; i++) {
            System.out.print("[");
            for (int j = 0; j < col; j++) {
                System.out.print((int) data[i][j]);
                System.out.print(",");
            }
            System.out.print("]");
            System.out.println();
        }
        System.out.println("]");
    }

}
