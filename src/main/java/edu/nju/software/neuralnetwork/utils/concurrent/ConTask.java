package edu.nju.software.neuralnetwork.utils.concurrent;

import java.util.concurrent.CountDownLatch;

public abstract class ConTask {
    private int num;

    public ConTask(int num) {
        this.num = num;
    }

    public void start() {
        int runCupu = (ConRunner.cpuNum >= num) ? 1 : ConRunner.cpuNum;

        int slice = (num - 1) / runCupu + 1;

        final CountDownLatch gate = new CountDownLatch(runCupu);
        for (int i = 0; i < runCupu; i++) {
            final int start = i * slice;
            int temp = (i + 1) * slice;
            final int end = (temp <= num) ? temp : num;

            Runnable task = new Runnable() {
                public void run() {
                    process(start, end);
                    gate.countDown();
                }
            };

            ConRunner.execute(task);
        }
        try {
            gate.await();
        } catch (InterruptedException e) {
            e.printStackTrace();
            throw new RuntimeException(e);
        }
    }

    public abstract void process(int start, int end);
}
