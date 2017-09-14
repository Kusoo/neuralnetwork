package edu.nju.software.neuralnetwork.utils.concurrent;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Created by LShuai on 2016/3/21.
 */
public class ConRunner {
    private static final ExecutorService executor;
    public static final int cpuNum;

    static {
        cpuNum = Runtime.getRuntime().availableProcessors();
        executor = Executors.newFixedThreadPool(cpuNum);
    }

    protected static void execute(Runnable task){
        executor.execute(task);
    }

    protected static void shutdown(){
        executor.shutdown();
    }
}
