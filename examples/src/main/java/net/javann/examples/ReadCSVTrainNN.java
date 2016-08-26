package net.javann.examples;

import net.javann.core.NeuralNetwork;
import net.javann.core.TrainingData;
import net.javann.util.LossFunction;
import net.javann.util.MetricFunction;

/**
 * This is a basic example to read data from CSV format and train Neural Network
 *
 */
public class ReadCSVTrainNN {

    public static void main(String[] args) throws Exception {
        TrainingData data = TrainingData.readFromCSV("/Users/bogatyy/javann/examples/resources/data.csv");
        int numExamples = data.getFeatures().length;
        int numFeatures = data.getFeatures()[0].length;
        System.out.println("Examples: " + numExamples + ", Features: " + numFeatures);
        NeuralNetwork gameNN = new NeuralNetwork(new int[] {numFeatures, 40, 10, 1});
        MetricFunction[] metricToLog = new MetricFunction[] {
                MetricFunction.RMSE,
                MetricFunction.NORMALIZED_ENTROPY,
                MetricFunction.AUC,
        };
        gameNN.enableLogging(metricToLog);
        gameNN.setLossFunction(LossFunction.LOG_LOSS);
        gameNN.feedData(data, 30);
        gameNN.writeTo("/Users/bogatyy/NeuralNet.ser");
    }
}