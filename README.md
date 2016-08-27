# javann
Light weight library with Neural Nets for classification and reinforcement learning in Java

It support training data both in CSV and binary format. Look for example in javann/examples/

```
    TrainingData data = TrainingData.readFromCSV("data.csv");
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
    gameNN.writeTo("NeuralNet.ser");
```
