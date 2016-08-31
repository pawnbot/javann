# javann
Light weight library with Neural Nets for classification and reinforcement learning in Java.

It supports training data both in CSV and binary format. Look for example in javann/examples/

```
        TrainingData data = TrainingData.readFromCSV("examples/resources/data.csv");
        System.out.println("#Examples: " + data.getFeatures().length + ", #Features: " + data.getFeatures()[0].length);
        Bucketizer bucketizer = new Bucketizer();
        bucketizer.enableLogging();
        bucketizer.feedData(data);
        bucketizer.forceOneHotEncoding(0);
        bucketizer.forceOneHotEncoding(1);
        System.out.println("#Features with one-hot encoding: " + bucketizer.getSizeWithOneHotEncoding());
        NeuralNetwork gameNN = new NeuralNetwork(new int[] {bucketizer.getSizeWithOneHotEncoding(), 80, 40, 20, 2, 1});
        gameNN.setBucketizer(bucketizer);
        gameNN.setDropoutRate(0.1);
        gameNN.setLearningRate(1.E-2);
        gameNN.setLossFunction(LossFunction.NORMALIZED_ENTROPY);
        MetricFunction[] metricToLog = new MetricFunction[] {
                MetricFunction.NORMALIZED_ENTROPY,
                MetricFunction.RMSE,
                MetricFunction.AUC,
                MetricFunction.L1_NORM,
                MetricFunction.L2_NORM
        };
        gameNN.enableLogging(metricToLog);
        gameNN.feedData(data, 20);
        gameNN.writeTo("examples/resources/NeuralNet.ser");
```
