package net.javann.core;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.DoubleSummaryStatistics;
import java.util.Random;

import net.javann.util.ValueLabel;
import net.javann.util.LossFunction;
import net.javann.util.MetricFunction;

public class NeuralNetwork implements Serializable {

	// serialver for backwards compatibility
	private static final long serialVersionUID = 2L;
	// the random number generator
	private final Random random;

	private double velocityMultiplier;
	private double momentumMultiplier;
	private double learningRate;
	private double l1Multiplier;
	private double l2Multiplier;
	private double dropoutRate;
	private int miniBatchSize;
	LossFunction lossFunction;
	MetricFunction[] loggedMetrics;

	int numInputs;
	int numOutputs;
	private double[][][] weights;
	private double[][] biases;
	private double[][] activations;

	private Bucketizer bucketizer;

	private double[][][] newNablaW;
	private double[][] newNablaB;
	private double[][][] oldNablaW;
	private double[][] oldNablaB;

	public NeuralNetwork(int[] structure) {
		random = new Random();
		velocityMultiplier = 0.;
		momentumMultiplier = 0.;
		learningRate = 1.E-2;
		l1Multiplier = 1.E-6;
		l2Multiplier = 1.E-5;
		dropoutRate = 0.;
		miniBatchSize = 10;
		lossFunction = LossFunction.SQUARED_ERROR;
		loggedMetrics = null;

		numInputs = structure[0];
		numOutputs = structure[structure.length - 1];
		weights = new double[structure.length - 1][][];
		biases = new double[structure.length - 1][];
		activations = new double[weights.length][];
		for (int depth = 0; depth < weights.length; ++depth) {
			if (depth + 1 < weights.length) {
				// ReLu layers
				weights[depth] = new double[structure[depth + 1]][];
				biases[depth] = new double[structure[depth + 1]];
				activations[depth] = new double[structure[1 + depth]];
			} else {
				// SoftMax layer
				weights[depth] = new double[structure[depth]][];
				biases[depth] = new double[structure[depth]];
				activations[depth] = new double[structure[depth]];
			}
			for (int unitId = 0; unitId < weights[depth].length; ++unitId) {
				if (depth + 1 < weights.length) {
					// ReLu layers
					weights[depth][unitId] = new double[structure[depth]];
				} else {
					// SoftMax Layer
					weights[depth][unitId] = new double[1];
				}
			}
		}
		randomizeWeights();

		bucketizer = new Bucketizer();

		newNablaW = null;
		newNablaB = null;
		oldNablaW = null;
		oldNablaB = null;
	}

	private double[] feedForward(double[] inputLayer, double activeRatio) {
		for (int depth = 0; depth + 1 < weights.length; ++depth) {
			for (int unitId = 0; unitId < weights[depth].length; ++unitId) {
				// Intermediate layers, using ReLu
				if (activeRatio < 1.0 && depth + 2 < weights.length && activeRatio < random.nextDouble()) {
					// Don't disable neurons right before SoftMax
					activations[depth][unitId] = 0.;
					continue;
				}
				double dotProduct = biases[depth][unitId];
				if (depth > 0) {
					dotProduct += innerProduct(weights[depth][unitId].length, weights[depth][unitId], activations[depth - 1]);
				} else {
					dotProduct += innerProduct(weights[depth][unitId].length, weights[depth][unitId], inputLayer);
				}
				activations[depth][unitId] = relu(dotProduct);
				if (activeRatio < 1.0 && depth + 2 < weights.length) {
					activations[depth][unitId] /= activeRatio;
				}
			}
		}
		int lastLayer = weights.length - 1;
		double sumExponents = softMaxDenominator(activations[lastLayer - 1]);
		for (int unitId = 0; unitId < activations[lastLayer].length; ++unitId) {
			if (sumExponents != Double.POSITIVE_INFINITY) {
				activations[lastLayer][unitId] = softMaxNumerator(activations[lastLayer - 1], unitId) / sumExponents;
			} else {
				if (Math.abs(activations[lastLayer - 1][unitId]) > Bucketizer.EPSILON) {
					activations[lastLayer][unitId] = 1.;
				} else {
					activations[lastLayer][unitId] = 0.;
				}
			}
		}
		return activations[lastLayer];
	}


	private void backPropagation(double[] inputLayer, double[] expectedOutputLayer) {
		double[][] nablaB = new double[weights.length][];
		double[][][] nablaW = new double[weights.length][][];
		for (int depth = 0; depth < weights.length; ++depth) {
			nablaB[depth] = new double [weights[depth].length];
			nablaW[depth] = new double [weights[depth].length][];
			for (int unitId = 0; unitId < weights[depth].length; ++unitId) {
				nablaW[depth][unitId] = new double [weights[depth][unitId].length];
			}
		}

		feedForward(inputLayer, 1. - dropoutRate);
		int lastLayer = weights.length - 1;
		double expectedAfterNumOutputs = 0.;
		for (int unitId = 0; unitId < weights[lastLayer].length; ++unitId) {
			if (unitId < numOutputs) {
				expectedAfterNumOutputs += Math.abs(expectedOutputLayer[unitId]) > Bucketizer.EPSILON ? 1. : 0.;
			} else {
				expectedAfterNumOutputs = (1. - expectedAfterNumOutputs) / (weights[lastLayer].length - numOutputs);
			}
		}
		for (int unitId = 0; unitId < weights[lastLayer].length; ++unitId) {
			double expectedOutput;
			if (unitId < numOutputs) {
				expectedOutput = expectedOutputLayer[unitId];
			} else {
				expectedOutput = expectedAfterNumOutputs;
			}
			double expectedProb = Math.abs(expectedOutput) > Bucketizer.EPSILON ? 1. : 0.;

			double derivative = 0.;
			if (lossFunction == LossFunction.NORMALIZED_ENTROPY) {
				derivative = activations[lastLayer][unitId] - expectedProb;
			} else if (lossFunction == LossFunction.CUMULATIVE_REWARD) {
				derivative = -expectedOutput * activations[lastLayer][unitId] * (1. - activations[lastLayer][unitId]);
			} else if (lossFunction == LossFunction.SQUARED_ERROR)
			{
				derivative = 2. * (activations[lastLayer][unitId] - expectedOutput);
				derivative *= activations[lastLayer][unitId] * (1. - activations[lastLayer][unitId]);
			}

			if (Double.isNaN(derivative)) {
				System.out.println("Activation: " + activations[lastLayer][unitId]);
				System.out.println("Prob: " + expectedProb);
				System.exit(0);
			}

			nablaB[lastLayer][unitId] = derivative;
			nablaW[lastLayer][unitId][0] = derivative * activations[lastLayer - 1][unitId];
		}
		for (int depth = lastLayer - 1; depth >= 0; --depth) {
			for (int unitId = 0; unitId < weights[depth].length; ++unitId) {
				if (activations[depth][unitId] == 0.)
					continue;
				double delta = 0.;
				if (depth + 1 != lastLayer) {
					for (int id = 0; id < nablaB[depth + 1].length; ++id) {
						delta += nablaB[depth + 1][id] * weights[depth + 1][id][unitId];
					}
				} else {
					delta = nablaB[depth + 1][unitId] * weights[depth + 1][unitId][0];
				}
				delta *= reluDerivative(activations[depth][unitId]);
				nablaB[depth][unitId] = delta;
				for (int inputId = 0; inputId < weights[depth][unitId].length; ++inputId) {
					if (depth > 0) {
						nablaW[depth][unitId][inputId] = delta * activations[depth - 1][inputId];
					} else {
						nablaW[depth][unitId][inputId] = delta * inputLayer[inputId];
					}
				}
			}
		}

		for (int depth = 0; depth < weights.length; ++depth) {
			for (int unitId = 0; unitId < weights[depth].length; ++unitId) {
				newNablaB[depth][unitId] += nablaB[depth][unitId];
				for (int inputId = 0; inputId < weights[depth][unitId].length; ++inputId) {
					newNablaW[depth][unitId][inputId] += nablaW[depth][unitId][inputId];
					// Adding L1 & L2 regularization
					newNablaW[depth][unitId][inputId] += l2Multiplier * weights[depth][unitId][inputId];
					newNablaW[depth][unitId][inputId] += l1Multiplier * Math.signum(weights[depth][unitId][inputId]);
				}
			}
		}
	}

	public void feedData(double[][] data, double[][] labels, int numEpochs) {
		if (!bucketizer.isInitialized()) {
			bucketizer.feedData(data, labels);
		}
		if (loggedMetrics.length > 0) {
			logResults(0, data, labels);
		}
		double[][] inputs = new double[data.length][];
		double[][] outputs = new double[labels.length][];
		int[] randomIndexes = new int[data.length];
		for (int id = 0; id < data.length; ++id) {
			randomIndexes[id] = id;
			inputs[id] = bucketizer.exampleToInput(data[id]);
			if (lossFunction != LossFunction.CUMULATIVE_REWARD) {
				outputs[id] = bucketizer.labelsToOutput(labels[id]);
			} else {
				outputs[id] = labels[id];
			}
		}
		double velocity = 0.;
		double momentum;
		double adaptiveLearningRate = learningRate;
		for (int epoch = 0; epoch < numEpochs; ++epoch) {
			shuffleArray(randomIndexes);
			for (int batchOffset = 0; batchOffset < data.length; batchOffset += miniBatchSize) {
				newNablaB = new double[weights.length][];
				newNablaW = new double[weights.length][][];
				for (int depth = 0; depth < weights.length; ++depth) {
					newNablaB[depth] = new double[weights[depth].length];
					newNablaW[depth] = new double[weights[depth].length][];
					for (int unitId = 0; unitId < weights[depth].length; ++unitId) {
						newNablaW[depth][unitId] = new double[weights[depth][unitId].length];
					}
				}
				int batchSize = Math.min(miniBatchSize, data.length - batchOffset);
				for (int id = 0; id < batchSize; ++id) {
					backPropagation(inputs[randomIndexes[batchOffset + id]], outputs[randomIndexes[batchOffset + id]]);
				}
				if (batchSize > 1) {
					scaleNablas(newNablaB, newNablaW, 1. / batchSize);
				}
				applyNablasToNN(newNablaB, newNablaW, learningRate);
				rescaleExplodedUnits();
				momentum = velocity;
				velocity = nablasInnerProduct();
				adaptiveLearningRate += velocityMultiplier * velocity + momentumMultiplier * momentum;
				oldNablaB = newNablaB;
				oldNablaW = newNablaW;
				newNablaB = null;
				newNablaW = null;
			}
			if (loggedMetrics.length > 0) {
				logResults(1 + epoch, data, labels);
			}
		}
		oldNablaB = null;
		oldNablaW = null;
	}

	public void feedData(TrainingData data, int numEpochs) {
		feedData(data.getFeatures(), data.getLabels(), numEpochs);
	}

	public double[] genProbabilities(double[] example) {
		double[] activations = feedForward(bucketizer.exampleToInput(example), 1.);
		double[] labels = new double[numOutputs];
		for (int labelId = 0; labelId < numOutputs; ++labelId) {
			labels[labelId] = activations[labelId];
		}
		return labels;
	}
	public double[] genLabels(double[] example) {
		return bucketizer.outputToPredictions(genProbabilities((example)));
	}

	public double getRMSE(double[][] data, double[][] labels) {
		double rmse = 0.;
		for (int id = 0; id < data.length; ++id) {
			double[] predictions = genLabels(data[id]);
			for (int labelId = 0; labelId < labels[id].length; ++labelId) {
				rmse += (predictions[labelId] - labels[id][labelId]) * (predictions[labelId] - labels[id][labelId]);
			}
		}
		rmse = Math.sqrt(rmse / data.length);
		return rmse;
	}

	public double getAUC(double[][] data, double[][] labels) {
		double result = 0.;
		for (int labelId = 0; labelId < numOutputs; ++labelId) {
			ArrayList<ValueLabel> auc = new ArrayList<>();
			for (int id = 0; id < data.length; ++id) {
				double[] predictions = genLabels(data[id]);
				auc.add(new ValueLabel(predictions[labelId], labels[id][labelId]));
			}
			Collections.sort(auc);
			double area = 0.;
			int height = 0;
			for (int id = 0; id < auc.size(); ++id) {
				if (auc.get(id).label[0] < bucketizer.EPSILON) {
					height++;
				} else {
					area += height;
				}
			}
			result +=  area / (height + Bucketizer.EPSILON) / (auc.size() - height + Bucketizer.EPSILON);
		}
		return  result / numOutputs;
	}

	public double getCalibration(double[][] data, double[][] labels) {
		double prediction = 0.;
		double observed = 0.;
		for (int id = 0; id < data.length; ++id) {
			double[] predictions = genLabels(data[id]);
			for (int labelId = 0; labelId < labels[id].length; ++labelId) {
				prediction += predictions[labelId];
				observed += labels[id][labelId];
			}
		}
		return prediction / (observed + Bucketizer.EPSILON);
	}

	public double getNormalizedEntropy(double[][] data, double[][] labels) {
		double[] ctr = new double [numOutputs];
		for (int id = 0; id < labels.length; ++id) {
			for (int labelId = 0; labelId < numOutputs; ++labelId) {
				ctr[labelId] += Math.abs(labels[id][labelId]) > Bucketizer.EPSILON ? 1. : 0.;
			}
		}
		for (int labelId = 0; labelId < numOutputs; ++labelId) {
			ctr[labelId] /= labels.length;
		}
		double loss = 0.;
		double constantLoss = 0.;
		for (int id = 0; id < data.length; ++id) {
			double[] predictions = genProbabilities(data[id]);
			for (int labelId = 0; labelId < labels[id].length; ++labelId) {
				double label = Math.abs(labels[id][labelId]) > Bucketizer.EPSILON ? 1. : 0.;
				loss += label * Math.log(predictions[labelId]);
				constantLoss += label * Math.log(ctr[labelId]);
				if (numOutputs == 1) {
					loss += (1. - label) * Math.log(1. - predictions[labelId]);
					constantLoss += (1. - label) * Math.log(1. - ctr[labelId]);
				}
			}
		}
		return loss / constantLoss;
	}

	public double getCumulativeReward(double[][] data, double[][] labels) {
		double totalReward = 0.;
		for (int id = 0; id < data.length; ++id) {
			double[] predictions = genProbabilities(data[id]);
			for (int labelId = 0; labelId < labels[id].length; ++labelId) {
				totalReward += labels[id][labelId] * predictions[labelId];
			}
		}
		return totalReward;
	}

	public double getL1Norm() {
		double result = 0.;
		int count = 0;
		for (int depth = 0; depth < weights.length; ++depth) {
			for (int unitId = 0; unitId < weights[depth].length; ++unitId) {
				result += Math.abs(biases[depth][unitId]);
				count++;
				for (int inputId = 0; inputId < weights[depth][unitId].length; ++inputId) {
					result += Math.abs(weights[depth][unitId][inputId]);
					count++;
				}
			}
		}
		return result / count;
	}

	public double getL2Norm() {
		double result = 0.;
		int count = 0;
		for (int depth = 0; depth < weights.length; ++depth) {
			for (int unitId = 0; unitId < weights[depth].length; ++unitId) {
				result += biases[depth][unitId] * biases[depth][unitId];
				count++;
				for (int inputId = 0; inputId < weights[depth][unitId].length; ++inputId) {
					result += weights[depth][unitId][inputId] * weights[depth][unitId][inputId];
					count++;
				}
			}
		}
		return result / count;
	}

	public Bucketizer getBucketizer() {
		return bucketizer;
	}

	public void setBucketizer(Bucketizer bucketizer) {
		this.bucketizer = bucketizer;
	}

	public double getLearningRate() {
		return learningRate;
	}

	public void setLearningRate(double learningRate) {
		this.learningRate = learningRate;
	}

	public double getL1Multiplier() {
		return l1Multiplier;
	}

	public void setL1Multiplier(double l1Multiplier) {
		this.l1Multiplier = l1Multiplier;
	}

	public double getL2Multiplier() {
		return l2Multiplier;
	}

	public void setL2Multiplier(double l2Multiplier) {
		this.l2Multiplier = l2Multiplier;
	}

	public double getDropoutRate() {
		return dropoutRate;
	}

	public void setDropoutRate(double dropoutRate) {
		this.dropoutRate = dropoutRate;
	}

	public int getMiniBatchSize() { return miniBatchSize; }

	public void setMiniBatchSize(int miniBatchSize) { this.miniBatchSize = miniBatchSize; }

	public void setLossFunction(LossFunction lossFunction) { this.lossFunction = lossFunction; }

	public void disableLogging() {
		loggedMetrics = null;
		bucketizer.disableLogging();
	}
	public void enableLogging(MetricFunction[] loggedMetrics) {
		this.loggedMetrics = loggedMetrics;
		bucketizer.enableLogging();
	}
	public void enableLogging() {
		enableLogging(MetricFunction.values());
	}

	private void logResults(int epochId, double[][] data, double[][] labels) {
		if (loggedMetrics == null)
			return;
		String message = "Epoch: " + epochId;
		for (int id = 0; id < loggedMetrics.length; ++id) {
			message += ", " + loggedMetrics[id].toString() + ": ";
			if (loggedMetrics[id] == MetricFunction.RMSE) {
				message += getRMSE(data, labels);
			}
			if (loggedMetrics[id] == MetricFunction.AUC) {
				message += getAUC(data, labels);
			}
			if (loggedMetrics[id] == MetricFunction.L1_NORM) {
				message += getL1Norm();
			}
			if (loggedMetrics[id] == MetricFunction.L2_NORM) {
				message += getL2Norm();
			}
			if (loggedMetrics[id] == MetricFunction.CALIBRATION) {
				message += getCalibration(data, labels);
			}
			if (loggedMetrics[id] == MetricFunction.NORMALIZED_ENTROPY) {
				message += getNormalizedEntropy(data, labels);
			}
			if (loggedMetrics[id] == MetricFunction.CUMULATIVE_REWARD) {
				message += getCumulativeReward(data, labels);
			}
		}
		System.out.println(message);
	}

	public void randomizeWeights() {
		for (int depth = 0; depth + 1 < weights.length; ++depth) {
			for (int unitId = 0; unitId < weights[depth].length; ++unitId) {
				biases[depth][unitId] = 0.;
				for (int inputId = 0; inputId < weights[depth][unitId].length; ++inputId) {
					weights[depth][unitId][inputId] = random.nextDouble() - 0.5;
					weights[depth][unitId][inputId] *= Math.sqrt(2. / weights[depth][unitId].length);
				}
			}
		}
		int lastLayer = weights.length - 1;
		for (int unitId = 0; unitId < weights[lastLayer].length; ++unitId) {
			biases[lastLayer][unitId] = 0.;
			weights[lastLayer][unitId][0] = 1.;
		}
	}

	private void applyNablasToNN(double[][] nablaB, double[][][] nablaW, double rate) {
		boolean shito = false;
		for (int depth = 0; depth < weights.length; ++depth) {
			for (int unitId = 0; unitId < weights[depth].length; ++unitId) {
				if (Double.isNaN(biases[depth][unitId])) {
					System.out.println("Depth: " + depth +", UnitId: " + unitId + ", Bias: " + biases[depth][unitId]);
					shito = true;
				}
				if (Double.isNaN(nablaB[depth][unitId])) {
					System.out.println("Depth: " + depth +", UnitId: " + unitId + ", nablaB: " + nablaB[depth][unitId]);
					shito = true;
				}
				biases[depth][unitId] -= rate * nablaB[depth][unitId];
				for (int inputId = 0; inputId < weights[depth][unitId].length; ++inputId) {
					if (Double.isNaN(weights[depth][unitId][inputId])) {
						System.out.println("Depth: " + depth +", UnitId: " + unitId + ", Weight: " + weights[depth][unitId][inputId]);
						shito = true;
					}
					if (Double.isNaN(nablaB[depth][unitId])) {
						System.out.println("Depth: " + depth +", UnitId: " + unitId + ", nablaW: " + nablaW[depth][unitId][inputId]);
						shito = true;
					}
					weights[depth][unitId][inputId] -= rate * nablaW[depth][unitId][inputId];
				}
			}
		}
		if (shito)
			System.exit(0);
	}
	private void scaleNablas(double[][] nablaB, double[][][] nablaW, double scale) {
		for (int depth = 0; depth < weights.length; ++depth) {
			for (int unitId = 0; unitId < weights[depth].length; ++unitId) {
				nablaB[depth][unitId] *= scale;
				for (int inputId = 0; inputId < weights[depth][unitId].length; ++inputId) {
					nablaW[depth][unitId][inputId] *= scale;
				}
			}
		}
	}
	private double nablasInnerProduct() {
		if (oldNablaB == null || oldNablaW == null || newNablaB == null || newNablaW == null) {
			return 0.;
		}
		double result = 0.;
		for (int depth = 0; depth < weights.length; ++depth) {
			for (int unitId = 0; unitId < weights[depth].length; ++unitId) {
				result += oldNablaB[depth][unitId] * newNablaB[depth][unitId];
				for (int inputId = 0; inputId < weights[depth][unitId].length; ++inputId) {
					result += oldNablaW[depth][unitId][inputId] * newNablaW[depth][unitId][inputId];
				}
			}
		}
		return result;
	}
	private void rescaleExplodedUnits() {
		for (int depth = 0; depth + 1 < weights.length; ++depth) {
			for (int unitId = 0; unitId < weights[depth].length; ++unitId) {
				double l2norm = biases[depth][unitId] * biases[depth][unitId];
				for (int inputId = 0; inputId < weights[depth][unitId].length; ++inputId) {
					l2norm += weights[depth][unitId][inputId] * weights[depth][unitId][inputId];
				}
				double scale = 3. / Math.sqrt(l2norm);
				if (scale < 1. - bucketizer.EPSILON) {
					biases[depth][unitId] *= scale;
					for (int inputId = 0; inputId < weights[depth][unitId].length; ++inputId) {
						weights[depth][unitId][inputId] *= scale;
					}
				}
			}
		}
	}

	private double softMaxDenominator(double[] input) {
		int lastLayer = weights.length - 1;
		double expSum = 0.;
		for (int unitId = 0; unitId < weights[lastLayer].length; ++unitId) {
			expSum += Math.exp(biases[lastLayer][unitId] + weights[lastLayer][unitId][0] * input[unitId]);
		}
		return expSum;
	}
	private double softMaxNumerator(double[] input, int unitId) {
		int lastLayer = weights.length - 1;
		return Math.exp(biases[lastLayer][unitId] + weights[lastLayer][unitId][0] * input[unitId]);
	}

	private double innerProduct(int size, double[] weights, double[] input) {
		double result = 0.;
		for (int id = 0; id < size; ++id)
			result += weights[id] * input[id];
		return result;
	}
	private double relu(double input) {
		return input > 0. ? input : 0.;
	}
	private double reluDerivative(double activation) {
		return activation > 0. ? 1. : 0.;
	}
	private void shuffleArray(int[] array)
	{
		int index;
		for (int id = array.length - 1; id > 0; id--)
		{
			index = random.nextInt(id + 1);
			if (index != id)
			{
				array[index] ^= array[id];
				array[id] ^= array[index];
				array[index] ^= array[id];
			}
		}
	}

	public static NeuralNetwork readFrom(String filename) throws IOException, ClassNotFoundException {
		ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filename));
		NeuralNetwork net = (NeuralNetwork) ois.readObject();
		ois.close();

		return net;
	}

	public void writeTo(String filename) throws IOException {
		ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filename));
		oos.writeObject(this);
		oos.flush();
		oos.close();
	}
}
