package net.javann.core;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

import net.javann.util.ValueLabel;
import net.javann.util.LossFunction;
import net.javann.util.MetricFunction;

public class NeuralNetwork implements Serializable {

	// serialver for backwards compatibility
	private static final long serialVersionUID = 1L;
	// the random number generator
	private final Random random = new Random();

	private double learningRate;
	private double l1Multiplier;
	private double l2Multiplier;
	private double activeUnitsRatio;
	private int miniBatchSize;
	LossFunction lossFunction;
	MetricFunction[] loggedMetrics;
	boolean useSoftMax;

	private double[][][] weights;
	private double[][] biases;
	private double[][] activations;

	private Bucketizer bucketizer;

	private double[][][] tmpNablaW;
	private double[][] tmpNablaB;


	/**
	 * Builds a neural network with the given number of input units, hidden
	 * units, and output units.
	 * 
	 * @param structure
	 *            Structure of NN
	 */
	public NeuralNetwork(int[] structure) {
		learningRate = 1.E-2;
		l1Multiplier = 1.E-7;
		l2Multiplier = 1.E-7;
		activeUnitsRatio = 1.;
		miniBatchSize = 100;
		lossFunction = LossFunction.SQUARED_ERROR;
		loggedMetrics = null;
		int lastLayer = structure.length - 1;
		useSoftMax = (structure[lastLayer] > 1) && structure[lastLayer] == structure[lastLayer - 1];

		weights = new double[structure.length - 1][][];
		biases = new double[structure.length - 1][];
		activations = new double[weights.length][];
		for (int depth = 0; depth < weights.length; ++depth) {
			weights[depth] = new double [structure[1 + depth]][];
			biases[depth] = new double [structure[1 + depth]];
			activations[depth] = new double[structure[1 + depth]];
			for (int unitId = 0; unitId < weights[depth].length; ++unitId) {
				weights[depth][unitId] = new double [structure[depth]];
			}
		}
		randomizeWeights();

		bucketizer = new Bucketizer();

		tmpNablaW = null;
		tmpNablaB = null;
	}

	private double[] feedForward(double[] inputLayer, double activeRatio) {
		for (int depth = 0; depth < weights.length; ++depth) {
			for (int unitId = 0; unitId < weights[depth].length; ++unitId) {
				if (depth < weights.length - 1) {
					double activeRatioForLayer = 1.;
					if (activeRatio < 1.0 && weights[depth].length > 10) {
						activeRatioForLayer = activeRatio;
					}
					if (activeRatioForLayer < 1. && activeRatioForLayer < random.nextDouble()) {
						activations[depth][unitId] = 0.;
						continue;
					}
					// Intermediate layers, using ReLu
					double dotProduct = biases[depth][unitId];
					if (depth > 0) {
						dotProduct += innerProduct(weights[depth][unitId].length, weights[depth][unitId], activations[depth - 1]);
					} else {
						dotProduct += innerProduct(weights[depth][unitId].length, weights[depth][unitId], inputLayer);
					}
					activations[depth][unitId] = relu(dotProduct) / activeRatioForLayer;
				} else {
					if (useSoftMax) {
						activations[depth][unitId] = softMax(biases[depth][unitId], weights[depth][unitId], activations[depth - 1], unitId);
					} else {
						double dotProduct = biases[depth][unitId] + innerProduct(weights[depth][unitId].length, weights[depth][unitId], activations[depth - 1]);
						activations[depth][unitId] = sigmoid(dotProduct);
					}
				}
			}
		}
		return activations[activations.length - 1];
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

		feedForward(inputLayer, activeUnitsRatio);
		int lastLayer = weights.length - 1;
		int numOutputs = weights[lastLayer].length;
		for (int unitId = 0; unitId < numOutputs; ++unitId) {
			double derivative;
			if (useSoftMax) {
				derivative = softMaxDerivative(biases[lastLayer][unitId], weights[lastLayer][unitId], activations[lastLayer - 1], unitId, unitId, activations[lastLayer][unitId]);
			} else {
				derivative = sigmoidDerivative(activations[lastLayer][unitId]);
			}
			double errorDerivative = 0.;
			if (lossFunction == LossFunction.SQUARED_ERROR) {
				errorDerivative = 2. * (activations[lastLayer][unitId] - expectedOutputLayer[unitId]);
			}
			if (lossFunction == LossFunction.LOG_LOSS) {
				if (expectedOutputLayer[unitId] < 0.5) {
					errorDerivative = 1. / (1. - activations[lastLayer][unitId]);
				} else {
					errorDerivative = -1. / activations[lastLayer][unitId];
				}
			}
			if (lossFunction == LossFunction.REINFORCEMENT_LEARNING) {
				errorDerivative = -expectedOutputLayer[unitId];
			}
			nablaB[lastLayer][unitId] = errorDerivative * derivative;
			// L2 regularization
			nablaB[lastLayer][unitId] += 2 * biases[lastLayer][unitId] * l2Multiplier;
			// L1 regularization
			nablaB[lastLayer][unitId] += biases[lastLayer][unitId] > 0. ? l1Multiplier : -l1Multiplier;
			for (int inputId = 0; inputId < weights[lastLayer][unitId].length; ++inputId) {
				if (useSoftMax) {
					derivative = softMaxDerivative(biases[lastLayer][unitId], weights[lastLayer][unitId], activations[lastLayer - 1], unitId, inputId, activations[lastLayer][unitId]);
				}
				nablaW[lastLayer][unitId][inputId] = errorDerivative * derivative * activations[lastLayer - 1][inputId];
				// L2 regularization
				nablaW[lastLayer][unitId][inputId] += 2 * weights[lastLayer][unitId][inputId] * l2Multiplier;
				// L1 regularization
				nablaW[lastLayer][unitId][inputId] += weights[lastLayer][unitId][inputId] > 0. ? l1Multiplier : -l1Multiplier;
			}
		}
		for (int depth = lastLayer - 1; depth >= 0; --depth) {
			for (int unitId = 0; unitId < weights[depth].length; ++unitId) {
				double delta = 0.;
				for (int id = 0; id < nablaB[depth + 1].length; ++id) {
					delta += nablaB[depth + 1][id] * weights[depth + 1][id][unitId];
				}
				delta *= reluDerivative(activations[depth][unitId]);
				nablaB[depth][unitId] = delta;
				// L2 regularization
				nablaB[depth][unitId] += 2 * biases[depth][unitId] * l2Multiplier;
				// L1 regularization
				nablaB[depth][unitId] += biases[depth][unitId] > 0. ? l1Multiplier : -l1Multiplier;
				for (int inputId = 0; inputId < weights[depth][unitId].length; ++inputId) {
					if (depth > 0) {
						nablaW[depth][unitId][inputId] = delta * activations[depth - 1][inputId];
					} else {
						nablaW[depth][unitId][inputId] = delta * inputLayer[inputId];
					}
					// L2 regularization
					nablaW[depth][unitId][inputId] += 2 * weights[depth][unitId][inputId] * l2Multiplier;
					// L1 regularization
					nablaW[depth][unitId][inputId] += weights[depth][unitId][inputId] > 0. ? l1Multiplier : -l1Multiplier;
				}
			}
		}

		if (tmpNablaB != null && tmpNablaW != null) {
			applyNablasToNN(nablaB, nablaW, learningRate);
		} else {
			for (int depth = 0; depth < weights.length; ++depth) {
				for (int unitId = 0; unitId < weights[depth].length; ++unitId) {
					tmpNablaB[depth][unitId] += learningRate * nablaB[depth][unitId];
					for (int inputId = 0; inputId < weights[depth][unitId].length; ++inputId) {
						tmpNablaW[depth][unitId][inputId] += learningRate * nablaW[depth][unitId][inputId];
					}
				}
			}
		}
	}

	public void feedData(double[][] data, double[][] labels, int numEpochs) {
		if (!bucketizer.isInitilized()) {
			bucketizer.feedData(data, labels);
		}
		if (loggedMetrics.length > 0) {
			logResults(-1, data, labels);
		}
		double[][] inputs = new double[data.length][];
		double[][] outputs = new double[labels.length][];
		int[] randomIndexes = new int[data.length];
		for (int id = 0; id < data.length; ++id) {
			randomIndexes[id] = id;
			inputs[id] = bucketizer.exampleToInput(data[id]);
			if (lossFunction != LossFunction.REINFORCEMENT_LEARNING) {
				outputs[id] = bucketizer.labelsToOutput(labels[id]);
			} else {
				outputs[id] = labels[id];
			}
		}
		for (int epoch = 0; epoch < numEpochs; ++epoch) {
			shuffleArray(randomIndexes);
			for (int batchOffset = 0; batchOffset < data.length; batchOffset += miniBatchSize) {
				tmpNablaB = new double[weights.length][];
				tmpNablaW = new double[weights.length][][];
				for (int depth = 0; depth < weights.length; ++depth) {
					tmpNablaB[depth] = new double[weights[depth].length];
					tmpNablaW[depth] = new double[weights[depth].length][];
					for (int unitId = 0; unitId < weights[depth].length; ++unitId) {
						tmpNablaW[depth][unitId] = new double[weights[depth][unitId].length];
					}
				}
				int batchSize = Math.min(batchOffset + miniBatchSize, data.length) - batchOffset;
				for (int id = 0; id < batchSize; ++id) {
					backPropagation(inputs[randomIndexes[batchOffset + id]], outputs[randomIndexes[batchOffset + id]]);
				}
				applyNablasToNN(tmpNablaB, tmpNablaW, 1. / batchSize);
				tmpNablaB = null;
				tmpNablaW = null;
			}
			if (loggedMetrics.length > 0) {
				logResults(epoch, data, labels);
			}
		}
	}

	public void feedData(TrainingData data, int numEpochs) {
		feedData(data.getFeatures(), data.getLabels(), numEpochs);
	}

	public double[] predictLabels(double[] example) {
		double[] activations = feedForward(bucketizer.exampleToInput(example), 1.);
		if (lossFunction != LossFunction.REINFORCEMENT_LEARNING) {
			return bucketizer.outputToPredictions(activations);
		} else {
			return activations;
		}
	}

	public double getRMSE(double[][] data, double[][] labels) {
		double rmse = 0.;
		for (int id = 0; id < data.length; ++id) {
			double[] predictions = predictLabels(data[id]);
			for (int labelId = 0; labelId < labels[id].length; ++labelId) {
				rmse += (predictions[labelId] - labels[id][labelId]) * (predictions[labelId] - labels[id][labelId]);
			}
		}
		rmse = Math.sqrt(rmse / data.length);
		return rmse;
	}

	public double getAUC(double[][] data, double[][] labels) {
		ArrayList<ValueLabel> auc = new ArrayList<>();
		for (int id = 0; id < data.length; ++id) {
			double[] predictions = predictLabels(data[id]);
			auc.add(new ValueLabel(predictions[0], labels[id][0]));
		}
		Collections.sort(auc);
		double area = 0.;
		int height = 0;
		for (int id = 0; id < auc.size(); ++id) {
			if (auc.get(id).label < 0.5) {
				height++;
			} else {
				area += height;
			}
		}
		return area / (height + Bucketizer.EPSILON) / (auc.size() - height + Bucketizer.EPSILON);
	}

	public double getCalibration(double[][] data, double[][] labels) {
		double prediction = 0.;
		double observed = 0.;
		for (int id = 0; id < data.length; ++id) {
			double[] predictions = predictLabels(data[id]);
			for (int labelId = 0; labelId < labels[id].length; ++labelId) {
				prediction += predictions[labelId];
				observed += labels[id][labelId];
			}
		}
		return prediction / (observed + Bucketizer.EPSILON);
	}

	public double getNormalizedEntropy(double[][] data, double[][] labels) {
		double ctr = 0.;
		for (int id = 0; id < labels.length; ++id) {
			ctr += labels[id][0];
		}
		ctr /= labels.length;
		double loss = 0.;
		double constantLoss = 0.;
		for (int id = 0; id < data.length; ++id) {
			double[] predictions = predictLabels(data[id]);
			for (int labelId = 0; labelId < labels[id].length; ++labelId) {
				loss += labels[id][labelId] * Math.log(predictions[labelId] + Bucketizer.EPSILON) + (1 - labels[id][labelId]) * Math.log(1. - predictions[labelId] + Bucketizer.EPSILON);
				constantLoss += labels[id][labelId] * Math.log(ctr + Bucketizer.EPSILON) + (1 - labels[id][labelId]) * Math.log(1. - ctr + Bucketizer.EPSILON);
			}
		}
		return loss / constantLoss;
	}

	public double getCumulativeReward(double[][] data, double[][] labels) {
		double totalReward = 0.;
		for (int id = 0; id < data.length; ++id) {
			double[] predictions = feedForward(bucketizer.exampleToInput(data[id]), 1.);
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

	public double getActiveUnitsRatio() {
		return activeUnitsRatio;
	}

	public void setActiveUnitsRatio(double activeUnitsRatio) {
		this.activeUnitsRatio = activeUnitsRatio;
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
		for (int depth = 0; depth < weights.length; ++depth) {
			for (int unitId = 0; unitId < weights[depth].length; ++unitId) {
				biases[depth][unitId] = 1.;
				for (int inputId = 0; inputId < weights[depth][unitId].length; ++inputId) {
					weights[depth][unitId][inputId] = (2 * random.nextDouble() - 1.) / weights[depth][unitId].length;
				}
			}
		}
	}

	private void applyNablasToNN(double[][] nablaB, double[][][] nablaW, double rate) {
		for (int depth = 0; depth < weights.length; ++depth) {
			for (int unitId = 0; unitId < weights[depth].length; ++unitId) {
				biases[depth][unitId] -= rate * nablaB[depth][unitId];
				for (int inputId = 0; inputId < weights[depth][unitId].length; ++inputId) {
					weights[depth][unitId][inputId] -= rate * nablaW[depth][unitId][inputId];
				}
			}
		}
	}

	private double innerProduct(int size, double[] weights, double[] input) {
		double result = 0.;
		for (int id = 0; id < size; ++id)
			result += weights[id] * input[id];
		return result;
	}
	private double softMax(double bias, double[] weights, double[] input, int outputId) {
		double value = Math.exp(bias + weights[outputId] * input[outputId]);
		double valueSum = 0.;
		for (int parentId = 0; parentId < weights.length; ++parentId) {
			if (parentId != outputId) {
				valueSum += Math.exp(-weights[parentId] * input[parentId]);
			} else {
				valueSum += value;
			}
		}
		return value / valueSum;
	}
	private double softMaxDerivative(double bias, double[] weights, double[] input, int outputId, int nablaId, double activation) {
		if (outputId == nablaId) {
			return activation * (1. - activation);
		} else {
			return activation * softMax(bias, weights, input, nablaId);
		}
	}
	private double sigmoid(double input) {
		return 1. / (1. + Math.exp(-input));
	}
	private double sigmoidDerivative(double activation) {
		return activation * (1. - activation);
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

	/**
	 * Method which reads and returns a network from the given file
	 *
	 * @param filename
	 *            The file to read from
	 */
	public static NeuralNetwork readFrom(String filename) throws IOException, ClassNotFoundException {
		ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filename));
		NeuralNetwork net = (NeuralNetwork) ois.readObject();
		ois.close();

		return net;
	}

	/**
	 * Method which writes this network to the given file
	 *
	 * @param filename
	 *            The file to write to
	 */
	public void writeTo(String filename) throws IOException {
		ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filename));
		oos.writeObject(this);
		oos.flush();
		oos.close();
	}
}
