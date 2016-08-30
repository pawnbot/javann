package net.javann.core;

import java.io.Serializable;

import java.util.HashSet;
import java.util.ArrayList;
import java.util.Collections;
import net.javann.util.ValueLabel;

public class Bucketizer implements Serializable {

	// serialver for backwards compatibility
	private static final long serialVersionUID = 2L;
	public static final double EPSILON = 1.E-9;

	private int numFeatures;
	private int numLabels;
	private ArrayList<ArrayList<Double>> bucketBoarders_;
	private double[] labelMinValue_;
	private double[] labelMaxValue_;
	private double maxCumErrorRatio_;
	private int minExamples_;
	private int discreteFeatureLimit_;
	private boolean initialized_;
	private boolean loggingEnabled_;
	HashSet<Integer> featureIdsWithOneHotEncoding_;

	public Bucketizer(double maxCumErrorRatio, int minExamples, int discreteFeatureLimit) {
		this.maxCumErrorRatio_ = maxCumErrorRatio;
		this.minExamples_ = minExamples;
		this.discreteFeatureLimit_ = discreteFeatureLimit;
		this.numFeatures = 0;
		this.numLabels = 0;
		this.bucketBoarders_ = null;
		this.labelMinValue_ = null;
		this.labelMaxValue_ = null;
		this.initialized_ = false;
		this.loggingEnabled_ = false;
		this.featureIdsWithOneHotEncoding_ = new HashSet<>();
	}

	public Bucketizer() {
		this(0.001, 10, 20);
	}

	public void feedData(TrainingData data) {
		feedData(data.getFeatures(), data.getLabels());
	}

	public void feedData(double[][] data, double[][] labels) {
		int numExamples = Math.min(data.length, labels.length);
		numFeatures = data[0].length;
		numLabels = labels[0].length;
		bucketBoarders_ = new ArrayList<>();
		for (int featureId = 0; featureId < numFeatures; ++featureId) {
			bucketBoarders_.add(new ArrayList<Double>());
		}

		labelMinValue_ = new double[numLabels];
		labelMaxValue_ = new double[numLabels];
		double[] labelMeanValue_ = new double[numLabels];
		for (int id = 0; id < numExamples; ++id) {
			for (int labelId = 0; labelId < numLabels; ++labelId) {
				labelMeanValue_[labelId] += labels[id][labelId];
				if (id > 0) {
					labelMinValue_[labelId] = Math.min(labelMinValue_[labelId], labels[id][labelId]);
					labelMaxValue_[labelId] = Math.max(labelMaxValue_[labelId], labels[id][labelId]);
				} else {
					labelMinValue_[labelId] = labels[id][labelId];
					labelMaxValue_[labelId] = labels[id][labelId];
				}
			}
		}
		for (int labelId = 0; labelId < numLabels; ++labelId) {
			labelMeanValue_[labelId] /= numExamples;
		}
		if (loggingEnabled_) {
			for (int labelId = 0; labelId < numLabels; ++labelId) {
				System.out.println("Label: " + labelId + ", MIN: " + labelMinValue_[labelId] + ", MAX: " + labelMaxValue_[labelId] + ", MEAN: " + labelMeanValue_[labelId]);
			}
		}

		double cumError = 0.;
		for (int id = 0; id < numExamples; ++id) {
			for (int labelId = 0; labelId < numLabels; ++labelId) {
				cumError += Math.pow(labels[id][labelId] - labelMeanValue_[labelId], 2);
			}
		}

		for (int featureId = 0; featureId < numFeatures; ++featureId) {
			ArrayList<ValueLabel> values = new ArrayList<>();
			for (int id = 0; id < numExamples; ++id) {
				values.add(new ValueLabel(data[id][featureId], labels[id]));
			}
			Collections.sort(values);

			int countInBucket = 1;
			for (int id = 1; id < values.size(); ++id) {
				countInBucket++;
				if (values.get(id - 1).value == values.get(id).value)
					continue;
				if (Math.abs(values.get(id - 1).value - values.get(id).value) < EPSILON)
					continue;
				if (countInBucket <= minExamples_)
					continue;
				bucketBoarders_.get(featureId).add(0.5 * (values.get(id - 1).value + values.get(id).value));
				countInBucket = 1;
				if (bucketBoarders_.get(featureId).size() > discreteFeatureLimit_) {
					bucketBoarders_.get(featureId).clear();
					break;
				}
			}
			if (!bucketBoarders_.get(featureId).isEmpty() || countInBucket == values.size()) {
				if (loggingEnabled_) {
					System.out.println("Discrete FeatureId: " + featureId + ", Buckets: " + (1 + bucketBoarders_.get(featureId).size()));
				}
				continue;
			}

			double labelError = 0.;
			countInBucket = 0;
			for (int id = 0; id < values.size(); ++id) {
				double errorDiff = 0.;
				for (int labelId = 0; labelId < numLabels; ++labelId) {
					errorDiff += Math.pow(values.get(id).label[labelId] - labelMeanValue_[labelId], 2);
				}
				countInBucket++;
				boolean extendBucket = labelError + errorDiff < maxCumErrorRatio_ * cumError;
				extendBucket &= countInBucket < maxCumErrorRatio_ * numExamples;
				extendBucket |= countInBucket <= minExamples_;
				if (id > 0) {
					extendBucket |= Math.abs(values.get(id - 1).value - values.get(id).value) < EPSILON;
					extendBucket |= values.get(id - 1).value == values.get(id).value;
					extendBucket &= Math.signum(values.get(id).value) == Math.signum(values.get(id - 1).value);
				}
				if (extendBucket) {
					labelError += errorDiff;
				} else {
					countInBucket = 1;
					labelError = errorDiff;
					bucketBoarders_.get(featureId).add(0.5 * (values.get(id - 1).value + values.get(id).value));
				}
			}
			if (loggingEnabled_) {
				System.out.println("Continuous FeatureId: " + featureId + ", Buckets: " + (1 + bucketBoarders_.get(featureId).size()));
			}
		}
		initialized_ = true;
	}

	public double[] labelsToOutput(double[] labels) {
		double[] outputLayer = new double[labels.length];
		for (int labelId = 0; labelId < labels.length; ++labelId) {
			if (!initialized_) {
				outputLayer[labelId] = labels[labelId];
			} else {
				double range = labelMaxValue_[labelId] - labelMinValue_[labelId];
				outputLayer[labelId] = (labels[labelId] - labelMinValue_[labelId]) / range;
			}
		}
		return outputLayer;
	}

	public double[] outputToPredictions(double[] outputLayer) {
		double[] predictions = new double[labelMinValue_.length];
		for (int labelId = 0; labelId < labelMinValue_.length; ++labelId) {
			if (!initialized_) {
				predictions[labelId] = outputLayer[labelId];
			} else {
				double range = labelMaxValue_[labelId] - labelMinValue_[labelId];
				predictions[labelId] = labelMinValue_[labelId] + outputLayer[labelId] * range;
			}
		}
		return predictions;
	}

	public double[] exampleToInput(double[] example) {
		double[] result = new double[getSizeWithOneHotEncoding()];
		int offset = 0;
		for (int featureId = 0; featureId < numFeatures; ++featureId) {
			if (bucketBoarders_.get(featureId).isEmpty()) {
				result[offset++] = 0.;
				continue;
			}
			if (featureIdsWithOneHotEncoding_.contains(featureId)) {
				int bucketId = findBucketId(bucketBoarders_.get(featureId), example[featureId]);
				result[offset + bucketId] = 1.;
				offset += bucketBoarders_.get(featureId).size() + 1;
			} else {
				int bucketId = findBucketId(bucketBoarders_.get(featureId), example[featureId]);
				result[offset++] = 2. * bucketId / bucketBoarders_.get(featureId).size() - 1.;
			}
		}
		return result;
	}

	public void disableLogging() {
		loggingEnabled_ = false;
	}
	public void enableLogging() {
		loggingEnabled_ = true;
	}
	public boolean isLoggingEnabled() {
		return loggingEnabled_;
	}
	public boolean isInitialized() {
		return initialized_;
	}
	public void enableOneHotUnderLimit(int oneHotEncodingLimit) {
		for (int featureId = 0; featureId < numFeatures; ++featureId) {
			if (bucketBoarders_.get(featureId).size() + 1 <= oneHotEncodingLimit) {
				featureIdsWithOneHotEncoding_.add(featureId);
			}
		}
	}
	public void forceOneHotEncoding(int featureId) {
		featureIdsWithOneHotEncoding_.add(featureId);
	}
	public int getSizeWithOneHotEncoding() {
		int cumSize = 0;
		for (int featureId = 0; featureId < numFeatures; ++featureId) {
			if (featureIdsWithOneHotEncoding_.contains(featureId)) {
				cumSize += bucketBoarders_.get(featureId).size() + 1;
			} else {
				cumSize++;
			}
		}
		return cumSize;
	}
	public int getNumBuckets(int featureId) {
		return bucketBoarders_.get(featureId).size() + 1;
	}
	private int findBucketId(ArrayList<Double> boarders, double featureValue) {
		int low = 0;
		int high = boarders.size() - 1;
		int bucketId = boarders.size();
		while (low <= high) {
			int mid = low + (high - low) / 2;
			if (featureValue < boarders.get(mid)) {
				bucketId = mid;
				high = mid - 1;
			} else {
				low = mid + 1;
			}
		}
		return bucketId;
	}
}
