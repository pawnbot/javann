package net.javann.core;

import java.io.Serializable;

import java.util.ArrayList;
import java.util.Collections;
import net.javann.util.ValueLabel;

public class Bucketizer implements Serializable {

	// serialver for backwards compatibility
	private static final long serialVersionUID = 1L;
	public static final double EPSILON = 1.E-9;

	private ArrayList<ArrayList<Double>> bucketBoarders_;
	private double[] labelMinValue_;
	private double[] labelMaxValue_;
	private double maxCumErrorRatio_;
	private int minExamples_;
	private boolean initilized_;
	private boolean loggingEnabled_;

	public Bucketizer(double maxCumErrorRatio, int minExamples) {
		this.bucketBoarders_ = null;
		this.labelMinValue_ = null;
		this.labelMaxValue_ = null;
		this.maxCumErrorRatio_ = maxCumErrorRatio;
		this.minExamples_ = minExamples;
		this.initilized_ = false;
		this.loggingEnabled_ = false;
	}

	public Bucketizer() {
		this.bucketBoarders_ = null;
		this.labelMinValue_ = null;
		this.labelMaxValue_ = null;
		this.maxCumErrorRatio_ = 0.001;
		this.minExamples_ = 10;
		this.initilized_ = false;
		this.loggingEnabled_ = false;
	}

	public void feedData(double[][] data, double[][] labels) {
		bucketBoarders_ = new ArrayList<>();
		for (int featureId = 0; featureId < data[0].length; ++featureId) {
			bucketBoarders_.add(new ArrayList<Double>());
		}

		labelMinValue_ = new double[labels[0].length];
		labelMaxValue_ = new double[labels[0].length];
		for (int id = 0; id < data.length; ++id) {
			for (int labelId = 0; labelId < labels[0].length; ++labelId) {
				if (id > 0) {
					labelMinValue_[labelId] = Math.min(labelMinValue_[labelId], labels[id][labelId]);
					labelMaxValue_[labelId] = Math.max(labelMaxValue_[labelId], labels[id][labelId]);
				} else {
					labelMinValue_[labelId] = labels[id][labelId];
					labelMaxValue_[labelId] = labels[id][labelId];
				}
			}
		}

		for (int featureId = 0; featureId < data[0].length; ++featureId) {
			ArrayList<ValueLabel> values = new ArrayList<>();

			double mean = 0.;
			for (int id = 0; id < data.length; ++id) {
				double combinedLabel = 0.;
				for (int out = 0; out < labels[id].length; ++out) {
					combinedLabel += (1. + out) * labels[id][out];
				}
				values.add(new ValueLabel(data[id][featureId], combinedLabel));
				mean += combinedLabel;
			}
			mean /= data.length;
			if (featureId == 0 && loggingEnabled_) {
				System.out.println("Average label: " + mean);
				for (int labelId = 0; labelId < labels[0].length; ++labelId) {
					System.out.println("Label: " + labelId + ", MIN: " + labelMinValue_[labelId] + ", MAX: " + labelMaxValue_[labelId]);
				}
			}
			double cumError = 0.;
			for (int id = 0; id < data.length; ++id) {
				cumError += (values.get(id).label - mean) * (values.get(id).label - mean);
			}
			Collections.sort(values);

			double valueMean = 0.;
			double valueError = 0.;
			double labelMean = 0.;
			double labelError = 0.;
			int count = 0;
			for (int id = 0; id < values.size(); ++id) {
				double newValueMean = (valueMean * count + values.get(id).value) / (1. + count);
				double newValueError = valueError + (newValueMean - values.get(id).value) * (newValueMean - values.get(id).value);

				double newLabelMean = (labelMean * count + values.get(id).label) / (1. + count);
				double newLabelError = labelError + (newLabelMean - values.get(id).label) * (newLabelMean - values.get(id).label);

				boolean extendBucket = count < minExamples_;
				extendBucket |= (newValueError - valueError) < EPSILON;
				extendBucket |= newLabelError < (maxCumErrorRatio_ * cumError);
				if (extendBucket) {
					valueMean = newValueMean;
					valueError = newValueError;
					labelMean = newLabelMean;
					labelError = newLabelError;
					count++;
				} else {
					bucketBoarders_.get(featureId).add(0.5 * (values.get(id - 1).value + values.get(id).value));
					valueMean = values.get(id).value;
					valueError = 0.;
					labelMean = values.get(id).label;
					labelError = 0.;
					count = 1;
				}
			}
			if (loggingEnabled_) {
				System.out.println("FeatureId: " + featureId + ", Buckets: " + (1 + bucketBoarders_.get(featureId).size()));
			}
		}
		initilized_ = true;
	}

	public double[] exampleToInput(double[] example) {
		double[] inputLayer = new double[example.length];
		for (int featureId = 0; featureId < example.length; ++featureId) {
			if (!initilized_) {
				inputLayer[featureId] = example[featureId];
			} else {
				int bucketId = findBucketId(bucketBoarders_.get(featureId), example[featureId]);
				inputLayer[featureId] = 2. * bucketId / (1. + bucketBoarders_.get(featureId).size()) - 1.;
			}
		}
		return inputLayer;
	}

	public double[] labelsToOutput(double[] labels) {
		double[] outputLayer = new double[labels.length];
		for (int labelId = 0; labelId < labels.length; ++labelId) {
			if (!initilized_) {
				outputLayer[labelId] = labels[labelId];
			} else {
				outputLayer[labelId] = labels[labelId] - labelMinValue_[labelId];
				outputLayer[labelId] /= (labelMaxValue_[labelId] - labelMinValue_[labelId]);
			}
		}
		return outputLayer;
	}

	public double[] outputToPredictions(double[] outputLayer) {
		double[] predictions = new double[outputLayer.length];
		for (int labelId = 0; labelId < outputLayer.length; ++labelId) {
			if (!initilized_) {
				predictions[labelId] = outputLayer[labelId];
			} else {
				predictions[labelId] = labelMinValue_[labelId];
				predictions[labelId] += outputLayer[labelId] * (labelMaxValue_[labelId] - labelMinValue_[labelId]);
			}
		}
		return predictions;
	}

	public double[] genOneHotEncoding(double[] features, int featurePos) {
		if (featurePos < 0) {
			int cumSize = 0;
			for (int featureId = 0; featureId < features.length; ++featureId) {
				cumSize += bucketBoarders_.size() + 1;
			}
			double[] result = new double[cumSize];
			int offset = 0;
			for (int featureId = 0; featureId < features.length; ++featureId) {
				int bucketId = findBucketId(bucketBoarders_.get(featureId), features[featureId]);
				result[offset + bucketId] = 1.;
				offset += 1. + bucketBoarders_.get(featureId).size();
			}
			return result;
		} else {
			double[] result = new double[bucketBoarders_.get(featurePos).size() + 1];
			int bucketId = findBucketId(bucketBoarders_.get(featurePos), features[featurePos]);
			result[bucketId] = 1.;
			return result;
		}
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
	public boolean isInitilized() {
		return initilized_;
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
