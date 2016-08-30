package net.javann.core;

import java.io.*;

import java.util.ArrayList;
import java.util.DoubleSummaryStatistics;

public class TrainingData implements Serializable, Cloneable {
	protected double[][] features_;
	protected double[][] labels_;

	public TrainingData(double[][] features, double[][] labels) {
		this.features_ = new double[features.length][];
		this.labels_ = new double[labels.length][];
		for (int id = 0; id < features.length; ++id) {
			this.features_[id] = new double [features[id].length];
			this.labels_[id] = new double [labels[id].length];
			System.arraycopy(features[id], 0, this.features_[id], 0, features[id].length);
			System.arraycopy(labels[id], 0, this.labels_[id], 0, labels[id].length);
		}
	}

	public TrainingData(String fileName, String separator, int numLabels) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(fileName));
		String line;
		ArrayList<double[]> parsedData = new ArrayList<>();
		int blobsInLine = 0;
		for (int id = 0; (line = reader.readLine()) != null; ++id) {
			String[] blobs = line.split(separator);
			if (blobsInLine > 0 && blobsInLine != blobs.length) {
				throw new IOException("Unequal number of features in training examples: " + blobsInLine + "and " + blobs.length);
			}
			blobsInLine = blobs.length;
			parsedData.add(new double[blobsInLine]);
			for (int blobId = 0; blobId < blobs.length; ++blobId) {
				try {
					parsedData.get(id)[blobId] = Double.parseDouble(blobs[blobId]);
				} catch (NumberFormatException ex) {
					parsedData.get(id)[blobId] = blobs[blobId].hashCode();
				}
				if (Double.isNaN(parsedData.get(id)[blobId])) {
					parsedData.get(id)[blobId] = 0.;
				}
			}
		}
		features_ = new double[parsedData.size()][blobsInLine - numLabels];
		labels_ = new double[parsedData.size()][numLabels];
		for (int id = 0; id < parsedData.size(); ++id) {
			for (int blobId = 0; blobId < blobsInLine; ++blobId) {
				if (blobId < numLabels) {
					labels_[id][blobId] = parsedData.get(id)[blobId];
				} else {
					features_[id][blobId - numLabels] = parsedData.get(id)[blobId];
				}
			}
		}
	}

	public double[][] getFeatures() {
		return features_;
	}
	public double[][] getLabels() {
		return labels_;
	}

	public static TrainingData readFromCSV(String fileName, String separator, int numLabels) throws IOException {
		return new TrainingData(fileName, separator, numLabels);
	}
	public static TrainingData readFromCSV(String fileName) throws IOException {
		return readFromCSV(fileName, ",", 1);
	}

	public static TrainingData readBinary(String filename) throws IOException, ClassNotFoundException {
		ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filename));
		TrainingData net = (TrainingData) ois.readObject();
		ois.close();
		return net;
	}

	public void saveBinary(String filename) throws IOException {
		ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filename));
		oos.writeObject(this);
		oos.flush();
		oos.close();
	}

	public TrainingData clone() {
		return new TrainingData(features_, labels_);
	}
}
