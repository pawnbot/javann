package net.javann.util;


public class ValueLabel implements Comparable<ValueLabel> {
	public double value;
	public double[] label;

	public ValueLabel(double value, double label) {
		this.value = value;
		this.label = new double[1];
		this.label[0] = label;
	}
	public ValueLabel(double value, double[] label) {
		this.value = value;
		this.label = label;
	}
	@Override
	public int compareTo(ValueLabel other) {
		return Double.compare(this.value, other.value);
	}
}
