package stcl.algo.brain.nodes;

import org.ejml.simple.SimpleMatrix;

public class Sensor extends Node {
	private SimpleMatrix input;
	
	public Sensor(int id, int inputLength) {
		this(id, inputLength, null);
	}

	public Sensor(int id, int inputLength, Node parent) {
		super(id, parent);
		feedforwardInputLength = inputLength;
	}

	@Override
	public void feedforward(double reward) {
		this.feedforwardOutput = input;
	}

	@Override
	public void feedback() {
		feedbackOutput = parent.getFeedbackOutputForChild(this.id);
	}
	
	public void setInput(double input){
		double[][] data = {{input}};
		SimpleMatrix m = new SimpleMatrix(data);
		this.setInput(m);
	}
	
	public void setInput(SimpleMatrix input){
		this.input = input;
	}

}
