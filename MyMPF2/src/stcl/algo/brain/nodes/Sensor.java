package stcl.algo.brain.nodes;

import org.ejml.simple.SimpleMatrix;

public class Sensor extends Node {
	private SimpleMatrix input;
	
	public Sensor(int id) {
		super(id);
	}

	public Sensor(int id, Node parent) {
		super(id, parent);
	}

	@Override
	public void feedforward() {
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
