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
		feedforwardOutputVectorLength = inputLength;
		needHelp = true;
		this.type = NodeType.SENSOR;
	}

	@Override
	public void feedforward(double reward, int actionPerformed) {
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
		if (input.numCols() != feedforwardInputLength || input.numRows() != 1) throw new IllegalArgumentException("The input vector has to have a length of " + feedforwardInputLength);
		this.input = input;
	}

}
