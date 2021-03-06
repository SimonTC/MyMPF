package stcl.algo.brain.nodes;

import org.ejml.simple.SimpleMatrix;

public class Sensor extends Node {
	private SimpleMatrix input;

	public Sensor(int id, int x, int y, int z){
		super(id, x, y, z);
		this.type = NodeType.SENSOR;
	}
	
	public void initialize(int inputLength){
		feedforwardInputLength = inputLength;
		feedforwardOutputVectorLength = inputLength;
		needHelp = true;
	}
	
	public Sensor(String s){
		super(s);
		needHelp = true;
		this.type = NodeType.SENSOR;
		String[] arr = s.split(" ");
		feedforwardInputLength = Integer.parseInt(arr[arr.length-1]);
		feedforwardOutputVectorLength = feedforwardInputLength;
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
	
	@Override
	public String toString(){
		String s = super.toString();
		s += " " + feedforwardInputLength;
		return s;
	}
	
	public void setInput(SimpleMatrix input){
		if (input.numCols() != feedforwardInputLength || input.numRows() != 1) throw new IllegalArgumentException("The input vector has to have a length of " + feedforwardInputLength);
		this.input = input;
	}

	@Override
	public void reinitialize() {
		if (input != null) input.set(0);
		
	}

	@Override
	public void reinitialize(String initializationString) {
		this.reinitialize();
		
	}

	@Override
	public void setLearning(boolean learning) {
		//No learning is performed in sensor, so change doesn't matter		
	}

}
