package stcl.algo.brain.nodes;

import java.util.ArrayList;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.NeoCorticalUnit;

public class UnitNode extends Node {
	
	private NeoCorticalUnit unit;
	private int temporalMapSize;

	public UnitNode(int id) {
		super(id);
	}

	public UnitNode(int id, Node parent) {
		super(id, parent);
	}

	public UnitNode(int id, Node parent, ArrayList<Node> children) {
		super(id, parent, children);
	}
	
	public void initializeUnit(Random rand, int ffInputLength, int spatialMapSize, int temporalMapSize, double initialPredictionLearningRate, boolean useMarkovPrediction, int markovOrder, boolean noTemporal){
		unit = new NeoCorticalUnit(rand, ffInputLength, spatialMapSize, temporalMapSize, initialPredictionLearningRate, useMarkovPrediction, markovOrder, noTemporal);
		this.temporalMapSize = temporalMapSize;
	}

	@Override
	public void feedforward(double reward) {
		if (childrenNeedHelp()){ 
			SimpleMatrix inputVector = collectInput();
			SimpleMatrix outputMatrix = unit.feedForward(inputVector);
			feedforwardOutput = new SimpleMatrix(outputMatrix);
			feedforwardOutput.reshape(1, outputMatrix.getNumElements());
			this.needHelp = unit.needHelp();
		}
	}
	
	private boolean childrenNeedHelp(){
		for (Node n : children){
			if (n.needHelp()) return true;
		}
		return false;
	}
	
	/**
	 * Combines the output from all the children into one vector
	 * @return
	 */
	private SimpleMatrix collectInput(){
		SimpleMatrix combinedInput = new SimpleMatrix(1, feedforwardInputLength);
		int currentStartCol = 0;
		for (Node n : children){
			combinedInput.insertIntoThis(0, currentStartCol, n.getFeedforwardOutput());
			currentStartCol += n.getFeedforwardOutputVectorLength();
		}
		return combinedInput;
	}

	@Override
	public void feedback() {
		SimpleMatrix inputVector = parent.getFeedbackOutputForChild(id);
		SimpleMatrix inputMatrix = new SimpleMatrix(inputVector);
		inputMatrix.reshape(temporalMapSize, temporalMapSize);
		feedbackOutput = unit.feedBackward(inputMatrix);
		needHelp = unit.needHelp();
	}
	
	public NeoCorticalUnit getUnit(){
		return unit;
	}

}
