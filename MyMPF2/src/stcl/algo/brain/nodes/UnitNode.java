package stcl.algo.brain.nodes;

import java.util.ArrayList;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.NeoCorticalUnit;
import stcl.algo.util.Normalizer;

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
		feedforwardOutputVectorLength = unit.getTemporalMapSize() * unit.getTemporalMapSize();
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
			SimpleMatrix feedforwardOutput = n.getFeedforwardOutput();
			combinedInput.insertIntoThis(0, currentStartCol, feedforwardOutput);
			currentStartCol += n.getFeedforwardOutputVectorLength();
		}
		return combinedInput;
	}

	@Override
	public void feedback() {
		SimpleMatrix inputMatrix = null;
		if (parent != null){
			SimpleMatrix inputVector = parent.getFeedbackOutputForChild(id);
			inputMatrix = new SimpleMatrix(inputVector);
			inputMatrix.reshape(temporalMapSize, temporalMapSize);
		} else {
			inputMatrix = new SimpleMatrix(temporalMapSize, temporalMapSize);
			inputMatrix.set(1);
			inputMatrix = Normalizer.normalize(inputMatrix);
		}
		
		feedbackOutput = unit.feedBackward(inputMatrix);
		needHelp = unit.needHelp();
	}
	
	public NeoCorticalUnit getUnit(){
		return unit;
	}

}
