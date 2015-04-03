package stcl.algo.brain.nodes;

import java.util.ArrayList;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.NeoCorticalUnit;
import stcl.algo.util.Normalizer;

public class UnitNode extends Node {
	
	private NeoCorticalUnit unit;
	private int temporalMapSize;
	protected Random rand;
	private int chosenAction;

	public UnitNode(int id) {
		super(id);
	}

	public UnitNode(int id, Node parent) {
		super(id, parent);
	}

	public UnitNode(int id, Node parent, ArrayList<Node> children) {
		super(id, parent, children);
	}
	
	/**
	 * 
	 * @param rand
	 * @param ffInputLength
	 * @param spatialMapSize
	 * @param temporalMapSize
	 * @param initialPredictionLearningRate
	 * @param useMarkovPrediction
	 * @param markovOrder
	 * @param noTemporal
	 */
	public void initializeUnit(Random rand, int ffInputLength, int spatialMapSize, int temporalMapSize, double initialPredictionLearningRate, boolean useMarkovPrediction, int markovOrder, boolean noTemporal, int numPossibleActions){
		unit = new NeoCorticalUnit(rand, ffInputLength, spatialMapSize, temporalMapSize, initialPredictionLearningRate, useMarkovPrediction, markovOrder, noTemporal, numPossibleActions);
		this.temporalMapSize = unit.getTemporalMapSize();
		feedforwardOutputVectorLength = unit.getTemporalMapSize() * unit.getTemporalMapSize();
		this.rand = rand;
	}
	
	/**
	 * Initialization method used when all children has already been added.
	 * @param rand
	 * @param spatialMapSize int > 0 
	 * @param temporalMapSize int >= 0 If 0 then now temporal pooling is used
	 * @param initialPredictionLearningRate
	 * @param markovOrder int >= 0 if zero then no prediction is performed
	 * @param numPossibleActions
	 */
	public void initializeUnit(Random rand, int spatialMapSize, int temporalMapSize, double initialPredictionLearningRate, int markovOrder, int numPossibleActions){
		int inputLength = feedforwardInputLength;
		this.initializeUnit(rand, inputLength, spatialMapSize, temporalMapSize, initialPredictionLearningRate, markovOrder > 0, markovOrder, temporalMapSize == 0, numPossibleActions);
	}

	@Override
	public void feedforward(double reward, int actionPerformed) {
		if (childrenNeedHelp()){ 
			forceHelpOnChildren();
			SimpleMatrix inputVector = collectInput();
			SimpleMatrix outputMatrix = unit.feedForward(inputVector, reward, actionPerformed);
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
	
	public void resetActivityOfUnit(){
		unit.resetActivity();
	}
	
	/**
	 * Loop through all children and force them to need help.
	 * Use to make sure that action nodes doesn't go on in a loop where they always predict the same without influence from above
	 */
	private void forceHelpOnChildren(){
		for (Node n : children){
			n.setNeedHelp(true);
		}
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
	
	@Override
	public void setNeedHelp(boolean needHelp){
		super.setNeedHelp(needHelp);
		unit.setNeedHelp(needHelp);
	}
	
	public void setChosenAction(int chosenAction){
		this.chosenAction = chosenAction;
	}
	
	public int getActionVote(){
		return unit.getNextAction();
	}


}
