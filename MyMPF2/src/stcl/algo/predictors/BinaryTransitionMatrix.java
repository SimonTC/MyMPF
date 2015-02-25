package stcl.algo.predictors;

import java.util.LinkedList;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.util.Normalizer;

/**
 * This class is used by the predictors to keep track of transitions.
 * The binary trasition matrix only work with binary transitions, and only when there is one significant transition pr. tick
 * @author Simon
 *
 */
public class BinaryTransitionMatrix {
	
	private SimpleMatrix transitionCount;
	private SimpleMatrix transitionProbabilityMatrix;
	private LinkedList<Transition> transitions;
	private int memorySize;

	public BinaryTransitionMatrix(int inputMatrixSize, int memorySize) {
		transitionCount = new SimpleMatrix(inputMatrixSize, inputMatrixSize);
		transitionProbabilityMatrix = new SimpleMatrix(inputMatrixSize, inputMatrixSize);
		transitions = new LinkedList<>();
		this.memorySize = memorySize;
	}
	
	/**
	 * Adds the given transition to the short term memory if the value is > 0
	 * @param fromState
	 * @param toState
	 * @param value
	 */
	public void addTransition(int fromState, int toState, double value){
		if (value > 0){
			if (transitions.size() >= memorySize){
				removeFirstTransition();			
			}
			
			addNewTransition(fromState, toState, value);	
		}
	}
	
	private Transition removeFirstTransition(){
		Transition oldTransition = transitions.removeFirst();
		int toState = oldTransition.getToState();
		int fromState = oldTransition.getFromState();
		double value = transitionCount.get(toState, fromState);
		value-= oldTransition.value;
		transitionCount.set(toState, fromState, value);
		return oldTransition;
	}
	
	private Transition addNewTransition(int fromState, int toState, double value){
		Transition newTransition = new Transition(toState, fromState, value);
		transitions.addLast(newTransition);
		double oldCountValue = transitionCount.get(toState, fromState);
		double newCountValue = oldCountValue +  newTransition.getValue();
		transitionCount.set(toState, fromState, newCountValue);
		return newTransition;
	}
	
	/**
	 * Updates the columns of the TransitionProbabilityMatrix where there has been changes
	 * @param removedTransition
	 * @param newTransition
	 */
	public void updateTransitionProbabilityMatrix(double learningRate){
		SimpleMatrix delta = new SimpleMatrix(transitionCount);
		Normalizer.normalizeColumns(delta);
		for (int i = 0; i < delta.getNumElements(); i++){
			double d = delta.get(i) - transitionProbabilityMatrix.get(i);
			if (d < 0) d = 0;
			delta.set(i, d);
		}
		//delta = delta.minus(transitionProbability);
		delta = delta.scale(learningRate);
		transitionProbabilityMatrix = transitionProbabilityMatrix.plus(delta);
		Normalizer.normalizeColumns(transitionProbabilityMatrix);
	}
	

	
	public SimpleMatrix getTransitionProbabilityMatrix(){
		return transitionProbabilityMatrix;
	}
	
	private class Transition{
		private int fromState, toState;
		private double value;
		
		public Transition(int toState, int fromState){
			this(fromState, toState, 1);
		}
		
		public Transition(int toState, int fromState, double value){
			this.fromState = fromState;
			this.toState = toState;
			this.value = value;
		}
		
		public int getFromState(){
			return fromState;
		}
		
		public int getToState(){
			return toState;
		}
		
		public double getValue(){
			return value;
		}
		
	}

	public SimpleMatrix extractVector(boolean extractRow , int element) {
		SimpleMatrix vector = transitionProbabilityMatrix.extractVector(extractRow, element);
		return vector;
	}
	
	public void decayProbabilityMatrix(double decay){
		transitionProbabilityMatrix = transitionProbabilityMatrix.scale(decay);
		Normalizer.normalizeColumns(transitionProbabilityMatrix);
	}

}
