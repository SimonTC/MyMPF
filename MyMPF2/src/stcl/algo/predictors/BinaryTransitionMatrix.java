package stcl.algo.predictors;

import java.util.LinkedList;

import org.ejml.simple.SimpleMatrix;

/**
 * This class is used by the predictors to keep track of transitions.
 * The binary trasition matrix only work with binary transitions, and only when there is one significant transition pr. tick
 * @author Simon
 *
 */
public class BinaryTransitionMatrix {
	
	private SimpleMatrix transitionCount;
	private SimpleMatrix transitionProbability;
	private LinkedList<Transition> transitions;
	private int memorySize;

	public BinaryTransitionMatrix(int inputMatrixSize, int memorySize) {
		transitionCount = new SimpleMatrix(inputMatrixSize, inputMatrixSize);
		transitionProbability = new SimpleMatrix(inputMatrixSize, inputMatrixSize);
		transitions = new LinkedList<>();
		this.memorySize = memorySize;
	}
	
	public void addTransition(int fromState, int toState){
		Transition removedTransition = null;
		if (transitions.size() >= memorySize){
			removedTransition = removeFirstTransition();			
		}
		
		Transition newTransition = addNewTransition(fromState, toState);
		
		updateTransitionrobabilityMatrix(removedTransition, newTransition);
		
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
	
	private Transition addNewTransition(int fromState, int toState){
		Transition newTransition = new Transition(fromState, toState);
		transitions.addLast(newTransition);
		double value = transitionCount.get(fromState, toState);
		value += newTransition.getValue();
		transitionCount.set(fromState, toState, value);
		return newTransition;
	}
	
	/**
	 * Updates the columns of the TransitionProbabilityMatrix where there has been changes
	 * @param removedTransition
	 * @param newTransition
	 */
	private void updateTransitionrobabilityMatrix(Transition removedTransition, Transition newTransition){
		int newColumn = newTransition.getFromState();
		
		updateProbabilityColumn(newColumn);
		
		if (removedTransition != null){
			int oldColumn = removedTransition.getFromState();
			if (newColumn != oldColumn){
				updateProbabilityColumn(oldColumn);
			}
		}		
	}
	
	private void updateProbabilityColumn(int columnID){
		SimpleMatrix column = transitionCount.extractVector(false, columnID);
		double sum = column.elementSum();
		if (sum != 0) column = column.divide(sum);
		transitionProbability.setColumn(columnID, 0, column.getMatrix().data);
	}
	
	public SimpleMatrix getTransitionrobabilityMatrix(){
		return transitionProbability;
	}
	
	private class Transition{
		private int fromState, toState;
		private double value;
		
		public Transition(int fromState, int toState){
			this(fromState, toState, 1);
		}
		
		public Transition(int fromState, int toState, double value){
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

}
