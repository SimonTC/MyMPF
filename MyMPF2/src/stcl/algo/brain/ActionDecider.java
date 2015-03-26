package stcl.algo.brain;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.util.Normalizer;
import stcl.algo.util.Orthogonalizer;

public class ActionDecider {
	
	private SimpleMatrix correlationMatrix;
	private int stateBefore;
	private int numPossibleStates;
	private int numPossibleActions;
	private double decayFactor;
	
	public ActionDecider(int numPossibleActions, int numPossibleStates, double decayFactor) {
		correlationMatrix = new SimpleMatrix(numPossibleActions, numPossibleStates);
		this.numPossibleActions = numPossibleActions;
		this.numPossibleStates = numPossibleStates;
		this.decayFactor = 0.1;//decayFactor;
		stateBefore = -1;
	}
	
	public int decideNextAction(int currentStateID, int actionToGetHere, double reward){
		if (stateBefore != -1){
			correlateActionAndReward(actionToGetHere, reward);
		}
		int actionToDo = chooseBestAction(currentStateID);
		stateBefore = currentStateID;
		return actionToDo;		
	}
	
	private int chooseBestAction(int currentStateID){
		int bestAction = -1;
		double highestReward = Double.NEGATIVE_INFINITY;
		for (int action = 0; action < numPossibleActions; action++){
			double reward = correlationMatrix.get(action, currentStateID);
			if (reward > highestReward){
				highestReward = reward;
				bestAction = action;
			}
		}
		return bestAction;
	}
	
	private void correlateActionAndReward(int actionPerformed, double reward){
		//Decay values in correlation matrix
		correlationMatrix = correlationMatrix.scale(1-decayFactor);
		double oldValue = correlationMatrix.get(actionPerformed, stateBefore);
		double newValue = oldValue + reward;
		correlationMatrix.set(actionPerformed, stateBefore, newValue);
		
		/*
		SimpleMatrix correlationVector = correlationMatrix.extractVector(true, actionPerformed);
		
		//Decay old rewards
		correlationVector = correlationVector.scale(1-decayFactor);
		
		//Add new rewards
		double oldValue = correlationVector.get(stateBefore);
		double newValue = oldValue + reward;
		correlationVector.set(newValue);
		
		correlationMatrix.insertIntoThis(actionPerformed, 0, correlationVector);
		*/
		//Normalize columns of correlationMatrix
		//correlationMatrix = Normalizer.normalizeColumns(correlationMatrix);
	}
	
	public void printCorrelationMatrix(){
		correlationMatrix.print();
	}

}
