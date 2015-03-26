package stcl.algo.brain;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.util.Normalizer;
import stcl.algo.util.Orthogonalizer;

public class ActionDecider {
	
	private SimpleMatrix correlationMatrix;
	private SimpleMatrix stateProbabilitiesBefore;
	private int numPossibleStates;
	private int numPossibleActions;
	private double decayFactor;
	
	public ActionDecider(int numPossibleActions, int numPossibleStates, double decayFactor) {
		correlationMatrix = new SimpleMatrix(numPossibleActions, numPossibleStates);
		this.numPossibleActions = numPossibleActions;
		this.numPossibleStates = numPossibleStates;
		this.decayFactor = 0.1;//decayFactor;
	}
	
	public int decideNextAction(SimpleMatrix currentStateProbabilities, int actionToGetHere, double reward){
		if (stateProbabilitiesBefore != null){
			correlateActionAndReward(actionToGetHere, reward);
		}
		int actionToDo = chooseBestAction(currentStateProbabilities);
		stateProbabilitiesBefore = currentStateProbabilities;
		return actionToDo;		
	}
	
	private int chooseBestAction(SimpleMatrix currentStateProbabilities){
		int bestAction = -1;
		double highestReward = Double.NEGATIVE_INFINITY;
		SimpleMatrix stateVector = new SimpleMatrix(1, numPossibleStates, true, currentStateProbabilities.getMatrix().data);
		for (int action = 0; action < numPossibleActions; action++){
			SimpleMatrix correlationVector = correlationMatrix.extractVector(true, action);
			SimpleMatrix rewardVector = correlationVector.elementMult(stateVector);
			double reward = rewardVector.elementSum();
			if (reward > highestReward){
				highestReward = reward;
				bestAction = action;
			}
		}
		return bestAction;
	}
	
	private void correlateActionAndReward(int actionPerformed, double reward){
		//Correlate state we were in before with the action done and reward received
		SimpleMatrix stateVector = new SimpleMatrix(1, numPossibleStates, true, stateProbabilitiesBefore.getMatrix().data);
		SimpleMatrix correlationVector = correlationMatrix.extractVector(true, actionPerformed);
		
		//Decay old rewards
		correlationVector = correlationVector.scale(1-decayFactor);
		
		//Add new rewards
		correlationVector = correlationVector.plus(reward, stateVector);
		
		correlationMatrix.insertIntoThis(actionPerformed, 0, correlationVector);
		
		//Normalize columns of correlationMatrix
		//correlationMatrix = Normalizer.normalizeColumns(correlationMatrix);
	}
	
	public void printCorrelationMatrix(){
		correlationMatrix.print();
	}

}
