package stcl.algo.brain;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.util.Normalizer;
import stcl.algo.util.Orthogonalizer;

public class ActionDecider {
	
	private SimpleMatrix correlationMatrix;
	private SimpleMatrix stateProbabilitiesBefore;
	private int actionDoneBefore;
	private int numPossibleStates;
	private int numPossibleActions;
	private double decayFactor;
	
	public ActionDecider(int numPossibleActions, int numPossibleStates, double decayFactor) {
		correlationMatrix = new SimpleMatrix(numPossibleActions, numPossibleStates);
		this.numPossibleActions = numPossibleActions;
		this.numPossibleStates = numPossibleStates;
		this.decayFactor = 0.1;//decayFactor;
	}
	
	/**
	 * Correlates reward with action done before and saves the info about the action to be done now and the current state probabilities
	 * @param actionToBeDone
	 * @param reward
	 */
	public void feedForward(SimpleMatrix currentStateProbabilities, int actionToBeDone, double reward){
		if (stateProbabilitiesBefore != null){
			correlateActionAndReward(reward);
		}
		actionDoneBefore = actionToBeDone;
		stateProbabilitiesBefore = currentStateProbabilities;
	}
	
	/**
	 * Decide what actions to do next based on the state we are expecting to be in
	 * @return the id of the action we want to do
	 */
	public int feedBack(SimpleMatrix probabilitiesOfNextState){
		int actionToDo = chooseBestAction(probabilitiesOfNextState);
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
	
	/**
	 * Correlates the reward received at step t+1 with the action performed at step t from the state we where in at step t
	 * @param actionToBeDone
	 * @param reward
	 */
	private void correlateActionAndReward(double reward){
		//Correlate state we were in before with the action done and reward received
		SimpleMatrix stateVector = new SimpleMatrix(1, numPossibleStates, true, stateProbabilitiesBefore.getMatrix().data);
		SimpleMatrix correlationVector = correlationMatrix.extractVector(true, actionDoneBefore);
		
		//Decay old rewards
		correlationVector = correlationVector.scale(1-decayFactor);
		
		//Add new rewards
		correlationVector = correlationVector.plus(reward, stateVector);
		
		correlationMatrix.insertIntoThis(actionDoneBefore, 0, correlationVector);
		
		//Normalize columns of correlationMatrix
		correlationMatrix = Normalizer.normalizeColumns(correlationMatrix);
	}
	
	public void printCorrelationMatrix(){
		correlationMatrix.print();
	}

}
