package stcl.algo.brain;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.util.Normalizer;
import stcl.algo.util.Orthogonalizer;

public class ActionDecider {
	
	private SimpleMatrix correlationMatrix;
	private SimpleMatrix stateProbabilitiesBefore;
	private int actionPerformedBefore;
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
	 * Correlates the given reward with the action performed at t-1	
	 * @param currentStateProbabilities
	 * @param actionPerformedNow
	 * @param reward
	 */
	public void feedForward(SimpleMatrix currentStateProbabilities, int actionPerformedNow, double reward){
		if (stateProbabilitiesBefore != null) correlateActionAndReward(reward);
		actionPerformedBefore = actionPerformedNow;
		stateProbabilitiesBefore = currentStateProbabilities;
	}
	
	/**
	 * Chooses which action to do at t+1 given the expected state of t+1
	 * @param expectedNextStateProbabilities
	 * @return
	 */
	public int feedback(SimpleMatrix expectedNextStateProbabilities){
		int action = chooseBestAction(expectedNextStateProbabilities);
		return action;
	}
	
	private int chooseBestAction(SimpleMatrix stateProbabilities){
		int bestAction = -1;
		double highestReward = Double.NEGATIVE_INFINITY;
		SimpleMatrix stateVector = new SimpleMatrix(1, numPossibleStates, true, stateProbabilities.getMatrix().data);
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
	
	private void correlateActionAndReward(double reward){
		//Correlate state we were in before with the action done and reward received
		SimpleMatrix stateVector = new SimpleMatrix(1, numPossibleStates, true, stateProbabilitiesBefore.getMatrix().data);
		SimpleMatrix correlationVector = correlationMatrix.extractVector(true, actionPerformedBefore);
		
		//Decay old rewards
		correlationVector = correlationVector.scale(1-decayFactor);
		
		//Add new rewards
		correlationVector = correlationVector.plus(reward, stateVector);
		
		correlationMatrix.insertIntoThis(actionPerformedBefore, 0, correlationVector);

	}
	
	public void printCorrelationMatrix(){
		correlationMatrix.print();
	}

}
