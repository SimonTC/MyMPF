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
	
	private double externalRewardBefore;
	private double externalRewardNow;
	private double maxReward;
	private double alpha;
	private double internalRewardBefore;
	
	public ActionDecider(int numPossibleActions, int numPossibleStates, double decayFactor) {
		correlationMatrix = new SimpleMatrix(numPossibleActions, numPossibleStates);
		this.numPossibleActions = numPossibleActions;
		this.numPossibleStates = numPossibleStates;
		this.decayFactor = 0.1;//decayFactor;
		
		this.maxReward = 1;
		this.alpha = decayFactor;
	}
	
	/**
	 * Correlates the given reward with the action performed at t-1	
	 * @param currentStateProbabilities
	 * @param actionPerformedNow
	 * @param reward
	 */
	public void feedForward(SimpleMatrix currentStateProbabilities, int actionPerformedNow, double reward){
		double internalReward = calculateInternaleward(reward);
		if (stateProbabilitiesBefore != null) correlateActionAndReward(internalReward);
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
	
	private double calculateInternaleward(double externalReward){
		/*externalRewardNow = externalReward;
		
		double exponentialWeightedMovingAverage = (externalRewardNow - externalRewardBefore) / maxReward;
		
		double internalReward = alpha * exponentialWeightedMovingAverage + (1-alpha) * internalRewardBefore;
		
		internalRewardBefore = internalReward;
		externalRewardBefore = externalRewardNow;
		
		return internalReward;
		*/
		return externalReward;

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
	
	private void correlateActionAndReward(double internalReward){
		//Correlate state we were in before with the action done and reward received
		SimpleMatrix stateVector = new SimpleMatrix(1, numPossibleStates, true, stateProbabilitiesBefore.getMatrix().data);
		SimpleMatrix correlationVector = correlationMatrix.extractVector(true, actionPerformedBefore);
		
		SimpleMatrix tau = stateVector.scale(0.1);
		
		//Multiply the tau value with the given reward
		//This is the influence that the actions at time t-1 have on the reward (I think)
		SimpleMatrix tmp = tau.scale(internalReward);
		
		//Calculate the other part of the equation
		SimpleMatrix learnedCorrelation = new SimpleMatrix(stateVector.numRows(), stateVector.numCols());
		learnedCorrelation.set(1);
		learnedCorrelation = learnedCorrelation.minus(tau);
		learnedCorrelation = learnedCorrelation.elementMult(correlationVector);
		
		//Calculate correlation matrix
		correlationVector = tmp.plus(learnedCorrelation);		
		
		/*
		//Decay old rewards
		correlationVector = correlationVector.scale(1-decayFactor);
		
		//Add new rewards
		correlationVector = correlationVector.plus(internalReward, stateVector);
		*/
		correlationMatrix.insertIntoThis(actionPerformedBefore, 0, correlationVector);

	}
	
	public void printCorrelationMatrix(){
		correlationMatrix.print();
	}

}
