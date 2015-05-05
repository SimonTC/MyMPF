package stcl.algo.brain;

import java.io.Serializable;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.util.Normalizer;
import stcl.algo.util.Orthogonalizer;

public class ActionDecider implements Serializable {
	private static final long serialVersionUID = 1L;
	private SimpleMatrix qMatrix;
	private SimpleMatrix traceMatrix;
	private int stateBefore;
	private int actionBefore;
	private int numPossibleStates;
	private int numPossibleActions;
	private double decayFactor;
	private double lambda;

	private double learningRate;
	
	public ActionDecider(int numPossibleActions, int numPossibleStates, double decayFactor, Random rand) {
		qMatrix = new SimpleMatrix(numPossibleActions, numPossibleStates);
		for (int i = 0; i < qMatrix.getNumElements(); i++) qMatrix.set(i, rand.nextDouble());
		traceMatrix = new SimpleMatrix(numPossibleActions, numPossibleStates);
		this.numPossibleActions = numPossibleActions;
		this.numPossibleStates = numPossibleStates;
		this.decayFactor = decayFactor;
		this.stateBefore = -1;
		lambda = 1;//decayFactor; //TODO:Should be parameter
		
		this.learningRate = 0.1;
	}
	
	/**
	 * Correlates the given reward with the action performed at t-1	
	 * @param currentStateProbabilities
	 * @param actionToBePerformedNow
	 * @param rewardForCurrentState
	 */
	public void feedForward(int currentState, int actionToBePerformedNow, double rewardForCurrentState){
		double internalReward = calculateInternaleward(rewardForCurrentState);
		if (stateBefore != -1) updateQMatrix(currentState, actionToBePerformedNow, lambda, decayFactor, learningRate, internalReward); //TODO: Lambda, gamma and alpha should be given as parameters and changed during training
		actionBefore = actionToBePerformedNow;
		stateBefore = currentState;
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
		
		//externalRewardNow = externalReward;
		/*
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
			SimpleMatrix qVector = qMatrix.extractVector(true, action);
			SimpleMatrix rewardVector = qVector.elementMult(stateVector);
			double reward = rewardVector.elementSum();
			if (reward > highestReward){
				highestReward = reward;
				bestAction = action;
			}
		}
		assert bestAction != -1 : "No best action could be found. There is something wrong";
		return bestAction;
	}
	
	private void updateQMatrix(int stateNow, int actionNow, double lambda, double decayFactor, double learningRate, double reward){
		updateTraceMatrix(stateNow, actionNow, lambda, decayFactor);
		double error = calculateTDError(stateNow, actionNow, stateBefore, actionBefore, decayFactor, reward);
		qMatrix = qMatrix.scale(1-learningRate);
		qMatrix = qMatrix.plus(learningRate * error, traceMatrix);

	}
	
	private double calculateTDError(int stateNow, int actionNow, int stateBefore, int actionBefore, double gamma, double rewardNow){
		double error = rewardNow + gamma * qMatrix.get(actionNow, stateNow) - qMatrix.get(actionBefore, stateBefore);
		return error;
	}
	
	private void updateTraceMatrix(int state, int action, double lambda, double gamma){
		//Decay all traces
		traceMatrix = traceMatrix.scale(gamma * lambda);
		
		//Calculate trace for the current state-action pair
		double newValue = 1 + traceMatrix.get(action, state);
		
		//Set trace to zero for actions not taken in current state
		SimpleMatrix actionVector = traceMatrix.extractVector(false, state);
		actionVector.set(0);
		traceMatrix.insertIntoThis(0, state, actionVector);
		
		//Set trace value for current state-action pair
		traceMatrix.set(action, state, newValue);		
		
	}
	
	public void printQMatrix(){
		qMatrix.print();
	}
	
	public void printTraceMatrix(){
		traceMatrix.print();
	}
	
	public void setLearningRate(double learningRate){
		this.learningRate = learningRate;
	}
	
	public void newEpisode(){
		traceMatrix.set(0);
	}

}
