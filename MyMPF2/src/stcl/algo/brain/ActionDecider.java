package stcl.algo.brain;

import java.io.Serializable;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import dk.itu.stcl.agents.QLearner;
import dk.itu.stcl.agents.SARSALearner;

public class ActionDecider implements Serializable {
	private static final long serialVersionUID = 1L;
	private SARSALearner sarsa;
	private int stateBefore, actionBefore;
	private boolean learning;

	public ActionDecider(int numPossibleActions, int numPossibleStates, double decayFactor, Random rand, boolean offlineLearning) {
		sarsa = new SARSALearner(numPossibleStates, numPossibleActions, 0.1, decayFactor, offlineLearning, 0.9);
		stateBefore = -1;
		actionBefore = -1;
		learning = true;
	}
	
	/**
	 * Correlates the given reward with the action performed at t-1	
	 * @param currentStateProbabilities
	 * @param actionToBePerformedNow
	 * @param rewardForCurrentState
	 */
	public void feedForward(int currentState, int actionToBePerformedNow, double rewardForCurrentState){
		double internalReward = calculateInternaleward(rewardForCurrentState);
		if(stateBefore != -1 && learning){
			sarsa.updateQMatrix(stateBefore, actionBefore, currentState, actionToBePerformedNow, internalReward);
		}
		stateBefore = currentState;
		actionBefore = actionToBePerformedNow;
		
	}
	
	
	public void updateQMatrix(int originState, int action, int nextState, int nextAction,
			double reward) {
		sarsa.updateQMatrix(originState, action, nextState, nextAction, reward);
	}
	
	/**
	 * Chooses which action to do at t+1 given the expected state of t+1
	 * @param expectedNextStateProbabilities
	 * @return
	 */
	public int feedBack(int originState){
		int action = sarsa.selectBestAction(originState);
		
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
	

	
	public void printQMatrix(){
		sarsa.getQMatrix().print();
	}
	
	public SimpleMatrix getQMatrix(){
		return sarsa.getQMatrix();
	}
	
	
	public void printTraceMatrix(){
		sarsa.getTraceMatrix().print();
	}
	
	public void setLearningRate(double learningRate){
		sarsa.setAlpha(learningRate);
	}
	
	public void newEpisode(){
		stateBefore = -1;
		actionBefore = -1;
		sarsa.newEpisode();
	}
	
	public void setLearning(boolean learning){
		this.learning = learning;
	}

}
