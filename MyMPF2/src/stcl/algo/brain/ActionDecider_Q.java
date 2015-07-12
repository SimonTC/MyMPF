package stcl.algo.brain;

import java.io.Serializable;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;
import dk.itu.stcl.agents.QLearner;
import dk.itu.stcl.agents.SARSALearner;

public class ActionDecider_Q implements Serializable {
	private static final long serialVersionUID = 1L;
	protected QLearner learner;
	protected int stateBefore, actionBefore;
	protected boolean learning;

	public ActionDecider_Q(int numPossibleActions, int numPossibleStates, double decayFactor, boolean offlineLearning) {
		learner = new QLearner(numPossibleStates, numPossibleActions, 0.1, decayFactor, offlineLearning);
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
			learner.updateQMatrix(stateBefore, actionBefore, currentState, internalReward);
			//learner.updateQMatrix(stateBefore, actionBefore, currentState, actionToBePerformedNow, internalReward);
		}
		stateBefore = currentState;
		actionBefore = actionToBePerformedNow;
		
	}
	
	
	public void updateQMatrix(int originState, int action, int nextState, int nextAction,
			double reward) {
		learner.updateQMatrix(originState, action, nextState, nextAction, reward);
	}
	
	/**
	 * Chooses which action to do at t+1 given the expected state of t+1
	 * @param expectedNextStateProbabilities
	 * @return
	 */
	public int feedBack(int originState){
		int action = learner.selectBestAction(originState);
		
		return action;
	}
	
	protected double calculateInternaleward(double externalReward){
		
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
		learner.getQMatrix().print();
	}
	
	public SimpleMatrix getQMatrix(){
		return learner.getQMatrix();
	}
	
	public void setLearningRate(double learningRate){
		learner.setAlpha(learningRate);
	}
	
	public void newEpisode(){
		stateBefore = -1;
		actionBefore = -1;
		learner.newEpisode();
	}
	
	public void setLearning(boolean learning){
		this.learning = learning;
	}
	
	public SimpleMatrix getPolicyMap(){
		SimpleMatrix qMatrix = learner.getQMatrix();
		SimpleMatrix map = new SimpleMatrix(qMatrix.numRows(), 1);
		
		for (int state = 0; state < qMatrix.numRows(); state++){
			double action = learner.selectBestAction(state);
			map.set(state,action);
		}
		return map;
	}

}
