package stcl.algo.brain;

import java.io.Serializable;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import dk.itu.stcl.agents.QLearner;
import dk.itu.stcl.agents.SARSALearner;

public class ActionDecider_Q_Reactionary implements Serializable {
	private static final long serialVersionUID = 1L;
	private QLearner learner;
	private int stateBefore;
	private boolean learning;
	private int actionToDoNext;

	public ActionDecider_Q_Reactionary(int numPossibleActions, int numPossibleStates, double decayFactor, Random rand, boolean offlineLearning) {
		learner = new QLearner(numPossibleStates, numPossibleActions, 0.1, decayFactor, offlineLearning);
		stateBefore = -1;
		learning = true;
	}
	
	/**
	 * Correlates the given reward with the action performed at t-1	
	 * @param currentStateProbabilities
	 * @param actionToGetToCurrentState
	 * @param rewardForCurrentState
	 */
	public void feedForward(int currentState, int actionToGetToCurrentState, double rewardForCurrentState){
		double internalReward = calculateInternaleward(rewardForCurrentState);
		if(stateBefore != -1 && learning){
			learner.updateQMatrix(stateBefore, actionToGetToCurrentState, currentState, internalReward);
		}
		actionToDoNext = learner.selectBestAction(currentState);
		stateBefore = currentState;
		
	}
	
	
	public void updateQMatrix(int originState, int action, int nextState, int nextAction,
			double reward) {
		learner.updateQMatrix(originState, action, nextState, nextAction, reward);
	}
	
	/**
	 * Returns the action to perform now (time t) based on the current state.
	 * NB: Doesn't make use of the originState parameter
	 * @param originState - Not used
	 * @return
	 */
	public int feedBack(int originState){
		return actionToDoNext;
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
		learner.newEpisode();
	}
	
	public void setLearning(boolean learning){
		this.learning = learning;
	}

}
