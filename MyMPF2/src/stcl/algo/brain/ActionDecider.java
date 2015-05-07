package stcl.algo.brain;

import java.io.Serializable;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import dk.itu.stcl.agents.SARSALearner;
import stcl.algo.util.Normalizer;
import stcl.algo.util.Orthogonalizer;

public class ActionDecider implements Serializable {
	private static final long serialVersionUID = 1L;
	private SARSALearner sarsa;
	private int stateBefore, actionBefore;
	
	public ActionDecider(int numPossibleActions, int numPossibleStates, double decayFactor, Random rand) {
		sarsa = new SARSALearner(numPossibleStates, numPossibleActions, 0.1, decayFactor, 0.9);
		stateBefore = -1;
		actionBefore = -1;
	}
	
	/**
	 * Correlates the given reward with the action performed at t-1	
	 * @param currentStateProbabilities
	 * @param actionToBePerformedNow
	 * @param rewardForCurrentState
	 */
	public void feedForward(int currentState, int actionToBePerformedNow, double rewardForCurrentState){
		double internalReward = calculateInternaleward(rewardForCurrentState);
		if(stateBefore != -1){
			sarsa.updateQMatrix(stateBefore, actionBefore, currentState, actionToBePerformedNow, internalReward);
		}
		stateBefore = currentState;
		actionBefore = actionToBePerformedNow;
		
	}
	
	/**
	 * Chooses which action to do at t+1 given the expected state of t+1
	 * @param expectedNextStateProbabilities
	 * @return
	 */
	public int feedback(int expectedNextState){
		int action = sarsa.selectBestAction(expectedNextState);
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
		System.out.println("Not implemented yet");//traceMatrix.print();
	}
	
	public void setLearningRate(double learningRate){
		sarsa.setAlpha(learningRate);
	}
	
	public void newEpisode(){
		sarsa.newEpisode();
	}

}
