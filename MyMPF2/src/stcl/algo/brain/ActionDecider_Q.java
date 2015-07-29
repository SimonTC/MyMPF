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
	protected SimpleMatrix prediction;
	private double maxError;
	private double noveltyInfluence;
	private double decayConstant;
	protected int counter;
	/**
	 * If true the reward given buy the game will also be the reward used in the Q-function.
	 * If false The reward used will be a reward that leads to exploration.
	 */
	private boolean useExternalReward;

	public ActionDecider_Q(int numPossibleActions, int numPossibleStates, double decayFactor, boolean offlineLearning) {
		learner = new QLearner(numPossibleStates, numPossibleActions, 0.1, decayFactor, offlineLearning);
		stateBefore = -1;
		actionBefore = -1;
		learning = true;
		maxError = 0;
		noveltyInfluence = 1; //TODO: Make parameter
		decayConstant = 1/(double)1000; //TODO: Make parameter
		counter = 0;
		useExternalReward = false;
	}
	
	/**
	 * Correlates the given reward with the action performed at t-1	
	 * @param currentStateProbabilities
	 * @param actionToBePerformedNow
	 * @param rewardForCurrentState
	 */
	public void feedForward(int currentState, int actionToBePerformedNow, double rewardForCurrentState, SimpleMatrix currentStateProbabilities){
		counter++;
		double internalReward = calculateInternaleward(rewardForCurrentState, currentStateProbabilities);
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
	public int feedBack(int originState, SimpleMatrix stateProbabilities){
		int action = learner.selectBestAction(originState);
		prediction = stateProbabilities;
		return action;
	}
	//TODO: Better citation
	/**
	 * Calculate internal reward by using method described in Incentivizing Exploration In Reinforcement Learning
	 * With Deep Predictive Models (Stadie, Levine and Abbeel)
	 * @param externalReward
	 * @param currentState
	 * @return
	 */
	//TODO: Is this used? Does it work? Wasn't there a problem with this?
	protected double calculateInternaleward(double externalReward, SimpleMatrix currentStateProbabilities){
		double internalReward = externalReward;
		if (prediction != null && !useExternalReward){
			SimpleMatrix diff = currentStateProbabilities.minus(prediction);
			double error = diff.normF();
			double normalizedError = error / maxError;
			double novelty = Math.pow(Math.abs(normalizedError / (counter * decayConstant)), 2);
			internalReward += noveltyInfluence * novelty;
			if (error > maxError) maxError = error;
		}

		return internalReward;

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
		prediction = null;
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
	
	public void setUseExternalReward(boolean flag){
		useExternalReward = flag;
	}

}
