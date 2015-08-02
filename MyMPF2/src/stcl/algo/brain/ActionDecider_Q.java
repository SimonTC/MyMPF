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

	public ActionDecider_Q(int numPossibleActions, int numPossibleStates, double decayFactor, boolean offlineLearning) {
		learner = new QLearner(numPossibleStates, numPossibleActions, 0.1, decayFactor, offlineLearning);
		stateBefore = -1;
		actionBefore = -1;
		learning = true;
		maxError = 0;
		noveltyInfluence = 1; //TODO: Make parameter
		decayConstant = 1/(double)1000; //TODO: Make parameter
		counter = 0;
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
	public int feedBack(SimpleMatrix stateProbabilities){
		int action = learner.selectBestAction(new SimpleMatrix(1, stateProbabilities.getNumElements(), true, stateProbabilities.getMatrix().data));
		prediction = stateProbabilities;
		return action;
	}

	protected double calculateInternaleward(double externalReward, SimpleMatrix currentStateProbabilities){
		double internalReward = externalReward;
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

}
