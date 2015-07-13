package stcl.algo.brain;

import java.io.Serializable;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import dk.itu.stcl.agents.QLearner;
import dk.itu.stcl.agents.SARSALearner;

public class ActionDecider_Q_Reactionary extends ActionDecider_Q {
	private static final long serialVersionUID = 1L;
	private int actionToDoNext;
	
	public ActionDecider_Q_Reactionary(int numPossibleActions, int numPossibleStates, double decayFactor, boolean offlineLearning) {
		super(numPossibleActions, numPossibleStates, decayFactor, offlineLearning);
	}
	
	@Override
	/**
	 * Correlates the given reward with the action performed at t-1	
	 * @param currentStateProbabilities
	 * @param actionToGetToCurrentState
	 * @param rewardForCurrentState
	 */
	public void feedForward(int currentState, int actionToGetToCurrentState, double rewardForCurrentState, SimpleMatrix currentStateProbabilities){
		counter++;
		double internalReward = calculateInternaleward(rewardForCurrentState, currentStateProbabilities);
		if(stateBefore != -1 && learning){
			learner.updateQMatrix(stateBefore, actionToGetToCurrentState, currentState, internalReward);
		}
		actionToDoNext = learner.selectBestAction(currentState);
		stateBefore = currentState;
		
	}
	
	@Override
	/**
	 * Returns the action to perform now (time t) based on the current state.
	 * NB: Doesn't make use of the originState parameter
	 * @param originState - Not used
	 * @return
	 */
	public int feedBack(int originState, SimpleMatrix stateProbabilities){
		prediction = stateProbabilities;
		return actionToDoNext;
	}

}
