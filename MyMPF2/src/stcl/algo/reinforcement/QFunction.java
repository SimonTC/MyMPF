package stcl.algo.reinforcement;

import java.io.Serializable;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

public class QFunction implements Serializable{

	private static final long serialVersionUID = 1L;
	private SimpleMatrix parameterVectorCurrentEpisode;
	private SimpleMatrix parameterVectorNextEpisode;
	private SimpleMatrix actionMatrix;
	private int actionPerformedBefore;
	private SimpleMatrix stateBefore;

	
	public QFunction(int numPossibleActions, int numPossibleStates, SimpleMatrix actionMatrix, Random rand){
		initializeParameterVector(numPossibleActions, numPossibleStates, rand);
		
		this.actionMatrix = actionMatrix;
	}
	
	public void setActionMatrix(SimpleMatrix actionMatrix){
		this.actionMatrix = actionMatrix;
	}
	
	public void feedForward(SimpleMatrix currentStateVector, int actionPerformedNow, double rewardForActionBefore){
		updateParameterVector(stateBefore, actionMatrix.extractVector(true, actionPerformedBefore), currentStateVector, rewardForActionBefore, 0.1, 0.9, parameterVectorNextEpisode);
		
		stateBefore = currentStateVector;
		actionPerformedBefore = actionPerformedNow;
	}
	
	public int feedback(SimpleMatrix expectedNextState){
		int action = chooseBestAction(expectedNextState);
		return action;
	}
	
	public void initializeParameterVector(int numPossibleActions, int numPossibleStates, Random rand){
		parameterVectorCurrentEpisode = new SimpleMatrix(1, numPossibleActions + numPossibleStates);
		for (int i = 0; i < parameterVectorCurrentEpisode.getNumElements(); i++){
			parameterVectorCurrentEpisode.set(rand.nextDouble());
		}
	}
	
	private double calculateQValue(SimpleMatrix currentState, SimpleMatrix actionPerformedNow){
		SimpleMatrix featureVector = createFeatureVector(currentState, actionPerformedNow);
		double qValue = parameterVectorCurrentEpisode.dot(featureVector.transpose());
		return qValue;		
	}
	
	private void updateParameterVector(SimpleMatrix stateNow, SimpleMatrix actionNow, SimpleMatrix stateNext, 
			double rewardNow, double learningRate, double decay, SimpleMatrix parameterVector){
			double valueThisState = calculateQValue(stateNow, actionNow);
			double maxValueNextState = calculateMaxQPossible(stateNext);
			double difference = rewardNow + decay * maxValueNextState - valueThisState;
			SimpleMatrix featureVector = createFeatureVector(stateNow, actionNow);
			SimpleMatrix delta = featureVector.scale(learningRate);
			delta = delta.scale(difference);
			parameterVector = parameterVector.plus(delta);		
	}
	
	private double calculateMaxQPossible(SimpleMatrix state){
		double max = Double.NEGATIVE_INFINITY;
		for (int action = 0; action < actionMatrix.numRows(); action++){
			SimpleMatrix actionVector = actionMatrix.extractVector(true, action);
			max = Math.max(max, calculateQValue(state, actionVector));
		}
		return max;
	}
	
	private int chooseBestAction(SimpleMatrix state){
		double max = Double.NEGATIVE_INFINITY;
		int bestState = -1;
		for (int action = 0; action < actionMatrix.numRows(); action++){
			SimpleMatrix actionVector = actionMatrix.extractVector(true, action);
			double value = calculateQValue(state, actionVector);
			if (value > max){
				max = value;
				bestState = action;
			}
		}
		return bestState;
	}
	
	private SimpleMatrix createFeatureVector(SimpleMatrix state, SimpleMatrix action){
		SimpleMatrix featureVector = new SimpleMatrix(1, state.getNumElements() + action.getNumElements());
		featureVector.insertIntoThis(0, 0, state);
		featureVector.insertIntoThis(0, state.getNumElements(), action);
		return featureVector;
	}
}
