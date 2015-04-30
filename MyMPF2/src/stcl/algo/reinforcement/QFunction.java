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
	private Random rand;

	
	public void setActionMatrix(SimpleMatrix actionMatrix){
		this.actionMatrix = actionMatrix;
	}
	
	public void feedForward(SimpleMatrix currentStateVector, int actionPerformedNow, double rewardForActionBefore){
		if(stateBefore != null){
			parameterVectorNextEpisode = updateParameterVector(stateBefore, actionMatrix.extractVector(true, actionPerformedBefore), currentStateVector, rewardForActionBefore, 0.5, 0.9, parameterVectorNextEpisode);
		}
		stateBefore = new SimpleMatrix(currentStateVector);
		actionPerformedBefore = actionPerformedNow;
	}
	
	public int feedback(SimpleMatrix expectedNextState){
		int action = chooseBestAction(expectedNextState);
		return action;
	}
	
	public int chooseNextAction(SimpleMatrix StateFromWhichToChoose, int temperature){
		SimpleMatrix probabilityVector = new SimpleMatrix(actionMatrix.numRows(),1);
		SimpleMatrix temperateQVector = new SimpleMatrix(actionMatrix.numRows(),1);
		double totalTemperateQ = 0;
		for (int action = 0; action < actionMatrix.numRows(); action++){
			double temperateQ = Math.pow(temperature, calculateQValue(StateFromWhichToChoose, actionMatrix.extractVector(true, action)));
			temperateQVector.set(action, temperateQ);
			totalTemperateQ += temperateQ;
		}
		
		probabilityVector = temperateQVector.divide(totalTemperateQ);
		double threshold = rand.nextDouble();
		boolean actionFound = false;
		int action = -1;
		double value = 0;
		while(!actionFound){
			action++;
			value += probabilityVector.get(action);
			if (value >= threshold) actionFound = true;
		}
		return action;
	}
	
	public void initialize(int actionVectorLength, int stateVectorLength, Random rand, SimpleMatrix actionMatrix){
		this.rand = rand;
		this.actionMatrix = actionMatrix;
		parameterVectorCurrentEpisode = new SimpleMatrix(1, actionVectorLength + stateVectorLength);
		for (int i = 0; i < parameterVectorCurrentEpisode.getNumElements(); i++){
			parameterVectorCurrentEpisode.set(i,rand.nextDouble());
		}
		parameterVectorNextEpisode = new SimpleMatrix(parameterVectorCurrentEpisode);
	}
	
	public void newEpisodedouble(double rewardForLastEpisode){
		SimpleMatrix tmpMatrix = new SimpleMatrix(stateBefore);
		tmpMatrix.set(0);
		parameterVectorNextEpisode = updateParameterVector(stateBefore, actionMatrix.extractVector(true, actionPerformedBefore), tmpMatrix, rewardForLastEpisode, 0.1, 0.9, parameterVectorNextEpisode);
		parameterVectorCurrentEpisode = new SimpleMatrix(parameterVectorNextEpisode);
		stateBefore = null;
	}
	
	private double calculateQValue(SimpleMatrix currentState, SimpleMatrix actionPerformedNow){
		SimpleMatrix featureVector = createFeatureVector(currentState, actionPerformedNow);
		double qValue = parameterVectorCurrentEpisode.dot(featureVector.transpose());
		return qValue;		
	}
	
	private SimpleMatrix updateParameterVector(SimpleMatrix stateNow, SimpleMatrix actionNow, SimpleMatrix stateNext, 
			double rewardNow, double learningRate, double decay, SimpleMatrix parameterVector){
			if (rewardNow > 0.5){
				System.out.println();
			}
			double valueThisState = calculateQValue(stateNow, actionNow);
			double maxValueNextState = calculateMaxQPossible(stateNext);
			double difference = rewardNow + decay * maxValueNextState - valueThisState;
			SimpleMatrix featureVector = createFeatureVector(stateNow, actionNow);
			SimpleMatrix delta = featureVector.scale(learningRate);
			delta = delta.scale(difference);
			parameterVector = parameterVector.plus(delta);	
		
			return parameterVector;
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
	
	public SimpleMatrix getParameterVectorNextEpisode(){
		return parameterVectorNextEpisode;
	}
}
