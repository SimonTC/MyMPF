package stcl.algo.reinforcement;

import java.io.Serializable;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

public class SARSA implements Serializable {
	private static final long serialVersionUID = 1L;
	private SimpleMatrix actionMatrix;
	private SimpleMatrix stateBefore;
	private Random rand;
	private SimpleMatrix weightVector;

	public void initialize(int actionVectorLength, int stateVectorLength, Random rand, SimpleMatrix actionMatrix){
		this.rand = rand;
		this.actionMatrix = actionMatrix;
		weightVector = new SimpleMatrix(1, actionVectorLength + stateVectorLength + 1); //One added for bias
		for (int i = 0; i < weightVector.getNumElements(); i++){
			weightVector.set(i,rand.nextDouble());
		}
	}

	
	public void setActionMatrix(SimpleMatrix actionMatrix){
		this.actionMatrix = actionMatrix;
	}
	
	public int feedForward(SimpleMatrix currentStateVector, int actionBefore, double rewardNow){
		int nextAction = chooseBestAction(currentStateVector);
		
		if (stateBefore != null){
			weightVector = updateParameterVector(stateBefore, actionMatrix.extractVector(true, actionBefore), currentStateVector, actionMatrix.extractVector(true, nextAction), rewardNow, 0.1, 0.9);
		}
		
		//weightVector.print();
		stateBefore = new SimpleMatrix(currentStateVector);
		actionBefore = nextAction;
		return nextAction;
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
			double q = calculateQValue(StateFromWhichToChoose, actionMatrix.extractVector(true, action), weightVector);
			double temperateQ = Math.pow(temperature, q);
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
	
	
	private double calculateQValue(SimpleMatrix currentState, SimpleMatrix actionPerformedNow, SimpleMatrix parameterVector){
		SimpleMatrix featureVector = createFeatureVector(currentState, actionPerformedNow);
		double qValue = parameterVector.dot(featureVector.transpose());
		return qValue;		
	}
	
	private SimpleMatrix updateParameterVector(SimpleMatrix stateNow, SimpleMatrix actionNow, SimpleMatrix stateNext, SimpleMatrix actionNext, 
			double rewardNow, double learningRate, double decay){
			double valueThisState = calculateQValue(stateNow, actionNow, weightVector);
			double valueNextState = calculateQValue(stateNext, actionNext, weightVector);
			double difference = rewardNow + decay * valueNextState - valueThisState;
			SimpleMatrix featureVector = createFeatureVector(stateNow, actionNow);
			SimpleMatrix delta = featureVector.scale(learningRate);
			delta = delta.scale(difference);
			weightVector = weightVector.plus(delta);	
		
			return weightVector;
	}
	
	private double calculateMaxQPossible(SimpleMatrix state, SimpleMatrix parameterVector){
		double max = Double.NEGATIVE_INFINITY;
		for (int action = 0; action < actionMatrix.numRows(); action++){
			SimpleMatrix actionVector = actionMatrix.extractVector(true, action);
			max = Math.max(max, calculateQValue(state, actionVector, parameterVector));
		}
		return max;
	}
	
	private int chooseBestAction(SimpleMatrix state){
		double max = Double.NEGATIVE_INFINITY;
		int bestState = -1;
		for (int action = 0; action < actionMatrix.numRows(); action++){
			SimpleMatrix actionVector = actionMatrix.extractVector(true, action);
			double value = calculateQValue(state, actionVector, weightVector);
			if (value > max){
				max = value;
				bestState = action;
			}
		}
		return bestState;
	}
	
	private SimpleMatrix createFeatureVector(SimpleMatrix state, SimpleMatrix action){
		SimpleMatrix featureVector = new SimpleMatrix(1, state.getNumElements() + action.getNumElements() + 1);
		featureVector.insertIntoThis(0, 1, state);
		featureVector.insertIntoThis(0, state.getNumElements() + 1, action);
		featureVector.set(0, 1); //Bias
		return featureVector;
	}
	
	public void newEpisode(){
		stateBefore = null;
	}


}
