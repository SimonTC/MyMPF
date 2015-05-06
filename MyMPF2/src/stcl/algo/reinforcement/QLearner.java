package stcl.algo.reinforcement;

import java.io.Serializable;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

public class QLearner implements Serializable {

	private static final long serialVersionUID = 1L;
	private SimpleMatrix qMatrix;
	private int stateBefore;
	private int actionBefore;
	private int numPossibleStates;
	private int numPossibleActions;
	private double decayFactor;
	private double lambda;

	private double learningRate;
	
	public QLearner(int numPossibleActions, int numPossibleStates, double decayFactor, Random rand) {
		qMatrix = new SimpleMatrix(numPossibleActions, numPossibleStates);
		//for (int i = 0; i < qMatrix.getNumElements(); i++) qMatrix.set(i, rand.nextDouble());
		
		this.numPossibleActions = numPossibleActions;
		this.numPossibleStates = numPossibleStates;
		this.decayFactor = decayFactor;
		this.stateBefore = -1;
		lambda = 0;// 0.9;//decayFactor; //TODO:Should be parameter
		
		this.learningRate = 0.1;
	}
	
	public int chooseBestAction(SimpleMatrix stateProbabilities){
		int bestAction = -1;
		double highestReward = Double.NEGATIVE_INFINITY;
		SimpleMatrix stateVector = new SimpleMatrix(1, numPossibleStates, true, stateProbabilities.getMatrix().data);
		for (int action = 0; action < numPossibleActions; action++){
			SimpleMatrix qVector = qMatrix.extractVector(true, action);
			SimpleMatrix rewardVector = qVector.elementMult(stateVector);
			double reward = rewardVector.elementSum();
			if (reward > highestReward){
				highestReward = reward;
				bestAction = action;
			}
		}
		assert bestAction != -1 : "No best action could be found. There is something wrong";
		return bestAction;
	}
	
	public void updateQMatrix(int stateNow, int actionNow, double decayFactor, double learningRate, double reward){
		if (stateBefore != -1){
			double maxQ = maxQ(stateNow);
			double delta = learningRate * (reward + decayFactor * maxQ - getQValue(stateBefore, actionBefore));
			delta += qMatrix.get(actionBefore, stateBefore);
			qMatrix.set(actionBefore, stateBefore, delta);
		}
		
		actionBefore = actionNow;
		stateBefore = stateNow;

	}

	
	private double maxQ(int state){
		double maxValue = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < numPossibleActions; i++){
			double value = getQValue(state, i);
			if (value > maxValue) maxValue = value;
		}
		return maxValue;
	}
	
	private double getQValue(int state, int action){
		return qMatrix.get(action, state);
	}
	
	
	
	
	public void printQMatrix(){
		qMatrix.print();
	}
	
	public SimpleMatrix getQMatrix(){
		return qMatrix;
	}
		
	public void setLearningRate(double learningRate){
		this.learningRate = learningRate;
	}

}
