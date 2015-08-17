package stcl.fun.rps;

import java.util.ArrayList;
import java.util.Random;

import javax.swing.JFrame;

import org.ejml.simple.SimpleMatrix;

import dk.itu.stcl.agents.QLearner;
import stcl.algo.brain.Network_DataCollector;
import stcl.algo.brain.nodes.Sensor;
import stcl.fun.rps.rewardfunctions.RewardFunction;
import stcl.fun.rps.rewardfunctions.RewardFunction_Inverse;
import stcl.fun.rps.rewardfunctions.RewardFunction_Standard;
import stcl.fun.rps.sequencecreation.SequenceBuilder;
import stcl.graphics.MPFGUI;

public class CopyOfSequenceRunner {
	
	private int[] sequence;
	private RewardFunction[] rewardFunctions;
	private RewardFunction curRewardFunction;
	private int curRewardFunctionID;
	private Random rand;
	
	private QLearner activator;
	
	//Variables have to be saved here to remember values between sequence runs
	private SimpleMatrix action;
	private SimpleMatrix prediction;
	
	
	public static void main(String[] args){
		RewardFunction[] functions = {new RewardFunction_Standard(), new RewardFunction_Inverse()};
		Random rand = new Random();
		int[] sequence = createSequences(rand);
		CopyOfSequenceRunner sr = new CopyOfSequenceRunner(sequence, functions, rand);
		
		int numSeq = 10000;
		for (int i = 0; i < numSeq; i++){
			double[] scores = sr.runSequence(1.0 - ((double)i / numSeq));
			double fitness = scores[1];
			System.out.println(i + ": " + fitness);
		}
	}
	
	private static int[] createSequences(Random rand){
		int[] mySequence ={2,1,0,2,2,1,0,1};
		return mySequence;		
	}

	public CopyOfSequenceRunner(int[] sequence, RewardFunction[] rewardFunctions, Random rand) {
		this.rand = rand;
		setSequence(sequence);
		setRewardFunctions(rewardFunctions);
		activator = new QLearner(4, 3, 0.1, 0.1, false);
	}
	

	
	/**
	 * Goes through the sequence once.
	 * Remember to call reset() if the evaluation should start from scratch
	 * @param activator
	 * @return Array containing prediction success and fitness in the form [prediction,fitness]
	 */
	public double[] runSequence(double explorationChance){
		double totalPredictionError = 0;
		double totalGameScore = 0;
		double reward_before = 0;
		int state_before = 0;
		int action_before = 0;
		
		int state = 1;
		
		for (int i = 0; i < sequence.length; i++){
			int myAction = activator.selectBestAction(state);
			if (rand.nextDouble() < explorationChance){
				myAction = rand.nextInt(3);
			}
			
			double reward_now = curRewardFunction.reward(state, myAction);
			totalGameScore += reward_now;	
			
			int nextState = sequence[i];
			
			activator.updateQMatrix(state_before, action_before, state, reward_before);
			
			state_before = state;
			action_before = myAction;
			reward_before = reward_now;
			
			state = nextState;
			
		}
		
		activator.updateQMatrix(state_before, action_before, state, reward_before);
		
		activator.newEpisode();
		
		//endSequence(activator, reward_before);
		
		double avgPredictionError = totalPredictionError / (double) sequence.length;
		double avgScore = totalGameScore / (double) sequence.length;
		double predictionSuccess = 1 - avgPredictionError;
		
		//Scores can't be less than zero as the evolutionary algorithm can't work with that
		
		double[] result = {predictionSuccess, avgScore};
		return result;
	}
	
	/**
	 * Goes through the sequence once.
	 * Remember to call reset() if the evaluation should start from scratch
	 * @param activator
	 * @return Array containing prediction success and fitness in the form [prediction,fitness]
	 */
	public double[] runSequence_2(double explorationChance){
		double totalPredictionError = 0;
		double totalGameScore = 0;
		double reward_before = 0;
		
		int state = 1;
		
		for (int i = 0; i < sequence.length; i++){
			int myAction = activator.selectBestAction(state);
			if (rand.nextDouble() < explorationChance){
				myAction = rand.nextInt(3);
			}
			
			double reward_now = curRewardFunction.reward(state, myAction);
			totalGameScore += reward_now;	
			
			int nextState = sequence[i];
			
			activator.updateQMatrix(state, myAction, nextState, reward_now);
			
			state = nextState;
			
		}
		
		activator.newEpisode();
		
		//endSequence(activator, reward_before);
		
		double avgPredictionError = totalPredictionError / (double) sequence.length;
		double avgScore = totalGameScore / (double) sequence.length;
		double predictionSuccess = 1 - avgPredictionError;
		
		//Scores can't be less than zero as the evolutionary algorithm can't work with that
		
		double[] result = {predictionSuccess, avgScore};
		return result;
	}
	
	private void initializeSequence(Network_DataCollector activator){
		//Give blank input and action to network
		SimpleMatrix initialInput = new SimpleMatrix(5, 5);
		SimpleMatrix initialAction = new SimpleMatrix(1, 3);
		giveInputsToActivator(activator, initialInput, initialAction);
		
		activator.step(0);

	}
	
	private void endSequence(Network_DataCollector activator, double reward){
		//Give blank input and action to network
		SimpleMatrix input = new SimpleMatrix(5, 5);
		SimpleMatrix[] output = collectOutput(activator);
		action = output[1];
		giveInputsToActivator(activator, input, action);
		
		activator.step(reward);
	}
	

	
	private void giveInputsToActivator(Network_DataCollector activator, SimpleMatrix input, SimpleMatrix action){
		SimpleMatrix inputVector = new SimpleMatrix(1, input.getNumElements(), true, input.getMatrix().data);
		SimpleMatrix actionVector = new SimpleMatrix(1, action.getNumElements(), true, action.getMatrix().data);
		setInput(inputVector.getMatrix().data, activator);		
		setAction(actionVector.getMatrix().data, activator);
	}
	
	public void setInput(double[] stimuli, Network_DataCollector network){
		ArrayList<Sensor> sensors = network.getSensors();
		 
		for (int i = 0; i < stimuli.length; i++){
			Sensor s = sensors.get(i);
			s.setInput(stimuli[i]);
		}

	}
	
	public void setAction(double[] action, Network_DataCollector network){
		ArrayList<Sensor> sensors = network.getSensors();
		Sensor actionSensor = sensors.get(sensors.size()-1);
		SimpleMatrix input = new SimpleMatrix(1, action.length, true, action);
		actionSensor.setInput(input);
	}
	
	/**
	 * Creates a noisy matrix based on the given matrix. The noise added is in the range [-0.25, 0.25]
	 * The input matrix is not altered in this method.
	 * Values in the matrix will be in the range [0,1] after adding noise
	 * @param m
	 * @param noiseMagnitude
	 * @return noisy matrix
	 */
	private SimpleMatrix addNoise(SimpleMatrix m, double magnitude){
		SimpleMatrix noisy = new SimpleMatrix(m);
		for (int i = 0; i < m.getNumElements(); i++){
			double d = m.get(i);
			double noise = magnitude * (rand.nextDouble() - 0.5) * 2;
			d = d + noise;
			if (d < 0) d = 0;
			if (d > 1) d = 1;
			noisy.set(i, d);
		}
		return noisy;
	}
	
	/**
	 * finds the action chosen by the player and returns the reward given for that action
	 * @param action
	 * @param inputID
	 * @return
	 */
	private double calculateReward(SimpleMatrix action, int inputID){
		if (action.elementSum() < 0.001) return -1; //Make sure that null actions are punished
		
		int actionID = -1;
		double maxValue = Double.NEGATIVE_INFINITY;
		for (int j = 0; j < action.getNumElements(); j++){
			double d = action.get(j);
			if (d > maxValue){
				maxValue = d;
				actionID = j;
			}
		}
		double reward = curRewardFunction.reward(inputID, actionID);
		return reward;
	}
	
	
	/**
	 * Collects the output from the activator
	 * @param activator
	 * @return prediction and action for the next time step
	 */
	private SimpleMatrix[] collectOutput(Network_DataCollector activator){
		double[] predictionData = getOutput(activator);
		SimpleMatrix prediction = new SimpleMatrix(1, predictionData.length, true, predictionData);
		prediction.reshape(5, 5);
		
		double[] actionData = getAction(activator);
		SimpleMatrix actionNextTimeStep = new SimpleMatrix(1, actionData.length, true, actionData);

		//Set max value of action to 1. The rest to zero
		int max = maxID(actionNextTimeStep);
		if (max != -1){
			actionNextTimeStep.set(0);
			actionNextTimeStep.set(max, 1);
		}				
		SimpleMatrix[] result = {prediction, actionNextTimeStep};
		return result;
	}
	
	public double[] getOutput(Network_DataCollector network){
		ArrayList<Sensor> sensors = network.getSensors();
		double[] output = new double[sensors.size()];
		for (int i = 0; i < sensors.size()-1; i++){ //Substracting one because we don't want the action
			Sensor s = sensors.get(i);
			output[i] = s.getFeedbackOutput().get(0);
		}
		return output;
	}
	
	public double[] getAction(Network_DataCollector network){
		ArrayList<Sensor> sensors = network.getSensors();
		Sensor actionSensor = sensors.get(sensors.size()-1);
		double[] output = actionSensor.getFeedbackOutput().getMatrix().data;
		return output;
	}
	
	/**
	 * Return the id of the element with the max value in the matrix
	 * @param m
	 * @return
	 */
	private int maxID(SimpleMatrix m){
		double[] vector = m.getMatrix().data;
		
		
		double max = Double.NEGATIVE_INFINITY;
		int maxID = -1;
		int id = 0;
		
		for (double d : vector){
			if (d > max){
				max = d;
				maxID = id;
			}
			id++;
		}
		return maxID;
	}
	
	public void setSequence(int[] sequence){
		this.sequence = sequence;
	}
	
	public void setRewardFunctions(RewardFunction[] rewardFunctions){
		this.rewardFunctions = rewardFunctions;
		curRewardFunctionID = 0;
		curRewardFunction = rewardFunctions[curRewardFunctionID];
	}

}
