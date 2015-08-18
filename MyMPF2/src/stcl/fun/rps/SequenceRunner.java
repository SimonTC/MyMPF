package stcl.fun.rps;

import java.util.ArrayList;
import java.util.Random;

import javax.swing.JFrame;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.Network_DataCollector;
import stcl.algo.brain.nodes.Sensor;
import stcl.algo.util.Normalizer;
import stcl.fun.rps.rewardfunctions.RewardFunction;
import stcl.graphics.MPFGUI;

public class SequenceRunner {
	
	protected SimpleMatrix[] possibleInputs;
	private SimpleMatrix[] patternsToTestAgainst;
	protected int[] sequence;
	private RewardFunction[] rewardFunctions;
	protected RewardFunction curRewardFunction;
	private int curRewardFunctionID;
	protected Random rand;
	private ArrayList<SimpleMatrix> possibleActions;
	private double noiseMagnitude;
	
	public SequenceRunner(int[] sequence, SimpleMatrix[] possibleInputs, RewardFunction[] rewardFunctions, Random rand, double noiseMagnitude) {
		this.possibleInputs = possibleInputs;
		this.rand = rand;
		setSequence(sequence);
		setRewardFunctions(rewardFunctions);
		reset(false);
		this.possibleActions = createPossibleActions();
		this.noiseMagnitude = noiseMagnitude;
		this.patternsToTestAgainst = createPatternsToTestAgainst(possibleInputs);
	}
	
	/**
	 * Reset all variables to their initial values.
	 */
	public void reset(boolean gotoNextRewardFunction){
		double[][] tmp = {{1,0,0}};
		if (gotoNextRewardFunction){
			curRewardFunctionID++;
			if (curRewardFunctionID == rewardFunctions.length) curRewardFunctionID = 0;
			curRewardFunction = rewardFunctions[curRewardFunctionID];
		}
	}
	
	private SimpleMatrix[] createPatternsToTestAgainst(SimpleMatrix[] possibleInputs){
		int cols = possibleInputs[0].numCols();
		int rows = possibleInputs[0].numRows();
		SimpleMatrix[] testPatterns = new SimpleMatrix[possibleInputs.length * 2];
		int counter = 0;
		for (int i = 0; i < possibleInputs.length; i++){
			testPatterns[i] = new SimpleMatrix(rows, cols);
			for (int j = 0; j < testPatterns[i].getNumElements(); j++){
				testPatterns[i].set(j, rand.nextDouble());
			}
			counter = i;
		}
		counter += 1;
		for (int i = 0; i < possibleInputs.length; i++){
			testPatterns[i + counter] = new SimpleMatrix(possibleInputs[i]);
			
		}
		return testPatterns;
	}
	
	private ArrayList<SimpleMatrix> createPossibleActions(){
		double[][] rock = {{1,0,0}};
		double[][] paper = {{0,1,0}};
		double[][] scissors = {{0,0,1}};
		double[][] empty = {{0,0,0}};
		
		SimpleMatrix r = new SimpleMatrix(rock);
		SimpleMatrix p = new SimpleMatrix(paper);
		SimpleMatrix s = new SimpleMatrix(scissors);
		SimpleMatrix e = new SimpleMatrix(empty);
		ArrayList<SimpleMatrix> arr = new ArrayList<SimpleMatrix>();
		arr.add(r);
		arr.add(p);
		arr.add(s);
		arr.add(e);
		return arr;
	}
	
	/**
	 * Goes through the sequence once.
	 * Remember to call reset() if the evaluation should start from scratch
	 * @param activator
	 * @return Array containing prediction success and fitness in the form [prediction,fitness]
	 */
	public double[] runSequence(Network_DataCollector activator, MPFGUI gui){
		double totalPredictionError = 0;
		double totalGameScore = 0;
		double reward_before = 0;
		
		int state = 1;
		activator.getActionNode().setPossibleActions(possibleActions);
		
		initializeSequence(activator);
		
		for (int i = 0; i < sequence.length; i++){
			
			//Get input			
			state = sequence[i];
			SimpleMatrix input = possibleInputs[state];
			SimpleMatrix noisyInput = addNoise(input, noiseMagnitude);
						
			//Collect output
			activator.feedback();
			
			//activator.collectFeedBackData();
			SimpleMatrix[] output = collectOutput(activator);
			SimpleMatrix prediction = output[0];
			SimpleMatrix myAction = output[1];
						
			activator.resetUnitActivity();
			
			double reward_now = calculateReward(myAction, state);
			totalGameScore += reward_now;	
			
			double predictionError = calculatePredictionError(prediction, input);
			totalPredictionError += predictionError;
			
			giveInputsToActivator(activator, noisyInput, myAction);
			activator.feedForward(reward_before);
			//activator.collectFeedForwardData();
			reward_before = reward_now;
			//activator.printDataToFiles();
			
		}
		
		endSequence(activator, reward_before);
		
		activator.newEpisode();
		
		double avgPredictionError = totalPredictionError / (double) sequence.length;
		double avgScore = totalGameScore / (double) sequence.length;
		double predictionSuccess = 1 - avgPredictionError;
		
		double[] result = {predictionSuccess, avgScore};
		return result;
	}
	
	public double[] runSequence(Network_DataCollector activator, MPFGUI gui, double explorationChance){
		double totalPredictionError = 0;
		double totalGameScore = 0;
		double reward_before = 0;
		
		int state;
		activator.getActionNode().setPossibleActions(possibleActions);
		
		initializeSequence(activator);
		
		for (int i = 0; i < sequence.length; i++){
			
			//Get input			
			state = sequence[i];
			SimpleMatrix input = possibleInputs[state];
			SimpleMatrix noisyInput = addNoise(input, noiseMagnitude);
						
			//Collect output
			activator.feedback();
			
			//activator.collectFeedBackData();
			SimpleMatrix[] output = collectOutput(activator);
			SimpleMatrix prediction = output[0];
			SimpleMatrix myAction = output[1];
			int realVote = activator.getUnitNodes().get(0).getUnit().getNextAction();
			if (rand.nextDouble() < explorationChance){
				realVote = rand.nextInt(3);
				myAction.set(0);
				if (realVote < 3) myAction.set(realVote, 1);
			}	
			
			
			activator.resetUnitActivity();
			
			double reward_now = calculateReward(myAction, state);
			totalGameScore += reward_now;	
			
			double predictionError = calculatePredictionError(prediction, input);
			totalPredictionError += predictionError;
			
			giveInputsToActivator(activator, noisyInput, myAction);
			activator.feedForward(reward_before);
			//activator.collectFeedForwardData();
			reward_before = reward_now;
			//activator.printDataToFiles();
			
		}
		
		endSequence(activator, reward_before);
		
		activator.newEpisode();
		
		double avgPredictionError = totalPredictionError / (double) sequence.length;
		double avgScore = totalGameScore / (double) sequence.length;
		double predictionSuccess = 1 - avgPredictionError;
		
		double[] result = {predictionSuccess, avgScore};
		return result;
	}
	
	private void initializeSequence(Network_DataCollector activator){
		//Give blank input and action to network
		SimpleMatrix initialInput = new SimpleMatrix(5, 5);
		SimpleMatrix initialAction = new SimpleMatrix(1, 3);
		giveInputsToActivator(activator, initialInput, initialAction);
		
		activator.feedForward(0);

	}
	
	private void endSequence(Network_DataCollector activator, double reward){
		//Give blank input and action to network
		SimpleMatrix input = new SimpleMatrix(5, 5);
		SimpleMatrix action = new SimpleMatrix(1, 3);
		giveInputsToActivator(activator, input, action);
		
		activator.feedForward(reward);
	}
	
	private double calculatePredictionError(SimpleMatrix prediction, SimpleMatrix actual){
		double minError = Double.POSITIVE_INFINITY;
		SimpleMatrix bestMatch = null;
		
		for (SimpleMatrix m : patternsToTestAgainst){
			SimpleMatrix diff = m.minus(prediction);
			double d = diff.normF();	
			if (d < minError){
				minError = d;
				bestMatch = m;
			}
		}		
		
		double predictionError = 1;
		if (bestMatch.isIdentical(actual, 0.001)) {
			predictionError = 0;
		}
		
		return predictionError;
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
		if (action.elementSum() < 0.001) return 0; //Make sure that null actions are punished
		
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
	
	public void setNoiseMagnitude(double d){
		this.noiseMagnitude = d;
	}
	
	public void setRewardFunctions(RewardFunction[] rewardFunctions){
		this.rewardFunctions = rewardFunctions;
		curRewardFunctionID = 0;
		curRewardFunction = rewardFunctions[curRewardFunctionID];
	}

}
