package stcl.fun.rps;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Random;

import javax.swing.JFrame;

import org.ejml.simple.SimpleMatrix;

import dk.itu.stcl.agents.QLearner;
import stcl.algo.brain.ActionDecider_Q;
import stcl.algo.brain.NeoCorticalUnit;
import stcl.algo.brain.Network_DataCollector;
import stcl.algo.brain.nodes.Sensor;
import stcl.algo.util.Normalizer;
import stcl.fun.rps.rewardfunctions.RewardFunction;
import stcl.fun.rps.rewardfunctions.RewardFunction_Inverse;
import stcl.fun.rps.rewardfunctions.RewardFunction_Standard;
import stcl.fun.rps.sequencecreation.SequenceBuilder;
import stcl.graphics.MPFGUI;

public class CopyOfSequenceRunner_NU_Stolen {
	
	private int[] sequence;
	private RewardFunction[] rewardFunctions;
	private RewardFunction curRewardFunction;
	private int curRewardFunctionID;
	private Random rand;
	private SimpleMatrix[] possibleInputs;
	
	private NeoCorticalUnit activator;
	
	//Variables have to be saved here to remember values between sequence runs
	private SimpleMatrix action;
	private SimpleMatrix prediction;
	
	boolean initializeRandomly = true;
	
	
	public static void main(String[] args) throws FileNotFoundException{
		String experimentRun = "C:/Users/Simon/Google Drev/Experiments/HTM/rps/Master data/evaluation/genomes/8 Simple Network";
		String genomeFile = experimentRun + "/SimpleNetwork_1_2.txt";
		
		
		RewardFunction[] functions = {new RewardFunction_Standard(), new RewardFunction_Inverse()};
		Random rand = new Random();
		int[] sequence = createSequences(rand);
		CopyOfSequenceRunner_NU_Stolen sr = new CopyOfSequenceRunner_NU_Stolen(sequence, functions, rand, genomeFile);
		double total = 0;
		for (int j = 0; j < 10; j++){
		
			int numSeq = 1000;
			for (int i = 0; i < numSeq; i++){
				double[] scores = sr.runSequence(1.0 - ((double)i / numSeq));
				double fitness = scores[1];
			}
			
			double itr_fitness = 0;
			for (int i = 0; i < 100; i++){
				double[] scores = sr.runSequence(0);
				double fitness = scores[1];
				itr_fitness += fitness;
			}
			
			total += itr_fitness / (double) 100;
		}
		
		double avg = total / (double) 10;
		System.out.println("Average: " + avg);
	}
	
	private static int[] createSequences(Random rand){
		int[] mySequence ={0};
		return mySequence;		
	}

	public CopyOfSequenceRunner_NU_Stolen(int[] sequence, RewardFunction[] rewardFunctions, Random rand, String networkfilename) throws FileNotFoundException {
		this.rand = rand;
		setSequence(sequence);
		setRewardFunctions(rewardFunctions);
		Network_DataCollector brain = setupBrain(networkfilename, rand);
		activator = brain.getUnitNodes().get(0).getUnit();
		//activator = new NeoCorticalUnit(25, 3, 3, 3, 3, true, false, true, rand);
		createInputs();
	}
	
	private Network_DataCollector setupBrain(String fileName, Random rand) throws FileNotFoundException{
		Network_DataCollector brain = new Network_DataCollector(fileName, rand);
		if (!initializeRandomly){
			brain.initialize(fileName, rand, true);
		}
		return brain;
	}
	
	private void createInputs(){
		double[][] rockData = {
				{0,0,0,0,0},
				{0,1,1,1,0},
				{0,1,1,1,0},
				{0,1,1,1,0},
				{0,0,0,0,0}
		};
		
		SimpleMatrix rock = new SimpleMatrix(rockData);
		rock.reshape(1, 25);
		
		double[][] paperData = {
				{1,1,1,1,1},
				{1,1,1,1,1},
				{1,1,1,1,1},
				{1,1,1,1,1},
				{1,1,1,1,1}
		};
		
		SimpleMatrix paper = new SimpleMatrix(paperData);
		paper.reshape(1, 25);
		
		double[][] scissorsData = {
				{0,0,0,1,0},
				{1,0,1,0,0},
				{0,1,0,0,0},
				{1,0,1,0,0},
				{0,0,0,1,0}
		};
		
		SimpleMatrix scissors = new SimpleMatrix(scissorsData);		
		scissors.reshape(1, 25);

		SimpleMatrix[] tmp = {rock, paper, scissors};

		possibleInputs = tmp;
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
		
		int state = 1;
		
		initializeSequence();
		
		for (int i = 0; i < sequence.length; i++){
			state = sequence[i];
			
			SimpleMatrix fbinputMatrix = new SimpleMatrix(5, 5);
			fbinputMatrix.set(1);
			fbinputMatrix = Normalizer.normalize(fbinputMatrix);
			
			activator.feedBackward(fbinputMatrix);
			int myAction = activator.getNextAction();
			if (rand.nextDouble() < explorationChance){
				myAction = rand.nextInt(3);
			}			
			
			double reward_now = curRewardFunction.reward(state, myAction);
			totalGameScore += reward_now;	
			
			SimpleMatrix ffInput = possibleInputs[state];
			
			activator.resetActivity();

			activator.feedForward(ffInput, reward_before, myAction);
			
			reward_before = reward_now;
			
		}
		endSequence(reward_before);
		
		activator.newEpisode();

		
		//endSequence(activator, reward_before);
		
		double avgPredictionError = totalPredictionError / (double) sequence.length;
		double avgScore = totalGameScore / (double) sequence.length;
		double predictionSuccess = 1 - avgPredictionError;
		
		//Scores can't be less than zero as the evolutionary algorithm can't work with that
		
		double[] result = {predictionSuccess, avgScore};
		return result;
	}
	
	private void initializeSequence(){
		//Give blank input and action to network
		SimpleMatrix initialInput = new SimpleMatrix(1, 25);
		SimpleMatrix initialAction = new SimpleMatrix(1, 3);
		activator.feedForward(initialInput, 0, 0);

	}
	
	private void endSequence(double reward){
		//Give blank input and action to network
		SimpleMatrix initialInput = new SimpleMatrix(1, 25);
		SimpleMatrix initialAction = new SimpleMatrix(1, 3);
		activator.feedForward(initialInput, reward, 0);
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
