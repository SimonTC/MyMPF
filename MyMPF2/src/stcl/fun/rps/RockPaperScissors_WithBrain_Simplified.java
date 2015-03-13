package stcl.fun.rps;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.Brain;
import stcl.algo.brain.Connection;
import stcl.algo.brain.NU;
import stcl.algo.brain.NeoCorticalUnit;
import stcl.algo.poolers.RSOM;

public class RockPaperScissors_WithBrain_Simplified {

	private Random rand = new Random(1234);
	private Brain brain;
	private SimpleMatrix rock, paper, scissors;
	private SimpleMatrix[] sequence;
	private int[] labelSequence;
	private SimpleMatrix rewardMatrix;
	
	private final int ITERATIONS = 1000;
	
	public static void main(String[] args) {
		RockPaperScissors_WithBrain_Simplified runner = new RockPaperScissors_WithBrain_Simplified();
		runner.run();

	}
	
	public RockPaperScissors_WithBrain_Simplified() {
		// TODO Auto-generated constructor stub
	}
	
	public void run(){
		setup(ITERATIONS);
		runExperiment(ITERATIONS);
	}
	
	private void setup(int maxIterations){
		createInputs();
		createRewardMatrix();
		int ffInputLength = 2; //Label and action
		int spatialMapSize = 5;
		int temporalMapSize = 5;
		int markovOrder = 2;
		brain = new Brain(2, rand, ffInputLength, spatialMapSize, temporalMapSize, markovOrder);
		
	}
	
	private void runExperiment(int maxIterations){
		int curInput = 0;
		double externalReward = 0;
		
		double[][] tmp = {{0}};
		SimpleMatrix actionNext = new SimpleMatrix(tmp); //m(t+1)
		int actionIDNow = 0;
		int actionIDNext = 0;
		int actionIDAfterNext = 0;
		
		SimpleMatrix input = new SimpleMatrix(tmp);
		int predictedLabel = 0;
		
		for (int i = 0; i < maxIterations; i++){
			//Update action chain
			actionIDNow = actionIDNext;
			actionIDNext = actionIDAfterNext;
			actionIDAfterNext = -1; //Just to keep track			
			
			//Get input
			int inputLabel = labelSequence[curInput];
			input.set(inputLabel);

			//int labelError = predictedLabel == inputLabel ? 0 : 1;		
			
			externalReward = reward(inputLabel, actionIDNow);
			System.out.println(inputLabel + " " + predictedLabel);
			
			//System.out.println(i + " " + inputLabel + " " + actionIDNow);
			//System.out.println("Iteration " + i + " Reward: " + externalReward);
			//System.out.println("Iteration " + i + " labelError: " + labelError);
			
			//System.out.println(i + " " + externalReward);

			//Combine with action vector
			actionNext.set(actionIDNext);
			SimpleMatrix combinedInput = input.combine(0, input.END, actionNext);
			
			SimpleMatrix fbOutput = brain.step(combinedInput, externalReward);
			
			//Collect action that will be done at timestep t + 2	
			
			actionIDAfterNext = Math.round(Math.round(fbOutput.get(fbOutput.getNumElements() - 1)));
			
			//Collect prediction and label of input that we expect to see at timestep t + 1
			predictedLabel = Math.round(Math.round(fbOutput.get(fbOutput.getNumElements() - 2)));			
			
			curInput++;
			if (curInput >= sequence.length){
				curInput = 0;
			}
		}
		
		//printInformation();
	}
	
	private void printInformation(){
		System.out.println();
		System.out.println("Spatial groups in unit 1");
		brain.getUnitList().get(0).getSpatialPooler().printModelWeigths();
		System.out.println();
		System.out.println("Temporal groups in unit 1");
		brain.getUnitList().get(0).getSequencer().printSequenceMemory();
		System.out.println();
		System.out.println("Prediction model, unit 1");
		brain.getUnitList().get(0).printPredictionModel();
		
		
		System.out.println();
		System.out.println("Spatial groups in unit 2");
		brain.getUnitList().get(1).getSpatialPooler().printModelWeigths();
		System.out.println();
		System.out.println("Temporal groups in unit 2");
		brain.getUnitList().get(1).getSequencer().printSequenceMemory();
		System.out.println("Prediction model, unit 2");
		brain.getUnitList().get(1).printPredictionModel();
	}
	
	/**
	 * 
	 * @param opponentSymbol Symbol played by opponent
	 * @param playerSymbol SYmbol played by AI
	 * @return
	 */
	private double reward(int opponentSymbol, int playerSymbol){
		double reward = rewardMatrix.get(playerSymbol, opponentSymbol);
		return reward;
	}
	
	protected SimpleMatrix addNoise(SimpleMatrix m, double noiseMagnitude){
		double noise = (rand.nextDouble() - 0.5) * 2 * noiseMagnitude;
		m = m.plus(noise);
		return m;
	}

	private void createInputs(){
		double[][] rockData = {
				{0,0,0,0,0},
				{0,1,1,1,0},
				{0,1,1,1,0},
				{0,1,1,1,0},
				{0,0,0,0,0}
		};
		
		rock = new SimpleMatrix(rockData);
		
		double[][] paperData = {
				{1,1,1,1,1},
				{1,1,1,1,1},
				{1,1,1,1,1},
				{1,1,1,1,1},
				{1,1,1,1,1}
		};
		
		paper = new SimpleMatrix(paperData);
		
		double[][] scissorsData = {
				{0,0,0,1,0},
				{1,0,1,0,0},
				{0,1,0,0,0},
				{1,0,1,0,0},
				{0,0,0,1,0}
		};
		
		scissors = new SimpleMatrix(scissorsData);
		
		SimpleMatrix[] tmp = {rock, paper, paper, scissors};
		int[] lbl = {0,1,1,2};
		labelSequence = lbl;
		sequence = tmp;			
	}
	
	private void createRewardMatrix(){
		double[][]data = {
				{0,-1,1},
				{1,0,-1},
				{-1,1,0}
		};
		
		rewardMatrix = new SimpleMatrix(data);
	}
}
