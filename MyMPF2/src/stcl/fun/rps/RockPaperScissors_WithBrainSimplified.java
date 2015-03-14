package stcl.fun.rps;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.Brain;

public class RockPaperScissors_WithBrainSimplified {

	private Random rand = new Random(1234);
	private Brain brain;
	private SimpleMatrix rock, paper, scissors;
	private SimpleMatrix[] sequence;
	private int[] labelSequence;
	private SimpleMatrix rewardMatrix;
	
	private final int ITERATIONS = 1000;
	
	public static void main(String[] args) {
		RockPaperScissors_WithBrainSimplified runner = new RockPaperScissors_WithBrainSimplified();
		runner.run();

	}
	
	public RockPaperScissors_WithBrainSimplified() {
		// TODO Auto-generated constructor stub
	}
	
	public void run(){
		setup(ITERATIONS);
		runExperiment(ITERATIONS);
	}
	
	private void setup(int maxIterations){
		createInputs();
		createRewardMatrix();
		int ffInputLength = rock.numCols() * rock.numRows() * 2; //Multiplying by two to make room for both input and action
		int spatialMapSize = 3;
		int temporalMapSize = 2;
		int markovOrder = 2;
		brain = new Brain(2, rand, ffInputLength, spatialMapSize, temporalMapSize, markovOrder);
		
	}
	
	private void runExperiment(int maxIterations){
		int curInput = 0;
		double externalReward = 0;
		
		double[][] tmp = {{0,0,0}};
		SimpleMatrix actionNow = new SimpleMatrix(tmp); //m(t)
		SimpleMatrix actionNext = new SimpleMatrix(tmp); //m(t+1)
		SimpleMatrix actionAfterNext = new SimpleMatrix(tmp); //m(t+2)

		SimpleMatrix prediction = new SimpleMatrix(tmp);
		int predictedLabel = 0;
		
		for (int i = 0; i < maxIterations; i++){
			//Update action chain
			actionNow = actionNext;
			actionNext = actionAfterNext;
			actionAfterNext = null;
			
			//Get input			
			SimpleMatrix input = new SimpleMatrix(sequence[curInput]);
			
			//Calculate prediction error
			SimpleMatrix diff = input.minus(prediction);
			double predictionError = diff.normF();
			
			//Calculate reward
			
			if ( i > 3){ //To get out of wrong actions
				int actionID = -1;
				if (actionNow.minus(rock).normF() < 0.001) actionID = 0;
				if (actionNow.minus(paper).normF() < 0.001) actionID = 1;
				if (actionNow.minus(scissors).normF() < 0.001) actionID = 2;
				
				externalReward = reward(labelSequence[curInput], actionID);
			}
			
			//Combine input with the action that will be performed at time t+1
			SimpleMatrix combinedInput = input.combine(0, input.END, actionNext);
			
			//Give combined input to brain and collect output
			SimpleMatrix output = brain.step(combinedInput, externalReward);
			
			//Extract prediction
			prediction = output.extractMatrix(0, 1, 0, 3);
			
			//Extract action to perform at t+2
			actionAfterNext = output.extractMatrix(0, 1, 3, output.END);
			
			//Set max value of action to 1. The rest to zero
			int max = maxID(actionAfterNext);
			if (max != -1){
				actionAfterNext.set(0);
				actionAfterNext.set(max, 1);
			}
			
			//Print info
			System.out.println(i + " Error: " + predictionError + " Reward: " + externalReward);
			
			curInput++;
			if (curInput >= sequence.length){
				curInput = 0;
			}
			
		}
		
		printInformation();
	}
	
	private int maxID(SimpleMatrix m){
		//Transform bias matrix into vector
				double[] vector = m.getMatrix().data;
				
				//Go through bias vector until value is >= random number
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
				{1,0,0}
		};
		
		rock = new SimpleMatrix(rockData);
		
		double[][] paperData = {
				{0,1,0}
		};
		
		paper = new SimpleMatrix(paperData);
		
		double[][] scissorsData = {
				{0,0,1}
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
