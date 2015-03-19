package stcl.fun.rps;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.Brain;
import stcl.algo.brain.Brain_DataCollector;

public class RockPaperScissors_WithBrainSimplified {

	private Random rand = new Random(1234);
	private Brain_DataCollector brain;
	private SimpleMatrix rock, paper, scissors;
	private SimpleMatrix[] sequence;
	private int[] labelSequence;
	private SimpleMatrix rewardMatrix;
	
	private final int ITERATIONS = 1000;
	
	public static void main(String[] args) {
		RockPaperScissors_WithBrainSimplified runner = new RockPaperScissors_WithBrainSimplified();
		String folder = "C:/Users/Simon/Documents/Experiments/RPS";
		runner.run(folder);
	}
	
	public RockPaperScissors_WithBrainSimplified() {
		// TODO Auto-generated constructor stub
	}
	
	public void run(String dataFolder){
		setup(ITERATIONS, dataFolder);
		
		//Show
		
		//Train
		runExperiment(ITERATIONS, 0.5);
		
		//Evaluate
		brain.flush();
		brain.setLearning(false);
		brain.openFiles(true);
		runExperiment(ITERATIONS, 0.0);
		
		brain.getConnectionList().get(0).getCorrelationMatrix().print();
		
		
	}
	
	private void setup(int maxIterations, String dataFolder){
		createInputs();
		createRewardMatrix();
		int ffInputLength = rock.numCols() * rock.numRows() * 2; //Multiplying by two to make room for both input and action
		int spatialMapSize = 3;
		int temporalMapSize = 2;
		int markovOrder = 2;
		brain = new Brain_DataCollector(2, rand, ffInputLength, spatialMapSize, temporalMapSize, markovOrder, dataFolder, false, true);
		
	}
	
	private void runExperiment(int maxIterations, double exploreChance){
		int curInput = 0;
		double externalReward = 0;
		
		double[][] tmp = {{1,0,0}};
		SimpleMatrix actionNow = new SimpleMatrix(tmp); //m(t)
		SimpleMatrix actionNext = new SimpleMatrix(tmp); //m(t+1)
		//SimpleMatrix actionAfterNext = new SimpleMatrix(tmp); //m(t+2)

		SimpleMatrix prediction = new SimpleMatrix(tmp);
		int predictedLabel = 0;
		
		for (int i = 0; i < maxIterations; i++){
			//Update action chain
			actionNow = actionNext;
			actionNext = null;// actionAfterNext;
			//actionAfterNext = null;
			
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
			SimpleMatrix combinedInput = input.combine(0, input.END, actionNow);
			
			//Give combined input to brain and collect output
			SimpleMatrix output = brain.step(combinedInput, externalReward);
			
			//Extract prediction
			prediction = output.extractMatrix(0, 1, 0, 3);
			
			//Extract action to perform at t+2
			actionNext = output.extractMatrix(0, 1, 3, output.END);
			if (rand.nextDouble() < exploreChance){
				//Do exploration by choosing random action
				actionNext.set(0);
				actionNext.set(rand.nextInt(actionNext.getNumElements()), 1);
				
			} else {
				//Set max value of action to 1. The rest to zero
				int max = maxID(actionNext);
				if (max != -1){
					actionNext.set(0);
					actionNext.set(max, 1);
				}
			}
		
			System.out.println(i + " Error: " + predictionError + " Reward: " + externalReward);
			
			curInput++;
			if (curInput >= sequence.length){
				curInput = 0;
			}
			
		}
		brain.closeFiles();
		//printInformation();
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
		System.out.println("Sequences observed by unit 1:");
		brain.getUnitList().get(0).getSequencer().printTrie();
		
		
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
		
		SimpleMatrix[] tmp = {rock, paper, scissors};
		int[] lbl = {0,1,2};
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
