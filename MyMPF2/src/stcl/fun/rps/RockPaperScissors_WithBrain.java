package stcl.fun.rps;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.Brain;
import stcl.algo.brain.Brain_DataCollector;

public class RockPaperScissors_WithBrain {

	private Random rand = new Random(1234);
	private Brain_DataCollector brain;
	private SimpleMatrix rock, paper, scissors, blank;
	private SimpleMatrix[] sequence;
	private int[] labelSequence;
	private SimpleMatrix rewardMatrix;
	private int inputLength, actionLength;
	
	private final int ITERATIONS = 20000;
	
	public static void main(String[] args) {
		RockPaperScissors_WithBrain runner = new RockPaperScissors_WithBrain();
		String folder = "C:/Users/Simon/Documents/Experiments/RPS";
		runner.run(folder);
	}
	
	public RockPaperScissors_WithBrain() {
		// TODO Auto-generated constructor stub
	}
	
	public void run(String dataFolder){
		setup(ITERATIONS, dataFolder);
		
		//Show
		
		//Train
		runExperiment(ITERATIONS, 0.5, false);
		
		//Evaluate
		brain.flush();
		brain.setLearning(false);
		brain.openFiles(true);
		runExperiment(1000, 0.0, true);
		
		
	}
	
	private void setup(int maxIterations, String dataFolder){
		createInputs();
		createRewardMatrix();
		inputLength = rock.getNumElements();
		actionLength = rock.getNumElements();
		int ffInputLength = inputLength + actionLength;
		int spatialMapSize = 3;
		int temporalMapSize = 3;
		int markovOrder = 2;
		brain = new Brain_DataCollector(2, rand, ffInputLength, spatialMapSize, temporalMapSize, markovOrder, dataFolder, false, false);
		
	}
	
	private void runExperiment(int maxIterations, double exploreChance, boolean printError){
		int curInput = 0;
		double externalReward = 0;
		SimpleMatrix actionNow = new SimpleMatrix(rock); //m(t)
		SimpleMatrix actionNext = new SimpleMatrix(rock); //m(t+1)
		//SimpleMatrix actionAfterNext = new SimpleMatrix(tmp); //m(t+2)

		SimpleMatrix prediction = blank;
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
				double minDiff = Double.POSITIVE_INFINITY;
				double diff_rock = actionNow.minus(rock).normF();
				double diff_paper = actionNow.minus(paper).normF();
				double diff_scissors = actionNow.minus(scissors).normF();
				
				if (diff_rock < minDiff){
					actionID = 0;
					minDiff = diff_rock;					
				}
				if (diff_paper < minDiff){
					actionID = 1;
					minDiff = diff_paper;					
				}
				if (diff_scissors < minDiff){
					actionID = 2;
					minDiff = diff_scissors;					
				}
				
				externalReward = reward(labelSequence[curInput], actionID);
			}
			
			//Combine input with the action that will be performed at time t+1
			SimpleMatrix combinedInput = new SimpleMatrix(1, input.getNumElements(), true, input.getMatrix().data);
			combinedInput = combinedInput.combine(0, input.END, actionNow);
			
			//Give combined input to brain and collect output
			SimpleMatrix output = brain.step(combinedInput, externalReward);
			
			//Extract prediction and reshape to matrix
			prediction = output.extractMatrix(0, 1, 0, output.numCols() - actionLength);
			prediction.reshape(5, 5);
			
			//Extract action to perform at t+1
			actionNext = output.extractMatrix(0, 1, output.numCols() - actionLength, output.numCols());
			actionNext.reshape(5, 5);
			if (rand.nextDouble() < exploreChance){
				//Do exploration by choosing random action
				int nextAction = rand.nextInt(3);
				switch(nextAction){
				case 0: actionNext = new SimpleMatrix(rock); break;
				case 1: actionNext = new SimpleMatrix(paper); break;
				case 2: actionNext = new SimpleMatrix(scissors); break;
				}				
			} else {
				int actionID = -1;
				double minDiff = Double.POSITIVE_INFINITY;
				double diff_rock = actionNow.minus(rock).normF();
				double diff_paper = actionNow.minus(paper).normF();
				double diff_scissors = actionNow.minus(scissors).normF();
				
				if (diff_rock < minDiff){
					actionID = 0;
					minDiff = diff_rock;					
				}
				if (diff_paper < minDiff){
					actionID = 1;
					minDiff = diff_paper;					
				}
				if (diff_scissors < minDiff){
					actionID = 2;
					minDiff = diff_scissors;					
				}
				switch(actionID){
				case 0: actionNext = new SimpleMatrix(rock); break;
				case 1: actionNext = new SimpleMatrix(paper); break;
				case 2: actionNext = new SimpleMatrix(scissors); break;
				}
			}
		
			if (printError) System.out.println(i + " Error: " + predictionError + " Reward: " + externalReward);
			
			curInput++;
			if (curInput >= sequence.length){
				curInput = 0;
			}
			
		}
		brain.closeFiles();
		System.out.println();
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
		//System.out.println("Sequences observed by unit 1:");
		//brain.getUnitList().get(0).getSequencer().printTrie();
		
		
		System.out.println();
		System.out.println("Spatial groups in unit 1");
		brain.getUnitList().get(0).getSpatialPooler().printModelWeigths();
		System.out.println();
		//System.out.println("Temporal groups in unit 1");
		//brain.getUnitList().get(0).getSequencer().printSequenceMemory();
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
		
		double[][] blankData = {
				{0,0,0,0,0},
				{0,0,0,0,0},
				{0,0,0,0,0},
				{0,0,0,0,0},
				{0,0,0,0,0}
		};
		
		blank = new SimpleMatrix(blankData);
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
