package stcl.fun.rps;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.Brain;

public class RockPaperScissors_WithBrain {

	private Random rand = new Random(1234);
	private Brain brain;
	private SimpleMatrix rock, paper, scissors;
	private SimpleMatrix[] sequence;
	private int[] labelSequence;
	private SimpleMatrix rewardMatrix;
	
	private final int ITERATIONS = 2000;
	
	public static void main(String[] args) {
		RockPaperScissors_WithBrain runner = new RockPaperScissors_WithBrain();
		runner.run();

	}
	
	public RockPaperScissors_WithBrain() {
		// TODO Auto-generated constructor stub
	}
	
	public void run(){
		setup(ITERATIONS);
		runExperiment(ITERATIONS);
	}
	
	private void setup(int maxIterations){
		createInputs();
		createRewardMatrix();
		int ffInputLength = rock.numCols() * rock.numCols() + 2; //Adding two for the label + action id
		int spatialMapSize = 5;
		int temporalMapSize = 5;
		int markovOrder = 2;
		brain = new Brain(2, rand, ffInputLength, spatialMapSize, temporalMapSize, markovOrder);
		
	}
	
	private void runExperiment(int maxIterations){
		int curInput = 0;
		double externalReward = 0;
		
		double[][] tmp = {{0}};
		SimpleMatrix actionNow = new SimpleMatrix(tmp); //m(t)
		SimpleMatrix actionNext = new SimpleMatrix(tmp); //m(t+1)
		SimpleMatrix actionAfterNext= new SimpleMatrix(tmp); //m(t+2)
		int actionIDNow = 0;
		int actionIDNext = 0;
		int actionIDAfterNext = 0;
		
		SimpleMatrix label = new SimpleMatrix(tmp);
		SimpleMatrix prediction = new SimpleMatrix(5, 5);
		int predictedLabel = 0;
		
		for (int i = 0; i < maxIterations; i++){
		if ( i == 500){
				System.out.println();;
			}
			
			//Update action chain
			actionIDNow = actionIDNext;
			actionIDNext = actionIDAfterNext;
			actionIDAfterNext = -1; //Just to keep track			
			
			//Get input
			SimpleMatrix input = new SimpleMatrix(sequence[curInput]);
			label.set(labelSequence[curInput]);
			
			SimpleMatrix diff = input.minus(prediction);
			double predictionError = diff.normF();
			int labelError = predictedLabel == labelSequence[curInput] ? 0 : 1;
			
			int inputLabel = labelSequence[curInput];
			externalReward = reward(inputLabel, actionIDNow);
			
			//System.out.println(i + " " + inputLabel + " " + actionIDNow);
			//System.out.println("Iteration " + i + " Reward: " + externalReward);
			//System.out.println("Iteration " + i + " spatialError: " + predictionError + " labelError: " + labelError);
			System.out.println(predictionError);
			//System.out.println(i + " " + externalReward);
			
			//Reshape input to vector
			input.reshape(1, 25);
			
			//Add noise to input
			input = addNoise(input, 0.0);
			
			//Combine with action vector
			actionNext.set(actionIDNext);
			SimpleMatrix combinedInput = input.combine(0, input.END, actionNext);
			
			//Combine with label
			combinedInput = combinedInput.combine(0, input.END, label);
			
			SimpleMatrix fbOutput = brain.step(combinedInput, externalReward);
			
			//Collect action that will be done at timestep t + 2	
			
			actionIDAfterNext = Math.round(Math.round(fbOutput.get(fbOutput.getNumElements() - 2)));
			
			//Collect prediction and label of input that we expect to see at timestep t + 1
			predictedLabel = Math.round(Math.round(fbOutput.get(fbOutput.getNumElements() - 1)));
			prediction = fbOutput.extractMatrix(0, 1, 0, fbOutput.getNumElements() - 2);
			prediction.reshape(5, 5);
			
			
			
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
