package stcl.fun.rps;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.Connection;
import stcl.algo.brain.NU;
import stcl.algo.brain.NeoCorticalUnit;
import stcl.algo.brain.NeoCorticalUnit4;

public class RockPaperScissors {

	private Random rand = new Random(1234);
	private NU unit1, unit2;
	private Connection conn;
	private SimpleMatrix rock, paper, scissors;
	private SimpleMatrix[] sequence;
	private int[] labelSequence;
	private SimpleMatrix rewardMatrix;
	
	private final int ITERATIONS = 1000;
	
	public static void main(String[] args) {
		RockPaperScissors runner = new RockPaperScissors();
		runner.run();

	}
	
	public RockPaperScissors() {
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
		int spatialMapSize = 10;
		int temporalMapSize = 10;
		double initialPredictionLearningRate = 0.1;
		boolean useMarkovPrediction = true;
		double decayFactor = 0.3;
		int markovOrder = 2;
		unit1 = new NeoCorticalUnit4(rand, ffInputLength, spatialMapSize, temporalMapSize, initialPredictionLearningRate, useMarkovPrediction,  markovOrder);
		unit2 = new NeoCorticalUnit4(rand, temporalMapSize * temporalMapSize, spatialMapSize, temporalMapSize, initialPredictionLearningRate, useMarkovPrediction, markovOrder);
		conn = new Connection(unit1, unit2, rand, 0.3, 2, 0.3);
	}
	
	private void runExperiment(int maxIterations){
		int curInput = 0;
		double externalReward = 0;
		double[][] tmp = {{0}};
		SimpleMatrix curAction = new SimpleMatrix(tmp);
		SimpleMatrix label = new SimpleMatrix(tmp);
		SimpleMatrix prediction = new SimpleMatrix(5, 5);
		int labelID = 0;
		int curActionID = 0;
		for (int i = 0; i < maxIterations; i++){
			//Get input
			SimpleMatrix input = new SimpleMatrix(sequence[curInput]);
			label.set(labelSequence[curInput]);
			SimpleMatrix diff = input.minus(prediction);
			double predictionError = diff.normF();
			int labelError = labelID == labelSequence[curInput] ? 0 : 1;
			
			System.out.println("Iteration " + i + " spatialError: " + predictionError + " labelError: " + labelError);
			
			externalReward = reward(labelID, curActionID);
			
			//System.out.println(i + " " + externalReward);
			
			//Reshape input to vector
			input.reshape(1, 25);
			
			//Add noise to input
			input = addNoise(input, 0.0);
			
			//Combine with action vector
			SimpleMatrix combinedInput = input.combine(0, input.END, curAction);
			
			//Combine with label
			combinedInput = combinedInput.combine(0, input.END, label);
			
			//Feed forward
			SimpleMatrix ffOutput1 = conn.feedForward(combinedInput, externalReward, 0.1);
			
			//Reshape ff output from unit 1
			SimpleMatrix out = new SimpleMatrix(1, 100, true, ffOutput1.getMatrix().data);
			
			//Feed through unit 2
			unit2.feedForward(out);
			SimpleMatrix ffOutput = unit2.getFfOutput();
			
			//Feed back through unit 2
			unit2.feedBackward(ffOutput);
			
			//Feed back through unit 1
			SimpleMatrix fbOutput = conn.feedBack(0.1);
			
			//Collect next action and prediction		
			curActionID = Math.round(Math.round(fbOutput.get(fbOutput.getNumElements() - 2)));
			
			labelID = Math.round(Math.round(fbOutput.get(fbOutput.getNumElements() - 1)));
						
			prediction = fbOutput.extractMatrix(0, 1, 0, fbOutput.getNumElements() - 2);
			
			prediction.reshape(5, 5);
			
			
			
			curInput++;
			if (curInput >= sequence.length){
				curInput = 0;
			}
		}
	}
	
	
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
