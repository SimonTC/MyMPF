package stcl.fun.rps;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.Connection;
import stcl.algo.brain.NeoCorticalUnit;

public class RockPaperScissors {

	private Random rand = new Random(1234);
	private NeoCorticalUnit unit1, unit2;
	private Connection conn;
	private SimpleMatrix rock, paper, scissors;
	private SimpleMatrix[] sequence;
	
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
		int ffInputLength = rock.numCols() * rock.numCols();
		int spatialMapSize = 10;
		int temporalMapSize = 10;
		double initialPredictionLearningRate = 0.1;
		boolean useMarkovPrediction = true;
		double decayFactor = 0.3;
		unit1 = new NeoCorticalUnit(rand, maxIterations, ffInputLength, spatialMapSize, temporalMapSize, initialPredictionLearningRate, useMarkovPrediction, decayFactor);
		unit2 = new NeoCorticalUnit(rand, maxIterations, temporalMapSize * temporalMapSize, spatialMapSize, temporalMapSize, initialPredictionLearningRate, useMarkovPrediction, decayFactor);
		conn = new Connection(unit1, unit2, rand, 0.3, 2, 0.3);
	}
	
	private void runExperiment(int maxIterations){
		int curInput = 0;
		double externalReward = 0;
		double[][] tmp = {{0}};
		SimpleMatrix curAction = new SimpleMatrix(tmp);
		SimpleMatrix prediction = new SimpleMatrix(5, 5);
		for (int i = 0; i < maxIterations; i++){
			//Get input
			SimpleMatrix input = new SimpleMatrix( sequence[curInput]);
			SimpleMatrix diff = input.minus(prediction);
			double predictionError = diff.normF();
			System.out.println("Iteration " + i + " error: " + predictionError);
			
			//Reshape input to vector
			input.reshape(1, 25);
			
			//Combine with action vector
			SimpleMatrix combinedInput = input;//input.combine(0, input.END, curAction);
			
			//Feed forward
			SimpleMatrix ffOutput1 = conn.feedForward(combinedInput, externalReward, 0.1);
			
			//Reshape ff output from unit 1
			ffOutput1.reshape(1, 100);
			
			//Feed throug unit 2
			SimpleMatrix ffOutput = unit2.feedForward(ffOutput1);
			
			//Feed back through unit 2
			unit2.feedBackward(ffOutput);
			
			//Feed back through unit 1
			SimpleMatrix fbOutput = conn.feedBack(0.1);
			
			//Collect next action and prediction
			prediction = fbOutput;
			prediction.reshape(5, 5);			
			
			curInput++;
			if (curInput >= sequence.length){
				curInput = 0;
			}
		}
	}

	private void createInputs(){
		double[][] rockData = {
				{0,0,0,0,0},
				{0,1,1,1,0},
				{0,1,1,1,0},
				{0,1,1,1,0},
				{0,1,1,1,1}
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
		sequence = tmp;			
	}
}
