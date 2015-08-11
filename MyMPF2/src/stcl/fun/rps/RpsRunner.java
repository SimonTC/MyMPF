package stcl.fun.rps;

import java.io.FileNotFoundException;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;



import stcl.algo.brain.Network_DataCollector;
import stcl.fun.rps.rewardfunctions.RewardFunction;
import stcl.fun.rps.rewardfunctions.RewardFunction_Inverse;
import stcl.fun.rps.rewardfunctions.RewardFunction_Standard;
import stcl.fun.rps.sequencecreation.SequenceBuilder;


public class RpsRunner {
	
	private boolean setSequenceManually = true;
	
	protected SimpleMatrix[] possibleInputs;
	protected int[][] sequences;
	protected SimpleMatrix rewardMatrix;
	protected int learningIterations;
	protected int trainingIterations;
	protected int evaluationIterations;
	protected int numDifferentSequences;
	protected int numExperimentsPerSequence;
	protected Random rand;
	protected boolean logTime;
	protected double noiseMagnitude;
	protected double exploreChance;

	public static void main(String[] args) throws FileNotFoundException {
		String experimentRun = "C:/Users/Simon/Google Drev/Experiments/HTM/rps_pc/1439281660206";
		String genomeFile = experimentRun + "/best_performing-final-1196.txt";;

		RpsRunner runner = new RpsRunner();
		runner.run(genomeFile);
		System.out.println();
		System.out.println();

	}
	
	public void run(String networkfilename) throws FileNotFoundException{
		this.init();
		Network_DataCollector brain = setupBrain(networkfilename, rand);
		double[][] result = this.evaluate(brain);
		
		double predictionSum = 0;
		double fitnessSum = 0;
		for (int i = 0; i < result.length; i++){
			System.out.println("Sequence " + i + " Prediction: " + result[i][0] + " Fitness: " + result[i][1]);
			predictionSum += result[i][0];
			fitnessSum += result[i][1];
			
		}
		double avgFitness = fitnessSum / (double) result.length;
		double avgPrediction = predictionSum / (double) result.length;
		System.out.println("Avg prediction: " + avgPrediction + " Avg fitness: " + avgFitness);
	}
	
	
	public void init() {
		rand = new Random();
		learningIterations = 100;
		trainingIterations = 1000;
		evaluationIterations = 100;
		numDifferentSequences = 1;
		numExperimentsPerSequence = 1;
		exploreChance = 0.1;
		noiseMagnitude = 0;
		
		sequences = createSequences(rand);
		
		createInputs();
		
	}
	
	private Network_DataCollector setupBrain(String fileName, Random rand) throws FileNotFoundException{
		Network_DataCollector brain = new Network_DataCollector(fileName, rand);
		return brain;
	}
	
	private double[][] evaluate(Network_DataCollector network) throws FileNotFoundException {
		RPS eval;
		RewardFunction[] functions = {new RewardFunction_Standard(), new RewardFunction_Inverse()};
		eval = new RPS(possibleInputs, sequences, functions, numExperimentsPerSequence, trainingIterations, evaluationIterations, rand.nextLong(), noiseMagnitude);
	
		eval.run(network, exploreChance);	
		double[][] result = eval.getSequenceScores();


		return result;
	}
	
	private RPS setupEvaluator(){
		RewardFunction[] functions = {new RewardFunction_Standard()};
		RPS eval = new RPS(possibleInputs, sequences, functions,  numExperimentsPerSequence, trainingIterations, evaluationIterations, rand.nextLong(), noiseMagnitude);
		return eval;
	}
	
	private int[][] createSequences(Random rand){

		int sequenceLevels = 3;
		int blockLengthMin = 2;
		int blockLengthMax = 2;
		int alphabetSize = 3;
		if (setSequenceManually){ //Only used for debugging
			int[][] mySequence ={{0,1,2}};
			return mySequence;
		} else {		
			Random sequenceRand = new Random();
			SequenceBuilder builder = new SequenceBuilder();
			int[][] sequences = new int[numDifferentSequences][];
			for ( int i = 0; i < numDifferentSequences; i++){
				sequences[i] = builder.buildSequence(sequenceRand, sequenceLevels, alphabetSize, blockLengthMin, blockLengthMax);
			}
			return sequences;
		}
		
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
		
		double[][] paperData = {
				{1,1,1,1,1},
				{1,1,1,1,1},
				{1,1,1,1,1},
				{1,1,1,1,1},
				{1,1,1,1,1}
		};
		
		SimpleMatrix paper = new SimpleMatrix(paperData);
		
		double[][] scissorsData = {
				{0,0,0,1,0},
				{1,0,1,0,0},
				{0,1,0,0,0},
				{1,0,1,0,0},
				{0,0,0,1,0}
		};
		
		SimpleMatrix scissors = new SimpleMatrix(scissorsData);		

		SimpleMatrix[] tmp = {rock, paper, scissors};

		possibleInputs = tmp;
	}

}
