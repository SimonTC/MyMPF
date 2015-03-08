package stcl.fun.sequenceprediction.experiments;

import java.util.ArrayList;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.NU;
import stcl.algo.brain.NeoCorticalUnit;
import stcl.algo.util.Normalizer;
import stcl.fun.sequenceprediction.SequenceBuilder;
import stcl.fun.sequenceprediction.SequenceTrainer;

public class HierarchicalTextPrediction {
	
	Random rand = new Random(1234);
	ArrayList<NU> brain;
	double[] sequence;
	
	SimpleMatrix uniformDistribution;

	public static void main(String[] args) {
		HierarchicalTextPrediction htp = new HierarchicalTextPrediction();
		htp.run();

	}
	
	public void run(){
		setupExperiment();
		runExperiment(100);
	}
	
	private void setupExperiment(){
		buildSequence();
		setupBrain(2);
	}
	
	private void runExperiment(int iterations){
		ArrayList<double[]> sequences = new ArrayList<double[]>();
		sequences.add(sequence);
		SequenceTrainer trainer = new SequenceTrainer(sequences, iterations, rand);
		boolean calculateErrorAsDistance = false;
		trainer.train(brain, 0, calculateErrorAsDistance);
	}
	
	private void setupBrain(int numUnits){
		int temporalMapSize = 4;
		int inputLength = 1;
		int spatialMapSize = 5;
		double predictionLearningRate = 0.1;
		int markovOrder = 5;

		brain = new ArrayList<NU>();
		NU nu1 = new NeoCorticalUnit(rand, inputLength, spatialMapSize, temporalMapSize, predictionLearningRate, true, markovOrder); //First one is special
		brain.add(nu1);
		for (int i = 0; i < numUnits - 1; i++){
			NU nu = new NeoCorticalUnit(rand, temporalMapSize * temporalMapSize, spatialMapSize, temporalMapSize, predictionLearningRate, true, markovOrder);
			brain.add(nu);
		}		
	}
	
	private void buildSequence(){
		SequenceBuilder builder = new SequenceBuilder();
		
		int minBlockLength = 3;
		int maxBlockLength = 9;
		int alphabetSize = 8;		
		int numLevels = 4;
		int[] intSequence = builder.buildSequence(rand, numLevels, alphabetSize, minBlockLength, maxBlockLength);
		double[] doubleSequence = new double[intSequence.length];
		for (int i = 0; i < intSequence.length; i++){
			doubleSequence[i] = intSequence[i];
		}
		sequence = doubleSequence;
	}
	

}
