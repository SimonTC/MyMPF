package stcl.fun.sequenceprediction.experiments;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import dk.stcl.core.basic.containers.SomNode;
import stcl.algo.brain.Brain;
import stcl.algo.brain.NU;
import stcl.algo.brain.NeoCorticalUnit;
import stcl.algo.util.FileWriter;
import stcl.fun.sequenceprediction.SequenceBuilder;
import stcl.fun.sequenceprediction.SequenceTrainer;

public class HierarchicalTextPrediction {
	
	Random rand = new Random();
	Brain brain;
	double[] sequence;
	FileWriter writer;
	
	SimpleMatrix uniformDistribution;

	public static void main(String[] args) throws IOException {
		String filepath = "D:/Users/Simon/Documents/Experiments/HierarchicalTextPrediction/Log";
		HierarchicalTextPrediction htp = new HierarchicalTextPrediction();
		htp.run(filepath);

	}
	
	public void run(String logFilepath) throws IOException{
		for (int i = 0; i < 10; i++){
			setupExperiment();
			writer = new FileWriter();
			writer.openFile(logFilepath + "_" + i, false);
			runExperiment(100);
			writer.closeFile();
		}
		System.out.println("finished");
	}
	
	private void setupExperiment(){
		buildSequence();
		setupBrain(1);
	}
	
	private void runExperiment(int iterations){
				
		ArrayList<double[]> sequences = new ArrayList<double[]>();
		sequences.add(sequence);
		SequenceTrainer trainer = new SequenceTrainer(sequences, iterations, rand);
		boolean calculateErrorAsDistance = false;
		//Train
		trainer.train(brain, 0, calculateErrorAsDistance, null);
		
		//Evaluate
		brain.setLearning(false);
		brain.flush();
		trainer.train(brain, 0, calculateErrorAsDistance, writer);
		
	}
	
	private void setupBrain(int numUnits){
		int temporalMapSize = 4;
		int inputLength = 1;
		int spatialMapSize = 3;
		double predictionLearningRate = 0.1;
		int markovOrder = 5;
		
		brain = new Brain(numUnits, rand, inputLength, spatialMapSize, temporalMapSize, markovOrder);
			
	}
	
	private void buildSequence(){
		SequenceBuilder builder = new SequenceBuilder();
		
		int minBlockLength = 3;
		int maxBlockLength = 3;
		int alphabetSize = 5;		
		int numLevels = 4;
		int[] intSequence = builder.buildSequence(rand, numLevels, alphabetSize, minBlockLength, maxBlockLength);
		double[] doubleSequence = new double[intSequence.length];
		for (int i = 0; i < intSequence.length; i++){
			doubleSequence[i] = intSequence[i];
		}
		sequence = doubleSequence;
	}
	

}
