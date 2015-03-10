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
	
	Random rand = new Random(1234);
	Brain brain;
	SimpleMatrix[] sequence;
	FileWriter writer;
	int bitStringMaxSize;
	
	SimpleMatrix uniformDistribution;

	public static void main(String[] args) throws IOException {
		String filepath = "C:/Users/Simon/Documents/Experiments/HierarchicalTextPrediction/Log";
		HierarchicalTextPrediction htp = new HierarchicalTextPrediction();
		htp.run(filepath);

	}
	
	public void run(String logFilepath) throws IOException{
		//for (int i = 0; i < 10; i++){
		int i = 0;	
		setupExperiment();
			writer = new FileWriter();
			writer.openFile(logFilepath + "_" + i, false);
			double error = runExperiment(100);
			writer.closeFile();
		//}
		System.out.printf("Error: %.3f", error );
	}
	
	private void setupExperiment(){
		buildSequence();
		setupBrain(2);
	}
	
	private double runExperiment(int iterations){
				
		ArrayList<SimpleMatrix[]> sequences = new ArrayList<SimpleMatrix[]>();
		sequences.add(sequence);
		SequenceTrainer trainer = new SequenceTrainer(sequences, iterations, rand, -1);
		boolean calculateErrorAsDistance = false;
		//Train
		trainer.train(brain, 0.0, calculateErrorAsDistance, null);
		
		//Evaluate
		brain.setLearning(false);
		brain.flush();
		ArrayList<Double> errors = trainer.train(brain, 0.0, calculateErrorAsDistance, writer);
		
		double error = 0;
		for (double d : errors){
			error += d;
		}
		
		error = error / (double) errors.size();
		return error;
		
	}
	
	private void setupBrain(int numUnits){
		int temporalMapSize = 4;
		int inputLength = bitStringMaxSize;
		int spatialMapSize = 3;
		double predictionLearningRate = 0.1;
		int markovOrder = 5;
		
		brain = new Brain(numUnits, rand, inputLength, spatialMapSize, temporalMapSize, markovOrder);
			
	}
	
	private void buildSequence(){
		SequenceBuilder builder = new SequenceBuilder();
		
		int minBlockLength = 3;
		int maxBlockLength = 3;
		int alphabetSize = 3;		
		int numLevels = 4;
		int[] intSequence = builder.buildSequence(rand, numLevels, alphabetSize, minBlockLength, maxBlockLength);
		
		String tmp = Integer.toBinaryString(alphabetSize);
		bitStringMaxSize = tmp.length();
		
		SimpleMatrix[] matrixSequence = new SimpleMatrix[intSequence.length];
		for (int i = 0; i < intSequence.length; i++){
			String bitString = intToBitString(intSequence[i], bitStringMaxSize);
			double[] vector = new double[bitString.length()];
			for (int j = 0; j < bitString.length(); j++) {
				int value = Integer.parseInt(bitString.substring(j, j+1));
				vector[j] = value;
			}
			
			double[][] data = {vector};
			SimpleMatrix m = new SimpleMatrix(data);			
			matrixSequence[i] = m;
		}
		sequence = matrixSequence;
	}
	
	/**
	 * 
	 * @param i integer to convert to bitstring
	 * @param length length of bit string. 0's will be added in front of the bit string if it is not long enough
	 * @return
	 */
	private String intToBitString(int i, int length){
		String s = Integer.toBinaryString(i);
		while (s.length() < length){
			s = "0" + s;
		}
		
		return s;
	}
	

}
