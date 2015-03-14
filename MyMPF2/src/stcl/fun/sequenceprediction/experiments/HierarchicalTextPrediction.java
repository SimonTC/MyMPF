package stcl.fun.sequenceprediction.experiments;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Locale;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.Brain_DataCollector;
import stcl.algo.util.FileWriter;
import stcl.fun.sequenceprediction.SequenceBuilder;
import stcl.fun.sequenceprediction.SequenceTrainer;

public class HierarchicalTextPrediction {
	
	Random rand = new Random();
	Brain_DataCollector brain;
	SimpleMatrix[] sequence;
	FileWriter writer;
	int bitStringMaxSize;
	
	SimpleMatrix uniformDistribution;

	public static void main(String[] args) throws IOException {
		String filepath = "c:/Users/Simon/Documents/Experiments/HierarchicalTextPrediction/Log";
		HierarchicalTextPrediction htp = new HierarchicalTextPrediction();
		htp.run(filepath);
		System.out.println();
		//htp.run_Staggered(filepath);
		//htp.run_big(filepath);
	}
	
	public void run(String logFilepath) throws IOException{
		double totalError = 0;
		int iterations = 50;
		for (int i = 0; i < iterations; i++){
		//int i = 0;	
			setupExperiment(6);
			totalError += runExperiment(100, true);
		}
		double error = totalError / (double) iterations;
		System.out.printf("Error: %.3f", error );
	}
	
	public void run_big(String logFilepath) throws IOException{
		System.out.println("Units;String presentations;Error");
		int iterations = 10;
		for (int numUnits = 1; numUnits <= 10; numUnits++){
			for (int stringPresentations = 100; stringPresentations <= 1000; stringPresentations += 100){
				double totalError = 0;
				for (int i = 0; i < iterations; i++){
					setupExperiment(numUnits);
					totalError += runExperiment(stringPresentations, true);
				}
				double error = totalError / (double) iterations;
				System.out.print(numUnits + ";" + stringPresentations + ";");
				System.out.printf("%.3f", error );
				System.out.println();
			}
		}
	}
	
	/*
	public void run_Staggered(String logFilepath) throws IOException{

		int i = 0;	
		//Train first level unit
		setupExperiment(1);
		double error = runExperiment(300, true);
		
		System.out.printf("Error, staggered unit 1: %.3f", error );
		System.out.println();
		
		//Train second level unit
		sequence = createSequenceForNextUnit();
		setupBrain(1);
		error = runExperiment(1, true);

		System.out.printf("Error, staggered unit 2: %.3f", error );
		System.out.println();
		
		//Train third level unit
		sequence = createSequenceForNextUnit();
		setupBrain(1);
		error = runExperiment(1, true);
		

		System.out.printf("Error, staggered unit 3: %.3f", error );

		
		
	}
	
	private SimpleMatrix[] createSequenceForNextUnit(){
		//Only works when old brain consists of one unit
		ArrayList<boolean[]> helpStatuses = brain.getHelpStatuses();
		ArrayList<SimpleMatrix[]> ffOutputs = brain.getFFOutputs();
		
		ArrayList<SimpleMatrix> newInputs = new ArrayList<SimpleMatrix>();
		for (int i = 0; i < ffOutputs.size(); i++){
			if (helpStatuses.get(i)[0]) newInputs.add(ffOutputs.get(i)[0]);
		}
		
		SimpleMatrix[] newSequence = new SimpleMatrix[newInputs.size()];
		for (int i = 0; i < newInputs.size(); i++){
			newSequence[i] = newInputs.get(i);
		}
		return newSequence;
	}
	*/
	private void setupExperiment(int numUnits){
		buildSequence();
		setupBrain(numUnits);
	}
	
	private double runExperiment(int iterations, boolean brainMemoryFlushBetweenTrainingAndEvaluation){
				
		ArrayList<SimpleMatrix[]> sequences = new ArrayList<SimpleMatrix[]>();
		sequences.add(sequence);
		SequenceTrainer trainer = new SequenceTrainer(sequences, iterations, rand, -1);
		boolean calculateErrorAsDistance = false;
		
		//brain.setBiasBeforePrediction(false);
		//brain.setUseBiasedInputToSequencer(true);
		
		//Train
		trainer.train(brain, 0.0, calculateErrorAsDistance);
		
		//Evaluate
		brain.setLearning(false);
		brain.flush();
		
		//brain.setEntropyThresholdFrozen(false);
		//brain.setBiasBeforePrediction(true);
		//brain.setUseBiasedInputToSequencer(true);
		ArrayList<Double> errors = trainer.train(brain, 0.0, calculateErrorAsDistance);
		
		double error = 0;
		for (double d : errors){
			error += d;
		}
		
		error = error / (double) errors.size();
		return error;
		
	}
	
	private void setupBrain(int numUnits){
		int temporalMapSize = 2;
		int inputLength = sequence[0].getNumElements();
		int spatialMapSize = 4;
		double predictionLearningRate = 0.1;
		int markovOrder = 3;
		
		brain = new Brain_DataCollector(numUnits, rand, inputLength, spatialMapSize, temporalMapSize, markovOrder);
			
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
