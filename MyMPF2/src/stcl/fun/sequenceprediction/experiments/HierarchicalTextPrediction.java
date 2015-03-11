package stcl.fun.sequenceprediction.experiments;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Locale;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import dk.stcl.core.basic.containers.SomNode;
import stcl.algo.brain.Brain;
import stcl.algo.brain.Brain_DataCollector;
import stcl.algo.brain.NU;
import stcl.algo.brain.NeoCorticalUnit;
import stcl.algo.util.FileWriter;
import stcl.fun.sequenceprediction.SequenceBuilder;
import stcl.fun.sequenceprediction.SequenceTrainer;

public class HierarchicalTextPrediction {
	
	Random rand = new Random(12345);
	Brain_DataCollector brain;
	SimpleMatrix[] sequence;
	FileWriter writer;
	int bitStringMaxSize;
	
	SimpleMatrix uniformDistribution;

	public static void main(String[] args) throws IOException {
		String filepath = "d:/Users/Simon/Documents/Experiments/HierarchicalTextPrediction/Log";
		HierarchicalTextPrediction htp = new HierarchicalTextPrediction();
		htp.run(filepath);
		System.out.println();
		htp.run_Staggered(filepath);
	}
	
	public void run(String logFilepath) throws IOException{
		double totalError = 0;
		int iterations = 10;
		for (int i = 0; i < iterations; i++){
		//int i = 0;	
			setupExperiment(1);
			totalError += runExperiment(200, true);
			writer = new FileWriter();
			writer.openFile(logFilepath + "_" + i, false);
			writeInfo(writer, brain);
			writer.closeFile();
			brain.getUnitList().get(0).getSequencer().printSequenceMemory();
			System.out.println();
			brain.getUnitList().get(0).getSequencer().printTrie();
		}
		double error = totalError / (double) iterations;
		System.out.printf("Error: %.3f", error );
	}
	
	public void run_Staggered(String logFilepath) throws IOException{

		int i = 0;	
		//Train first level unit
		setupExperiment(1);
		double error = runExperiment(100, true);
		
		writer = new FileWriter();
		writer.openFile(logFilepath + "Staggered_Level1", false);
		writeInfo(writer, brain);
		writer.closeFile();
		
		System.out.printf("Error, staggered unit 1: %.3f", error );
		System.out.println();
		
		//Train second level unit
		sequence = createSequenceForNextUnit();
		setupBrain(1);
		error = runExperiment(1, true);
		
		
		writer = new FileWriter();
		writer.openFile(logFilepath + "Staggered_Level2" , false);
		writeInfo(writer, brain);
		writer.closeFile();

		System.out.printf("Error, staggered unit 2: %.3f", error );
		System.out.println();
		
		//Train third level unit
		sequence = createSequenceForNextUnit();
		setupBrain(1);
		error = runExperiment(1, true);
		
		
		writer = new FileWriter();
		writer.openFile(logFilepath + "Staggered_Level3" , false);
		writeInfo(writer, brain);
		writer.closeFile();

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
	
	private void setupExperiment(int numUnits){
		buildSequence();
		setupBrain(numUnits);
	}
	
	private double runExperiment(int iterations, boolean brainMemoryFlushBetweenTrainingAndEvaluation){
				
		ArrayList<SimpleMatrix[]> sequences = new ArrayList<SimpleMatrix[]>();
		sequences.add(sequence);
		SequenceTrainer trainer = new SequenceTrainer(sequences, iterations, rand, -1);
		boolean calculateErrorAsDistance = false;
		
		//Train
		trainer.train(brain, 0.0, calculateErrorAsDistance);
		
		//Evaluate
		brain.setLearning(false);
		brain.flush();
		if (brainMemoryFlushBetweenTrainingAndEvaluation) brain.flushCollectedData();
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
		int spatialMapSize = 3;
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
	
	private void writeInfo(FileWriter writer, Brain_DataCollector brain){
		ArrayList<SimpleMatrix> brainInputs = brain.getReceivedInputs();
		ArrayList<SimpleMatrix> brainOutputs = brain.getReturnedOutputs();
		ArrayList<double[]> predictionEntropies = brain.getPredictionEntropies();	
		ArrayList<double[]> entropyThresholds = brain.getEntropiesThresholds();
		ArrayList<int[]> spatialBMUs = brain.getSpatialBMUs();
		ArrayList<int[]> temporalBMUs = brain.getTemporalBMUs();
		ArrayList<boolean[]> helpStatuses = brain.getHelpStatuses();
		ArrayList<boolean[]> activeStatuses = brain.getActiveStatuses();
		ArrayList<SimpleMatrix[]> ffOutputs = brain.getFFOutputs();
		
		//Write headers
		int numUnits = brain.getNumUnits();
		String header = "";
		header += writeRepeatedString("Input", 1, ";");
		header += writeRepeatedString("Output", 1, ";");
		header += writeRepeatedString("Prediction entropy",numUnits, ";");
		header += writeRepeatedString("Entropy threshold", numUnits, ";");
		header += writeRepeatedString("Spatial BMU", numUnits, ";");
		//header += writeRepeatedString("Temporal BMU", numUnits, ";");
		header += writeRepeatedString("Need help", numUnits, ";");
		header += writeRepeatedString("Was active", numUnits, ";");
		header += writeRepeatedString("FF Output", numUnits, ";");
		header = header.substring(0, header.length() - 1); //Remove last semi-colon
		try {
			writer.writeLine(header);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		//Write lines
		for (int k = 0; k < predictionEntropies.size(); k++){
			String line = "";
			line += writeMatrixArray(brainInputs.get(k)) + ";"; 
			line += writeMatrixArray(brainOutputs.get(k)) + ";";
			for (double d : predictionEntropies.get(k)){
				line += d + ";";
	 		}
			for (double d : entropyThresholds.get(k)){
				line += d + ";";
	 		}
			for (int i : spatialBMUs.get(k)){
				line += i + ";";
			}
			/*
			for (int i : temporalBMUs.get(k)){
				line += i + ";";
			}
			*/
			for (boolean b : helpStatuses.get(k)){
				int i = b ? 1 : 0;
				line += i + ";";
			}
			for (boolean b : activeStatuses.get(k)){
				int i = b ? 1 : 0;
				line += i + ";";
			}
			
			for (SimpleMatrix m : ffOutputs.get(k)){
				line += writeMatrixArray(m) + ";"; 
			}
			
			
			
			line = line.substring(0, line.length() - 1); //Remove last semi-colon
			try {
				writer.writeLine(line);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
	
	private String writeRepeatedString(String stringToRepeat, int numberOfTimes, String delimiter){
		String s = "";
		for (int i = 1; i <= numberOfTimes; i++){
			s += stringToRepeat;
			if (numberOfTimes > 1) s+= " " + i;
			s+= delimiter;
		}
		return s;
	}
	
	/**
	 * All is pretty much taken from the Matrix.toString() metods in simpleMatrix
	 * @param m
	 * @return
	 */
	private String writeMatrixArray(SimpleMatrix m){
		int numChar = 6;
		int precision = 3;
		String format = "%"+numChar+"."+precision+"f " + "  ";
		
		ByteArrayOutputStream stream = new ByteArrayOutputStream();
		PrintStream ps = new PrintStream(stream);
		
		for (double d : m.getMatrix().data){
			ps.printf(Locale.US, format, d);
		}
		
		return stream.toString();
	}
	

}
