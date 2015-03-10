package stcl.fun.sequenceprediction.experiments;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
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
	
	Random rand = new Random(1234);
	Brain_DataCollector brain;
	SimpleMatrix[] sequence;
	FileWriter writer;
	int bitStringMaxSize;
	
	SimpleMatrix uniformDistribution;

	public static void main(String[] args) throws IOException {
		String filepath = "d:/Users/Simon/Documents/Experiments/HierarchicalTextPrediction/Log";
		HierarchicalTextPrediction htp = new HierarchicalTextPrediction();
		htp.run(filepath);

	}
	
	public void run(String logFilepath) throws IOException{
		//for (int i = 0; i < 10; i++){
		int i = 0;	
		setupExperiment();
			double error = runExperiment(100);
			writer = new FileWriter();
			writer.openFile(logFilepath + "_" + i, false);
			writeInfo(writer, brain);
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
		trainer.train(brain, 0.0, calculateErrorAsDistance);
		
		//Evaluate
		brain.setLearning(false);
		brain.flush();
		brain.flushCollectedData();
		ArrayList<Double> errors = trainer.train(brain, 0.0, calculateErrorAsDistance);
		
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
		ArrayList<int[]> spatialBMUs = brain.getSpatialBMUs();
		ArrayList<int[]> temporalBMUs = brain.getTemporalBMUs();
		ArrayList<boolean[]> helpStatuses = brain.getHelpStatuses();
		ArrayList<boolean[]> activeStatuses = brain.getActiveStatuses();
		
		//Write headers
		String header = "";
		header += writeRepeatedString("Input", 1, ";");
		header += writeRepeatedString("Output", 1, ";");
		header += writeRepeatedString("Prediction entropy", 2, ";");
		header += writeRepeatedString("Spatial BMU", 2, ";");
		header += writeRepeatedString("Temporal BMU", 2, ";");
		header += writeRepeatedString("Need help", 2, ";");
		header += writeRepeatedString("Was active", 2, ";");
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
			for (int i : spatialBMUs.get(k)){
				line += i + ";";
			}
			for (int i : temporalBMUs.get(k)){
				line += i + ";";
			}
			for (boolean b : helpStatuses.get(k)){
				int i = b ? 1 : 0;
				line += i + ";";
			}
			for (boolean b : activeStatuses.get(k)){
				int i = b ? 1 : 0;
				line += i + ";";
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
			ps.printf(format,d);
		}
		
		return stream.toString();
	}
	

}
