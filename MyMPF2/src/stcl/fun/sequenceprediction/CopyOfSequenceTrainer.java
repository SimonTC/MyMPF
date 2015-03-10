package stcl.fun.sequenceprediction;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.Brain;
import stcl.algo.brain.NU;
import stcl.algo.poolers.RSOM;
import stcl.algo.poolers.SOM;
import stcl.algo.util.FileWriter;
import stcl.algo.util.Normalizer;
import dk.stcl.core.basic.containers.SomNode;

public class CopyOfSequenceTrainer {
	private ArrayList<double[]> possibleSequences; //All possible sequences
	protected ArrayList<double[]> trainingSet; //The list used when training
	private Random rand;
	
	public CopyOfSequenceTrainer(ArrayList<double[]> sequences, int numIterations, Random rand) {
		this.possibleSequences = sequences;
		this.rand = rand;
		
		buildRandomList(possibleSequences, numIterations);
	}
	
	private void buildRandomList(ArrayList<double[]> possibleSequences, int numSequences){
		trainingSet = new ArrayList<double[]>();
		for (int i = 0; i < numSequences; i++){
			int id = rand.nextInt(possibleSequences.size());
			trainingSet.add(possibleSequences.get(id));
		}
	}
	
	/**
	 * Trains the brain on a training set generated when the sequenceTrainer was instantiated
	 * @param brain
	 * @param noiseMagnitude
	 * @return a list with the MSQE of each sequence in the training set
	 */
	public ArrayList<Double> train(Brain brain, double noiseMagnitude, boolean calculateErrorAsDistance, FileWriter writer){
		return train(brain, noiseMagnitude, trainingSet, calculateErrorAsDistance, writer);
	}
	
	/**
	 * Trains the brain on the given trainingset.
	 * @param brain
	 * @param noiseMagnitude
	 * @return a list with the MSQE of each sequence in the training set
	 */
	public ArrayList<Double> train(Brain brain, double noiseMagnitude, ArrayList<double[]> givenTrainingSet, boolean calculateErrorAsDistance, FileWriter writer){
		ArrayList<Double> errors = new ArrayList<Double>(); 
		int counter = 1;
		int numSequences = givenTrainingSet.size();
		for (double[] sequence : givenTrainingSet){
			double error = doSequence(brain, noiseMagnitude, sequence, calculateErrorAsDistance, writer);
			errors.add(error);
			counter++;
		}
		return errors;
	}
	
	private void flush(ArrayList<NU> brain){
		for (NU nu : brain) nu.flush();
	}
	
	private void printSomMap(SOM som){
		for (SomNode n : som.getNodes()) System.out.printf("%1$.4f " ,n.getVector().get(0) );
		System.out.println();
		System.out.println();
	}
	
	private double doSequence(Brain brain, double noiseMagnitude, double[] sequence, boolean calculateErrorAsDistance, FileWriter writer){
		double prediction = 0;
		double totalError = 0;
		for (double d : sequence){
			
			if (calculateErrorAsDistance) {
				totalError += calculateErrorAsDistance(prediction, d);
			} else {
				totalError += calculateErrorAsBoolean(prediction, d);
			}
			SimpleMatrix output = step(brain, noiseMagnitude, d);
			prediction = output.get(0);
			if (writer != null) writeInfo(writer, brain, d, prediction);
		}
		double MSQE = totalError / (double) sequence.length;
		return MSQE;
	}
	
	private void writeInfo(FileWriter writer, Brain brain, double input, double prediction){
		double[] predictionEntropies = brain.collectPredictionEntropies();
		double[] spatialFFOutEntropies = brain.collectSpatialFFEntropies();
		int[] spatialBMUs = brain.collectBMUs(true);
		int[] temporalBMUs = brain.collectBMUs(false);
		int[] helpStatus = brain.collectHelpStatus();
		
		String line = "";
		line += input + ";";
		line += prediction + ";";
		for (double d : predictionEntropies){
			line += d + ";";
 		}
		for (double d : spatialFFOutEntropies){
			line += d + ";";
 		}
		for (int i : spatialBMUs){
			line += i + ";";
		}
		for (int i : temporalBMUs){
			line += i + ";";
		}
		for (int i : helpStatus){
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
	
	/**
	 * 
	 * @param brain
	 * @param noiseMagnitude
	 * @param input
	 * @return feed back output from the brain
	 */
	private SimpleMatrix step(Brain brain, double noiseMagnitude, double input){
		double d = addNoise(input, noiseMagnitude);
		double[][] inputData = {{d}};
		SimpleMatrix inputVector = new SimpleMatrix(inputData);		
		
		SimpleMatrix prediction =  brain.step(inputVector);
		
		return prediction;
	}
	
	private SimpleMatrix resizeToFitFBPass(SimpleMatrix matrixToResize, NU unitToFit){
		SimpleMatrix m = new SimpleMatrix(matrixToResize);
		RSOM rsom = unitToFit.getTemporalPooler().getRSOM();
		int rows = rsom.getHeight();
		int cols = rsom.getWidth();
		
		m.reshape(rows, cols);
		return m;
	}
	
	private SimpleMatrix resizeToFitFFPass(SimpleMatrix matrixToResize, NU unitToFit){
		SimpleMatrix m = new SimpleMatrix(matrixToResize);
		int rows = 1;
		int cols = unitToFit.getSOM().getInputVectorLength();
				
		m.reshape(rows, cols);
		return m;
	}
	
	private double calculateErrorAsDistance(double prediction, double actual){
		double error = Math.pow(prediction - actual, 2);
		return error;
	}
	
	private double calculateErrorAsBoolean(double prediction, double actual){
		float actual_int = Math.round(actual);
		float prediction_int = Math.round(prediction);
		if (actual_int != prediction_int) return 1;
		return 0;
	}
	
	private double addNoise(double value, double noiseMagnitude){
		double noise = (rand.nextDouble() - 0.5) * 2 * noiseMagnitude;
		return value + noise;
	}
	
	private SimpleMatrix createUniformDistribution(int rows, int columns){
		SimpleMatrix m = new SimpleMatrix(rows, columns);
		m.set(1);
		m = Normalizer.normalize(m);
		return m;
	}
	
	public ArrayList<double[]> getTrainingSet(){
		return trainingSet;
	}

}
