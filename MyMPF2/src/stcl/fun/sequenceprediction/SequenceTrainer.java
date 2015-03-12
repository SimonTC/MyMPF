package stcl.fun.sequenceprediction;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
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

public class SequenceTrainer {
	protected ArrayList<SimpleMatrix[]> trainingSet; //The list used when training
	private Random rand;
	
	/**
	 * Converts the arraylist of double arrays into an array list of simple matrices
	 * @param sequences
	 * @param numIterations
	 * @param rand
	 */
	public SequenceTrainer(ArrayList<double[]> sequences, int numIterations, Random rand) {		
		this.rand = rand;
		ArrayList<SimpleMatrix[]> possibleSequences = convertDoubleSequencesToMatrixSequences(sequences);
		buildTrainingSet(possibleSequences, numIterations);
	}	
	
	/**
	 * Builds a trainingset from the given sequences. The training set is built by choosing a random sequence for each iteration
	 * @param sequences
	 * @param numIterations
	 * @param rand
	 * @param blob Added because constructor apparently else would look like the constructor with a list of double arrays
	 */
	public SequenceTrainer(ArrayList<SimpleMatrix[]> sequences, int numIterations, Random rand, int blob) {
		this.rand = rand;
		buildTrainingSet(sequences, numIterations);
	}

	private ArrayList<SimpleMatrix[]> convertDoubleSequencesToMatrixSequences(ArrayList<double[]> sequences){
		ArrayList<SimpleMatrix[]> matrixSequences = new ArrayList<SimpleMatrix[]>();
		for (double[] seq : sequences){
			SimpleMatrix[] matrixSeq = new SimpleMatrix[seq.length];
			for (int i = 0; i < seq.length; i++){
				double d = seq[i];
				double data[][] = {{d}};
				SimpleMatrix m = new SimpleMatrix(data);
				matrixSeq[i] = m;
			}
			matrixSequences.add(matrixSeq);
		}
		return matrixSequences;
	}
	
	private void buildTrainingSet(ArrayList<SimpleMatrix[]> possibleSequences, int numSequences){
		trainingSet = new ArrayList<SimpleMatrix[]>();
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
	public ArrayList<Double> train(Brain brain, double noiseMagnitude, boolean calculateErrorAsDistance){
		return train(brain, noiseMagnitude, trainingSet, calculateErrorAsDistance);
	}
	
	/**
	 * Trains the brain on the given trainingset.
	 * @param brain
	 * @param noiseMagnitude
	 * @return a list with the MSQE of each sequence in the training set
	 */
	public ArrayList<Double> train(Brain brain, double noiseMagnitude, ArrayList<SimpleMatrix[]> givenTrainingSet, boolean calculateErrorAsDistance){
		ArrayList<Double> errors = new ArrayList<Double>(); 
		for (SimpleMatrix[] sequence : givenTrainingSet){
			double error = doSequence(brain, noiseMagnitude, sequence, calculateErrorAsDistance);
			errors.add(error);
		}
		return errors;
	}

	private double doSequence(Brain brain, double noiseMagnitude, SimpleMatrix[] sequence, boolean calculateErrorAsDistance){
		SimpleMatrix prediction = null;
		double totalError = 0;
		for (SimpleMatrix m : sequence){
			
			if (prediction != null){
				if (calculateErrorAsDistance){
					totalError += calculateErrorAsDistance(prediction, m);
				} else {
					totalError += calculateErrorAsBoolean(prediction, m, 0.01);
				}
			}
			SimpleMatrix output = step(brain, noiseMagnitude, m);
			prediction = output;
		}
		double MSQE = totalError / (double) sequence.length;
		return MSQE;
	}
	
	/**
	 * 
	 * @param brain
	 * @param noiseMagnitude 
	 * @param input
	 * @return feed back output from the brain
	 */
	private SimpleMatrix step(Brain brain, double noiseMagnitude, SimpleMatrix input){
		SimpleMatrix noisyInput = addNoise(input, noiseMagnitude);
				
		SimpleMatrix prediction =  brain.step(noisyInput);
		
		return prediction;
	}
	
	/**
	 * Calculates the NormF of the difference between the two matrices
	 */
	private double calculateErrorAsDistance(SimpleMatrix prediction, SimpleMatrix actual){
		SimpleMatrix diff = prediction.minus(actual);
		double error = diff.normF();
		return error;
	}
	
	/**
	 * If any of the values differs mpre than threshold from the corresponding value in the other matrix, the value 1 is returned. Otherwise 0
	 * @param prediction
	 * @param actual
	 * @param threshold
	 * @return
	 */
	private double calculateErrorAsBoolean(SimpleMatrix prediction, SimpleMatrix actual, double threshold){
		for (int i = 0; i < prediction.getNumElements(); i++){
			double diff = Math.abs(prediction.get(i) - actual.get(i));
			if (diff > threshold) return 1;
		}

		return 0;
	}
	
	private SimpleMatrix addNoise(SimpleMatrix m, double noiseMagnitude){
		SimpleMatrix noisyMatrix = new SimpleMatrix(m);
		for (int i = 0; i < noisyMatrix.getNumElements(); i++){
			double d = noisyMatrix.get(i);
			double noise = (rand.nextDouble() - 0.5) * 2 * noiseMagnitude;
			d += noise;
			noisyMatrix.set(i,d);
		}
		return noisyMatrix;
	}
	
	public ArrayList<SimpleMatrix[]> getTrainingSet(){
		return trainingSet;
	}

}
