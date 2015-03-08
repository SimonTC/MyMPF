package stcl.fun.sequenceprediction;

import java.util.ArrayList;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.NU;
import stcl.algo.poolers.RSOM;
import stcl.algo.poolers.SOM;
import stcl.algo.util.Normalizer;
import dk.stcl.core.basic.containers.SomNode;

public class SequenceTrainer {
	private ArrayList<double[]> possibleSequences; //All possible sequences
	protected ArrayList<double[]> trainingSet; //The list used when training
	private Random rand;
	
	public SequenceTrainer(ArrayList<double[]> sequences, int numIterations, Random rand) {
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
	 * Trains the brain on a trainingset generated when the sequenceTrainer was instantiated
	 * @param brain
	 * @param noiseMagnitude
	 * @return a list with the MSQE of each sequence in the training set
	 */
	public ArrayList<Double> train(ArrayList<NU> brain, double noiseMagnitude, boolean calculateErrorAsDistance){
		return train(brain, noiseMagnitude, trainingSet, calculateErrorAsDistance);
	}
	
	/**
	 * Trains the brain on the given trainingset.
	 * @param brain
	 * @param noiseMagnitude
	 * @return a list with the MSQE of each sequence in the training set
	 */
	public ArrayList<Double> train(ArrayList<NU> brain, double noiseMagnitude, ArrayList<double[]> givenTrainingSet, boolean calculateErrorAsDistance){
		ArrayList<Double> errors = new ArrayList<Double>(); 
		int counter = 1;
		int numSequences = givenTrainingSet.size();
		for (double[] sequence : givenTrainingSet){
			//flush(brain);
			double error = doSequence(brain, noiseMagnitude, sequence, calculateErrorAsDistance);
			//SOM som = brain.get(0).getSOM(); printSomMap(som);
			errors.add(error);
			//System.out.printf("Sequence " +counter + " of " + numSequences + " MSE: %1$.3f", error);
			System.out.println( error);
			//System.out.println();
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
	
	private double doSequence(ArrayList<NU> brain, double noiseMagnitude, double[] sequence, boolean calculateErrorAsDistance){
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
	private SimpleMatrix step(ArrayList<NU> brain, double noiseMagnitude, double input){
		double d = addNoise(input, noiseMagnitude);
		double[][] inputData = {{d}};
		SimpleMatrix inputVector = new SimpleMatrix(inputData);		
		SimpleMatrix uniformDistribution = null;
		
		//Feed forward
		SimpleMatrix ffInput = inputVector;
		int i = 0;
		boolean cont = true;
		do {
			NU nu = brain.get(i);
			SimpleMatrix m = resizeToFitFFPass(ffInput, nu);
			SimpleMatrix inputToNextLayer = nu.feedForward(m);
			//System.out.println( i + " Entropy " + nu.getEntropy() + " Threshold: " + nu.getEntropyThreshold());
			cont = nu.needHelp();
			if (cont) {
				ffInput = inputToNextLayer;
			} else {
				ffInput = null;
			}
			i++;
		} while (i < brain.size() && cont);
		
		//Feed back
		if (uniformDistribution == null){
			RSOM rsom = brain.get(brain.size() - 1).getTemporalPooler().getRSOM();
			int rows = rsom.getHeight();
			int cols = rsom.getWidth();
			uniformDistribution = createUniformDistribution(rows, cols);
		}
		
		SimpleMatrix fbInput = uniformDistribution;
		for (int j = brain.size()-1; j >= 0; j--){
			NU nu = brain.get(j);
			SimpleMatrix m = resizeToFitFBPass(fbInput, nu);
			fbInput = nu.feedBackward(m);
		}
		
		return fbInput; //The last fb input is the output of the brain
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
