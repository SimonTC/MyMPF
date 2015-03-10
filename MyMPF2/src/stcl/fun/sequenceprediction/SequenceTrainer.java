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
	
	public SequenceTrainer(ArrayList<double[]> sequences, int numIterations, Random rand) {
		
		this.rand = rand;
		ArrayList<SimpleMatrix[]> possibleSequences = convertDoubleSequencesToMatrixSequences(sequences);
		buildTrainingSet(possibleSequences, numIterations);
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
	public ArrayList<Double> train(Brain brain, double noiseMagnitude, boolean calculateErrorAsDistance, FileWriter writer){
		return train(brain, noiseMagnitude, trainingSet, calculateErrorAsDistance, writer);
	}
	
	/**
	 * Trains the brain on the given trainingset.
	 * @param brain
	 * @param noiseMagnitude
	 * @return a list with the MSQE of each sequence in the training set
	 */
	public ArrayList<Double> train(Brain brain, double noiseMagnitude, ArrayList<SimpleMatrix[]> givenTrainingSet, boolean calculateErrorAsDistance, FileWriter writer){
		ArrayList<Double> errors = new ArrayList<Double>(); 
		int counter = 1;
		int numSequences = givenTrainingSet.size();
		for (SimpleMatrix[] sequence : givenTrainingSet){
			double error = doSequence(brain, noiseMagnitude, sequence, writer);
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
	
	private double doSequence(Brain brain, double noiseMagnitude, SimpleMatrix[] sequence, FileWriter writer){
		SimpleMatrix prediction = null;
		double totalError = 0;
		for (SimpleMatrix m : sequence){
			
			if (prediction != null){
				totalError += calculateErrorAsDistance(prediction, m);
			}
			SimpleMatrix output = step(brain, noiseMagnitude, m);
			prediction = output;
			if (writer != null) writeInfo(writer, brain, m, prediction);
		}
		double MSQE = totalError / (double) sequence.length;
		return MSQE;
	}
	
	private void writeInfo(FileWriter writer, Brain brain, SimpleMatrix input, SimpleMatrix prediction){
		double[] predictionEntropies = brain.collectPredictionEntropies();
		double[] spatialFFOutEntropies = brain.collectSpatialFFEntropies();
		int[] spatialBMUs = brain.collectBMUs(true);
		int[] temporalBMUs = brain.collectBMUs(false);
		int[] helpStatus = brain.collectHelpStatus();
		
		String line = "";
		line += writeMatrixArray(input) + ";"; 
		line += writeMatrixArray(prediction) + ";";
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
		
		return ps.toString();
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
	
	/**
	 * Calculates the NormF of the difference between the two matrices
	 */
	private double calculateErrorAsDistance(SimpleMatrix prediction, SimpleMatrix actual){
		SimpleMatrix diff = prediction.minus(actual);
		double error = diff.normF();
		return error;
	}
	
	private double calculateErrorAsBoolean(double prediction, double actual){
		float actual_int = Math.round(actual);
		float prediction_int = Math.round(prediction);
		if (actual_int != prediction_int) return 1;
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
	
	private SimpleMatrix createUniformDistribution(int rows, int columns){
		SimpleMatrix m = new SimpleMatrix(rows, columns);
		m.set(1);
		m = Normalizer.normalize(m);
		return m;
	}
	
	public ArrayList<SimpleMatrix[]> getTrainingSet(){
		return trainingSet;
	}

}
