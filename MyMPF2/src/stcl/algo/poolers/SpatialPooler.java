package stcl.algo.poolers;

import java.util.Random;
import java.util.Vector;

import javax.swing.plaf.basic.BasicInternalFrameTitlePane.MaximizeAction;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.som.SOM;
import stcl.algo.som.SomNode;
import stcl.algo.util.ExponentialDecayFunction;

public class SpatialPooler {
	
	//Weight matrix
	protected SOM som;
	
	//Matrice used 
	protected SimpleMatrix errorMatrix; //Squared error of the activation in the SOM
	protected SimpleMatrix activationMatrix;
	
	//Variables for learning
	protected double curLearningRate;
	protected double curNeighborhoodRadius;
	protected double curNoiseMagnitude;
	private int tick;
	
	//Variables used for testing inputs
	protected int inputLength;
	protected int mapSize;
	
	//Decay function
	//TODO: They probably have to be changed at some point. Especially noiseDecay. It should be dependant on activitty from lower levels.
	//TODO: They should probably be moved to the Neocortical unit
	private ExponentialDecayFunction learningDecay;
	private ExponentialDecayFunction radiusDecay;
	private ExponentialDecayFunction noiseDecay;
	
	//Misc
	protected Random rand;
	
	
	public SpatialPooler(Random rand, int maxIterations, int inputLength, int mapSize) {
		this.rand = rand;
		som = new SOM(mapSize, mapSize, inputLength, rand);
		errorMatrix = new SimpleMatrix(mapSize, mapSize);
		activationMatrix = new SimpleMatrix(mapSize, mapSize);
		tick = 0;
		this.inputLength = inputLength;
		this.mapSize = mapSize;
		
		//TODO: change start rates to something from a parameter file / given as parameter to constructor
		curLearningRate = 1;
		curNeighborhoodRadius = (double) mapSize / 2;
		curNoiseMagnitude = 1;
		
		//TODO: Something has to be done about this
		learningDecay = new ExponentialDecayFunction(curLearningRate, 0.01, maxIterations);
		radiusDecay = new ExponentialDecayFunction(curNeighborhoodRadius, 0.01, curNeighborhoodRadius);
		noiseDecay = new ExponentialDecayFunction(curNoiseMagnitude, 0.01, maxIterations);
	}
	
	public void tick(){
		tick++;
		curLearningRate = learningDecay.decayValue(tick);
		curNeighborhoodRadius = radiusDecay.decayValue(tick);
		curNoiseMagnitude = noiseDecay.decayValue(tick);
	}
	
	public SimpleMatrix feedForward(SimpleMatrix inputVector){
		//Test input
		if (!inputVector.isVector()) throw new IllegalArgumentException("The feed forward input to the spatial pooler has to be a vector");
		if (inputVector.numCols() != inputLength) throw new IllegalArgumentException("The feed forward input to the spatial pooler has to be a 1 x " + inputLength + " vector");
		
		//Adjust weights of SOM
		som.step(inputVector, curLearningRate, curNeighborhoodRadius);
		
		//Collect error matrix
		errorMatrix = som.getErrorMatrix();
		
		//Compute ActivationMatrix
		activationMatrix = computeActivationMatrix(errorMatrix);
		
		//Normalize activation matrix
		activationMatrix = normalize(activationMatrix);
		
		return activationMatrix;
	}
	
	/**
	 * 
	 * @param inputMatrix
	 * @return vector
	 */
	public SimpleMatrix feedBackward(SimpleMatrix inputMatrix){
		//Test input
		if (inputMatrix.isVector()) throw new IllegalArgumentException("The feed back input to the spatial pooler has to be a matrix");
		if (inputMatrix.numCols() != mapSize || inputMatrix.numRows() != mapSize) throw new IllegalArgumentException("The feed back input to the spatial pooler has to be a " + mapSize + " x " + mapSize + " matrix");
		
		//Choose random model from som by roulette selection based on the input
		SimpleMatrix model = chooseRandom(inputMatrix);
		
		//Add noise
		model = addNoise(model, curNoiseMagnitude);
		
		return model;
		
	}
	
	protected SimpleMatrix normalize(SimpleMatrix matrix){
		double sum = matrix.elementSum();
		SimpleMatrix m = matrix.scale(1/sum);
		return m;
	}
	
	protected SimpleMatrix chooseRandom(SimpleMatrix input){
		//Transform bias matrix into vector
		double[] vector = input.getMatrix().data;
		
		//Choose random number between 0 and 1
		double d = rand.nextDouble();
		
		//Go through bias vector until value is >= random number
		double tmp = 0;
		int id = 0;
		while (tmp < d && id < vector.length){
			tmp += vector[id++];
		}
		
		id--; //We have t subtract to be sure we get the correct model
		
		//Choose model from som
		SimpleMatrix model = som.getModel(id).getVector();
		
		return model;
		
	}
	
	protected SimpleMatrix computeActivationMatrix( SimpleMatrix errorMatrix){
		double maxError = errorMatrix.elementMaxAbs();
		SimpleMatrix m = errorMatrix.divide(maxError);
		SimpleMatrix activation = new SimpleMatrix(errorMatrix.numRows(), errorMatrix.numCols());
		activation.set(1);
		activation = activation.minus(m);	
		return activation;
	}
	
	/**
	 * Adds noise to the given matrix and returns the matrix.
	 * The matrix is altered in this method.
	 * @param m
	 * @param noiseMagnitude
	 * @return
	 */
	protected SimpleMatrix addNoise(SimpleMatrix m, double noiseMagnitude){
		double noise = (rand.nextDouble() - 0.5) * 2 * noiseMagnitude;
		m = m.plus(noise);
		return m;
	}
	
	public SOM getSOM(){
		return som;
	}
}
