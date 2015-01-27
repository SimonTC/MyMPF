package stcl.algo.poolers;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.util.ExponentialDecayFunction;
import dk.stcl.som.som.SOM;

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
	
	//Variables used for testing inputs
	protected int inputLength;
	protected int mapSize;
	
	//Misc
	protected Random rand;
	
	/**
	 * Constructor used when initial learning rate, neighboorhood radius and noise magnitude should be default
	 * @param rand
	 * @param maxIterations
	 * @param inputLength
	 * @param mapSize
	 */
	public SpatialPooler(Random rand, int inputLength, int mapSize) {
		this(rand, inputLength, mapSize, 0.1, 2 , 0.125);
	}

	/**
	 * Constructor used when all parameters are given
	 * @param rand
	 * @param inputLength
	 * @param mapSize
	 * @param initialLearningRate
	 * @param stddev
	 * @param activationCodingFactor
	 */
	public SpatialPooler(Random rand, int inputLength, int mapSize, double initialLearningRate, double stddev, double activationCodingFactor ) {
		this.rand = rand;
		som = new SOM(mapSize, mapSize, inputLength, rand, initialLearningRate, stddev, activationCodingFactor);
		errorMatrix = new SimpleMatrix(mapSize, mapSize);
		activationMatrix = new SimpleMatrix(mapSize, mapSize);
		this.inputLength = inputLength;
		this.mapSize = mapSize;		
		curNoiseMagnitude = 0; //TODO: Be aware of noise magnitude
	}

	
	/**
	 * 
	 * @param feedForwardInputVector
	 * @return Returns a probability matrix. The value of cell (i,j) in the output matrix is the probability that SOM-model (i,j) is an accurate model of the observed input
	 */
	public SimpleMatrix feedForward(SimpleMatrix feedForwardInputVector){
		//Test input
		if (!feedForwardInputVector.isVector()) throw new IllegalArgumentException("The feed forward input to the spatial pooler has to be a vector");
		if (feedForwardInputVector.numCols() != inputLength) throw new IllegalArgumentException("The feed forward input to the spatial pooler has to be a 1 x " + inputLength + " vector");
		
		//Adjust weights of SOM
		som.step(feedForwardInputVector);
		
		//Compute ActivationMatrix
		activationMatrix = som.computeActivationMatrix();
		
		//Normalize activation matrix
		activationMatrix = normalize(activationMatrix);
		
		//Orthogonalize activation matrix
		//activationMatrix = orthogonalize(activationMatrix);
		
		return activationMatrix;
	}
	
	/**
	 * 
	 * @param feedBackwardInputMatrix
	 * @return vector Returns a vector with the values that are expected to be observed at time t+1
	 */
	public SimpleMatrix feedBackward(SimpleMatrix feedBackwardInputMatrix){
		//Test input
		if (feedBackwardInputMatrix.isVector()) throw new IllegalArgumentException("The feed back input to the spatial pooler has to be a matrix");
		if (feedBackwardInputMatrix.numCols() != mapSize || feedBackwardInputMatrix.numRows() != mapSize) throw new IllegalArgumentException("The feed back input to the spatial pooler has to be a " + mapSize + " x " + mapSize + " matrix");
		
		//Choose random model from som by roulette selection based on the input
		SimpleMatrix model = chooseRandom(feedBackwardInputMatrix);
		
		//Add noise
		model = addNoise(model, curNoiseMagnitude);
		
		return model;
		
	}
	
	protected SimpleMatrix orthogonalize(SimpleMatrix m) {
		SimpleMatrix activation = m.divide(-2 * Math.pow(0.125, 2));	 //TODO: Change the activation coding factor 0.125 to a parameter
		activation = activation.elementExp();
		return activation;
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
		SimpleMatrix model = som.getNode(id).getVector();
		
		return model;
		
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
	
	public SimpleMatrix getActivationMatrix(){
		return this.activationMatrix;
	}
	
	public void setLearning(boolean learning){
		som.setLearning(learning);
	}
}
