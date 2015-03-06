package stcl.algo.poolers;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.util.Orthogonalizer;

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
		som = new SOM(mapSize, inputLength, rand, initialLearningRate, activationCodingFactor, stddev);
		errorMatrix = new SimpleMatrix(mapSize, mapSize);
		activationMatrix = new SimpleMatrix(mapSize, mapSize);
		this.inputLength = inputLength;
		this.mapSize = mapSize;		
		curNoiseMagnitude = 0; //TODO: Be aware of noise magnitude
	}

	
	/**
	 * 
	 * @param feedForwardInputVector
	 * @param orthogonalize if true output is orthogonalized. 
	 * @return Returns a probability matrix. The value of cell (i,j) in the output matrix is the probability that SOM-model (i,j) is an accurate model of the observed input
	 */
	public SimpleMatrix feedForward(SimpleMatrix feedForwardInputVector, boolean orthogonalize){
		//Test input
		if (!feedForwardInputVector.isVector()) throw new IllegalArgumentException("The feed forward input to the spatial pooler has to be a vector");
		if (feedForwardInputVector.numCols() != inputLength) throw new IllegalArgumentException("The feed forward input to the spatial pooler has to be a 1 x " + inputLength + " vector");
		
		//Adjust weights of SOM
		som.step(feedForwardInputVector);
		
		//Compute ActivationMatrix
		activationMatrix = som.computeActivationMatrix();
		
		//Normalize activation matrix
		//activationMatrix = normalize(activationMatrix);
		
		//Orthogonalize activation matrix
		if (orthogonalize){
			activationMatrix = orthogonalize(activationMatrix);
		}
		
		return activationMatrix;
	}
	
	/**
	 * 
	 * @param feedForwardInputVector
	 * @return orthogonalized probability matrix. The value of cell (i,j) in the output matrix is the probability that SOM-model (i,j) is an accurate model of the observed input
	 */
	public SimpleMatrix feedForward(SimpleMatrix feedForwardInputVector){
		return feedForward(feedForwardInputVector, true);
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
		return Orthogonalizer.orthogonalize(m);
		/*
		double max = Double.NEGATIVE_INFINITY;
		int maxID = -1;
		for (int i = 0; i < m.getNumElements(); i++){
			double value = m.get(i);
			if (value > max){
				max = value;
				maxID = i;
			}
		}
		
		SimpleMatrix ortho = new SimpleMatrix(m.numRows(), m.numCols());
		ortho.set(maxID, 1);
		return ortho;
		*/
	}
	
	protected SimpleMatrix normalize(SimpleMatrix matrix){
		double sum = matrix.elementSum();
		SimpleMatrix m = matrix.scale(1/sum);
		return m;
	}
	
	/**
	 * Use roulette emthod to choose a random model
	 * @param input
	 * @return
	 */
	protected SimpleMatrix chooseRandom(SimpleMatrix input){
		//Transform matrix into vector
		double[] vector = input.getMatrix().data;
		
		//Choose random number between 0 and 1
		double d = rand.nextDouble();
		
		//Go through bias vector until value is >= random number
		double tmp = 0;
		int id = 0;
		while (tmp < d && id < vector.length){
			tmp += vector[id++];
		}
		
		id--; //We have to subtract to be sure we get the correct model
		
		//Choose model from som
		SimpleMatrix model = som.getNode(id).getVector();
		
		//System.out.println("Chose model: " + id);
		
		return model;
		
	}
	
	protected SimpleMatrix chooseMax(SimpleMatrix input){
		//Transform bias matrix into vector
		double[] vector = input.getMatrix().data;
		
		//Go through bias vector until value is >= random number
		double max = Double.NEGATIVE_INFINITY;
		int maxID = -1;
		int id = 0;
		
		for (double d : vector){
			if (d > max){
				max = d;
				maxID = id;
			}
			id++;
		}
		
		//System.out.println("Chose model: " + maxID);
		
		//Choose model from som
		SimpleMatrix model = som.getNode(maxID).getVector();
		
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
	
	public int getMapSize(){
		return this.mapSize;
	}
	
	public void sensitize(int iteration){
		som.sensitize(iteration);
	}
}
