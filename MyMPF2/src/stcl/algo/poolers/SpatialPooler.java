package stcl.algo.poolers;

import java.io.Serializable;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.util.Normalizer;
import dk.stcl.core.basic.SomBasics;
import dk.stcl.core.basic.containers.SomNode;
import dk.stcl.core.utils.SomConstants;

public class SpatialPooler implements Serializable {
	private static final long serialVersionUID = 1L;
	//Weight matrix
	private SOM som;
	
	//Matrice used 
	private SimpleMatrix activationMatrix;
	
	//Variables used for testing inputs
	private int inputLength;
	private int mapSize;
	
	//Misc
	private Random rand;

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
		activationMatrix = new SimpleMatrix(mapSize, mapSize);
		this.inputLength = inputLength;
		this.mapSize = mapSize;		
	}
	
	public SpatialPooler(String initializationString, int startLine, Random rand){
		String[] lines = initializationString.split(SomConstants.LINE_SEPARATOR);
		String[] poolerInfo = lines[startLine].split(" ");
		inputLength = Integer.parseInt(poolerInfo[0]);
		mapSize = Integer.parseInt(poolerInfo[1]);
		activationMatrix = new SimpleMatrix(mapSize, mapSize);
		som = new SOM(initializationString, startLine + 1);
		this.rand = rand;
	}
	
	public String toInitializationString(){
		String s = inputLength + " " + mapSize + SomConstants.LINE_SEPARATOR;
		s += som.toFileString();
		return s;
	}
	
	/**
	 * 
	 * @param feedForwardInputVector
	 * @return Returns a probability matrix. The value of cell (i,j) in the output matrix is the probability that SOM-model (i,j) is an accurate model of the observed input.
	 */
	public SimpleMatrix feedForward(SimpleMatrix feedForwardInputVector){
		//Test input
		if (!feedForwardInputVector.isVector()) throw new IllegalArgumentException("The feed forward input to the spatial pooler has to be a vector");
		if (feedForwardInputVector.numCols() != inputLength) throw new IllegalArgumentException("The feed forward input to the spatial pooler has to be a 1 x " + inputLength + " vector");
		
		//Adjust weights of SOM
		som.step(feedForwardInputVector);
		
		//Compute ActivationMatrix
		activationMatrix = som.computeActivationMatrix();
		
		//Normalize
		activationMatrix = Normalizer.normalize(activationMatrix);
		
		return activationMatrix;
	}
	
		
	/**
	 * 
	 * @param feedBackwardInputMatrix
	 * @return vector Returns a vector with the values that are expected to be observed at time t+1. The vector is choosed by roulette based on the weights in the feedBackwardInputMatrix.
	 */
	public SimpleMatrix feedBackward(SimpleMatrix feedBackwardInputMatrix){
		//Test input
		if (feedBackwardInputMatrix.isVector()) throw new IllegalArgumentException("The feed back input to the spatial pooler has to be a matrix");
		if (feedBackwardInputMatrix.numCols() != mapSize || feedBackwardInputMatrix.numRows() != mapSize) throw new IllegalArgumentException("The feed back input to the spatial pooler has to be a " + mapSize + " x " + mapSize + " matrix");
		
		//Choose random model from som by roulette selection based on the input
		//SimpleMatrix model = chooseRandom(feedBackwardInputMatrix, som);
		SimpleMatrix model = chooseMax(feedBackwardInputMatrix, som);
		
		return model;		
	}
	
	/**
	 * Use roulette method to choose a random model from the given SomBasics object
	 * @param input
	 * @return
	 */
	private SimpleMatrix chooseRandom(SimpleMatrix input, SomBasics map){
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
		SimpleMatrix model = map.getNode(id).getVector();
		
		//System.out.println("Chose model: " + id);
		
		return model;
		
	}
	
	/**
	 * Choose the most probable model from the given SomBasics object
	 * @param input
	 * @param map
	 * @return
	 */
	private SimpleMatrix chooseMax(SimpleMatrix input, SomBasics map){
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
		SimpleMatrix model = map.getNode(maxID).getVector();
		
		return model;
		
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
	
	public void printModelWeigths(){
		for (SomNode n :som.getNodes()){
			for (double d : n.getVector().getMatrix().data){
				System.out.printf("%.3f  ", d);
			}
			System.out.println();
		}
	}
}
