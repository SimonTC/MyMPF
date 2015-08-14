package stcl.algo.poolers;

import java.io.Serializable;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.util.Normalizer;
import dk.stcl.core.basic.SomBasics;
import dk.stcl.core.basic.containers.SomNode;
import dk.stcl.core.utils.SomConstants;

public class SpatialPooler extends Pooler implements Serializable {
	private static final long serialVersionUID = 1L;
	//Weight matrix
	protected SOM som;

	/**
	 * Constructor used when all parameters are given
	 * @param rand
	 * @param inputLength
	 * @param mapSize
	 * @param initialLearningRate
	 * @param stddev
	 * @param activationCodingFactor
	 */
	public SpatialPooler(int inputLength, int mapSize, double initialLearningRate, double stddev, double activationCodingFactor ) {
		super(inputLength, mapSize);
		som = new SOM(mapSize, inputLength, initialLearningRate, activationCodingFactor, stddev);
	}
	
	public SpatialPooler(int inputLength, int mapSize, double initialLearningRate, double stddev, double activationCodingFactor, Random rand ) {
		super(inputLength, mapSize);
		som = new SOM(mapSize, inputLength, rand, initialLearningRate, activationCodingFactor, stddev);
	}
	
	public SpatialPooler(String initializationString, int startLine){
		super(initializationString, startLine);
		som = new SOM(initializationString, startLine + 1);
	}
	
	@Override
	public String toInitializationString(){
		String s = super.toInitializationString();
		s += som.toInitializationString();
		return s;
	}
	
	/**
	 * 
	 * @param feedForwardInputVector
	 * @return Returns a probability matrix. The value of cell (i,j) in the output matrix is the probability that SOM-model (i,j) is an accurate model of the observed input.
	 */
	public SimpleMatrix feedForward(SimpleMatrix feedForwardInputVector){
		//Test input
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
		if (feedBackwardInputMatrix.numCols() != mapSize || feedBackwardInputMatrix.numRows() != mapSize) throw new IllegalArgumentException("The feed back input to the spatial pooler has to be a " + mapSize + " x " + mapSize + " matrix");
		
		//Choose random model from som by roulette selection based on the input
		//SimpleMatrix model = chooseRandom(feedBackwardInputMatrix, som);
		SimpleMatrix model = chooseMax(feedBackwardInputMatrix, som);
		
		return model;		
	}
	
	public SOM getSOM(){
		return som;
	}

	public void setLearning(boolean learning){
		som.setLearning(learning);
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
