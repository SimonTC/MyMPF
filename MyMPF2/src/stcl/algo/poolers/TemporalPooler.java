package stcl.algo.poolers;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.som.SOM;

public class TemporalPooler extends SpatialPooler  {

	private SOM leakyDifferenceMap; //TODO: FInd another name
	private double curLeakyCoefficient; 
	
	public TemporalPooler(Random rand, int maxIterations, int inputLength,
			int mapSize) {
		super(rand, maxIterations, inputLength, mapSize);
		
		curLeakyCoefficient = 1; //TODO: Initial leaky coefficient should be a parameter. Does it change at all?
		
		leakyDifferenceMap = new SOM(mapSize, mapSize, inputLength, rand);
		
	}
	
	@Override
	public SimpleMatrix feedForward(SimpleMatrix inputVector){
		//Update matrix with leaky differences
		
		//Find BMU
		
		//Update error matrix
		
		//Update SOM
		
		//TODO: OBS I am not sure this alogrithm is correct. Has to check up on it
		
		//Update weight matrix (SOM)
		som.step(inputVector, curLearningRate, curNeighborhoodRadius);
		
		//Collect error from the SOM (They are used in the update of the leaky differences)
		SimpleMatrix somErrors = som.getErrorMatrix();
		
		//Update matrix of temporal differences
		updateLeakyDifferenceMatrix(somErrors);
		
		//Find BMU in Differences. Te BMU is the element with the lowest error
			//TODO: Is this used for anything?
				 // I am not implementing this right now
		
		//Collect error
		errorMatrix = new SimpleMatrix(leakyDifferenceMap);
		
		//Compute activation
		activationMatrix = computeActivationMatrix(errorMatrix);
		
		//Compute output
		SimpleMatrix output = addNoise(activationMatrix, curNoiseMagnitude);
		
		return output;
	}
	
	private SimpleMatrix updateLeakyDifferenceMatrix(SimpleMatrix inputVector, SimpleMatrix som, SimpleMatrix leakyDifferencesMatrix){
		for (int row = 0; row < leakyDifferencesMatrix.numRows(); row++){
			for (int col = 0; col < leakyDifferencesMatrix.numCols(); col++){
				
			}
			SimpleMatrix leakyRow = leakyDifferencesMatrix.extractVector(true, row);
			SimpleMatrix somRow = som.extractVector(true, row);
			
		}
	}

	

}
