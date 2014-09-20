package stcl.algo.poolers;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.som.SOM;

public class TemporalPooler extends SpatialPooler  {

	private SimpleMatrix differences; //TODO: FInd another name
	private double curLeakyCoefficient; 
	
	public TemporalPooler(Random rand, int maxIterations, int inputLength,
			int mapSize) {
		super(rand, maxIterations, inputLength, mapSize);
		
		curLeakyCoefficient = 1; //TODO: Initial leaky coefficient should be a parameter. Does it change at all?
		
		differences = new SimpleMatrix(mapSize, mapSize);	
		
	}
	
	@Override
	public SimpleMatrix feedForward(SimpleMatrix input){
		//TODO: OBS I am not sure this alorithm is correct. Has to check up on it
		
		//Update weight matrix (SOM)
		som.step(input, curLearningRate, curNeighborhoodRadius);
		
		//Collect error from the SOM (They are used in the update of the leaky differences)
		SimpleMatrix somErrors = som.getErrorMatrix();
		
		//Update matrix of temporal differences
		updateTemporalDifferences(somErrors);
		
		//Find BMU in Differences. THe BMU is the element with the lowest error
			//TODO: Is this used for anything?
				 // I am not implementing this right now
		
		//Collect error
		SimpleMatrix errorMatrix = new SimpleMatrix(differences);
		
		//Compute activation
		double maxError = errorMatrix.elementMaxAbs();
		computeActivationMatrix(maxError, errorMatrix);
		
		//Compute output
		SimpleMatrix output = addNoise(matrix_Activation, curNoiseMagnitude);
		
		return output;
	}
	
	private void updateTemporalDifferences(SimpleMatrix somErrors){
		SimpleMatrix scaledErros = somErrors.scale(curLeakyCoefficient);
		SimpleMatrix scaledDifferences = differences.scale(1-curLeakyCoefficient);
		differences = scaledErros.elementMult(scaledDifferences);
	}

	

}
