package stcl.algo.poolers;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.som.RSOM;
import stcl.algo.som.SOM;

public class TemporalPooler extends SpatialPooler  {

	private RSOM rsom; //TODO: FInd another name
	private double curLeakyCoefficient; 
	
	public TemporalPooler(Random rand, int maxIterations, int inputLength,
			int mapSize, double leakyCoefficient) {
		super(rand, maxIterations, inputLength, mapSize);
		
		curLeakyCoefficient = leakyCoefficient; //TODO: Initial leaky coefficient should be a parameter. Does it change at all?
		
		rsom = new RSOM(mapSize, mapSize, inputLength, rand, curLeakyCoefficient);
		
	}
	
	@Override
	public SimpleMatrix feedForward(SimpleMatrix inputVector){
		//Update RSOM
		rsom.step(inputVector, curLearningRate, curNeighborhoodRadius);
		
		//Collect error
		errorMatrix = rsom.getErrorMatrix();
		
		//Compute activation
		activationMatrix = computeActivationMatrix(errorMatrix);
		
		//Normalize activation matrix
		activationMatrix = normalize(activationMatrix);
		
		return activationMatrix;
	}
}
