package stcl.algo.poolers;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.som.RSOM;
import stcl.algo.som.SOM;

public class TemporalPooler extends SpatialPooler  {

	private RSOM rsom; //TODO: FInd another name
	private double curLeakyCoefficient; 
	
	public TemporalPooler(Random rand, int maxIterations, int inputLength,
			int mapSize) {
		super(rand, maxIterations, inputLength, mapSize);
		
		curLeakyCoefficient = 1; //TODO: Initial leaky coefficient should be a parameter. Does it change at all?
		
		rsom = new SOM(mapSize, mapSize, inputLength, rand);
		
	}
	
	@Override
	public SimpleMatrix feedForward(SimpleMatrix inputVector){
		//Update RSOM
		rsom.step(inputVector, curLeakyCoefficient, curLearningRate, curNeighborhoodRadius);
		
		//Collect error
		errorMatrix = rsom.getErrorMatrix();
		
		//Compute activation
		activationMatrix = computeActivationMatrix(errorMatrix);
		
		return activationMatrix;
	}
}
