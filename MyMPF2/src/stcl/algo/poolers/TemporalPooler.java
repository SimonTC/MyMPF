package stcl.algo.poolers;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import dk.stcl.som.rsom.RSOM;

public class TemporalPooler extends SpatialPooler  {

	private RSOM rsom; 
	private double curLeakyCoefficient; 
	
	public TemporalPooler(Random rand, int maxIterations, int inputLength,
			int mapSize, double leakyCoefficient) {
		super(rand, maxIterations, inputLength, mapSize);
		
		curLeakyCoefficient = leakyCoefficient; //TODO: Initial leaky coefficient should be a parameter. Does it change at all?
		
		rsom = new RSOM(mapSize, mapSize, inputLength, rand, curLeakyCoefficient);
		
	}
	
	@Override
	public SimpleMatrix feedForward(SimpleMatrix inputVector){
		//Test input
		if (!inputVector.isVector()) throw new IllegalArgumentException("The feed forward input to the temporal pooler has to be a vector");
		if (inputVector.numCols() != inputLength) throw new IllegalArgumentException("The feed forward input to the temporal pooler has to be a 1 x " + inputLength + " vector");				
		
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
	
	/**
	 * 
	 * @param inputMatrix
	 * @return vector
	 */
	public SimpleMatrix feedBackward(SimpleMatrix inputMatrix){
		//Test input
		if (inputMatrix.isVector()) throw new IllegalArgumentException("The feed back input to the temporal pooler has to be a matrix");
		if (inputMatrix.numCols() != mapSize || inputMatrix.numRows() != mapSize) throw new IllegalArgumentException("The feed back input to the temporal pooler has to be a " + mapSize + " x " + mapSize + " matrix");
		
		//Choose random model from som by roulette selection based on the input
		SimpleMatrix model = chooseRandom(inputMatrix);
		
		//Add noise
		model = addNoise(model, curNoiseMagnitude);
		
		//Normalize
		double sum = model.elementSum();
		model = model.scale(1/sum);
		
		return model;		
	}
	
	public void resetLeakyDifferences(){
		//TODO: SHould that be the name of the method?
		rsom.resetLeakyDifferencesMap();
	}
}
