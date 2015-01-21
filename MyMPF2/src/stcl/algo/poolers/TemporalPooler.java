package stcl.algo.poolers;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import dk.stcl.som.rsom.RSOM;

public class TemporalPooler extends SpatialPooler  {

	private RSOM rsom; 
	
	public TemporalPooler(Random rand, int inputLength, int mapSize,
			double initialLearningRate, double stddev,
			double activationCodingFactor, double decay) {
		
		super(rand, inputLength, mapSize, initialLearningRate, stddev,
				activationCodingFactor);
		
		rsom = new RSOM(mapSize, mapSize, inputLength, rand, initialLearningRate, stddev, activationCodingFactor, decay);
		
	}	
	
	public TemporalPooler(Random rand, int inputLength, int mapSize, double decay) {
		super(rand, inputLength, mapSize);
		
		rsom = new RSOM(mapSize, mapSize, inputLength, rand, 0.1, 1, 0.125, decay);
	}
	
	
	@Override
	public SimpleMatrix feedForward(SimpleMatrix inputVector){
		//Test input
		if (!inputVector.isVector()) throw new IllegalArgumentException("The feed forward input to the temporal pooler has to be a vector");
		if (inputVector.numCols() != inputLength) throw new IllegalArgumentException("The feed forward input to the temporal pooler has to be a 1 x " + inputLength + " vector");				
		
		//Update RSOM
		rsom.step(inputVector);
		
		//Compute activation
		activationMatrix = rsom.computeActivationMatrix();
		
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
	
	public void flushTemporalMemory(){
		//TODO: SHould that be the name of the method?
		rsom.flush();	}
}
