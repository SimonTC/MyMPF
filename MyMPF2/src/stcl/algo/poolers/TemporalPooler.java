package stcl.algo.poolers;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

public class TemporalPooler extends SpatialPooler  {

	public TemporalPooler(Random rand, int inputLength, int mapSize,
			double initialLearningRate, double stddev,
			double activationCodingFactor, int maxIterations, double decay) {
		super(rand, inputLength, mapSize, initialLearningRate, stddev,
				activationCodingFactor, maxIterations);

		rsom = new RSOM(mapSize, inputLength, rand, maxIterations, initialLearningRate, activationCodingFactor, decay);
	}

	private RSOM rsom; 
	
	
	
	
	@Override
	public SimpleMatrix feedForward(SimpleMatrix inputVector){
		//Test input
		//if (!inputVector.isVector()) throw new IllegalArgumentException("The feed forward input to the temporal pooler has to be a vector");
		if (inputVector.numCols() != inputLength) throw new IllegalArgumentException("The feed forward input to the temporal pooler has to be a 1 x " + inputLength + " vector");				
		
		//Update RSOM
		rsom.step(inputVector);
		
		//Compute activation
		activationMatrix = rsom.computeActivationMatrix();
		
		//Normalize activation matrix
		activationMatrix = normalize(activationMatrix);
		
		//Orthogonalize activation matrix
		//activationMatrix = orthogonalize(activationMatrix);
		
		return activationMatrix;
	}
	
	/**
	 * 
	 * @param inputMatrix
	 * @return vector
	 */
	public SimpleMatrix feedBackward(SimpleMatrix inputMatrix){
		//Test input
		//if (inputMatrix.isVector()) throw new IllegalArgumentException("The feed back input to the temporal pooler has to be a matrix");
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
		rsom.flush();	
	}
	
	@Override
	public void setLearning(boolean learning){
		super.setLearning(learning);
		rsom.setLearning(learning);
	}
	
	public void sensitize(int iteration){
		rsom.sensitize(iteration);
	}
}
