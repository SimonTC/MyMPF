package stcl.algo.poolers;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.util.Normalizer;

public class TemporalPooler extends SpatialPooler  {

	private RSOM rsom; 	
	
	public TemporalPooler(Random rand, int inputLength, int mapSize,
			double initialLearningRate, double stddev,
			double activationCodingFactor, double decay) {
		super(rand, inputLength, mapSize, initialLearningRate, stddev, activationCodingFactor);

		rsom = new RSOM(mapSize, inputLength, rand, initialLearningRate, activationCodingFactor, stddev, decay);
	}

	
	
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
		activationMatrix = Normalizer.normalize(activationMatrix);
		
		return activationMatrix;
	}
	
	/**
	 * 
	 * @param inputMatrix containing probabiliites of being in the dfferent temporal groups in the model
	 * @return vector containing probabilities of seeing the different spatial models in the current temporal group
	 */
	public SimpleMatrix feedBackward(SimpleMatrix inputMatrix){
		//Test input
		//if (inputMatrix.isVector()) throw new IllegalArgumentException("The feed back input to the temporal pooler has to be a matrix");
		if (inputMatrix.numCols() != mapSize || inputMatrix.numRows() != mapSize) throw new IllegalArgumentException("The feed back input to the temporal pooler has to be a " + mapSize + " x " + mapSize + " matrix");
		
		//Choose random model from som by roulette selection based on the input
		SimpleMatrix model = chooseRandom(inputMatrix, rsom);
		
		//Add noise
		model = addNoise(model, curNoiseMagnitude);
		
		//Normalize
		model = Normalizer.normalize(model);
		
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
	
	public RSOM getRSOM(){
		return this.rsom;
	}
}
