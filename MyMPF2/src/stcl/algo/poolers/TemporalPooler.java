package stcl.algo.poolers;

import org.ejml.simple.SimpleMatrix;

import dk.stcl.core.basic.containers.SomNode;
import dk.stcl.core.utils.SomConstants;
import stcl.algo.util.Normalizer;

public class TemporalPooler extends Pooler  {

	private RSOM rsom; 	
	
	public TemporalPooler(int inputLength, int mapSize,
			double initialLearningRate, double stddev,
			double activationCodingFactor, double decay) {
		super(inputLength, mapSize);

		rsom = new RSOM(mapSize, inputLength,initialLearningRate, activationCodingFactor, stddev, decay);
	}
	
	public TemporalPooler (String initializationString, int startLine){
		super(initializationString, startLine);
		rsom = new RSOM(initializationString, startLine + 1);
	}

	@Override
	public String toInitializationString(){
		String s = super.toInitializationString();
		s += rsom.toInitializationString();
		return s;
	}
	
	@Override
	public SimpleMatrix feedForward(SimpleMatrix inputVector){
		//Test input
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
		if (inputMatrix.numCols() != mapSize || inputMatrix.numRows() != mapSize) throw new IllegalArgumentException("The feed back input to the temporal pooler has to be a " + mapSize + " x " + mapSize + " matrix");
		
		//Choose random model from som by roulette selection based on the input
		//SimpleMatrix model = chooseRandom(inputMatrix, rsom);
		SimpleMatrix model = chooseMax(inputMatrix, rsom);
		
		//Add noise
		//model = addNoise(model, curNoiseMagnitude);
		
		//Normalize
		model = Normalizer.normalize(model);
		
		return model;		
	}
	
	public void flushTemporalMemory(){
		rsom.flush();	
	}
	
	@Override
	public void setLearning(boolean learning){
		rsom.setLearning(learning);
	}
	
	public void sensitize(int iteration){
		rsom.sensitize(iteration);
	}
	
	public RSOM getRSOM(){
		return this.rsom;
	}

	@Override
	public void printModelWeigths(){
		for (SomNode n :rsom.getNodes()){
			for (double d : n.getVector().getMatrix().data){
				System.out.printf("%.3f  ", d);
			}
			System.out.println();
		}
	}
}
	