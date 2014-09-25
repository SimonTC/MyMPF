package stcl.algo.brain.rewardCorrelators;

import org.ejml.simple.SimpleMatrix;

public class RewardCorrelator {
	
	private SimpleMatrix correlationMatrix;
	
	public RewardCorrelator(int inputMatrixSize) {
		correlationMatrix = new SimpleMatrix(inputMatrixSize, inputMatrixSize);
	}
	
	/**
	 * The reward correlator is placed between layers in the brain. It correlates the reward at time t with the FF output from the layer below at time t-1.
	 * 
	 * C_ij(t) = tau_ij(t) x R(t) + (1-tau_ij(t)) * C_ij(t-1)
	 * tau_ij(t) = w_c * input_ij
	 * w_c = curLearningRate
	 * @param inputMatrix the FF output from the layer below this unit at t-1
	 * @param reward reward given to the brain at times t
	 * @param curLearningRate
	 * @return
	 */
	public SimpleMatrix correlateReward(SimpleMatrix inputMatrix, double reward, double curLearningRate){
		//Multiply input matrix by the learning rate input matrix
		SimpleMatrix tau = inputMatrix.scale(curLearningRate); 
		
		//Multiply the tau value with the given reward
		//This is the influence that the actions at time t-1 have on the reward (I think)
		SimpleMatrix tmp = tau.scale(reward);
		
		//Calculate the other part of the equation
		SimpleMatrix learnedCorrelation = new SimpleMatrix(inputMatrix.numRows(), inputMatrix.numCols());
		learnedCorrelation.set(1);
		learnedCorrelation = learnedCorrelation.minus(tau);
		learnedCorrelation = learnedCorrelation.elementMult(correlationMatrix);
		
		//Calculate correlation matrix
		correlationMatrix = tmp.plus(learnedCorrelation);
		
		return correlationMatrix;
		
		
	}

}
