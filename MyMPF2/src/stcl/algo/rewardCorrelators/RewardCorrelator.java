package stcl.algo.rewardCorrelators;

import org.ejml.simple.SimpleMatrix;

public class RewardCorrelator {
	
	private SimpleMatrix correlationMatrix;
	
	public RewardCorrelator(int size) {
		correlationMatrix = new SimpleMatrix(size, size);
	}
	
	/**
	 * The reward correlator is placed between layers in the brain. It correlates the reward at time t with the FF output from the layer below at time t-1
	 * @param inputMatrix the FF output from the layer below this unit at t-1
	 * @param reward reward given to the brain at times t
	 * @param curLearningRate
	 * @return
	 */
	public SimpleMatrix correlateReward(SimpleMatrix inputMatrix, double reward, double curLearningRate){
		SimpleMatrix t = inputMatrix.scale(curLearningRate); //Find other name
		
		SimpleMatrix tmp = t.scale(reward);
		SimpleMatrix learnedCorrelation = new SimpleMatrix(inputMatrix.numRows(), inputMatrix.numCols());
		learnedCorrelation.set(1);
		learnedCorrelation = learnedCorrelation.minus(t);
		learnedCorrelation = learnedCorrelation.elementMult(correlationMatrix);
		correlationMatrix = tmp.plus(learnedCorrelation);
		
		return correlationMatrix;
		
		
	}

}
