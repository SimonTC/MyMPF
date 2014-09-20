package stcl.algo.rewardCorrelators;

import org.ejml.simple.SimpleMatrix;

public class RewardCorrelator {
	
	private SimpleMatrix correlationMatrix;
	
	public RewardCorrelator(int size) {
		correlationMatrix = new SimpleMatrix(size, size);
	}
	
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
