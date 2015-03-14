package stcl.algo.brain;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.biasunits.BiasUnit;
import stcl.algo.brain.rewardCorrelators.RewardCorrelator;
import stcl.algo.brain.rewardCorrelators.RewardFunction;

public class Connection {
	private BiasUnit bias;
	private RewardCorrelator correlator;
	private SimpleMatrix correlationMatrix;
	private RewardFunction rewardFunction;
	private SimpleMatrix ffOutputBefore;
	
	/**
	 * A connection connects two neocortical units.
	 * Signals sent through the connection are biased by the bias unit used
	 */
	public Connection(NeoCorticalUnit in, NeoCorticalUnit out, Random rand, double biasInfluence, double maxReward, double alpha) {
		int inputMatrixSize = in.getTemporalMapSize();
		this.bias = new BiasUnit(inputMatrixSize, biasInfluence, rand);
		this.correlator = new RewardCorrelator(inputMatrixSize);
		this.rewardFunction = new RewardFunction(maxReward, alpha);
		ffOutputBefore = in.getFFOutput();
	}
	
	/**
	 * Feeds the input matrix forward without changing it
	 * @param ffOutputFromUnit1
	 * @return
	 */
	public SimpleMatrix feedForward(SimpleMatrix ffOutputFromUnit1, double externalReward, double curLearningRate){
		//Do reward correlation
		double internalReward = rewardFunction.calculateReward(externalReward);
		correlationMatrix = correlator.correlateReward(ffOutputBefore, internalReward, curLearningRate);
		
		ffOutputBefore = ffOutputFromUnit1;
		
		return ffOutputFromUnit1;
	}
	
	public SimpleMatrix feedBack(SimpleMatrix fbOutputFromUnit2, double noiseMagnitude){
				
		//Bias output
		SimpleMatrix biasedMatrix = bias.biasFBSpatialOutput(fbOutputFromUnit2, correlationMatrix, noiseMagnitude);
		
		return biasedMatrix;
	}

}
