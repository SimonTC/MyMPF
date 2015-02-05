package stcl.algo.brain;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.biasunits.BiasUnit;
import stcl.algo.brain.rewardCorrelators.RewardCorrelator;
import stcl.algo.brain.rewardCorrelators.RewardFunction;

public class Connection {
	private NeoCorticalUnit in, out;
	private BiasUnit bias;
	private RewardCorrelator correlator;
	private SimpleMatrix correlationMatrix;
	private RewardFunction rewardFunction;
	private SimpleMatrix oldInputMatrix;
	
	/**
	 * A connection connects two neocortical units.
	 * Signals sent through the connection are biased by the bias unit used
	 */
	public Connection(NeoCorticalUnit in, NeoCorticalUnit out, Random rand, double biasInfluence, double maxReward, double alpha) {
		this.in = in;
		this.out = out;
		int inputMatrixSize = in.getTemporalPooler().getMapSize();
		this.bias = new BiasUnit(inputMatrixSize, biasInfluence, rand);
		this.correlator = new RewardCorrelator(inputMatrixSize);
		this.rewardFunction = new RewardFunction(maxReward, alpha);
	}
	
	/**
	 * Feeds the input matrix forward without changing it
	 * @param input
	 * @return
	 */
	public SimpleMatrix feedForward(SimpleMatrix input, double externalReward, double curLearningRate){
		double internalReward = rewardFunction.calculateReward(externalReward);
		correlationMatrix = correlator.correlateReward(oldInputMatrix, internalReward, curLearningRate);
		oldInputMatrix = input;
		
		return input;
	}
	
	public SimpleMatrix feedBack(SimpleMatrix fbInputMatrix, double noiseMagnitude){
		SimpleMatrix biasedMatrix = bias.biasFBSpatialOutput(fbInputMatrix, correlationMatrix, noiseMagnitude);
		
		return biasedMatrix;
	}

}
