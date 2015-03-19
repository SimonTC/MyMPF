package stcl.algo.predictors;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.biasunits.BiasUnit;
import stcl.algo.brain.rewardCorrelators.RewardCorrelator;
import stcl.algo.brain.rewardCorrelators.RewardFunction;

public class Decider extends Predictor_VOMM {
	
	private BiasUnit bias;
	private RewardCorrelator correlator;
	private SimpleMatrix correlationMatrix;
	private RewardFunction rewardFunction;
	private double externalReward;

	public Decider(int markovOrder, double learningRate, Random rand, double biasInfluence, double maxReward, double alpha, int inputMatrixSize) {
		super(markovOrder, learningRate, rand);
		
		this.bias = new BiasUnit(inputMatrixSize, biasInfluence, rand);
		this.correlator = new RewardCorrelator(inputMatrixSize);
		this.rewardFunction = new RewardFunction(maxReward, alpha);
		externalReward = 0;
	}
	
	@Override
	/**
	 * Biases the prediction towards a state that leads to a higher reward (exploitation) or a random state (exploration)
	 */
	public SimpleMatrix predict(SimpleMatrix inputMatrix) {
		SimpleMatrix prediction = super.predict(inputMatrix);
		
		//Do reward correlation
		if (learning){
			double internalReward = rewardFunction.calculateReward(externalReward);
			correlationMatrix = correlator.correlateReward(inputMatrix, internalReward, 0.1);
		}
		
		//Bias prediction toward a better choice
		SimpleMatrix decision = bias.biasFBSpatialOutput(prediction, correlationMatrix, 0);
		return decision;		
	}
	
	public void giveExternalReward(double reward){
		this.externalReward = reward;
	}

}
