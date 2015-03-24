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
	private Random rand;

	public Decider(int markovOrder, double learningRate, Random rand, double biasInfluence, double maxReward, double alpha, int inputMatrixSize) {
		super(markovOrder, learningRate, rand);
		this.rand = rand;
		this.bias = new BiasUnit(inputMatrixSize, biasInfluence, rand);
		this.correlator = new RewardCorrelator(inputMatrixSize);
		this.rewardFunction = new RewardFunction(maxReward, alpha);
		externalReward = 0;
		learning = true;
	}
	
	@Override
	/**
	 * Biases the prediction towards a state that leads to a higher reward (exploitation) or a random state (exploration)
	 */
	public SimpleMatrix predict(SimpleMatrix inputMatrix) {
		return this.predict(inputMatrix, 0);	
	}
	
	public SimpleMatrix predict(SimpleMatrix inputMatrix, double explorationChance){
		SimpleMatrix prediction = super.predict(inputMatrix);
		
		//Do reward correlation
		if (learning){
			double internalReward = rewardFunction.calculateReward(externalReward);
			correlationMatrix = correlator.correlateReward(inputMatrix, internalReward, 0.1);
		}
		
		SimpleMatrix decision = null;
		if (rand.nextDouble() < explorationChance){
			//Choose random action
			decision = new SimpleMatrix(prediction);
			decision.set(0);
			decision.set(rand.nextInt(decision.getNumElements()), 1);
		} else {
			//Bias prediction toward a better choice
			decision = bias.biasFBSpatialOutput(prediction, correlationMatrix, 0);
		}
		probabilityMatrix = decision;
		return decision;
	}
	
	public void giveExternalReward(double reward){
		this.externalReward = reward;
	}
	
	public void printCorrelationMatrix(){
		if (correlationMatrix != null) correlationMatrix.print();
	}
	
	

}
