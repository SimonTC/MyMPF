package stcl.algo.brain;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.biasunits.BiasUnit;
import stcl.algo.brain.rewardCorrelators.RewardCorrelator;
import stcl.algo.brain.rewardCorrelators.RewardFunction;

public class Brain {
	
	private NeoCorticalUnit nu1;
	private NeoCorticalUnit nu2;
	
	private Random rand;
	private int maxIterations;
	private int ffInputLength1;
	private int ffInputLength2;
	private int spatialMapSize1;
	private int spatialMapSize2;
	private int temporalMapSize1;
	private int temporalMapSize2;
	
	private double biasInfluence;
	private double constantPredictionLearningRate;
	private boolean useMarkovPrediction;
	private double constantLeakyCoefficient;
	
	private double noiseMagnitude;
	
	private RewardCorrelator rewardCorrelator1;
	private RewardFunction rewardFunction;
	private BiasUnit bias;
	
	private SimpleMatrix ffOutputU1Before;
	
	public Brain() {
		rand = new Random();
		maxIterations = 100;
		ffInputLength1 = 16 * 16;
		spatialMapSize1 = 5;
		temporalMapSize1 = 5;
		ffInputLength2 = temporalMapSize1 * temporalMapSize1;
		temporalMapSize2 = 5;
		
		biasInfluence = 0.1;
		constantPredictionLearningRate = 0.6;
		useMarkovPrediction = true;
		constantLeakyCoefficient = 0.3;
		
		noiseMagnitude = 0.05;
		
		double maxReward = 1;
		double historyInfluence = 0.3;		
		rewardFunction = new RewardFunction(maxReward, historyInfluence);
		
		rewardCorrelator1 = new RewardCorrelator(temporalMapSize1);
		
		ffOutputU1Before = new SimpleMatrix(temporalMapSize1, temporalMapSize1);
		
		bias = new BiasUnit(ffOutputU1Before.numRows(), historyInfluence, rand);
		
		nu1 = new NeoCorticalUnit(rand, maxIterations, ffInputLength1, spatialMapSize1, temporalMapSize1, constantPredictionLearningRate, useMarkovPrediction, constantLeakyCoefficient);
		
		nu2 = new NeoCorticalUnit(rand, maxIterations, ffInputLength2, spatialMapSize2, temporalMapSize2, constantPredictionLearningRate, useMarkovPrediction, constantLeakyCoefficient);
	}
	
	public SimpleMatrix activate(SimpleMatrix inputVector, double reward){
		//Feed input vector forward
		SimpleMatrix ffOutputU1 = nu1.feedForward(inputVector);
				
		//Do reward correlation
		double internalReward = rewardFunction.calculateReward(reward);
		SimpleMatrix correlationMatrix = rewardCorrelator1.correlateReward(ffOutputU1Before, internalReward, 0.3);
		ffOutputU1Before = ffOutputU1;
		
		//Transform output matrix to vector
		SimpleMatrix ffInputU2 = new SimpleMatrix(ffOutputU1);
		ffInputU2.reshape(1, ffInputU2.numCols() * ffInputU2.numRows());
		
		//Send unit one's output to unit 2
		SimpleMatrix ffOutputU2 = nu2.feedBackward(ffInputU2);
		
		//Give some fb input to unit 2
		SimpleMatrix fbInputU2 = SimpleMatrix.random(ffOutputU2.numRows(), ffOutputU2.numCols(), 0, 1, rand);
		
		//Collect FB output from unit 2
		SimpleMatrix fbOutputU2 = nu2.feedBackward(fbInputU2);
		
		//Bias fb output from unit 2
		SimpleMatrix biasedFBOutputU2 = bias.biasFBSpatialOutput(fbOutputU2, correlationMatrix, noiseMagnitude);
		
		//Give FB output from unit 2 to unit 1
		SimpleMatrix fbOutputU1 = nu1.feedBackward(biasedFBOutputU2);
		
		//Return something
		return fbOutputU1;		
	}
	
	public void newIteration(){
		nu1.resetTemporalDifferences();
		nu2.resetTemporalDifferences();
	}

}
