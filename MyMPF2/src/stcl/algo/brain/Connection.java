package stcl.algo.brain;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.biasunits.BiasUnit;
import stcl.algo.brain.rewardCorrelators.RewardCorrelator;
import stcl.algo.brain.rewardCorrelators.RewardFunction;

public class Connection {
	private NU in, out;
	private BiasUnit bias;
	private RewardCorrelator correlator;
	private SimpleMatrix correlationMatrix;
	private RewardFunction rewardFunction;
	private SimpleMatrix temporalFFActivationMatrixNow, temporalFFActivationMatrixBefore;
	
	/**
	 * A connection connects two neocortical units.
	 * Signals sent through the connection are biased by the bias unit used
	 */
	public Connection(NU in, NU out, Random rand, double biasInfluence, double maxReward, double alpha) {
		this.in = in;
		this.out = out;
		int inputMatrixSize = in.getTemporalPooler().getMapSize();
		this.bias = new BiasUnit(inputMatrixSize, biasInfluence, rand);
		this.correlator = new RewardCorrelator(inputMatrixSize);
		this.rewardFunction = new RewardFunction(maxReward, alpha);
		this.temporalFFActivationMatrixNow = new SimpleMatrix(inputMatrixSize, inputMatrixSize);
	}
	
	/**
	 * Feeds the input matrix forward without changing it
	 * @param ffInputVector
	 * @return
	 */
	public SimpleMatrix feedForward(SimpleMatrix ffInputVector, double externalReward, double curLearningRate){
		temporalFFActivationMatrixBefore = temporalFFActivationMatrixNow;
		
		//Feed input vector to first unit
		in.feedForward(ffInputVector);
		temporalFFActivationMatrixNow = in.getFFOutput();
		
		//Do reward correlation
		double internalReward = rewardFunction.calculateReward(externalReward);
		correlationMatrix = correlator.correlateReward(temporalFFActivationMatrixBefore, internalReward, curLearningRate);
				
		return new SimpleMatrix(temporalFFActivationMatrixNow);
	}
	
	public SimpleMatrix feedBack(double noiseMagnitude){
		//Get fb outut from out unit
		SimpleMatrix fbInputMatrix = out.getFBOutput();
		
		//Convert to matrix
		int size = in.getTemporalPooler().getMapSize();
		SimpleMatrix fb = new SimpleMatrix(size, size, true, fbInputMatrix.getMatrix().data);
		
		//Bias output
		SimpleMatrix biasedMatrix = bias.biasFBSpatialOutput(fb, correlationMatrix, noiseMagnitude);
		
		//Feed to in unit
		in.feedBackward(biasedMatrix);
		SimpleMatrix ffOutput = in.getFBOutput();
		
		return ffOutput;
	}

}
