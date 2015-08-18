package stcl.fun.rps;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.NeoCorticalUnit;
import stcl.algo.brain.Network_DataCollector;
import stcl.algo.util.Normalizer;
import stcl.fun.rps.rewardfunctions.RewardFunction;
import stcl.graphics.MPFGUI;

public class SequenceRunner_NU extends SequenceRunner {
	
	private NeoCorticalUnit my_Activator;

	public SequenceRunner_NU(int[] sequence, SimpleMatrix[] possibleInputs,
			RewardFunction[] rewardFunctions, Random rand, double noiseMagnitude) {
		super(sequence, possibleInputs, rewardFunctions, rand, noiseMagnitude);
		// TODO Auto-generated constructor stub
	}
	
	
	/**
	 * Goes through the sequence once.
	 * Remember to call reset() if the evaluation should start from scratch
	 * @param my_Activator
	 * @return Array containing prediction success and fitness in the form [prediction,fitness]
	 */
	public double[] runSequence(Network_DataCollector activator, MPFGUI gui, double explorationChance){
		this.my_Activator = activator.getUnitNodes().get(0).getUnit();
		double totalPredictionError = 0;
		double totalGameScore = 0;
		double reward_before = 0;
		
		int state = 1;
		
		initializeSequence();
		
		for (int i = 0; i < sequence.length; i++){
			state = sequence[i];
			
			SimpleMatrix fbinputMatrix = new SimpleMatrix(5, 5);
			fbinputMatrix.set(1);
			fbinputMatrix = Normalizer.normalize(fbinputMatrix);
			
			my_Activator.feedBackward(fbinputMatrix);
			int myAction = my_Activator.getNextAction();
			if (rand.nextDouble() < explorationChance){
				myAction = rand.nextInt(3);
			}			
			
			double reward_now = curRewardFunction.reward(state, myAction);
			totalGameScore += reward_now;	
			
			SimpleMatrix ffInput = new SimpleMatrix(possibleInputs[state]);
			ffInput.reshape(1, 25);
			
			my_Activator.resetActivity();

			my_Activator.feedForward(ffInput, reward_before, myAction);
			
			reward_before = reward_now;
			
		}
		endSequence(reward_before);
		
		my_Activator.newEpisode();

		
		//endSequence(activator, reward_before);
		
		double avgPredictionError = totalPredictionError / (double) sequence.length;
		double avgScore = totalGameScore / (double) sequence.length;
		double predictionSuccess = 1 - avgPredictionError;
		
		//Scores can't be less than zero as the evolutionary algorithm can't work with that
		
		double[] result = {predictionSuccess, avgScore};
		return result;
	}
	
	private void initializeSequence(){
		//Give blank input and action to network
		SimpleMatrix initialInput = new SimpleMatrix(1, 25);
		SimpleMatrix initialAction = new SimpleMatrix(1, 3);
		my_Activator.feedForward(initialInput, 0, 0);

	}
	
	private void endSequence(double reward){
		//Give blank input and action to network
		SimpleMatrix initialInput = new SimpleMatrix(1, 25);
		SimpleMatrix initialAction = new SimpleMatrix(1, 3);
		my_Activator.feedForward(initialInput, reward, 0);
	}

}
