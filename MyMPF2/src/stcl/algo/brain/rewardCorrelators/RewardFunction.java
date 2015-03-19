package stcl.algo.brain.rewardCorrelators;

public class RewardFunction {
	
	private double externalRewardBefore;
	private double externalRewardNow;
	private double maxReward;
	private double alpha;
	private double internalRewardBefore;
	
	/**
	 * 
	 * @param maxReward maximum possible reward given. Used in normalizing the weighted average of the rewards
	 * @param alpha Influence of the historic reward signals
	 */
	public RewardFunction(double maxReward, double alpha) {
		externalRewardBefore = 0;
		externalRewardNow = 0;
		internalRewardBefore = 0;
		this.maxReward = maxReward;
		this.alpha = alpha;
	}

	
	public double calculateReward(double externalReward){
		/*
		externalRewardNow = externalReward;
		
		double exponentialWeightedMovingAverage = (externalRewardNow - externalRewardBefore) / maxReward;
		
		double internalReward = alpha * exponentialWeightedMovingAverage + (1-alpha) * internalRewardBefore;
		
		internalRewardBefore = internalReward;
		externalRewardBefore = externalRewardNow;
		
		return internalReward;
		*/
		return externalReward;
	}
}
