package stcl.algo.brain.rewardCorrelators;

public class RewardFunction {
	
	private double externalRewardBefore;
	private double externalRewardNow;
	private double maxReward;
	private double historyInfluence;
	private double internalRewardBefore;
	
	public RewardFunction(double maxReward, double historyInfluence) {
		externalRewardBefore = 0;
		externalRewardNow = 0;
		internalRewardBefore = 0;
		this.maxReward = maxReward;
		this.historyInfluence = historyInfluence;
	}

	
	public double calculateReward(double externalReward){
		externalRewardNow = externalReward;
		
		double evma = (externalRewardNow - externalRewardBefore) / maxReward;
		
		double internalReward = historyInfluence * evma + (1-historyInfluence) * internalRewardBefore;
		
		internalRewardBefore = internalReward;
		externalRewardBefore = externalRewardNow;
		
		return internalReward;
	}
}
