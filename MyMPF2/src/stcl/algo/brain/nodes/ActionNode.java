package stcl.algo.brain.nodes;

import java.util.ArrayList;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.NeoCorticalUnit;
import stcl.algo.poolers.SpatialPooler;
import stcl.algo.util.Normalizer;

public class ActionNode extends Node {
	private double explorationChance;
	private ArrayList<UnitNode> voters;
	private SpatialPooler pooler;
	private Sensor actionSensor;
	private int currentAction;
	private int nextActionID;
	private Random rand;
	private ArrayList<Double> influenceVector;
	private SimpleMatrix votes;
	private SimpleMatrix nextAction;

	public ActionNode(int id, double initialExplorationChance, Sensor actionSensor) {
		super(id);
		voters = new ArrayList<UnitNode>();
		influenceVector = new ArrayList<Double>();
		explorationChance = initialExplorationChance;
		this.actionSensor = actionSensor;
	}
	
	public void initialize(Random rand, int actionVectorLength, int actionGroupMapSize, double initialLearningRate){
		pooler = new SpatialPooler(rand, actionVectorLength, actionGroupMapSize, 0.1, Math.sqrt(actionGroupMapSize), 0.125); //TODO: Move all parameters out
		votes = new SimpleMatrix(actionGroupMapSize, actionGroupMapSize);
		this.rand = rand;
	}
	
	public void setExplorationChance(double explorationChance){
		this.explorationChance = explorationChance;
	}
	
	@Override
	public void feedback() {
		
		if (rand.nextDouble() < explorationChance){
			nextActionID = rand.nextInt(votes.getNumElements());
			//TODO: Implement better exploration policy
		} else {
			//Collect action votes
			int mostPopularAction = -1;
			double highestVote = Double.NEGATIVE_INFINITY;
			for (int i = 0;i < voters.size(); i++){
				UnitNode n = voters.get(i);
				NeoCorticalUnit unit = n.getUnit();
				if (unit.active() && !unit.needHelp()){
					int vote = unit.getActionVote();
					double influence = influenceVector.get(i);
					double currentVoteValue = votes.get(vote);
					double newValue = currentVoteValue + influence;
					votes.set(newValue);
					if (newValue > highestVote){
						highestVote = newValue;
						mostPopularAction = vote;
					}
				}				
			}			
			nextActionID = mostPopularAction;
		}		
		
		nextAction = pooler.getSOM().getNode(nextActionID).getVector();
		
	}

	@Override
	public void feedforward(double reward) {
		pooler.feedForward(actionSensor.getFeedforwardOutput());
		currentAction = pooler.getSOM().getBMU().getId();
		
		//TODO: Implement weighting of votes
		//Update weights of voters
			//Good votes --> higher influence of future votes
			//Bad votes --> lower influence of future votes
	}
	
	public void addVoter(UnitNode voter){
		voters.add(voter);
		influenceVector.add(1.0);
	}
	
	public int getNextActionID(){
		return nextActionID;
	}
	
	public SimpleMatrix getNextAction(){
		return nextAction;
	}

}
