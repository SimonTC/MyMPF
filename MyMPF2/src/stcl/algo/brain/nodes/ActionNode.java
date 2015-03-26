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
	private ArrayList<Integer>givenVotes;
	private ArrayList<Double> influenceVector;
	private SimpleMatrix votesForActions;
	private SimpleMatrix nextAction;
	private double rewardInfluence;

	public ActionNode(int id, double initialExplorationChance, Sensor actionSensor) {
		super(id);
		voters = new ArrayList<UnitNode>();
		influenceVector = new ArrayList<Double>();
		givenVotes = new ArrayList<Integer>();
		explorationChance = initialExplorationChance;
		this.actionSensor = actionSensor;
		this.rewardInfluence = 0.2; //TODO: Make a parameter
	}
	
	public void initialize(Random rand, int actionVectorLength, int actionGroupMapSize, double initialLearningRate){
		pooler = new SpatialPooler(rand, actionVectorLength, actionGroupMapSize, 0.1, Math.sqrt(actionGroupMapSize), 0.125); //TODO: Move all parameters out
		votesForActions = new SimpleMatrix(actionGroupMapSize, actionGroupMapSize);
		this.rand = rand;
	}
	
	public void setExplorationChance(double explorationChance){
		this.explorationChance = explorationChance;
	}
	
	@Override
	public void feedback() {
		
		if (rand.nextDouble() < explorationChance){
			nextActionID = doExploration();
		} else {
			//Collect action votes
			int mostPopularAction = -1;
			double highestVote = Double.NEGATIVE_INFINITY;
			for (int i = 0; i < voters.size(); i++){
				UnitNode n = voters.get(i);
				NeoCorticalUnit unit = n.getUnit();
				givenVotes.set(i, -1);
				boolean mayVote = unit.active();// && !unit.needHelp();
				if (mayVote){
					int vote = unit.getNextAction();
					givenVotes.set(i, vote);
					double voteInfluence = influenceVector.get(i);
					double currentVoteValue = votesForActions.get(vote);
					double newValue = currentVoteValue + voteInfluence;
					votesForActions.set(newValue);
					if (newValue > highestVote){
						highestVote = newValue;
						mostPopularAction = vote;
					}
				}				
			}
			if (mostPopularAction < 0){
				nextActionID = doExploration(); //Might happen if all nodes needs help
			} else {
				nextActionID = mostPopularAction;
			}
		}		
		
		nextAction = pooler.getSOM().getNode(nextActionID).getVector();
		feedbackOutput = nextAction;
		
	}
	
	private int doExploration(){
		//TODO: Implement better exploration policy
		int nextAction = rand.nextInt(votesForActions.getNumElements());
		return nextAction;
	}

	@Override
	public void feedforward(double reward, int actionPerformed) {
		pooler.feedForward(actionSensor.getFeedforwardOutput());
		currentAction = pooler.getSOM().getBMU().getId();
		
		
		//TODO: Implement weighting of votes
		//Update weights of voters
		//If you voted for the action that was performed your influenced is changed based on how good the outcome was
		for (int voter = 0; voter < voters.size(); voter++){
			int vote = givenVotes.get(voter);
			if (vote != -1){
				//Voter did vote in last election
				if (vote == actionPerformed){
					double oldInfluence = influenceVector.get(voter);
					double newInfluence = oldInfluence + reward * rewardInfluence;
					if (newInfluence < 0) newInfluence = 0;
					influenceVector.set(voter, newInfluence);
				}
			}
		}
		
			//Good votes --> higher influence of future votes
			//Bad votes --> lower influence of future votes
	}
	
	public void addVoter(UnitNode voter){
		voters.add(voter);
		influenceVector.add(1.0);
		givenVotes.add(-1);
	}
	
	public int getNextActionID(){
		return nextActionID;
	}
	
	public SimpleMatrix getNextAction(){
		return nextAction;
	}
	
	public int getCurrentAction(){
		return currentAction;
	}
	
	public void printSomModels(){
		pooler.printModelWeigths();
	}

}
