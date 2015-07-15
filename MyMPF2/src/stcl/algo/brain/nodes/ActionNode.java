package stcl.algo.brain.nodes;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Random;
import java.util.TreeMap;

import org.ejml.simple.SimpleMatrix;

import dk.stcl.core.basic.containers.SomNode;
import stcl.algo.brain.NeoCorticalUnit;
import stcl.algo.poolers.SpatialPooler;
import stcl.algo.util.Normalizer;

public class ActionNode extends Node {

	private static final long serialVersionUID = 1L;
	private double explorationChance;
	private ArrayList<UnitNode> voters;
	private SpatialPooler pooler;
	private Sensor actionSensor;
	private int currentAction;
	private int nextActionID;
	private Random rand;
	private TreeMap<Integer, Integer> givenVotes;
	private TreeMap<Integer, Double> influenceMap;
	private SimpleMatrix votesForActions;
	private SimpleMatrix nextAction;
	private double rewardInfluence;
	private boolean learningActions;
	private String initializationDescription;
	private boolean updateVoterInfluence;
	private String poolerInitializationString;
	private int lastExplorativeAction;

	public ActionNode(int id) {
		super(id, -1, -1, -1);
		voters = new ArrayList<UnitNode>();
		influenceMap = new TreeMap<Integer, Double>();
		givenVotes = new TreeMap<Integer, Integer>();
		this.rewardInfluence = 0.01; //TODO: Make a parameter
		this.type = NodeType.ACTION;
		learningActions = true;
		updateVoterInfluence = true;
	}
	
	/**
	 * Creates and initializes an Action node based on the given string.
	 * The string is created by the toString() method.
	 * Remember to set the sensor afterwards;
	 * @param s
	 * @param rand
	 */
	public ActionNode(String s, Random rand){
		super(s);
		voters = new ArrayList<UnitNode>();
		influenceMap = new TreeMap<Integer, Double>();
		givenVotes = new TreeMap<Integer, Integer>();
		this.rewardInfluence = 0.01; //TODO: Make a parameter
		learningActions = true;
		updateVoterInfluence = true;
		String[] arr = s.split(" ");
		int length = arr.length;
		this.initialize(rand, Integer.parseInt(arr[length-3]), Integer.parseInt(arr[length-2]), Double.parseDouble(arr[length - 1]), Double.parseDouble(arr[length-4]));
	}
	
	public void initialize(Random rand, int actionVectorLength, int actionGroupMapSize, double initialLearningRate, double initialExplorationChance){
		pooler = new SpatialPooler(rand, actionVectorLength, actionGroupMapSize, initialLearningRate, Math.sqrt(actionGroupMapSize), 0.125); //TODO: Move all parameters out
		votesForActions = new SimpleMatrix(actionGroupMapSize, actionGroupMapSize);
		this.rand = rand;
		initializationDescription = initialExplorationChance +  " " + actionVectorLength + " " + actionGroupMapSize + " " + initialLearningRate;
		explorationChance = initialExplorationChance;
		poolerInitializationString = pooler.toInitializationString();
		lastExplorativeAction = -1;
	}
	
	public void setExplorationChance(double explorationChance){
		this.explorationChance = explorationChance;
	}
	
	@Override
	public void addChild(Node child){
		super.addChild(child);
		this.actionSensor = (Sensor) child;
	}
	
	@Override
	public void feedback() {
		
		if(rand.nextDouble() < explorationChance){
			nextActionID = doExploration();
		} else {
		 
			//Collect action votes
			int mostPopularAction = -1;
			double highestVote = Double.NEGATIVE_INFINITY;
			votesForActions.set(0);
			for (int i = 0; i < voters.size(); i++){
				UnitNode n = voters.get(i);
				int voterID = n.getID();
				NeoCorticalUnit unit = n.getUnit();
				givenVotes.put(voterID, -1);
				boolean mayVote = unit.active() && !unit.needHelp();// && !unit.needHelp();
				if (mayVote){
					int vote = unit.getNextAction();
					givenVotes.put(n.getID(), vote);
					double voteInfluence = influenceMap.get(voterID);
					double currentVoteValue = votesForActions.get(vote);
					double newValue = currentVoteValue + voteInfluence;
					votesForActions.set(vote, newValue);
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
		//int nextAction = rand.nextInt(votesForActions.getNumElements());
		int nextAction = ++lastExplorativeAction;
		if (nextAction == votesForActions.getNumElements()) nextAction = 0;
		lastExplorativeAction = nextAction;
		return nextAction;
	}

	@Override
	public void feedforward(double reward, int actionPerformed) {
		pooler.feedForward(actionSensor.getFeedforwardOutput());
		currentAction = pooler.getSOM().getBMU().getId();
		
		
		//TODO: Implement weighting of votes
		//Update weights of voters
		//If you voted for the action that was performed your influenced is changed based on how good the outcome was
		if (updateVoterInfluence){
			for (UnitNode voter : voters){
				int voterID = voter.getID();
				int vote = givenVotes.get(voterID);
				if (vote != -1){
					//Voter did vote in last election
					if (vote == currentAction){
						double oldInfluence = influenceMap.get(voterID);
						double newInfluence = oldInfluence + reward * rewardInfluence;
						if (newInfluence < 0) newInfluence = 0;
						if (newInfluence > 1) newInfluence = 1;
						influenceMap.put(voterID, newInfluence);
					}
				}
			}
		}
	}
	
	public void addVoter(UnitNode voter){
		voters.add(voter);
		int voterID = voter.getID();
		influenceMap.put(voterID, 1.0);
		givenVotes.put(voterID, -1);
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
	
	public void printVoterInfluence(){
		System.out.println(influenceMap.toString());
	}
	
	public void printActionModels(){
		pooler.printModelWeigths();
	}
	
	public TreeMap<Integer, Double> getInfluenceMap(){
		return influenceMap;
	}
	
	public void setInfluenceMap(TreeMap<Integer, Double> influenceMap){
		this.influenceMap = influenceMap;
	}
	
	/**
	 * Use this method if you want to give the network a set of specific actions it can perform without having to learn them first.
	 * @param actions
	 */
	public void setPossibleActions(ArrayList<SimpleMatrix> actions){
		if (learningActions){ //We don't want to go through this if it has already been done before
			SomNode[] poolerNodes = pooler.getSOM().getNodes();
			assert poolerNodes.length == actions.size() : "Number of given actions is different from the size of the action pooler map";
			assert poolerNodes[0].getVector().getNumElements() == actions.get(0).getNumElements() : "The given actions are of different length than the vectors in the action pooler map";

			for (int i = 0; i < poolerNodes.length; i++){
				SomNode n = poolerNodes[i];
				n.setVector(actions.get(i));
			}
			pooler.setLearning(false);
			learningActions = false;
		}
		
	}
	
	public void setUpdateVoterInfluence(boolean flag){
		this.updateVoterInfluence = flag;
	}
	
	@Override
	public String toString(){
		String s = super.toString();
		s += " " + initializationDescription;
		return s;
	}

	@Override
	public void reinitialize() {
		givenVotes = new TreeMap<Integer, Integer>();
		for (Integer i : influenceMap.keySet()){
			influenceMap.put(i, 1.0);
			givenVotes.put(i, -1);
		}
		learningActions = true;
		updateVoterInfluence = true;
		votesForActions.set(0);
		String[] arr = initializationDescription.split(" ");
		int length = arr.length;
		String orgPoolerInitializationString = poolerInitializationString; //Save string here as it is changed in initialization
		this.initialize(rand, Integer.parseInt(arr[length-3]), Integer.parseInt(arr[length-2]), Double.parseDouble(arr[length - 1]), Double.parseDouble(arr[length-4]));
		poolerInitializationString = orgPoolerInitializationString;
		pooler = new SpatialPooler(poolerInitializationString, 0, rand);
	}
	
	public SpatialPooler getPooler(){
		return pooler;
	}
}
