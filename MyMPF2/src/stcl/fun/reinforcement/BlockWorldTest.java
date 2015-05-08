package stcl.fun.reinforcement;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.ActionDecider;


public class BlockWorldTest {
	private SimpleMatrix world;
	private enum ACTIONS {N,S,E,W};
	private final double GOAL_REWARD = 1;
	private final double HOLE_REWARD = -1;
	private State goal;
	private State hole;
	private Random rand = new Random();
	private ActionDecider agent;
	
	public static void main(String[] args){
		BlockWorldTest bwt = new BlockWorldTest();
		bwt.setup(4);
		bwt.run(1000);
	}
	
	public void run(int numEpisodes){
		for (int i = 1; i <= numEpisodes; i++){
			agent.setLearningRate(0.1);
			agent.newEpisode();
			runEpisode(agent, 1 - (double) i / numEpisodes);
		}
		
		System.out.println("Q matrix: ");
		agent.printQMatrix();
		System.out.println();
		
		System.out.println("Policy map:");
		printPolicyMap(agent);
		System.out.println();
		
		System.out.println("Reward map:");
		world.print();
	}
	
	public void setup(int worldSize){
		world = new SimpleMatrix(worldSize, worldSize);
		goal = selectRandomState(true);
		//hole = selectRandomState(false);
		world.set(goal.row, goal.col, GOAL_REWARD);
		//world.set(hole.row, hole.col, HOLE_REWARD);
		agent = new ActionDecider(4, worldSize * worldSize, 0.9, rand);
	}
	
	public void runEpisode(ActionDecider agent, double explorationChance){
		agent.newEpisode();
		State state = selectRandomState(true);
		int actionID = chooseAction(state, explorationChance);
		agent.feedForward(state.id, actionID, 0);//Reward not important for this first feed forward. Only used to save initial state and action
				
		while(!isTerminalState(state)){
			State nextState = move(state, ACTIONS.values()[actionID]);
			double reward = world.get(nextState.row, nextState.col);
			int nextActionID = chooseAction(nextState, explorationChance);
			agent.feedForward(nextState.id, nextActionID, reward);
			//agent.updateQMatrix(state.id, actionID, nextState.id, nextActionID, reward);
			state = nextState;		
			actionID = nextActionID;
		}
	}
	
	private int chooseAction(State state, double explorationChance){
		int actionID;
		if (rand.nextDouble() < explorationChance){
			actionID = rand.nextInt(ACTIONS.values().length);
		} else {
			actionID = agent.feedBack(state.id);
		}
		return actionID;
	}
	
	public void printPolicyMap(ActionDecider agent){
		for (int row = 0; row < world.numRows(); row++){
			for (int col = 0; col < world.numCols(); col++){
				State s = new State(row, col, world);
				if (s.equals(goal)){
					System.out.print("*  ");
				} else if (s.equals(hole)){
					System.out.print("/  ");
				} else {
					int bestAction = agent.feedBack(s.id);
					System.out.print(ACTIONS.values()[bestAction].name() + "  ");
				}
			}
			System.out.println();
		}
	}

	private State move(State state, ACTIONS action){
		int rowChange = 0, colChange = 0;
		switch(action){
		case E: rowChange = 0; colChange = 1;  break;
		case N: rowChange = -1; colChange = 0;  break;
		case S: rowChange = 1; colChange = 0;  break;
		case W: rowChange = 0; colChange = -1;  break;		
		}
		
		int newRow = state.getRow() + rowChange;
		int newCol = state.getCol() + colChange;
		
		if(newRow < 0 || newRow > world.numRows() - 1) newRow = state.getRow();
		if(newCol < 0 || newCol > world.numCols() - 1) newCol = state.getCol();
		
		State newState = new State(newRow, newCol, world);
		return newState;
	}
	
	private State selectRandomState(boolean includeGoalState){
		boolean stateFound = false;
		State s = null;
		int id = -1;
		while(!stateFound){
			id = rand.nextInt(world.getNumElements());
			if (includeGoalState){
				stateFound = true;
			} else {
				if (id != goal.id) stateFound = true;
			}
		}
		s = new State(id, world);
		return s;
	}
	
	private boolean isTerminalState(State s){
		if (s.equals(goal)) return true;
		if (s.equals(hole)) return true;
		return false;
	}
	
	public int getNumActions(){
		return ACTIONS.values().length;
	}
	
	public int getNumStates(){
		return world.getNumElements();
	}
	
	private class State{
		private int row, col, id;
		public State(int row, int col, SimpleMatrix world){
			this.row = row;
			this.col = col;
			this.id = world.getIndex(row, col);
		}
		public State(int id, SimpleMatrix world){
			int numCols = world.numCols();
			row = (int) Math.floor(id / (double)numCols);
			col = id - row * numCols;
			this.id = id;
			
		}
		public int getRow(){
			return row;
		}
		public int getCol(){
			return col;
		}
		public int getID(){
			return id;
		}
		@Override
		public int hashCode() {
			final int prime = 31;
			int result = 1;
			result = prime * result + getOuterType().hashCode();
			result = prime * result + col;
			result = prime * result + id;
			result = prime * result + row;
			return result;
		}
		@Override
		public boolean equals(Object obj) {
			if (this == obj)
				return true;
			if (obj == null)
				return false;
			if (getClass() != obj.getClass())
				return false;
			State other = (State) obj;
			if (!getOuterType().equals(other.getOuterType()))
				return false;
			if (col != other.col)
				return false;
			if (id != other.id)
				return false;
			if (row != other.row)
				return false;
			return true;
		}
		private BlockWorldTest getOuterType() {
			return BlockWorldTest.this;
		}
		
	}
	
	
}
