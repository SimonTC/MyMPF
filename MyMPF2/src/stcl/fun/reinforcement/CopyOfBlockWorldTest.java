package stcl.fun.reinforcement;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;
import stcl.algo.brain.ActionDecider;

public class CopyOfBlockWorldTest {
	private enum ACTIONS {N,S,E,W};
	private Random rand = new Random();
	private SimpleMatrix world;
	private State goal;
	private ActionDecider agent;
	
	public static void main(String[] args){
		CopyOfBlockWorldTest bwt = new CopyOfBlockWorldTest(4);
		bwt.run(1000);
	}
	
	public CopyOfBlockWorldTest(int worldSize){
		setupWorld(worldSize);
		setupAgent(worldSize);
	}
	
	private void setupAgent(int worldSize){
		agent = new ActionDecider(4, worldSize * worldSize, 0.9, rand);
	}
	
	private void setupWorld(int worldSize){
		world = new SimpleMatrix(worldSize, worldSize);
		int goalID = rand.nextInt(worldSize * worldSize);
				
		goal = new State(goalID, world);
		System.out.println("End: (" + goal.row + "," + goal.col + ") ID: " + goal.id);
		world.set(0);
		world.set(goalID, 1);

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
	
	public void run(int numEpisodes){
		for (int i = 1; i <= numEpisodes; i++){
			agent.setLearningRate(0.1);
			agent.newEpisode();
			runEpisode(1 - (double) i / numEpisodes);
		}
		
		System.out.println("Q matrix: ");
		agent.printQMatrix();
		System.out.println();
		
		System.out.println("Policy map:");
		printPolicyMap();
		System.out.println();
		
		System.out.println("Reward map:");
		world.print();
	}
	
	private void printPolicyMap(){
		for (int row = 0; row < world.numRows(); row++){
			for (int col = 0; col < world.numCols(); col++){
				State s = new State(row, col, world);
				if (s.equals(goal)){
					System.out.print("*  ");
				} else {
					int bestAction = agent.feedBack(s.id);
					System.out.print(ACTIONS.values()[bestAction].name() + "  ");
				}
			}
			System.out.println();
		}
	}
	
	private void runEpisode(double explorationChance){
		agent.newEpisode();
		State state = selectRandomState(true);
		int actionID = chooseAction(state, explorationChance);
		agent.feedForward(state.id, actionID, 0); //Reward is unimportant in the first update. Update only used to save start state and action
		
		while(!isTerminalState(state)){
			State nextState = move(state, ACTIONS.values()[actionID]);
			double reward = world.get(nextState.row, nextState.col);
			if(reward > 0){
				System.out.println();
			}
			int nextActionID = chooseAction(state, explorationChance);
			
			agent.feedForward(nextState.id, nextActionID, reward);
			
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
	
	private boolean isTerminalState(State s){
		if (s.equals(goal)) return true;

		return false;
	}
	
	private State move(State state, ACTIONS action){
		int rowChange = 0, colChange = 0;
		switch(action){
		case E: rowChange = 0; colChange = 1;  break;
		case N: rowChange = 1; colChange = 0;  break;
		case S: rowChange = -1; colChange = 0;  break;
		case W: rowChange = 0; colChange = -1;  break;		
		}
		
		int newRow = state.getRow() + rowChange;
		int newCol = state.getCol() + colChange;
		
		if(newRow < 0 || newRow > world.numRows() - 1) newRow = state.getRow();
		if(newCol < 0 || newCol > world.numCols() - 1) newCol = state.getCol();
		
		State newState = new State(newRow, newCol, world);
		return newState;
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
		private CopyOfBlockWorldTest getOuterType() {
			return CopyOfBlockWorldTest.this;
		}
		
	}
	
	
}
