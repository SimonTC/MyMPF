package stcl.fun.reinforcement;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.ActionDecider;

public class BlockWorldTest {
	private enum ACTIONS {N,S,E,W};
	private Random rand = new Random();
	private SimpleMatrix world;
	private State start, end;
	private ActionDecider agent;
	private int worldSize;
	
	public static void main(String[] args){
		BlockWorldTest bwt = new BlockWorldTest(4);
		bwt.run(100);
	}
	
	public BlockWorldTest(int worldSize){
		this.worldSize = worldSize;
		setupWorld(worldSize);
		setupAgent(worldSize);
	}
	
	private void setupAgent(int worldSize){
		agent = new ActionDecider(4, worldSize * worldSize, 0.9, rand);
	}
	
	private void setupWorld(int worldSize){
		world = new SimpleMatrix(worldSize, worldSize);
		int startID = rand.nextInt(worldSize * worldSize);
		int endID = rand.nextInt(worldSize * worldSize);
		while (endID == startID){
			endID = rand.nextInt(worldSize * worldSize);
		};
		
		start = new State(startID, world);
		end = new State(endID, world);
		System.out.println("Start: (" + start.col + "," + start.row + ") ID: " + start.id);
		System.out.println("End: (" + end.col + "," + end.row + ") ID: " + end.id);
		world.set(0);
		world.set(endID, 1);

	}
	
	public void run(int numEpisodes){
		int minSteps = Integer.MAX_VALUE;
		for (int i = 1; i <= numEpisodes; i++){
			agent.setLearningRate(0.1);
			agent.newEpisode();
			int steps = runEpisode(1 - (double) i / numEpisodes);
			if (steps < minSteps) minSteps = steps;
			System.out.println("Finished episode " + i + " in " + steps + " steps");	
			/*
			System.out.println();
			System.out.println("Q matrix: ");
			agent.printQMatrix();
			System.out.println();
			
			System.out.println("Trace matrix:");
			agent.printTraceMatrix();
			*/
		}
		
		//agent.printQMatrix();
		
		//Evaluation
		agent.setLearningRate(0);
		agent.newEpisode();
		int steps = runEpisode(0);
		System.out.println("Finished evaluation in " + steps + " steps");
		System.out.println("Minimum number of steps used: " + minSteps);
		
		System.out.println("Q matrix: ");
		agent.printQMatrix();
		System.out.println();
		
		System.out.println("Trace matrix:");
		agent.printTraceMatrix();
	}
	
	private int runEpisode(double explorationChance){
		boolean goalFound = false;
		ACTIONS action = ACTIONS.N;
		State state = start;
		double reward = 0;
		int counter = 0;
		//System.out.println("Starts in: (" + state.getCol() + ", " + state.getRow() + ")");
		while (!goalFound && counter < 1000){
			agent.feedForward(state.getID(), action.ordinal(), reward);
			SimpleMatrix stateProbs = new SimpleMatrix(worldSize, worldSize);
			State nextState = move(state, action);
			state = nextState;
			//System.out.println("Moved to: (" + state.getCol() + ", " + state.getRow() + ")");
			if(state.equals(end)) goalFound = true;
			reward = world.get(state.id);
			stateProbs.set(state.getID(), 1);
			int nextAction = agent.feedback(state.getID());
			if (rand.nextDouble() < explorationChance){
				nextAction = rand.nextInt(ACTIONS.values().length);
			}
			action = ACTIONS.values()[nextAction];	
			counter++;
		}
		agent.feedForward(state.getID(), action.ordinal(), reward); //Need to give it the reward for finishing
		
		return counter;
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
		private BlockWorldTest getOuterType() {
			return BlockWorldTest.this;
		}
		
	}
	
	
}
