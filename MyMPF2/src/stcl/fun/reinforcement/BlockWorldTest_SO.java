package stcl.fun.reinforcement;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.ActionDecider;
/**
 * This test is based on the following Stack Overflow answer: http://stackoverflow.com/a/13153633
 * @author Simon
 *
 */
public class BlockWorldTest_SO {
	private enum ACTIONS {N,S,E,W};
	private Random rand = new Random(); //12345678
	private SimpleMatrix world;
	private State end;
	private ActionDecider agent;
	private int worldSize;
	private SimpleMatrix visitCounter;
	
	public static void main(String[] args){
		BlockWorldTest_SO bwt = new BlockWorldTest_SO(4);
		bwt.run(100);
	}
	
	public BlockWorldTest_SO(int worldSize){
		this.worldSize = worldSize;
		setupWorld(worldSize);
		setupAgent(worldSize);
	}
	
	private void setupAgent(int worldSize){
		agent = new ActionDecider(4, worldSize * worldSize, 0.9, rand);
	}
	
	private SimpleMatrix createPolicyMap(){
		SimpleMatrix qValues = agent.getQMatrix();
		SimpleMatrix policyMap = new SimpleMatrix(world);
		for (int state = 0; state < policyMap.getNumElements(); state++){
			SimpleMatrix actionValues = qValues.extractVector(false, state);
			double maxValue = Double.NEGATIVE_INFINITY;
			int bestAction = -1;
			for (int i = 0; i < actionValues.getNumElements(); i++){
				double d = actionValues.get(i);
				if (d > maxValue){
					maxValue = d;
					bestAction = i;
				}
				policyMap.set(state, bestAction);
			}
		}
		return policyMap;
	}
	
	private State findStartState(){
		int endID = end.getID();
		int startID = endID;		
		while (endID == startID){
			startID = rand.nextInt(worldSize * worldSize);
		};
		State start = new State(startID, world);
		return start;
		
	}
	
	private void setupWorld(int worldSize){
		world = new SimpleMatrix(worldSize, worldSize);
		visitCounter = new SimpleMatrix(world);
		visitCounter.set(1);
		int endID = 0; //Upper left corner
		end = new State(endID, world);
		world.set(1);
		world.set(endID, 10);

	}
	
	private int calculateMinVisits(){
		int minVisits = Integer.MAX_VALUE;
		for (double d : visitCounter.getMatrix().data){
			int i = (int) d;
			if (i < minVisits) minVisits = i;
		}
		return minVisits;
	}
	
	public void run(int numEpisodes){
		int minSteps = Integer.MAX_VALUE;
		int totalSteps = 0;
		boolean cont = true;
		SimpleMatrix oldMap = createPolicyMap();
		int noChange = 0;
		while(cont){
			agent.newEpisode();
			int steps = runEpisode(0.1);
			totalSteps += steps;
			SimpleMatrix map = createPolicyMap();
			printPolicyMap(map);
			if (map.isIdentical(oldMap, 0.001)){
				noChange++;
			} else {
				noChange = 0;
			}
			
			System.out.println("Steps: " + totalSteps + " Episodes without changes: " + noChange);
			
			if (noChange > 500){
				if (calculateMinVisits() > 30) cont = false;				
			}
			
			oldMap = map;
		}
		System.out.println("Finished after " + totalSteps + " steps");
		System.out.println("Final policy map:");
		printPolicyMap(oldMap);
		
		System.out.println();
		System.out.println("Visit counts:");
		visitCounter.print();
		
		System.out.println();
		System.out.println("Q matrix:");
		agent.printQMatrix();
		
		System.out.println();
		System.out.println("Trace matrix:");
		agent.printTraceMatrix();
	}
	
	private void printPolicyMap(SimpleMatrix policyMap){
		String[][] map = new String[policyMap.numRows()][policyMap.numCols()];
		for (int row = 0; row < map.length; row++){
			for (int col = 0; col < map[row].length; col++){
				int action = (int) policyMap.get(row, col);
				System.out.print(ACTIONS.values()[action].name() + "  ");
			}
			System.out.println();
		}
	}
	
	private int runEpisode(double explorationChance){
		boolean goalFound = false;
		ACTIONS action = ACTIONS.N;
		
		State state = findStartState();
		double reward = 0;
		int counter = 0;
		//System.out.println("Starts in: (" + state.getCol() + ", " + state.getRow() + ")");
		while (!goalFound){
			int visits = (int) visitCounter.get(state.id);
			double learningRate = 1.0 / (double) visits;
			agent.setLearningRate(learningRate);
			agent.feedForward(state.getID(), action.ordinal(), reward);
			SimpleMatrix stateProbs = new SimpleMatrix(worldSize, worldSize);
			State nextState = move(state, action);
			visits++;
			visitCounter.set(state.id, visits);
			state = nextState;
			//System.out.println("Moved to: (" + state.getCol() + ", " + state.getRow() + ")");
			if(state.equals(end)) goalFound = true;
			reward = world.get(state.id);
			stateProbs.set(state.getID(), 1);
			int nextAction = agent.feedback(stateProbs);
			if (rand.nextDouble() < explorationChance){
				nextAction = rand.nextInt(ACTIONS.values().length);
			}
			action = ACTIONS.values()[nextAction];	
			counter++;
		}
		agent.feedForward(state.getID(), action.ordinal(), reward); //Need to give it the reward for finishing
		int visits = (int) visitCounter.get(state.id);
		visits++;
		visitCounter.set(state.id, visits);
		
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
		private BlockWorldTest_SO getOuterType() {
			return BlockWorldTest_SO.this;
		}
		
	}
	
	
}
