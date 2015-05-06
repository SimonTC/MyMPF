package stcl.fun.reinforcement;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.ActionDecider;
import stcl.algo.reinforcement.QLearner;
/**
 * This test is based on the following Stack Overflow answer: http://stackoverflow.com/a/13153633
 * @author Simon
 *
 */
public class BlockWorldTest_SO_Qlearner {
	private enum ACTIONS {N,S,E,W};
	private Random rand = new Random(1234); //12345678
	private SimpleMatrix world;
	private State end;
	private QLearner agent;
	private int worldSize;
	private SimpleMatrix visitCounter;
	
	public static void main(String[] args){
		BlockWorldTest_SO_Qlearner bwt = new BlockWorldTest_SO_Qlearner(4);
		bwt.run(100);
	}
	
	public BlockWorldTest_SO_Qlearner(int worldSize){
		this.worldSize = worldSize;
		setupWorld(worldSize);
		setupAgent(worldSize);
	}
	
	private void setupAgent(int worldSize){
		agent = new QLearner(4, worldSize * worldSize, 0.9, rand);
		visitCounter = new SimpleMatrix(agent.getQMatrix());
		visitCounter.set(1);
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
		int endID = 0; //Upper left corner
		end = new State(endID, world);
		world.set(-0.1);
		world.set(endID, 1);

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
		SimpleMatrix orgQ = new SimpleMatrix(agent.getQMatrix());
		int totalSteps = 0;
		boolean cont = true;
		SimpleMatrix oldMap = createPolicyMap();
		int noChange = 0;
		double explorationChance = 0.5;
		State state = findStartState();
		boolean goalFound = false;
		while(cont){
			if (totalSteps % 1000 == 0) System.out.println("Step " + totalSteps);
			if (goalFound){
				state = findStartState();
				goalFound = false;
			}

			SimpleMatrix stateProbs = new SimpleMatrix(worldSize, worldSize);
			stateProbs.set(state.getID(), 1);
			int nextAction = agent.chooseBestAction(stateProbs);
			if (rand.nextDouble() < explorationChance){
				nextAction = rand.nextInt(ACTIONS.values().length);
			}
			int visits = (int) visitCounter.get(nextAction, state.id);
			visits++;
			visitCounter.set(nextAction, state.id, visits);
			ACTIONS action = ACTIONS.values()[nextAction];	
			State nextState = move(state, action);
			
			double learningRate = 0.1;//1.0 / (double) visits;
			agent.setLearningRate(learningRate);
			state = nextState;
			//System.out.println("Moved to: (" + state.getCol() + ", " + state.getRow() + ")");
			if(state.equals(end)){
				goalFound = true;
				for (int i = 0; i < 4; i++){
					visits = (int) visitCounter.get(i, state.id);
					visits++;
					visitCounter.set(i, state.id, visits);
					agent.getQMatrix().set(i, state.id, 1);
				}
				
			}
			double reward = world.get(state.id);
			agent.updateQMatrix(state.id, nextAction, 0.9, learningRate, reward);
			
			SimpleMatrix map = createPolicyMap();
		//	printPolicyMap(map);
			if (map.isIdentical(oldMap, 0.001)){
				noChange++;
			} else {
				noChange = 0;
			}
			
			//System.out.println("Steps: " + totalSteps + " Episodes without changes: " + noChange);
			
			if (noChange > 10000){
				if (calculateMinVisits() > 30) cont = false;				
			}
			if (totalSteps > 1200000) cont = false;
			
			totalSteps++;
			oldMap = map;
		}
		System.out.println("Finished after " + totalSteps + " steps");
		System.out.println("Final policy map:");
		printPolicyMap(oldMap);
		
		System.out.println();
		System.out.println("Visit counts:");
		visitCounter.print();
		
		System.out.println();
		System.out.println("Q matrix now:");
		agent.printQMatrix();
		
		System.out.println();
		System.out.println("Q matrix start:");
		orgQ.print();
	}
	
	private void printPolicyMap(SimpleMatrix policyMap){
		String[][] map = new String[policyMap.numRows()][policyMap.numCols()];
		for (int row = 0; row < map.length; row++){
			for (int col = 0; col < map[row].length; col++){
				if(row == end.row && col == end.col){
					System.out.print("*  ");
				} else {
					int action = (int) policyMap.get(row, col);
					System.out.print(ACTIONS.values()[action].name() + "  ");
				}
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
			SimpleMatrix stateProbs = new SimpleMatrix(worldSize, worldSize);
			stateProbs.set(state.getID(), 1);
			int nextAction = agent.chooseBestAction(stateProbs);
			if (rand.nextDouble() < explorationChance){
				nextAction = rand.nextInt(ACTIONS.values().length);
			}
			action = ACTIONS.values()[nextAction];	
			State nextState = move(state, action);
			visits++;
			visitCounter.set(state.id, visits);
			state = nextState;
			//System.out.println("Moved to: (" + state.getCol() + ", " + state.getRow() + ")");
			if(state.equals(end)) goalFound = true;
			reward = world.get(state.id);
			agent.updateQMatrix(state.id, nextAction, 0.9, learningRate, reward);
			
			counter++;
		}		
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
		
		public String toString(){
			return "" + id;
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
		private BlockWorldTest_SO_Qlearner getOuterType() {
			return BlockWorldTest_SO_Qlearner.this;
		}
		
	}
	
	
}
