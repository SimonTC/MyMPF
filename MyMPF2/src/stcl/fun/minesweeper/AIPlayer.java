package stcl.fun.minesweeper;

import java.util.LinkedList;
import java.util.Queue;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

public class AIPlayer extends Player {
	private Brain brain;
	private Random rand;
	private double reward;
	
	private Queue<int[]> actionQueue;

	public AIPlayer(String name, PlayerType type, Brain brain, Random rand) {
		super(name, type);
		this.brain = brain;
		this.rand = rand;
		reward = 0;
		actionQueue = new LinkedList<int[]>();
	}

	@Override
	public int[] getMove(int[] board) {
		//TODO: Normalize input??
		int boardSize = (int) Math.sqrt(board.length);
		while (actionQueue.size() < 3){
			actionQueue.add(randomActions(boardSize));
		}
		
		int[] curActions = actionQueue.poll();
		int[] nextActions = actionQueue.peek();
		
		SimpleMatrix input = getInputVector(board, nextActions);
		
		SimpleMatrix output = brain.activate(input, reward);
		
		actionQueue.add(convertOutpuIntoActions(output));
		
		return curActions;
	}
	
	public void giveReward(double reward){
		this.reward = reward;
	}
	
	private int[] convertOutpuIntoActions(SimpleMatrix output){
		int lastID = output.getNumElements() - 1;
		
		int y = (int) output.get(lastID);
		int x = (int) output.get(lastID - 1);
		
		int[] actions = {x,y};
		
		return actions;
	}
	
	private int[] randomActions(int boardSize){
		//Select x
		int x = rand.nextInt(boardSize);
		int y = rand.nextInt(boardSize);
		
		int[] actions = {x,y};
		return actions;
	}
	
	/**
	 * 
	 * @param board
	 * @param nextActions contains the actions that will be performed at t + 1
	 * @return
	 */
	private SimpleMatrix getInputVector(int[] board, int[] nextActions){
		//Convert array to double
		double[] input = new double[board.length + 2];
		
		for (int i = 0; i < board.length; i++){
			input[i] = board[i];
		}
		
		input[board.length] = nextActions[0];
		input[board.length + 1] = nextActions[1];
		
		double[][] inputVector = {input};
		
		SimpleMatrix m = new SimpleMatrix(inputVector);
		
		return m;
	}
	
	public void makeReadyForNewGame(){
		brain.newIteration();
	}

}
