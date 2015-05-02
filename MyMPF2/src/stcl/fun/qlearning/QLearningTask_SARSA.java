package stcl.fun.qlearning;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.reinforcement.QFunction;
import stcl.algo.reinforcement.SARSA;

public class QLearningTask_SARSA {
	
	private SARSA sarsa;
	private Random rand = new Random(1234);
	private SimpleMatrix rewardMatrix, actionMatrix;
	private int numEpisodes = 100000;

	public static void main(String[] args) {
		QLearningTask_SARSA qt = new QLearningTask_SARSA();
		qt.setup();
		qt.run();

	}
	
	public void setup(){
		sarsa = new SARSA();
		actionMatrix = createActionMatrix();
		sarsa.initialize(4, 2, rand, actionMatrix);
		rewardMatrix = createRewardMatrix();
	}
	
	private SimpleMatrix createActionMatrix(){
		double[][] data = {
				{1,0,0,0},
				{0,1,0,0},
				{0,0,1,0},
				{0,0,0,1},
				{0,0,0,0}
		};
		SimpleMatrix m = new SimpleMatrix(data);
		return m;
	}
	
	private SimpleMatrix createRewardMatrix(){
		double[][] data = {
				{0,0,0,1},
				{0,0,0,0},
				{0,0,0,0},
				{0,0,0,0}
		};
		SimpleMatrix m = new SimpleMatrix(data);
		return m;
	}
	
	public void run(){
		
		for (int i = 0; i < numEpisodes; i++){
			int result = runEpisode(i);
			System.out.println("Finished episode " + i + " Result: " + result);
		}
		
		System.out.println();
	}
	
	public int runEpisode(int temperature){
		double[][] posData = {{2,0}}; //{Row, col}
		SimpleMatrix position = new SimpleMatrix(posData);
		boolean goalFound = false;
		boolean dead = false;
		int maxSteps = 20;
		int step = 0;
		int action = 4;
		double reward = 0;
		boolean stop = false;
		sarsa.newEpisode();
		int nextAction = sarsa.feedForward(position, -1, reward);
		double threshold = 1 * Math.exp(-temperature / numEpisodes);
		if (rand.nextDouble() < threshold) nextAction = rand.nextInt(5);
		action = nextAction;
		while(!goalFound && !dead && !stop && step < maxSteps){
			//System.out.println("Position: " + position.get(0) + ", " + position.get(1));
			SimpleMatrix actionToPerform = actionMatrix.extractVector(true, action);
			
			reward = rewardMatrix.get((int) position.get(0), (int) position.get(1));
			if ((int)position.get(0) == 0 && (int)position.get(1) == 3){
				reward = 0;
				goalFound = true;
			}
						
			if (!goalFound){			
				double rowChange = actionToPerform.get(0) - actionToPerform.get(1);
				double colChange = actionToPerform.get(2) - actionToPerform.get(3);
				double oldRow = position.get(0);
				double oldCol = position.get(1);
				double newRow = oldRow + rowChange;
				double newCol = oldCol + colChange;
				
				if (newRow > 3){
					newRow = 3;
					reward = -1;
					//dead = true;
					//stop = true;
				}
				if (newRow < 0){
					newRow = 0;
					reward = -1;
					//dead = true;
					//stop = true;

				}
				if (newCol > 3){
					newCol = 3;
					reward = -1;
					//dead = true;
					//stop = true;

				}
				if (newCol < 0){
					newCol = 0;
					reward = -1;
					//dead = true;
					//stop = true;

				}
				
				position.set(0, newRow);
				position.set(1, newCol);
				step++;
			}
			nextAction = 0;
			nextAction = sarsa.feedForward(position, action, reward);
			//nextAction = qfunction.chooseNextAction(position, 10 + temperature);
			threshold = 1 * Math.exp(-temperature / numEpisodes);
			if (rand.nextDouble() < threshold) nextAction = rand.nextInt(5);
			action = nextAction;
		}
		int endResult = 0;
		if (goalFound) endResult = 1;
		if (dead) endResult = -1;
		
		

		
		return endResult;
	}

}
