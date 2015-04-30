package stcl.fun.qlearning;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.reinforcement.QFunction;

public class QLearningTask {
	
	private QFunction qfunction;
	private Random rand = new Random(1234);
	private SimpleMatrix rewardMatrix, actionMatrix;

	public static void main(String[] args) {
		QLearningTask qt = new QLearningTask();
		qt.setup();
		qt.run();

	}
	
	public void setup(){
		qfunction = new QFunction();
		actionMatrix = createActionMatrix();
		qfunction.initialize(4, 2, rand, actionMatrix);
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
				{0,0,0,10},
				{0,0,0,0},
				{0,0,0,0},
				{0,0,0,0}
		};
		SimpleMatrix m = new SimpleMatrix(data);
		return m;
	}
	
	public void run(){
		int numEpisodes = 200000;
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
		int maxSteps = 100;
		int step = 0;
		int action = 4;
		double reward = 0;
		while(!goalFound && !dead && step < maxSteps){
			qfunction.feedForward(position, action, reward);
			SimpleMatrix actionToPerform = actionMatrix.extractVector(true, action);
			
			reward = rewardMatrix.get((int) position.get(0), (int) position.get(1));
			if (reward >= 9.99 && reward <= 10.01){
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
					dead = true;
				}
				if (newRow < 0){
					newRow = 0;
					dead = true;
				}
				if (newCol > 3){
					newCol = 3;
					dead = true;
				}
				if (newCol < 0){
					newCol = 0;
					dead = true;
				}
				
				position.set(0, newRow);
				position.set(1, newCol);
				int nextAction = 0;
				//nextAction = qfunction.feedback(position);
				nextAction = qfunction.chooseNextAction(position, 10 + temperature);
				//if (rand.nextDouble() < 0.2) nextAction = rand.nextInt(5);
				action = nextAction;
				step++;
			}
		}
		int endResult = 0;
		if (goalFound) endResult = 1;
		if (dead) endResult = -1;
		
		qfunction.newEpisodedouble(endResult);
		
		return endResult;
	}

}
