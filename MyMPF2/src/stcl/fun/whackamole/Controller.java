package stcl.fun.whackamole;

import java.util.Random;

import stcl.fun.whackamole.players.Player;
import stcl.fun.whackamole.players.ReactiveHTM;

public class Controller {
	
	private Player player;
	private Model model;
	private Random rand = new Random(1234);
	
	public static void main(String[] args) {
		Controller c = new Controller();
		c.setup();
		c.run(100);
	}
	
	public void run(int numRounds){
		for (int i = 0; i < numRounds; i++){
			runRound(player, model, i);
			//player.getBrain().getUnitNodes().get(0).getUnit().getDecider().printQMatrix();
			//System.out.println();
			//player.getBrain().getActionNode().printActionModels();
		}
		
		System.out.println("Q Matrix");
		player.getBrain().getUnitNodes().get(0).getUnit().getDecider().printQMatrix();
		System.out.println();
		System.out.println("Model weights:");
		player.getBrain().getUnitNodes().get(0).getUnit().getSpatialPooler().printModelWeigths();
	}
	
	public void setup(){
		model = new Model();
		int worldSize = 2;
		model.initialize(worldSize, 3, 2, 5, 1, rand);
		player = new ReactiveHTM((int) Math.pow(worldSize, 2), rand);
		player.getBrain().getActionNode().setExplorationChance(0);
	}
	
	private void runRound(Player player, Model model, int roundID){
		model.start();
		
		do {
			player.giveInfo(model);
			player.step();
			int action = player.getAction();
			boolean hit = action == 1;
			int nextStateID = model.nextStateID();
			int score = model.step(nextStateID, hit);
			player.giveScore(score);			
		} while (!model.isGameOver());
		
		int totalScore = model.getRunningScore();
		int maxPossibleScore = model.getMaxPossibleScore();
		player.endRound();
		
		System.out.println("Round " + roundID + " finished. Player got " + totalScore + " points out of " + maxPossibleScore + " points possible");
	}

}
