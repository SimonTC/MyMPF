package stcl.fun.whackamole;

import java.util.Random;

import stcl.fun.whackamole.players.Player;

public class Controller {
	
	private Player player;
	private Model model;
	private Random rand = new Random(1234);
	
	public static void main(String[] args) {
		Controller c = new Controller();
		c.setup();

	}
	
	public void setup(){
		model = new Model();
		model.initialize(4, 3, 2, 5, 0.1, rand);
	}
	
	public void runRound(Player player, Model model){
		model.start();
		
		do {
			player.giveInfo(model);
			int[] action = player.action();
			int activeField = action[0];
			boolean hit = action[1] == 1;
			int score = model.step(activeField, hit);
			player.giveScore(score);			
		} while (!model.isGameOver());
		
		int totalScore = model.getRunningScore();
		int maxPossibleScore = model.getMaxPossibleScore();
		
		System.out.println("Round finished. Player got " + totalScore + " points out of " + maxPossibleScore + " points possible");
	}

}
