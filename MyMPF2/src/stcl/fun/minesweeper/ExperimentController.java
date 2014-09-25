package stcl.fun.minesweeper;

import java.util.Random;

import stcl.Board;
import stcl.Controller;
import stcl.MineGui;
import stcl.Model;
import stcl.players.Player;
import stcl.players.RandomAI;
import stcl.players.Player.PlayerType;

public class ExperimentController extends Controller {

	private AIPlayer aiPlayer;
	private boolean visual;
	
	public static void main(String[] args){
		int maxIterations = 1000;
		Random rand = new Random();
		Brain b = new Brain(maxIterations, N_COLS * N_ROWS + 2, rand);
		
		AIPlayer p = new AIPlayer("Bob", Player.PlayerType.AI, b, rand);
		
		boolean visual = false;
		
		ExperimentController c = new ExperimentController(p, visual);
		c.start(maxIterations);
	}
	
	
	
	public ExperimentController(Player player, boolean visual) {
		super(player, visual);
		aiPlayer = (AIPlayer) player;
		this.visual = visual;
	}
	
	public void start(int numberOfGames){
		System.out.println("Starting experiment");
		int totalVictories = 0;
		for (int game = 1; game <= numberOfGames; game++){
			boolean won = runAIGame(visual);
			if (won){
				totalVictories++;
				aiPlayer.giveReward(1);
			} else {
				aiPlayer.giveReward(-1);
			}
			aiPlayer.makeReadyForNewGame();
			System.out.println("Game " + game + " Current victories: " + totalVictories + "/" + game);
		}
	}
	
	/** 
	 * @return true if game is won. False otherwise
	 */
	public boolean runAIGame(boolean visual){
		if (visual){
			return runAIGameVisual();
		}
		
		boolean running = true;
		int counter = 0;
		int maxCount = model.getFields().length * 2;
		while (running && counter++ < maxCount){
			int[] action = aiPlayer.getMove(model.getFields());
			running = fieldChosen(action[1], action[0], true);
			if (running ){
				aiPlayer.giveReward(0.5);
			}
		}
		
		if (counter >= maxCount) return false;
		return !model.gameLost();
	}
	@Override
	public boolean runAIGameVisual(){
		int FRAMES_PER_SECOND = 1000;
	    int SKIP_TICKS = 1000 / FRAMES_PER_SECOND;
	   
	    float next_game_tick = System.currentTimeMillis();
	    float sleepTime = 0;
	    
		boolean running = true;
		while (running){
			int[] action = aiPlayer.getMove(model.getFields());
			running = fieldChosen(action[1], action[0], true);
			if (running ){
				aiPlayer.giveReward(0.5);
			}
			System.out.println("Chose row " + action[1] + " col " + action[0] + " leftclicked? " + (1 == action[2]));
			if (running){
				next_game_tick+= SKIP_TICKS;
				sleepTime = next_game_tick - System.currentTimeMillis();
				if (sleepTime >= 0){
					try {
						Thread.sleep(SKIP_TICKS);
					} catch (InterruptedException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
			}
		}
		
		return !model.gameLost();
	}
	
	@Override
	public boolean fieldChosen(int row, int col, boolean uncover){
		int marksLeft = 0;
		boolean running = true;
		
		if (!model.isInGame()){
			model.setupGame();
			model.startGame();
		}
		
		//Do action
		if (uncover){
			running = model.uncoverField(col, row);	
			marksLeft = model.marksLeft();
		} else {
			marksLeft = model.markField(col, row);
		}
		
		//Update visuals
		if (visual){			
			board.updateMarksLeft(marksLeft);
			board.repaint();
		}
		
		return running;
	}

}
