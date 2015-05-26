package stcl.fun.whackamole.players;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.Network_DataCollector;
import stcl.fun.whackamole.Model;

public abstract class Player {

	/**
	 * Used to collect the action to perform. 
	 * @return array containing two elements: the state that the player is aiming at, whether the player will hit (1) or not (0)
	 */
	public abstract void step();
	
	public abstract int getAction();
	
	public abstract SimpleMatrix getPrediction();
	
	public abstract void giveScore(int score);
	
	public abstract void giveInfo(Model model);
	
	public abstract void endRound();
	
	public abstract Network_DataCollector getBrain();
}
