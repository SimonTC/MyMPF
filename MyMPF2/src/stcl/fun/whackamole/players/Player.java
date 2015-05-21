package stcl.fun.whackamole.players;

import stcl.fun.whackamole.Model;

public abstract class Player {

	/**
	 * Used to colelct the action to perform. 
	 * @return array containing two elements: the state that the player is aiming at, whether the player will hot (1) or not (0)
	 */
	public abstract int[] action();
	
	public abstract void giveScore(int score);
	
	public abstract void giveInfo(Model model);
}
