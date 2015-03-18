package stcl.algo.brain.reinforcementlearning;

public class Transition {
	private int curState, action, nextState;
	
	/**
	 * 
	 * @param curState The state we are currently in
	 * @param action action taken to get out of the current state
	 * @param nextState state we end up in by doing the action
	 */
	public Transition(int curState, int action, int nextState) {
		this.curState = curState;
		this.action = action;
		this.nextState = nextState;
	}
	
	public int getCurState(){
		return curState;
	}
	
	public int getAction(){
		return action;
	}
	
	public int getNextState(){
		return nextState;
	}

	

}
