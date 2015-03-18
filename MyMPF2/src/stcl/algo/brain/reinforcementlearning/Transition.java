package stcl.algo.brain.reinforcementlearning;

public class Transition {
	private int state, action;
	
	/**
	 * 
	 * @param state The state we are currently in
	 * @param action action taken to get out of the current state
	 */
	public Transition(int state, int action) {
		this.state = state;
		this.action = action;
	}

	/* (non-Javadoc)
	 * @see java.lang.Object#hashCode()
	 */
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + action;
		result = prime * result + state;
		return result;
	}

	/* (non-Javadoc)
	 * @see java.lang.Object#equals(java.lang.Object)
	 */
	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Transition other = (Transition) obj;
		if (action != other.action)
			return false;
		if (state != other.state)
			return false;
		return true;
	}

}
