package stcl.algo.brain.reinforcementlearning.trie;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;

import stcl.algo.brain.reinforcementlearning.Transition;


public class TrieNode{
	private int stateID;
	private int count; //Number of times a sequence ending with this node is observed in the data
	private ArrayList<HashMap<Integer, TrieNode>> actionChilds; //Map of states you can end up in by doing actions from this state
	private double probability; //Using when predicting symbols
	private int sequenceID;
	private int numActions;
	
	public TrieNode(int stateID, int numActions) {
		this.count = 0;
		this.stateID = stateID;
		this.numActions = numActions;
		actionChilds = new ArrayList<HashMap<Integer,TrieNode>>();
		//Set all children of actions to null
		for (int i = 0; i < numActions; i++){
			actionChilds.add(null);
		}
		sequenceID = -2;
	}
	
	/**
	 * Adds an occurrence of this sequence to the node.
	 * If the sequence doesn't end in this node, the remaining sequence is added to the child node corresponding to the next symbol in the sequence.
	 * If no child-node corresponds to the next symbol, a new child is created.
	 * @param transitionSequence the sequence of symbols coming after this node
	 */
	public LinkedList<TrieNode> addSequence(LinkedList<Transition> transitionSequence, LinkedList<TrieNode> nodeSequence){
		if (!transitionSequence.isEmpty()){
			Transition transition = transitionSequence.removeFirst();
			int action = transition.getAction();
			int nextState = transition.getNextState();
			
			HashMap<Integer,TrieNode> possibleChildren = actionChilds.get(action);
			if (possibleChildren == null) possibleChildren = new HashMap<Integer, TrieNode>();
			TrieNode nextStateNode = possibleChildren.get(nextState);
			if (nextStateNode == null){
				nextStateNode = addNewChild(action, nextState);
			} 
			nodeSequence.addFirst(nextStateNode);
			nodeSequence = nextStateNode.addSequence(transitionSequence, nodeSequence);
		} else {
			count++;
		}
		return nodeSequence;
		
	}
	
	/**
	 * Updates the count of all nodes along the sequence
	 * @param sequence
	 */
	public void updateCount(LinkedList<Transition> sequence){
		count++;
		if (!sequence.isEmpty()){
			Transition transition = sequence.removeFirst();
			int action = transition.getAction();
			int nextState = transition.getNextState();
			
			HashMap<Integer,TrieNode> possibleChildren = actionChilds.get(action);
			if (possibleChildren == null) return;
			
			TrieNode correctChild = possibleChildren.get(nextState);
			if (correctChild == null) return;
			correctChild.updateCount(sequence);
		}
	}
	
	/**
	 * Removes an occurrence of the given sequence from this node.
	 * If the sequence doesn't end in this node, the remaining sequence is removed from the child node corresponding to the next symbol in the sequence.
	 * If the child-node is empty (count == 0) after removing the sequence, the child node is deleted.
	 * @param sequence
	 * @return true if the count of this node is > 0, false otherwise
	 */
	public boolean removeSequence(LinkedList<Transition> sequence){
		count--;
		if(!sequence.isEmpty()){
			Transition transition = sequence.removeFirst();
			int action = transition.getAction();
			int nextState = transition.getNextState();
			
			HashMap<Integer,TrieNode> possibleChildren = actionChilds.get(action);
			TrieNode correctChild = possibleChildren.get(nextState);

			boolean childIsEmpty = !correctChild.removeSequence(sequence);				
			if (childIsEmpty) possibleChildren.remove(nextState);
			if (possibleChildren.isEmpty()) actionChilds.set(action, null);
		}
		
		return count > 0;			
	}
	
	/**
	 * Returns how many occurrences of sequences ending with the symbol of this node has been observed in the data.
	 * If the sequence doesn't end in this node it returns the findSequenceCount() of the child node corresponding to the next symbol in the sequence
	 * 
	 * @param sequence
	 * @return 
	 */
	public int findSequenceCount(LinkedList<Transition> sequence){
		if(!sequence.isEmpty()){
			Transition transition = sequence.removeFirst();
			int action = transition.getAction();
			int nextState = transition.getNextState();
			
			HashMap<Integer,TrieNode> possibleChildren = actionChilds.get(action);
			if (possibleChildren == null) return 0;
			TrieNode correctChild = possibleChildren.get(nextState);

			if (correctChild == null) return 0;
			return correctChild.findSequenceCount(sequence);
		} else {
			return count;
		}
	}
	
	/**
	 * Increments the count of the last node in the sequence
	 * @param sequence
	 */
	public void incrementSequenceCount(LinkedList<Transition> sequence){
		if(!sequence.isEmpty()){
			Transition transition = sequence.removeFirst();
			int action = transition.getAction();
			int nextState = transition.getNextState();			
			HashMap<Integer,TrieNode> possibleChildren = actionChilds.get(action);
			TrieNode correctChild = possibleChildren.get(nextState);
			correctChild.incrementSequenceCount(sequence);
		} else {
			count++;
			return;
		}
	}
	
	/**
	 * Creates a new child with the given symbol.
	 * The child is added to the map of children and returned.
	 * The count of the child is NOT incremented.
	 * @param symbolOfChild
	 * @return
	 */
	private TrieNode addNewChild(int actionToGetToChildState, int childStateID){
		TrieNode child = new TrieNode(childStateID, numActions);
		actionChilds.get(actionToGetToChildState).put(childStateID, child);
		return child;
	}

	/**
	 * Returns the number of times this node has been visited while traversing sequences in the data
	 * @return
	 */
	public int getCount(){
		return count;
	}
	
	/**
	 * Calculate the escape chance from this node.
	 * The escape chance is calculated as (number of sequences visiting child nodes) / (Number of sequences visiting this node)
	 * @return
	 */
	public double calculateEscapeChance(int action){
		int childrenCount = 0;
		//TODO: I think calculating escape like this is probably wrong.
		HashMap<Integer, TrieNode> childrenFromThisAction = actionChilds.get(action);
		
		for (TrieNode n : childrenFromThisAction.values()){
			childrenCount += n.getCount();
		}		
		double escapeChance = childrenCount / (double)count;
		
		return escapeChance;
	}
	
	/**
	 * Returns the children of the last node in the sequence
	 * @param sequence
	 * @return null if the sequence is unknown. Otherwise it returns the hashmap of children including the probabilities that they will be the next ones to be active.
	 */
	public HashMap<Integer, TrieNode> findChildrenOfLastNode(LinkedList<Integer> sequence){
		if(!sequence.isEmpty()){
			//Send question further down
			Integer childSymbol = sequence.removeFirst();
			TrieNode<Integer> correctChild = actionChilds.get(childSymbol);
			if (correctChild == null) return null;
			return correctChild.findChildrenOfLastNode(sequence);
		} else {
			//Calculate probabilities
			for (TrieNode<Integer> n : actionChilds.values()){
				double probability = n.getCount() / (double) count;
				n.setProbability(probability);				
			}
			return actionChilds;				
		}
	}
	
	/**
	 * Recursively converts the given symbol sequence into a node sequence and calculates the probabiities of seeing the children of the last node in the sequence
	 * @param symbolSequence
	 * @param nodeSequence
	 * @return
	 */
	public LinkedList<TrieNode<Integer>> findNodeSequence(LinkedList<Integer> symbolSequence, LinkedList<TrieNode<Integer>> nodeSequence){
		if(!symbolSequence.isEmpty()){
			//Send question further down
			Integer childSymbol = symbolSequence.removeFirst();
			TrieNode<Integer> correctChild = actionChilds.get(childSymbol);
			if (correctChild == null) return null;
			nodeSequence.addFirst(correctChild);;
			return correctChild.findNodeSequence(symbolSequence, nodeSequence);
		} else {
			//Calculate probabilities
			for (TrieNode<Integer> n : actionChilds.values()){
				double probability = n.getCount() / (double) count;
				n.setProbability(probability);				
			}
			return nodeSequence;				
		}
	}
	
	public Integer getSymbol(){
		return symbol;
	}
	
	/**
	 * Adds the symbol of this node to the sequenceSoFar String and the number of times a seuence ending in this node has occured in the data.
	 * If this is not a leaf node and the length of the sequenceSoFar is not maxSequenceLength, the string is sent further down and the returned string is a concatenation of the sequences ending at the lower levels.
	 * @param sequenceSoFar
	 * @param maxSequenceLength
	 * @return
	 */
	public String writeSequence(String sequenceSoFar, int maxSequenceLength){
		if (symbol != null){ //Root node will have its symbol == null
			int sequenceLength = sequenceSoFar.length();
			if (sequenceLength == maxSequenceLength - 1) {
				sequenceSoFar += symbol.toString() + " " + count + "    ";
				return sequenceSoFar;
			}
			sequenceSoFar+=symbol.toString();
		}			
		
		String sequenceToReturn = "";
		if (actionChilds.isEmpty()){
			return sequenceSoFar + " " + count + "   ";
		}
		
		for (TrieNode<Integer> n : actionChilds.values()){
			String returnedSequence = n.writeSequence(sequenceSoFar, maxSequenceLength);
			sequenceToReturn+=returnedSequence;
		}
		
		return sequenceToReturn;
	}
	
	public void setProbability(double probability){
		this.probability = probability;
	}
	
	public double getProbability(){
		return probability;
	}

	/**
	 * @return the sequenceID
	 */
	public int getSequenceID() {
		return sequenceID;
	}

	/**
	 * @param sequenceID the sequenceID to set
	 */
	public void setSequenceID(int sequenceID) {
		this.sequenceID = sequenceID;
	}
}
