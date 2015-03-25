package stcl.algo.util.trie;

import java.util.HashMap;
import java.util.LinkedList;


public class TrieNode<T>{
	private T symbol;
	private int count; //Number of times a sequence ending with this node is observed in the data
	private HashMap<T, TrieNode<T>> children;
	private double probability; //Using when predicting symbols
	private int sequenceID;
	private TrieNode<T> parent;
	private double reward;
	private double rewardDecay;
	
	public TrieNode(T symbol, TrieNode<T> parent) {
		this.count = 0;
		this.symbol = symbol;
		children = new HashMap<T, TrieNode<T>>();
		sequenceID = -2;
		this.parent = parent;
		reward = 0;
		rewardDecay = 0.2; //TODO: Should be a parameter
	}
	
	/**
	 * Adds an occurrence of this sequence to the node.
	 * If the sequence doesn't end in this node, the remaining sequence is added to the child node corresponding to the next symbol in the sequence.
	 * If no child-node corresponds to the next symbol, a new child is created.
	 * @param symbolSequence the sequence of symbols coming after this node
	 * @param nodeSequence the the symbol sequence converted to a node sequence. Is returned in reverse order of the symbol sequence
	 * @param rreward reward given for this sequence. Is added to last node in sequence
	 */
	public LinkedList<TrieNode<T>> addSequence(LinkedList<T> symbolSequence, LinkedList<TrieNode<T>> nodeSequence, double reward){
		if (!symbolSequence.isEmpty()){
			T childSymbol = symbolSequence.removeFirst();
			TrieNode<T> correctChild = children.get(childSymbol);
			if (correctChild == null){
				correctChild = addNewChild(childSymbol);
			} 
			nodeSequence.addFirst(correctChild);
			nodeSequence = correctChild.addSequence(symbolSequence, nodeSequence);
			this.reward = calculateReward();
		} else {
			count++;
			double oldReward = this.reward;
			this.reward = oldReward * (1-rewardDecay) + reward;
		}
		return nodeSequence;		
	}
	
	/**
	 * Adds an occurrence of this sequence to the node.
	 * If the sequence doesn't end in this node, the remaining sequence is added to the child node corresponding to the next symbol in the sequence.
	 * If no child-node corresponds to the next symbol, a new child is created.
	 * @param symbolSequence the sequence of symbols coming after this node
	 */
	public LinkedList<TrieNode<T>> addSequence(LinkedList<T> symbolSequence, LinkedList<TrieNode<T>> nodeSequence){
		return this.addSequence(symbolSequence, nodeSequence, 0);
	}
	
	/**
	 * Calculates the reward of a node with children.The reward is equal to the average of the reward of the children
	 * @return
	 */
	private double calculateReward(){
		double totalReward = 0;
		for (TrieNode<T> child : children.values()){
			totalReward += child.getReward();
		}
		
		if (totalReward == 0) return 0;
		double avgReward = totalReward / (double)children.size();
		return avgReward;
	}
	
	/**
	 * Updates the count of all nodes along the sequence
	 * @param sequence
	 */
	public void updateCount(LinkedList<T> sequence){
		count++;
		if (!sequence.isEmpty()){
			T childSymbol = sequence.poll();
			TrieNode<T> correctChild = children.get(childSymbol);
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
	public boolean removeSequence(LinkedList<T> sequence){
		count--;
		if(!sequence.isEmpty()){
			T childSymbol = sequence.removeFirst();
			TrieNode<T> correctChild = children.get(childSymbol);
			boolean childIsEmpty = !correctChild.removeSequence(sequence);				
			if (childIsEmpty){
				children.remove(childSymbol);
			}
			this.reward = calculateReward();
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
	public int findSequenceCount(LinkedList<T> sequence){
		if(!sequence.isEmpty()){
			T childSymbol = sequence.removeFirst();
			TrieNode<T> correctChild = children.get(childSymbol);
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
	public void incrementSequenceCount(LinkedList<T> sequence){
		if(!sequence.isEmpty()){
			T childSymbol = sequence.removeFirst();
			TrieNode<T> correctChild = children.get(childSymbol);
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
	private TrieNode<T> addNewChild(T symbolOfChild){
		TrieNode<T> child = new TrieNode<T>(symbolOfChild, this);
		children.put(symbolOfChild, child);
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
	public double calculateEscapeChance(){
		int childrenCount = 0;
		for (TrieNode<?> n : children.values()){
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
	public HashMap<T, TrieNode<T>> findChildrenOfLastNode(LinkedList<T> sequence){
		if(!sequence.isEmpty()){
			//Send question further down
			T childSymbol = sequence.removeFirst();
			TrieNode<T> correctChild = children.get(childSymbol);
			if (correctChild == null) return null;
			return correctChild.findChildrenOfLastNode(sequence);
		} else {
			//Calculate probabilities
			for (TrieNode<T> n : children.values()){
				double probability = n.getCount() / (double) count;
				n.setProbability(probability);				
			}
			return children;				
		}
	}
	
	/**
	 * Recursively converts the given symbol sequence into a node sequence and calculates the probabiities of seeing the children of the last node in the sequence
	 * @param symbolSequence
	 * @param nodeSequence
	 * @return
	 */
	public LinkedList<TrieNode<T>> findNodeSequence(LinkedList<T> symbolSequence, LinkedList<TrieNode<T>> nodeSequence){
		if(!symbolSequence.isEmpty()){
			//Send question further down
			T childSymbol = symbolSequence.removeFirst();
			TrieNode<T> correctChild = children.get(childSymbol);
			if (correctChild == null) return null;
			nodeSequence.addFirst(correctChild);;
			return correctChild.findNodeSequence(symbolSequence, nodeSequence);
		} else {
			//Calculate probabilities
			for (TrieNode<T> n : children.values()){
				double probability = n.getCount() / (double) count;
				n.setProbability(probability);				
			}
			return nodeSequence;				
		}
	}
	
	public T getSymbol(){
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
		if (children.isEmpty()){
			return sequenceSoFar + " " + count + "   ";
		}
		
		for (TrieNode<T> n : children.values()){
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
	
	public double getReward(){
		return this.reward;
	}
	
	public TrieNode<T> getParent(){
		return parent;
	}
}
