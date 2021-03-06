package stcl.algo.predictors.trie;

import java.io.Serializable;
import java.util.HashMap;
import java.util.LinkedList;

public class Trie<T> implements Serializable{
	private static final long serialVersionUID = 1L;
	private TrieNode<T> root;
	
	public Trie() {
		root = new TrieNode<T>(null, null);
	}
	
	/**
	 * Add the given sequence to the Trie
	 * @param sequence
	 * @param reward the reward given t this sequence. Added to the last node in the sequence
	 * @return a list of trieNodes corresponding to the symbol sequence. 
	 * The first node in the returned list corresponds to the last symbol in the input sequence
	 */
	public LinkedList<TrieNode<T>> add(LinkedList<T> sequence, double reward){
		LinkedList<TrieNode<T>> nodeSequence = new LinkedList<TrieNode<T>>();
		LinkedList<T> copy = copySequence(sequence);
		nodeSequence = root.addSequence(copy, nodeSequence, reward);
		return nodeSequence;
	}
	
	/**
	 * Add the given sequence to the Trie
	 * @param sequence
	 * @return a list of trieNodes corresponding to the symbol sequence. 
	 * The first node in the returned list corresponds to the last symbol in the input sequence
	 */
	public LinkedList<TrieNode<T>> add(LinkedList<T> sequence){
		return this.add(sequence,0);
	}
	
	/**
	 * Remove the given sequence from the Trie
	 * @param sequences
	 */
	public void remove(LinkedList<T> sequence){
		LinkedList<T> copy = copySequence(sequence);
		root.removeSequence(copy);
	}
	
	/**
	 * Returns the number of times the given sequence has been observed in the data set.
	 * Does not change the trie.
	 * @param sequence
	 * @return
	 */
	public int findSequenceCount(LinkedList<T> sequence){
		LinkedList<T> copy = copySequence(sequence);
		return root.findSequenceCount(copy);
	}
	
	/**
	 *  Recursively converts the given symbol sequence into a node sequence and calculates the probabiities of seeing the children of the last node in the sequence.
	 *  Doesn't change the Data structure other than calulating probabilities
	 *  @param sequence
	 * 	@return a list of trieNodes corresponding to the symbol sequence. 
	 * 	The first node in the returned list corresponds to the last symbol in the input sequence
	 *  
	 */
	public LinkedList<TrieNode<T>> findNodeSequence(LinkedList<T> sequence){
		LinkedList<TrieNode<T>> nodeSequence = new LinkedList<TrieNode<T>>();
		LinkedList<T> copy = copySequence(sequence);
		nodeSequence = root.findNodeSequence(copy, nodeSequence);
		return nodeSequence;
	}
	
	public void printTrie(int maxDepth){
		for (int depth = 1; depth <= maxDepth; depth++){
			String s = root.writeSequence("", depth);
			System.out.println(s);
		}
	}
	
	/**
	 * Increments the count of the last node in the given sequence
	 * @param sequence
	 */
	public void incrementSequenceCount(LinkedList<T> sequence){
		LinkedList<T> copy = copySequence(sequence);
		root.incrementSequenceCount(copy);
	}
	
	/**
	 * Updates the counts for all nodes in the sequence
	 * @param sequence
	 */
	public void updateCounts(LinkedList<T> sequence){
		LinkedList<T> copy = copySequence(sequence);
		root.updateCount(copy);
	}
	
	/**
	 * Given the sequence the trie returns all possible next nodes. The probability of each node being active next is saved in the node.
	 * @param sequence
	 * @return
	 */
	public HashMap<T, TrieNode<T>> findChildrenOfLastNode(LinkedList<T> sequence){
		LinkedList<T> copy = copySequence(sequence);
		HashMap<T, TrieNode<T>> prediction = root.findChildrenOfLastNode(copy);
		return prediction;
	}
	
	/**
	 * Creates a copy of the sequence
	 * @param sequence
	 * @return
	 */
	private LinkedList<T> copySequence(LinkedList<T> sequence){
		LinkedList<T> copy = new LinkedList<T>(sequence);
		return copy;
	}
}
