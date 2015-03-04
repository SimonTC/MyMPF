package stcl.algo.util.trie;

import java.util.HashMap;
import java.util.LinkedList;

public class Trie<T> {
	
	private TrieNode<T> root;
	
	public Trie() {
		root = new TrieNode<T>(null);
	}
	
	/**
	 * Add the given sequence to the Trie
	 * @param sequence
	 * @return true if the sequence is new
	 */
	public void add(LinkedList<T> sequence){
		LinkedList<T> copy = copySequence(sequence);
		root.addSequence(copy);
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
	
	public void printTrie(int maxDepth){
		for (int depth = 1; depth <= maxDepth; depth++){
			String s = root.writeSequence("", depth);
			System.out.println(s);
		}
	}
	
	public void incrementSequenceCount(LinkedList<T> sequence){
		LinkedList<T> copy = copySequence(sequence);
		root.incrementSequenceCount(copy);
	}
	
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
	
	private LinkedList<T> copySequence(LinkedList<T> sequence){
		LinkedList<T> copy = new LinkedList<T>(sequence);
		return copy;
	}
}
