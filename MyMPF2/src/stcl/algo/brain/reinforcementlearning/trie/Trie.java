package stcl.algo.brain.reinforcementlearning.trie;

import java.util.HashMap;
import java.util.LinkedList;

public class Trie<Transition> {
	
	private TrieNode<Transition> root;
	
	public Trie() {
		root = new TrieNode<Transition>(null);
	}
	
	/**
	 * Add the given sequence to the Trie
	 * @param sequence
	 * @return a list of trieNodes corresponding to the symbol sequence. 
	 * The first node in the returned list corresponds to the last symbol in the input sequence
	 */
	public LinkedList<TrieNode<Transition>> add(LinkedList<Transition> sequence){
		LinkedList<TrieNode<Transition>> nodeSequence = new LinkedList<TrieNode<Transition>>();
		LinkedList<Transition> copy = copySequence(sequence);
		nodeSequence = root.addSequence(copy, nodeSequence);
		return nodeSequence;
	}
	
	/**
	 * Remove the given sequence from the Trie
	 * @param sequences
	 */
	public void remove(LinkedList<Transition> sequence){
		LinkedList<Transition> copy = copySequence(sequence);
		root.removeSequence(copy);
	}
	
	/**
	 * Returns the number of times the given sequence has been observed in the data set.
	 * Does not change the trie.
	 * @param sequence
	 * @return
	 */
	public int findSequenceCount(LinkedList<Transition> sequence){
		LinkedList<Transition> copy = copySequence(sequence);
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
	public LinkedList<TrieNode<Transition>> findNodeSequence(LinkedList<Transition> sequence){
		LinkedList<TrieNode<Transition>> nodeSequence = new LinkedList<TrieNode<Transition>>();
		LinkedList<Transition> copy = copySequence(sequence);
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
	public void incrementSequenceCount(LinkedList<Transition> sequence){
		LinkedList<Transition> copy = copySequence(sequence);
		root.incrementSequenceCount(copy);
	}
	
	/**
	 * Updates the counts for all nodes in the sequence
	 * @param sequence
	 */
	public void updateCounts(LinkedList<Transition> sequence){
		LinkedList<Transition> copy = copySequence(sequence);
		root.updateCount(copy);
	}
	
	/**
	 * Given the sequence the trie returns all possible next nodes. The probability of each node being active next is saved in the node.
	 * @param sequence
	 * @return
	 */
	public HashMap<Transition, TrieNode<Transition>> findChildrenOfLastNode(LinkedList<Transition> sequence){
		LinkedList<Transition> copy = copySequence(sequence);
		HashMap<Transition, TrieNode<Transition>> prediction = root.findChildrenOfLastNode(copy);
		return prediction;
	}
	
	/**
	 * Creates a copy of the sequence
	 * @param sequence
	 * @return
	 */
	private LinkedList<Transition> copySequence(LinkedList<Transition> sequence){
		LinkedList<Transition> copy = new LinkedList<Transition>(sequence);
		return copy;
	}
}
