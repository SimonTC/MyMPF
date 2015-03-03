package stcl.algo.predictors;

import java.util.HashMap;
import java.util.LinkedList;

public class Trie<T> {
	
	private Node root;
	
	public Trie() {
		root = new Node(null);
	}
	
	/**
	 * Add the given sequence to the Trie
	 * @param sequence
	 * @return true if the sequence is new
	 */
	public void add(LinkedList<T> sequence){
		LinkedList<T> copy = new LinkedList<T>(sequence);
		root.addSequence(copy);
	}
	
	/**
	 * Remove the given sequence from the Trie
	 * @param sequences
	 */
	public void remove(LinkedList<T> sequence){
		LinkedList<T> copy = new LinkedList<T>(sequence);
		root.removeSequence(copy);
	}
	
	/**
	 * Returns the number of times the given sequence has been observed in the data set.
	 * Does not change the trie.
	 * @param sequence
	 * @return
	 */
	public int findSequenceCount(LinkedList<T> sequence){
		LinkedList<T> copy = new LinkedList<T>(sequence);
		return root.findSequenceCount(copy);
	}
	
	public void printTrie(int maxDepth){
		for (int depth = 1; depth <= maxDepth; depth++){
			String s = root.writeSequence("", depth);
			System.out.println(s);
		}
	}
	
	public void incrementSequenceCount(LinkedList<T> sequence){
		LinkedList<T> copy = new LinkedList<T>(sequence);
		root.incrementSequenceCount(copy);
	}
	
	public void updateCounts(LinkedList<T> sequence){
		LinkedList<T> copy = new LinkedList<T>(sequence);
		root.updateCount(copy);
	}
	
	
	private class Node{
		private T symbol;
		private int count; //Number of times a sequence ending with this node is observed in the data
		private HashMap<T, Node> children;
		
		public Node(T symbol) {
			this.count = 0;
			this.symbol = symbol;
			children = new HashMap<T, Trie<T>.Node>();
		}
		
		/**
		 * Adds an occurrence of this sequence to the node.
		 * If the sequence doesn't end in this node, the remaining sequence is added to the child node corresponding to the next symbol in the sequence.
		 * If no child-node corresponds to the next symbol, a new child is created.
		 * @param sequence the sequence of symbols coming after this node
		 */
		public void addSequence(LinkedList<T> sequence){
		//	if (increment) count++;			
			if (!sequence.isEmpty()){
				T childSymbol = sequence.poll();
				Node correctChild = children.get(childSymbol);
				if (correctChild == null){
					correctChild = addNewChild(childSymbol);
				} 
	
				correctChild.addSequence(sequence);
			} else {
				count++;
			}
			
		}
		
		public void updateCount(LinkedList<T> sequence){
			count++;
			if (!sequence.isEmpty()){
				T childSymbol = sequence.poll();
				Node correctChild = children.get(childSymbol);
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
				Node correctChild = children.get(childSymbol);
				boolean childIsEmpty = !correctChild.removeSequence(sequence);				
				if (childIsEmpty) children.remove(childSymbol);
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
				Node correctChild = children.get(childSymbol);
				if (correctChild == null) return 0;
				return correctChild.findSequenceCount(sequence);
			} else {
				return count;
			}
		}
		
		public void incrementSequenceCount(LinkedList<T> sequence){
			if(!sequence.isEmpty()){
				T childSymbol = sequence.removeFirst();
				Node correctChild = children.get(childSymbol);
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
		private Node addNewChild(T symbolOfChild){
			Node child = new Node(symbolOfChild);
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
			for (Node n : children.values()){
				childrenCount += n.getCount();
			}
			
			double escapeChance = childrenCount / (double)count;
			
			return escapeChance;
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
			
			for (Node n : children.values()){
				String returnedSequence = n.writeSequence(sequenceSoFar, maxSequenceLength);
				sequenceToReturn+=returnedSequence;
			}
			
			return sequenceToReturn;
		}
	}

}
