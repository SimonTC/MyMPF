package stcl.algo.poolers;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.util.Normalizer;
import stcl.algo.util.trie.Trie;
import stcl.algo.util.trie.TrieNode;

public class NewSequencer {
	private Trie<Integer> trie;
	private int markovOrder;
	private LinkedList<Integer> currentSequence;
	private LinkedList<SimpleMatrix> currentInputProbabilitites;
	private SimpleMatrix sequenceProbabilities; //Used as feed forward matrix
	private ArrayList<LinkedList<TrieNode<Integer>>> sequenceMemory; //List holding the last node in each of the sequences currently in memory  
													     //Has to be arraylist as it is not possible to instantiate a normal array with them
	//Statistics about the sequences in memory
	private int currentMinCount;
	private int currentMinID;
	private int maxNumberOfSequencesInMemory;
	private int inputLength;
	private int elementsInMemory;
	
	private boolean learning;
	
	
	public NewSequencer(int markovOrder, int temporalGroupMapSize, int inputLength) {
		this.markovOrder = markovOrder;
		this.trie = new Trie<Integer>();
		currentMinCount = Integer.MAX_VALUE;
		currentMinID = -1;
		this.inputLength = inputLength;
		this.maxNumberOfSequencesInMemory = temporalGroupMapSize;
		sequenceProbabilities = new SimpleMatrix(temporalGroupMapSize, temporalGroupMapSize);
		reset();
		setLearning(true);
		//Fill the sequence memory up with null values
		sequenceMemory = new ArrayList<LinkedList<TrieNode<Integer>>>();
		for (int i = 0; i < maxNumberOfSequencesInMemory; i++) sequenceMemory.add(null);
		elementsInMemory = 0;
	}
	
	public void reset(){
		currentSequence = new LinkedList<Integer>();
		currentInputProbabilitites = new LinkedList<SimpleMatrix>();
		
	}
	
	private int findNextFreeSpotInMemory(){
		for (int i = 0; i < sequenceMemory.size(); i++){
			if (sequenceMemory.get(i) == null) return i;
		}
		return -1;
	}
	
	public SimpleMatrix feedForward(SimpleMatrix probabilityVector, int spatialBMUID, boolean startNewSequence){
		//Add input if we are still within the maximum length of a sequence
		if (currentSequence.size() < markovOrder) {
			currentSequence.addLast(spatialBMUID);
			currentInputProbabilitites.addLast(probabilityVector);
		}
		
		if (learning){
			if (startNewSequence){
				//Add the current sequence to our trie of sequences
				LinkedList<TrieNode<Integer>> nodeList = trie.add(currentSequence);
				//printSequence(nodeList);
				TrieNode<Integer> lastNodeInSequence = nodeList.peekFirst(); //First node in the node list corresponds to last node in the symbol sequence
				int count = lastNodeInSequence.getCount();
				int id = lastNodeInSequence.getSequenceID();
				
				//Add sequence to sequence memory if there is room / it is important enough
				if (id < 0){
					if (elementsInMemory < maxNumberOfSequencesInMemory){
						//System.out.println(" --- Added");
						//We still have room
						int newID = findNextFreeSpotInMemory();
						sequenceMemory.set(newID, nodeList);
						lastNodeInSequence.setSequenceID(newID);					
						elementsInMemory++;
					} else {
						//We have to see if it can get a place by kicking somebody else out
						if (count > currentMinCount){
							//System.out.println(" --- Added");
							//It will kick out the other one
							TrieNode<Integer> oldNode = sequenceMemory.get(currentMinID).peekFirst();
							//printSequence(sequenceMemory.get(currentMinID));
							//System.out.println(" --- Removed");
							oldNode.setSequenceID(-1);
							sequenceMemory.set(currentMinID, nodeList);
							lastNodeInSequence.setSequenceID(currentMinID);
						} else {
							//System.out.println(" --- Not added");
						}
					}
				} 
				
				//Update counts
				currentMinCount = Integer.MAX_VALUE;
				int totalCount = 0;
				int[] counts = new int[sequenceMemory.size()];
				for (int i = 0; i < sequenceMemory.size(); i++){
					LinkedList<TrieNode<Integer>> sequence = sequenceMemory.get(i);
					if (sequence != null){
						TrieNode<Integer> lastNode = sequence.peekFirst();
						int nodeCount = lastNode.getCount();
						totalCount += nodeCount;
						counts[i] = nodeCount;
						if (nodeCount < currentMinCount){
							currentMinCount = nodeCount;
							currentMinID = lastNode.getSequenceID();
						}
					}
					
				}
				
				//Go through the sequences to see if some of them should be removed
				//Sequences will be removed if the have appeared less than 1 % of the time
				for (int i = 0; i < counts.length; i++){
					double frequency = (double)counts[i] / totalCount;
					if (frequency < 0.01){
						LinkedList<TrieNode<Integer>> sequence = sequenceMemory.get(i);
						if (sequence != null){
							TrieNode<Integer> oldNode = sequence.peekFirst();
							oldNode.setSequenceID(-1);
							sequenceMemory.set(i, null);
							elementsInMemory--;
						}
						
					}
				}
			}
		}
		
		//Calculate probability of having just exited the different sequences
		for (int i = 0; i < sequenceMemory.size(); i++){
			LinkedList<TrieNode<Integer>> sequence = sequenceMemory.get(i);
			double probability = 0;
			if (sequence != null){
				probability = calculateProbabilityOfSequence(sequence);
			}
			sequenceProbabilities.set(i, probability);
		}
		
		sequenceProbabilities = Normalizer.normalize(sequenceProbabilities);
		
		//If the chance of beeing in a sequence is less than 10% we set it to zero
		//This should make the output a bit more clean
		for (int i = 0; i < sequenceProbabilities.getNumElements(); i++){
			double probability = sequenceProbabilities.get(i);
			if (probability < 0.1) probability = 0;
			sequenceProbabilities.set(i, probability);
		}
		
		sequenceProbabilities = Normalizer.normalize(sequenceProbabilities);
		
		if (startNewSequence) reset();
		
		return sequenceProbabilities;
	}
	
	private double calculateProbabilityOfSequence(LinkedList<TrieNode<Integer>> sequence){
		double sequenceProbability = 1;
		int count = 0;
		Iterator<TrieNode<Integer>> sequenceIterator = sequence.iterator();
		Iterator<SimpleMatrix> probabilityIterator = currentInputProbabilitites.descendingIterator(); //The probability list holds the inputs in the reverse form of the sequence list
		while (sequenceIterator.hasNext()){
			double probability;
			int inputID = sequenceIterator.next().getSymbol(); //Do this here to avoid unending loop
			if (probabilityIterator.hasNext()){
				SimpleMatrix inputProbabilityMatrix = probabilityIterator.next();
				probability = inputProbabilityMatrix.get(inputID);
			} else {
				probability = 0; //If the sequence we just exited is shorter than the sequence we are looking at, we didn't exit that sequence
			}
			sequenceProbability *= probability;
			count++;
		}
				
		//If the length of the sequence is less than the markov order, the missing probabilitites are taken to be 1 / #possible inputs
		int numPossibleInputs = currentInputProbabilitites.peek().getNumElements();
		double generalProbability = 1.0 / numPossibleInputs;
		while (count < markovOrder){
			sequenceProbability *= generalProbability;
			count++;
		}
		
		return sequenceProbability;
	}
	
	public SimpleMatrix feedBackward(SimpleMatrix inputMatrix){
		//Returns the probabilities of the next input given the probabilities of starting the different sequences
		SimpleMatrix probabilities = new SimpleMatrix(1, inputLength);
		Iterator<LinkedList<TrieNode<Integer>>> sequenceIterator = sequenceMemory.iterator();
		while (sequenceIterator.hasNext()){
			LinkedList<TrieNode<Integer>> sequence = sequenceIterator.next();
			if (sequence != null){
				TrieNode<Integer> firstNodeInSequence = sequence.peekLast(); //TODO: Check that it is really correct
				TrieNode<Integer> lastNodeInSequence = sequence.peekFirst(); //It doesn't make sense, but I think it is correct 
				int sequenceID = lastNodeInSequence.getSequenceID();
				int inputID = firstNodeInSequence.getSymbol();
				double probability = inputMatrix.get(sequenceID);
				double oldValue = probabilities.get(inputID);
				double newValue = oldValue + probability;
				probabilities.set(inputID, newValue);
				
			}
			
		}
		
		probabilities = Normalizer.normalize(probabilities);
		
		return probabilities;
	}
	
	public void printTrie(){
		trie.printTrie(markovOrder);
	}
	
	public void printSequenceMemory(){
		System.out.println("Sequence memory:");
		for (LinkedList<TrieNode<Integer>> sequence : sequenceMemory){
			if (sequence != null){
				printSequence(sequence);
				System.out.println();
			} else {
				System.out.println("Null");
			}
		}
	}
	
	private void printSequence(LinkedList<TrieNode<Integer>> sequence){
		Iterator<TrieNode<Integer>> iterator = sequence.descendingIterator();
		while (iterator.hasNext()){
			TrieNode<Integer> n = iterator.next();
			System.out.print(n.getSymbol() + " ");
		}
		int count = sequence.peekFirst().getCount();
		System.out.print(" ---- " + count);
	}

	public boolean isLearning() {
		return learning;
	}

	public void setLearning(boolean learning) {
		this.learning = learning;
	}
}
