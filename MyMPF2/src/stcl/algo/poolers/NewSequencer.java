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
	
	
	public NewSequencer(int markovOrder, int maxNumberOfSequencesInMemory, int inputLength) {
		this.markovOrder = markovOrder;
		this.trie = new Trie<Integer>();
		sequenceMemory = new ArrayList<LinkedList<TrieNode<Integer>>>();
		currentMinCount = Integer.MAX_VALUE;
		currentMinID = -1;
		this.inputLength = inputLength;
		this.maxNumberOfSequencesInMemory = maxNumberOfSequencesInMemory;
		sequenceProbabilities = new SimpleMatrix(1, maxNumberOfSequencesInMemory);
		reset();
		
		
	}
	
	public void reset(){
		currentSequence = new LinkedList<Integer>();
		currentInputProbabilitites = new LinkedList<SimpleMatrix>();
		
	}
	
	public SimpleMatrix feedForward(SimpleMatrix probabilityVector, int spatialBMUID, boolean startNewSequence){
		//Add input if we are still within the maximum length of a sequence
		if (currentSequence.size() < markovOrder) {
			currentSequence.addLast(spatialBMUID);
			currentInputProbabilitites.addLast(probabilityVector);
		}
		
		if (startNewSequence){
			//Add the current sequence to our trie of sequences
			LinkedList<TrieNode<Integer>> nodeList = trie.add(currentSequence);
			TrieNode<Integer> lastNodeInSequence = nodeList.peekFirst(); //First node in the node list corresponds to last node in the symbol sequence
			int count = lastNodeInSequence.getCount();
			int id = lastNodeInSequence.getSequenceID();
			
			//Add sequence to sequence memory if there is room
			if (id < 0){
				if (sequenceMemory.size() < maxNumberOfSequencesInMemory){
					//We still have room
					sequenceMemory.add(nodeList);
					int newID = sequenceMemory.size() - 1;
					lastNodeInSequence.setSequenceID(newID);
					if (count < currentMinCount){
						currentMinCount = count;
						currentMinID = newID;
					}
				} else {
					//We have to see if it can get a place by kicking somebody else out
					if (count > currentMinCount){
						//It will kick out the other one
						TrieNode<Integer> oldNode = sequenceMemory.get(currentMinID).peekFirst();
						oldNode.setSequenceID(-1);
						sequenceMemory.set(currentMinID, nodeList);
						lastNodeInSequence.setSequenceID(currentMinID);
					}
				}
			}		
			
			//Calculate probability of having just exited the different sequences
			for (int i = 0; i < sequenceMemory.size(); i++){
				LinkedList<TrieNode<Integer>> sequence = sequenceMemory.get(i);
				double probability = calculateProbabilityOfSequence(sequence);
				sequenceProbabilities.set(i, probability);
			}
			
			sequenceProbabilities = Normalizer.normalize(sequenceProbabilities);
			
			reset();
		}
		
		return sequenceProbabilities;
	}
	
	private double calculateProbabilityOfSequence(LinkedList<TrieNode<Integer>> sequence){
		double sequenceProbability = 1;
		int count = 0;
		Iterator<TrieNode<Integer>> sequenceIterator = sequence.iterator();
		Iterator<SimpleMatrix> probabilityIterator = currentInputProbabilitites.descendingIterator(); //The probability list holds the inputs in the reverse form of the sequence list
		while (sequenceIterator.hasNext() && probabilityIterator.hasNext()){
			SimpleMatrix inputProbabilityMatrix = probabilityIterator.next();
			int inputID = sequenceIterator.next().getSymbol();
			double probability = inputProbabilityMatrix.get(inputID);
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
			TrieNode<Integer> firstNodeInSequence = sequence.peekLast(); //TODO: Check that it is really correct
			TrieNode<Integer> lastNodeInSequence = sequence.peekFirst(); //It doesn't make sense, but I think it is correct 
			int sequenceID = lastNodeInSequence.getSequenceID();
			int inputID = firstNodeInSequence.getSymbol();
			double probability = inputMatrix.get(sequenceID);
			probabilities.set(inputID, probability);
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
			for (TrieNode<Integer> n : sequence){
				System.out.print(n.getSymbol() + " ");
			}
			System.out.println();
		}
	}
}
