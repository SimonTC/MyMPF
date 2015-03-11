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
	
	
	
	public NewSequencer(int markovOrder, int mapSize) {
		this.markovOrder = markovOrder;
		this.trie = new Trie<Integer>();
		sequenceMemory = new ArrayList<LinkedList<TrieNode<Integer>>>();
		currentMinCount = Integer.MAX_VALUE;
		currentMinID = -1;
		
		maxNumberOfSequencesInMemory = mapSize * mapSize;
		sequenceProbabilities = new SimpleMatrix(1, maxNumberOfSequencesInMemory);
		
		reset();
		
		
	}
	
	private void reset(){
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
			LinkedList<TrieNode<Integer>> nodeSequence = trie.add(currentSequence);
			TrieNode<Integer> lastNode = nodeSequence.peekFirst();
			int count = lastNode.getCount();
			int id = lastNode.getSequenceID();
			
			//Add sequence to sequence memory if there is room
			if (id == -1){
				if (sequenceMemory.size() < maxNumberOfSequencesInMemory){
					//We still have room
					sequenceMemory.add(nodeSequence);
					int newID = sequenceMemory.size() - 1;
					lastNode.setSequenceID(newID);
					if (count < currentMinCount){
						currentMinCount = count;
						currentMinID = newID;
					}
				} else {
					//We have to see if it can get a place by kicking somebody else out
					if (count > currentMinCount){
						//It will kick out the other one
						TrieNode<Integer> oldNode = sequenceMemory.get(currentMinID).peekLast();
						oldNode.setSequenceID(-1);
						sequenceMemory.set(currentMinID, nodeSequence);
						lastNode.setSequenceID(currentMinID);
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
		Iterator<SimpleMatrix> probabilityIterator = currentInputProbabilitites.iterator();
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
		//Returns the probabilitites of the next input given the probabilitites of starting the different sequences
		return null;
	}
	
	public void printTrie(){
		trie.printTrie(markovOrder);
	}
}
