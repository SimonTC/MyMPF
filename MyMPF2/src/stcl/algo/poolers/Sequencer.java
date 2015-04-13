package stcl.algo.poolers;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.predictors.trie.Trie;
import stcl.algo.predictors.trie.TrieNode;
import stcl.algo.util.Normalizer;

public class Sequencer implements Serializable {
	private static final long serialVersionUID = 1L;
	private Trie<Integer> trie;
	private int markovOrder;
	private LinkedList<Integer> currentSequence;
	private LinkedList<SimpleMatrix> currentInputProbabilitites;
	private SimpleMatrix sequenceProbabilities; //Used as feed forward matrix
	private ArrayList<LinkedList<TrieNode<Integer>>> sequenceMemory; //List holding the last node in each of the sequences currently in memory  
													     //Has to be arraylist as it is not possible to instantiate a normal array with them
	//Statistics about the sequences in memory
	private int currentMinFrequency;
	private int currentMinID;
	private int combinedCountOfSequencesInMemory;
	private int maxNumberOfSequencesInMemory;
	private int inputLength;
	private int elementsInMemory;
	
	private boolean learning;
	
	
	public Sequencer(int markovOrder, int temporalGroupMapSize, int inputLength) {
		this.markovOrder = markovOrder;
		this.trie = new Trie<Integer>();
		currentMinFrequency = Integer.MAX_VALUE;
		currentMinID = -1;
		this.inputLength = inputLength;
		this.maxNumberOfSequencesInMemory = temporalGroupMapSize * temporalGroupMapSize;
		sequenceProbabilities = new SimpleMatrix(temporalGroupMapSize, temporalGroupMapSize);
		reset();
		setLearning(true);
		//Fill the sequence memory up with null values
		sequenceMemory = new ArrayList<LinkedList<TrieNode<Integer>>>();
		for (int i = 0; i < maxNumberOfSequencesInMemory; i++) sequenceMemory.add(null);
		elementsInMemory = 0;
	}
	
	/**
	 * 
	 * @param probabilityVector
	 * @param spatialBMUID
	 * @param startNewSequence
	 * @return Probability matrix containing the probabilities of having just exited each of the known sequences
	 */
	public SimpleMatrix feedForward(SimpleMatrix probabilityVector, int spatialBMUID, boolean startNewSequence){
		//Add input if we are still within the maximum length of a sequence
		if (currentSequence.size() < markovOrder) {
			currentSequence.addLast(spatialBMUID);
			currentInputProbabilitites.addLast(probabilityVector);
		}
		
		if (learning){
			if (startNewSequence){
				addFinishedSequenceToMemory();
			}
		}
		
		//Calculate probability of having just exited the different sequences
		calculateExitProbabilities();
		
		//If the chance of being in a sequence is less than 10% we set it to zero
		//This should make the output a bit more clean
		removeLowChanceSequences();
		
		if (startNewSequence) reset();
		
		return sequenceProbabilities;
	}
	
	/**
	 * 
	 * @param inputMatrix matrix containing the probabilities of each of the sequences being the ones that start at time t + 1
	 * @return probabilities of the next input given the probabilities of starting the different sequences
	 */
	public SimpleMatrix feedBackward(SimpleMatrix inputMatrix){
		//Returns the probabilities of the next input given the probabilities of starting the different sequences
		SimpleMatrix symbolProbabilities = new SimpleMatrix(1, inputLength);
		Iterator<LinkedList<TrieNode<Integer>>> sequenceIterator = sequenceMemory.iterator();
		while (sequenceIterator.hasNext()){
			LinkedList<TrieNode<Integer>> sequence = sequenceIterator.next();
			if (sequence != null){
				TrieNode<Integer> firstNodeInSequence = sequence.peekLast(); //The sequence is inverted
				TrieNode<Integer> lastNodeInSequence = sequence.peekFirst(); 
				int sequenceID = lastNodeInSequence.getSequenceID();
				int symbolID = firstNodeInSequence.getSymbol();
				double probability = inputMatrix.get(sequenceID);
				double oldValue = symbolProbabilities.get(symbolID);
				double newValue = oldValue + probability;
				symbolProbabilities.set(symbolID, newValue);				
			}
			
		}
		
		symbolProbabilities = Normalizer.normalize(symbolProbabilities);
		
		return symbolProbabilities;
	}
	
	public void reset(){
		currentSequence = new LinkedList<Integer>();
		currentInputProbabilitites = new LinkedList<SimpleMatrix>();
		
	}
	
	/**
	 * Finds the first spot in the sequence memory that contains a null value.
	 * Returns -1 if no spots are free.
	 * @return
	 */
	private int findNextFreeSpotInMemory(){
		for (int i = 0; i < sequenceMemory.size(); i++){
			if (sequenceMemory.get(i) == null) return i;
		}
		return -1;
	}
	
	private void calculateExitProbabilities(){
		for (int i = 0; i < sequenceMemory.size(); i++){
			LinkedList<TrieNode<Integer>> sequence = sequenceMemory.get(i);
			double probability = 0;
			if (sequence != null){
				probability = calculateProbabilityOfSequence(sequence);
			}
			sequenceProbabilities.set(i, probability);
		}
		
		sequenceProbabilities = Normalizer.normalize(sequenceProbabilities);
		
	}
	
	private void removeLowChanceSequences(){
		for (int i = 0; i < sequenceProbabilities.getNumElements(); i++){
			double probability = sequenceProbabilities.get(i);
			if (probability < 0.1) probability = 0;
			sequenceProbabilities.set(i, probability);
		}
		
		sequenceProbabilities = Normalizer.normalize(sequenceProbabilities);
	}
	
	/**
	 * Adds the sequence that has just ended to the sequence memory if there is room. Also updates sequence statistics and clean memory of rare sequences.
	 */
	private void addFinishedSequenceToMemory(){
		//Add the current sequence to our trie of sequences
		LinkedList<TrieNode<Integer>> nodeList = trie.add(currentSequence);
		TrieNode<Integer> lastNodeInSequence = nodeList.peekFirst(); //First node in the node list corresponds to last node in the symbol sequence
		int count = lastNodeInSequence.getCount();
		int id = lastNodeInSequence.getSequenceID();
		
		//Add sequence to sequence memory if there is room / it is important enough
		if (id < 0){
			addSequenceToMemory(count, nodeList);
		} 
		
		//Update frequency counters
		int[] frequencies = updateFrequencyCounts();
		
		//Go through the sequences to see if some of them should be removed
		//Sequences will be removed if the have appeared less than 1 % of the time
		cleanMemory(frequencies);
	}
	
	/**
	 * Removes sequences from memory that has appeared less than 1% of the time
	 * @param frequencies
	 */
	private void cleanMemory(int[] frequencies){
		for (int i = 0; i < frequencies.length; i++){
			double frequency = (double)frequencies[i] / combinedCountOfSequencesInMemory;
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
	
	/**
	 * Updates the statistics on the frequencies of the sequences in memory.
	 * Fields updated: currentMinFrequency,  currentMinID, combinedCountOfSequencesInMemory
	 * @return arraylist of the frequencies of each of the sequences in memory
	 */
	private int[] updateFrequencyCounts(){
		currentMinFrequency = Integer.MAX_VALUE;
		combinedCountOfSequencesInMemory = 0;
		int[] frequencies = new int[sequenceMemory.size()];
		for (int i = 0; i < sequenceMemory.size(); i++){
			LinkedList<TrieNode<Integer>> sequence = sequenceMemory.get(i);
			if (sequence != null){
				TrieNode<Integer> lastNode = sequence.peekFirst();
				int sequenceFrequency = lastNode.getCount();
				combinedCountOfSequencesInMemory += sequenceFrequency;
				frequencies[i] = sequenceFrequency;
				if (sequenceFrequency < currentMinFrequency){
					currentMinFrequency = sequenceFrequency;
					currentMinID = lastNode.getSequenceID();
				}
			}			
		}
		return frequencies;
	}
	
	/**
	 * Adds the given node sequence to the sequence memory.
	 * If there is a free spot in the sequence memory the sequence is added.
	 * If there is no free spot, the sequence is only added if it has a higher frequency than at least one of the sequences currently in memory.
	 * The sequence with lower frequency is removed from memory.
	 * @param sequenceFrequency
	 * @param nodeSequence
	 */
	private void addSequenceToMemory(int sequenceFrequency, LinkedList<TrieNode<Integer>> nodeSequence){
		TrieNode<Integer> lastNodeInSequence = nodeSequence.peekFirst(); //The sequence is in reverse order
		if (elementsInMemory < maxNumberOfSequencesInMemory){
			//We still have room
			int newID = findNextFreeSpotInMemory();
			sequenceMemory.set(newID, nodeSequence);
			lastNodeInSequence.setSequenceID(newID);					
			elementsInMemory++;
		} else {
			//We have to see if it can get a place by kicking somebody else out
			if (sequenceFrequency > currentMinFrequency){
				//It will kick out the other one
				TrieNode<Integer> oldNode = sequenceMemory.get(currentMinID).peekFirst();
				oldNode.setSequenceID(-1);
				sequenceMemory.set(currentMinID, nodeSequence);
				lastNodeInSequence.setSequenceID(currentMinID);
			}
		}

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


	/**
	 * @return the sequenceProbabilities
	 */
	public SimpleMatrix getSequenceProbabilities() {
		return sequenceProbabilities;
	}
}
