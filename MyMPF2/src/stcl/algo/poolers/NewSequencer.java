package stcl.algo.poolers;

import java.util.LinkedList;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.util.Normalizer;
import stcl.algo.util.trie.Trie;

public class NewSequencer {
	private Trie<Integer> trie;
	private int markovOrder;
	private LinkedList<Integer> currentSequence;
	private SimpleMatrix sequenceProbabilities;
	
	
	
	public NewSequencer(int markovOrder, int mapSize) {
		this.markovOrder = markovOrder;
		this.trie = new Trie<Integer>();
		sequenceProbabilities = new SimpleMatrix(mapSize, mapSize);
		sequenceProbabilities.set(1);
		sequenceProbabilities = Normalizer.normalize(sequenceProbabilities);
		reset();
		
		
	}
	
	private void reset(){
		currentSequence = new LinkedList<Integer>();
	}
	
	public SimpleMatrix feedForward(SimpleMatrix probabilityVector, int spatialBMUID, boolean resetCount){
		if (currentSequence.size() <= markovOrder) {
			currentSequence.addLast(spatialBMUID);
		}
		
		if (resetCount){
			trie.add(currentSequence);
			reset();
		}
		
		return sequenceProbabilities;
	}
	
	public SimpleMatrix feedBackward(SimpleMatrix inputMatrix){
		//Returns the probabilitites of the next input given the probabilitites of starting the different sequences
		return null;
	}
	
	public void printTrie(){
		trie.printTrie(markovOrder);
	}
}
