package stcl.algo.predictors;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;

import stcl.algo.util.trie.Trie;
import stcl.algo.util.trie.TrieNode;

/**
 * AMAPPM is an implementation of the Adaptive Model Averaged PPM algorithm
 * presented by Saishankar K.P. and Sheetal Kalyani in 
 * "A Modified PPM Algorithm for Online Sequence Prediction using Short Data Records"
 * @author Simon
 *
 */
//TODO: Better citation
public class AMAPPM<T> {
	private Trie<T> trie;
	private LinkedList<T> memory = new LinkedList<T>();
	private int markovOrder;
	private ArrayList<HashMap<T, Double>> predictions; //Would be better to use normal array, but don't know how to create generic array
	
	public AMAPPM(int markovOrder) {
		trie = new Trie<T>();
		this.markovOrder = markovOrder;
		predictions = new ArrayList<HashMap<T,Double>>(3);
	}
	
	/**
	 * Predicts the next symbol in the sequence given symbol.
	 * The given symbol is added to the sequence and used in the prediction
	 * @param symbol
	 * @return
	 */
	public T predict(T symbol){
		//Update error vector with prediction from t-1
		
		//Make predictions
		LinkedList<T> context = new LinkedList<T>();
		for ( int i = 0; i < markovOrder; i++){
			T c = memory.get(memory.size() - i);
			context.addFirst(c);
			HashMap<T, TrieNode<T>> possibleNextNodes = trie.findChildrenOfLastNode(context);
			
			for (T s : possibleNextNodes.keySet()){
				
			}
			
			predictions.set(i, possibleNextNodes);
		}
		
		
		return null;
	}
	
	/**
	 * Adds the given symbol to the memory and adds all possible contexts in the memory to the trie
	 * @param symbol
	 */
	public void addSymbol(T symbol){		
		memory.add(symbol);			
		if (memory.size() > markovOrder) memory.removeFirst();
		
		//Add all possible update trie based on all contexts in the memory to the trie
		LinkedList<T> context = new LinkedList<T>();
		for (int j = 1; j <= memory.size(); j++){
			T c = memory.get(memory.size() - j);
			context.addFirst(c);
			trie.add(context);
		}
	}
}
