package stcl.algo.predictors;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.util.trie.Trie;
import stcl.algo.util.trie.TrieNode;

/**
 * VOMM is an implementation of the Adaptive Model Averaged PPM algorithm
 * presented by Saishankar K.P. and Sheetal Kalyani in 
 * "A Modified PPM Algorithm for Online Sequence Prediction using Short Data Records"
 * @author Simon
 *
 */
//TODO: Better citation
public class VOMM<T> {
	private Trie<T> trie;
	private LinkedList<T> memory;
	private int markovOrder;
	private int trieDepth;
	private ArrayList<T> predictions; //List off predictions by each markovorder model
	private SimpleMatrix weightVector;
	private double learningRate;
	private boolean learning;
	private T currentSymbol;
	private HashMap<T, Double> probabilityDistribution;
	
	public VOMM(int markovOrder, double learningRate) {
		trie = new Trie<T>();
		this.markovOrder = markovOrder;
		trieDepth = markovOrder + 1;
		this.learningRate = learningRate;

		weightVector = new SimpleMatrix(1, markovOrder);
		weightVector.set(0, 1);
		//weightVector.set((double) 1 / markovOrder);
		predictions = new ArrayList<T>();
		memory  = new LinkedList<T>();
		
		for (int i = 0; i < markovOrder; i++){
			predictions.add(null);
		}
		
		learning = true;
	}
	
	/**
	 * Predicts the next symbol in the sequence.
	 * Call {@link #addSymbol(Object)} before calling this to add a new symbol to the memory.
	 * @param symbol
	 * @return
	 */
	public T predict(){
		
		if (learning){
			SimpleMatrix errorVector = new SimpleMatrix(1, markovOrder);
			errorVector.set(1);
			boolean clean;
			boolean[] notUsingModel = new boolean[markovOrder];
			int modelsUsed = markovOrder;
			do{
				clean = true;
				for (int i = 0; i < markovOrder; i++){
					//Compute error
					if (notUsingModel[i]){
						errorVector.set(i, 0); //We don't want it error to influence the differential computation
					}else{
						T prediction = predictions.get(i);
						errorVector.set(i, 1);
						if (prediction != null){
							if (prediction.equals(currentSymbol)) {
								errorVector.set(i, 0);
							}
						}
					}
				}
					
				//Compute differential error vector
				SimpleMatrix differentialErrorVector = new SimpleMatrix(errorVector);
				double errorSum = errorVector.elementSum();
				double delta = (double) 1 / modelsUsed * errorSum;
				differentialErrorVector = differentialErrorVector.minus(delta);
					
				//Update weight vector
				for (int j = 0; j < weightVector.getNumElements(); j++){
					if (!notUsingModel[j]){
						double newValue = weightVector.get(j) - learningRate * differentialErrorVector.get(j);
						if (newValue <= 0) {
							newValue = 0;
							clean = false;
							notUsingModel[j] = true;
							modelsUsed--;
						}
						if (newValue > 1) newValue = 1;
						weightVector.set(j, newValue);
					}
				}				
				
			} while (!clean);
			
			//Normalize weight vector
			double sum = weightVector.elementSum();
			weightVector = weightVector.divide(sum);
		}

		//Do prediction
		probabilityDistribution = new HashMap<T, Double>();
		T predictedNextSymbol = null;
		double maxCombinedProbability = Double.NEGATIVE_INFINITY;
		LinkedList<T> context = new LinkedList<T>();
		for ( int i = 0; i < markovOrder; i++){ //Make sure we don't get out of bounds in the start of the training
			if (memory.size() >= i + 1){
				double maxMarkovProbability = Double.NEGATIVE_INFINITY;			
				T c = memory.get(memory.size() - (i + 1));
				context.addFirst(c);
				HashMap<T, TrieNode<T>> possibleNextNodes = trie.findChildrenOfLastNode(context);
				if (possibleNextNodes != null){
					for (T s : possibleNextNodes.keySet()){
						double probability = possibleNextNodes.get(s).getProbability();
						
						if (probability > maxMarkovProbability){
							maxMarkovProbability = probability;
							predictions.set(i, s);
						}
						
						probability = probability * weightVector.get(i);
						Double d = probabilityDistribution.get(s);
						double oldValue = 0;
						if (d != null){
							oldValue = d.doubleValue();
						}
						probability += oldValue;
						if (probability > maxCombinedProbability){
							maxCombinedProbability = probability;
							predictedNextSymbol = s;
						}
						probabilityDistribution.put(s, probability);
					}
				}	
			}
		}

		return predictedNextSymbol;
	}
	
	/**
	 * Adds the given symbol to the memory and adds all possible contexts in the memory to the trie
	 * @param symbol
	 */
	public void addSymbol(T symbol){		
		currentSymbol = symbol;
		memory.addLast(symbol);			
		if (memory.size() > trieDepth) memory.removeFirst();
		
		if (learning) {
			//Update trie based on all contexts in the memory to the trie
			LinkedList<T> context = new LinkedList<T>();
			for (int j = 1; j <= memory.size(); j++){
				T c = memory.get(memory.size() - j);
				context.addFirst(c);
				trie.add(context);
			}
		}
	}
	
	/**
	 * Calculates the entropy of the current prediction.
	 * Using log base e
	 * @return
	 */
	public double calculateEntropy(){
		double sum = 0;
		for (T s : probabilityDistribution.keySet()){
			double d = probabilityDistribution.get(s);
			if (d != 0) sum += d * Math.log(d);
		}
		return -sum;
	}
	
	public HashMap<T, Double> getCurrentProbabilityDistribution(){
		return probabilityDistribution;
	}
	
	public void printTrie(){
		trie.printTrie(memory.size());
	}
	
	public void setLearning(boolean learning){
		this.learning = learning;
	}
	
	public void flushMemory(){
		memory = new LinkedList<T>();
	}
}
