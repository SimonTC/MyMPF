package stcl.algo.predictors;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.predictors.trie.Trie;
import stcl.algo.predictors.trie.TrieNode;

/**
 * VOMM is an implementation of the Adaptive Model Averaged PPM algorithm from 
 * S. Pulliyakode and S. Kalyani, �A Modified PPM Algorithm for Online Sequence Prediction using Short Data Records,� IEEE Commun. Lett., vol. 7798, no. c, pp. 1�1, 2014.
 * @author Simon
 *
 */
public class VOMM<T> implements Serializable{
	private static final long serialVersionUID = 1L;
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
	private LinkedList<TrieNode<T>> currentNodeSequence;
	
	
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
	 * @return the next most probable symbol
	 */
	public T predict(){
		
		if (learning){
			doLearning();
		}

		//Do prediction
		probabilityDistribution = new HashMap<T, Double>();
		T predictedNextSymbol = null;
		double maxCombinedProbability = Double.NEGATIVE_INFINITY;
		LinkedList<T> context = new LinkedList<T>();
		for ( int i = 0; i < markovOrder; i++){ //Make sure we don't get out of bounds in the start of the training
			if (memory.size() >= i + 1){
				double maxMarkovProbability = Double.NEGATIVE_INFINITY;			
				//Update context
				T c = memory.get(memory.size() - (i + 1));
				context.addFirst(c);
				
				//Find the set of possible next symbols
				HashMap<T, TrieNode<T>> possibleNextNodes = trie.findChildrenOfLastNode(context);
				if (possibleNextNodes != null){
					for (T s : possibleNextNodes.keySet()){ //Calculate probability of each symbol given the current context
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
	 * Performs the learning algorithm for the VOMM and updates the weight vector
	 */
	private void doLearning(){
		SimpleMatrix errorVector;
		boolean clean;
		boolean[] notUsingModel = new boolean[markovOrder];
		int modelsUsed = markovOrder;
		do{
			clean = true;
			errorVector = computeErrorVector(notUsingModel);
		
			SimpleMatrix differentialErrorVector = computeDifferentialErrorVector(errorVector, modelsUsed);
				
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
	
	/**
	 * Computes a binary error vector for each of the markov models
	 * @param listOfModelsNotInUse boolean list of same length as the number of markov models. If the entry for a particular markov model is true, its error is set to zero such that it doesn't influence later calculations
	 * @return
	 */
	private SimpleMatrix computeErrorVector(boolean[] listOfModelsNotInUse){
		SimpleMatrix errorVector = new SimpleMatrix(1, markovOrder);
		errorVector.set(1);
		for (int i = 0; i < markovOrder; i++){
			//Compute error
			if (listOfModelsNotInUse[i]){
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
		
		return errorVector;
	}
	
	/**
	 * Calculates the differential error vector
	 * @param errorVector vector of error values
	 * @param modelsUsed number of models currently in use
	 * @return 
	 */
	private SimpleMatrix computeDifferentialErrorVector(SimpleMatrix errorVector, int modelsUsed){
		SimpleMatrix differentialErrorVector = new SimpleMatrix(errorVector);
		double errorSum = errorVector.elementSum();
		double delta = (double) 1 / modelsUsed * errorSum;
		differentialErrorVector = differentialErrorVector.minus(delta);
		return differentialErrorVector;
	}
	
	/**
	 * Adds the given symbol to the memory and adds all possible contexts in the memory to the trie.
	 * Updates the current node sequence if learning.
	 * @param symbol
	 */
	public void addSymbol(T symbol, double reward){		
		currentSymbol = symbol;
		memory.addLast(symbol);			
		if (memory.size() > trieDepth) memory.removeFirst();
		
		LinkedList<T> context = new LinkedList<T>();
		if (learning) {
			//Update trie based on all contexts in the memory to the trie
			for (int j = 1; j <= memory.size(); j++){
				T c = memory.get(memory.size() - j);
				context.addFirst(c);
				currentNodeSequence = trie.add(context, reward);
			}
		} else {
			currentNodeSequence = trie.findNodeSequence(memory);
		}
	}
	
	public void addSymbol(T symbol){
		this.addSymbol(symbol, 0);
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
	
	/**
	 * Return a map containing all known symbols together with the probabilities of seeing the symbols at next timestep
	 * @return
	 */
	public HashMap<T, Double> getCurrentProbabilityDistribution(){
		return probabilityDistribution;
	}
	
	/**
	 * Prints the trie
	 */
	public void printTrie(){
		trie.printTrie(memory.size());
	}
	
	public void setLearning(boolean learning){
		this.learning = learning;
	}
	
	public void setLearningRate(double newLearningRate){
		learningRate = newLearningRate;
	}
	
	/**
	 * Clear out the memory of the VOMM.
	 * The Trie is not changed
	 */
	public void flushMemory(){
		memory = new LinkedList<T>();
	}
	
	public LinkedList<T> getMemory(){
		return memory;
	}
	
	/**
	 * Returns the current active node sequence.
	 * This first node in the list corresponds to the last node in the sequence
	 * @return
	 */
	public LinkedList<TrieNode<T>> getCurrentNodeSequence(){
		return currentNodeSequence;
	}
}
