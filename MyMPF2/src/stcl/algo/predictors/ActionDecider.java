package stcl.algo.predictors;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.util.Normalizer;
import stcl.algo.util.trie.TrieNode;

/**
 * VOMM that is capable of working with fuzzy inputs (probability distributions as inputs)
 * @author Simon
 *
 * @param <T>
 */
public class ActionDecider{
	
	private VOMM<Integer> vomm;
	private int markovOrder;
	private LinkedList<Double> stateProbabilities;
	private int predictedNextSymbol;
	private SimpleMatrix actionMatrix;
	private SimpleMatrix probabilityMatrix;
	private Random rand;
	private HashMap<Integer, TrieNode<Integer>> actionsFromCurrentState;

	public ActionDecider(int markovOrder, double learningRate, Random rand, int actionMatrixSize) {
		int markovWithActions = markovOrder * 2; //Tree should be double as deep as markov order to account for actions
		vomm = new VOMM<Integer>(markovWithActions, learningRate);
		stateProbabilities = new LinkedList<Double>();
		this.markovOrder = markovOrder;
		predictedNextSymbol = -1;
		this.rand = rand;
		actionMatrix = new SimpleMatrix(actionMatrixSize, actionMatrixSize);
	}

	/**
	 * Returns the ID of the action it would be best to perform to get out of the current state
	 * @param currentState
	 * @param actionToGetHere
	 * @return
	 */
	public SimpleMatrix chooseNextAction(SimpleMatrix currentState) {	
			probabilityMatrix = new SimpleMatrix(currentState.numRows(), currentState.numCols()); //Reset the matrix of probabilities of seeing states next turn
			
			//Find the id of that symbol which we are most probably observing now
			int mostProbableStateID = findMostProbableInput(currentState);	
			
			///Add probability that the most probable symbol is indeed the symbol we are observing
			stateProbabilities.addLast(currentState.get(mostProbableStateID));
			if (stateProbabilities.size() > markovOrder) stateProbabilities.removeFirst();
			
			//Add action and state to the tree
			vomm.addSymbol(mostProbableStateID);
			
			//Collect the currently observed node sequence
			LinkedList<TrieNode<Integer>> currentSequence =  vomm.getCurrentNodeSequence();
			
			//Collect possible actions from the current state
			TrieNode<Integer> currentStateNode = currentSequence.peekFirst();
			actionsFromCurrentState = currentStateNode.getChildren();
			
			//Create weight matrix for the different possible actions
			actionMatrix.set(0);
			for (TrieNode<Integer> n : actionsFromCurrentState.values()){
				actionMatrix.set(n.getSymbol(), n.getReward());
			}
			actionMatrix = Normalizer.normalize(actionMatrix);
			
			return actionMatrix;
			
	}
	
	/**
	 * Returns a probability matrix over all possible state we could end up in next turn by doing the given action in the state we are currently in
	 * @param chosenAction
	 * @return
	 */
	public SimpleMatrix predictNextState(int chosenAction) {			
		//Predict the id of the next input
		vomm.addSymbol(chosenAction);
		Integer prediction = vomm.predict();
		
		//Calculate the probability distribution over all possible symbols
		if (prediction == null){
			probabilityMatrix.set(1);
			predictedNextSymbol = 0; //Set prediction to some value.
		} else {
			predictedNextSymbol = prediction.intValue();
			double sequenceProbability = calculateProbabilityOfNodeSequence();
			
			double scaling = (1-sequenceProbability) * ((double) 1 / probabilityMatrix.getNumElements());
			
			HashMap<Integer, Double> probabilityOfSymbols =  vomm.getCurrentProbabilityDistribution();
			
			for (int i : probabilityOfSymbols.keySet()){
				double symbolProbability = probabilityOfSymbols.get(i);
				symbolProbability = symbolProbability * sequenceProbability + scaling;
				probabilityMatrix.set(i, symbolProbability);				
			}
		}			
		
		//Normalize
		probabilityMatrix = Normalizer.normalize(probabilityMatrix);
		
		return probabilityMatrix;
	}
	
	private double calculateProbabilityOfNodeSequence(){
		double d = 1;
		for (int i = 0; i < stateProbabilities.size(); i++){
			d = d * stateProbabilities.get(i);
		}
		return d;
	}
	
	private int findInputByRoulette(SimpleMatrix probabilityMatrix){
		double v = rand.nextDouble();
		double sum = 0;
		boolean found = false;
		int id = -1;
		int i = 0;
		do{
			double d = probabilityMatrix.get(i);
			sum += d;
			if (sum >= v){
				found = true;
				id = 1;
			}
			i++;
		} while (!found && i < probabilityMatrix.getNumElements());
		return id;
	}
	
	/**
	 * Finds the id of the element with the highest value
	 * @param probabilityMatrix
	 * @return
	 */
	private int findMostProbableInput(SimpleMatrix probabilityMatrix){
		double max = Double.NEGATIVE_INFINITY;
		int maxID = -1;
		for (int i = 0; i < probabilityMatrix.getNumElements(); i++){
			double value = probabilityMatrix.get(i);
			if (value > max){
				maxID = i;
				max = value;
			}
		}		
		return maxID;
		
	}

	public void flush() {
		vomm.flushMemory();		
	}

	public SimpleMatrix getActionMatrix() {
		return actionMatrix;
	}
	
	public int getNextPredictedSymbol(){
		return predictedNextSymbol;
	}
	
	public double calculateEntropy(){
		return vomm.calculateEntropy();
	}
	
	public void printModel(){
		vomm.printTrie();
	}

	public void setLearning(boolean learning) {
		vomm.setLearning(learning);
		
	}

	public void setLEarningRate(double learningRate) {
		vomm.setLearningRate(learningRate);
		
	}
	

}
