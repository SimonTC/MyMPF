package stcl.algo.predictors;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.util.Normalizer;

/**
 * VOMM that is capable of working with fuzzy inputs (probability distributions as inputs)
 * @author Simon
 *
 * @param <T>
 */
public class Predictor_VOMM implements Predictor{
	
	private VOMM<Integer> vomm;
	private int markovOrder;
	private LinkedList<Double> inputProbabilities;
	private int predictedNextSymbol;
	private SimpleMatrix probabilityMatrix;
	private Random rand;

	public Predictor_VOMM(int markovOrder, double learningRate, Random rand) {
		vomm = new VOMM<Integer>(markovOrder, learningRate);
		inputProbabilities = new LinkedList<Double>();
		this.markovOrder = markovOrder;
		predictedNextSymbol = -1;
		this.rand = rand;
		
	}

	@Override
	/**
	 * Returns a probability matrix over all possible symbols.
	 * Contains the probability of seeing a symbol given the current context.
	 * 
	 */
	public SimpleMatrix predict(SimpleMatrix inputMatrix) {	
			probabilityMatrix = new SimpleMatrix(inputMatrix.numRows(), inputMatrix.numCols());
			
			//Find the id of that symbol which we are most probably observing now
			int mostProbableInputID = findMostProbableInput(inputMatrix);	
			
			///Add probability that the most probable symbol is indeed the symbol we are observing
			inputProbabilities.addLast(inputMatrix.get(mostProbableInputID));
			if (inputProbabilities.size() > markovOrder) inputProbabilities.removeFirst();
			
			//Predict the id of the next input
			vomm.addSymbol(mostProbableInputID);
			Integer prediction = vomm.predict();
			
			//Calculate the probability distribution over all possible symbols
			if (prediction == null){
				probabilityMatrix.set(1);
				predictedNextSymbol = 0; //Set prediction to some value.
			} else {
				predictedNextSymbol = prediction.intValue();
				double sequenceProbability = calculateProbabilityOfNodeSequence();
				
				double scaling = (1-sequenceProbability) * ((double) 1 / inputMatrix.getNumElements());
				
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
		for (int i = 0; i < inputProbabilities.size(); i++){
			d = d * inputProbabilities.get(i);
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

	@Override
	public void flush() {
		vomm.flushMemory();		
	}

	@Override
	public SimpleMatrix getConditionalPredictionMatrix() {
		return probabilityMatrix;
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

	@Override
	public void setLearning(boolean learning) {
		vomm.setLearning(learning);
		
	}

	@Override
	public void setLEarningRate(double learningRate) {
		vomm.setLearningRate(learningRate);
		
	}
	

}
