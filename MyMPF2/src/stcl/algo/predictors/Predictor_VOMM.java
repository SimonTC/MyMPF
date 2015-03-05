package stcl.algo.predictors;

import java.util.HashMap;
import java.util.LinkedList;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.util.Normalizer;
import stcl.algo.util.trie.TrieNode;

/**
 * VOMM that is capable of working with fuzzy inputs (probability distribuions as inputs
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

	public Predictor_VOMM(int markovOrder, double learningRate) {
		vomm = new VOMM<Integer>(markovOrder, learningRate);
		inputProbabilities = new LinkedList<Double>();
		this.markovOrder = markovOrder;
		predictedNextSymbol = -1;
		
	}

	@Override
	public SimpleMatrix predict(SimpleMatrix inputMatrix, double curLearningRate, boolean associate) {	
			vomm.setLearning(associate);
			vomm.setLearningRate(curLearningRate);
			probabilityMatrix = new SimpleMatrix(inputMatrix.numRows(), inputMatrix.numCols());
			inputProbabilities.addLast(inputMatrix.elementMaxAbs());
			if (inputProbabilities.size() > markovOrder) inputProbabilities.removeFirst();
			
			int mostProbableInput = findMostProbableInput(inputMatrix);			
			vomm.addSymbol(mostProbableInput);
			Integer prediction = vomm.predict();
			if (prediction == null){
				probabilityMatrix.set(1);
				predictedNextSymbol = 0; //Set prediction some value.
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
	
	

}
