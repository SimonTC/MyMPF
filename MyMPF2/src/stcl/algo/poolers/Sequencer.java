package stcl.algo.poolers;

import java.util.LinkedList;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import dk.stcl.core.basic.containers.SomNode;
import stcl.algo.predictors.Predictor_VOMM;

public class Sequencer {
	private Predictor_VOMM predictor;
	private LinkedList<SimpleMatrix> currentSequence;
	private RSOM rsom; 	
	private SimpleMatrix prediction;
	private SimpleMatrix cumulativeProbabilityOfHavingSeenInput; 
	
	private double entropyThreshold;
	private double currentEntropy;
	private int markovOrder;
	
	private LinkedList<Double> entropyHistory;
	private int entropyHistoryLength;
	private boolean learning;
	private double predictionLearningRate;
	
	
	public Sequencer(int markovOrder, double predictionLearningRate, int poolerMapSize, int inputLength, Random rand, double poolerLearningRate, double activationCodingFactor, double stdDev, double decayFactor ) {
		predictor = new Predictor_VOMM(markovOrder, predictionLearningRate);
		rsom = new RSOM(poolerMapSize, inputLength, rand, poolerLearningRate, activationCodingFactor, stdDev, decayFactor);
		this.markovOrder = markovOrder;
		cumulativeProbabilityOfHavingSeenInput = new SimpleMatrix(1, inputLength);
		
		entropyHistoryLength = markovOrder * 10; //Arbitrarily chosen number
		entropyHistory = new LinkedList<Double>();
		learning = true;
		this.predictionLearningRate = predictionLearningRate;
		currentSequence = new LinkedList<SimpleMatrix>();
	}
	
	/**
	 * 
	 * @param inputVector probability vector over the probability of each model being active at time t
	 * @return Probability matrix over the probability of each temporal group being the group that started the current sequence. Returns null if sequence isn't ended
	 */
	public SimpleMatrix feedForward(SimpleMatrix inputVector){
		
		//Save inputvector in sequence memory
		currentSequence.addLast(inputVector);
		
		if (currentSequence.size() <= markovOrder){
			cumulativeProbabilityOfHavingSeenInput = cumulativeProbabilityOfHavingSeenInput.plus(inputVector);
		}
		
		//Update temporal pooler
		rsom.step(inputVector);
		
		//Predict next input
		if (learning){
			prediction = predictor.predict(inputVector, predictionLearningRate, learning);
		} else {
			prediction = predictor.predict(inputVector, 0, false);
		}
		
		//Calculate entropy
		currentEntropy = predictor.calculateEntropy();
		entropyHistory.addLast(currentEntropy);
		if (entropyHistory.size() > entropyHistoryLength) entropyHistory.removeFirst();
		entropyThreshold = calculateAverage(entropyHistory);
		
		SimpleMatrix output = null; //Shoudl be set to null but now we just use activation
		output = rsom.computeActivationMatrix();
		if (currentEntropy > entropyThreshold){
			//Calculate probabilities of each of the temporal groups being the ones that started the sequence
				//Done by calculating the certainty of having observed each of the inputs in the start and multiplying it with the frequency in the temporal groups
				int length = currentSequence.size() > markovOrder ? markovOrder : currentSequence.size();
				SimpleMatrix inputProbability = cumulativeProbabilityOfHavingSeenInput.divide(length);
			
			//Find probability that each temporal group started the current sequence by looking at the first k inputs in the current sequence and 
			//multiplying them with the frequencies of the groups in the temporal pooler
			SimpleMatrix temporalGroupCertaintyMatrix = new SimpleMatrix(rsom.getHeight(), rsom.getWidth());
			for (SomNode temporalGroup : rsom.getNodes()){
				SimpleMatrix inputFrequencies = temporalGroup.getVector();
				SimpleMatrix inputProbabilityMatrix = inputFrequencies.elementMult(inputProbability);
				double groupCertainty = 1;
				for (double d : inputProbabilityMatrix.getMatrix().data){
					groupCertainty *= d; //TODO: This might lead to very low values
				}
				temporalGroupCertaintyMatrix.set(temporalGroup.getId(), groupCertainty);
			}
			
			output = temporalGroupCertaintyMatrix;
			
			//OR: return current activation matrix from pooler signifying which temporal group we are in now
			//output = rsom.computeActivationMatrix();
						
			//Reset current sequence and set last input to start of new sequence (not sure about last part)
			currentSequence = new LinkedList<SimpleMatrix>(); 
			cumulativeProbabilityOfHavingSeenInput = new SimpleMatrix(inputVector.numRows(), inputVector.numCols());
			predictor.flush(); //Not sure about this
			//currentSequence.addFirst(inputVector);
		}		
		
		return output;
	}
	
	/**
	 * 
	 * @param inputMatrix matrix containing probabilities of entering the different temporal groups next time step
	 * @return probability of seeing the different spatial models
	 */
	public SimpleMatrix feedBackward(SimpleMatrix inputMatrix){
		
		SimpleMatrix output = prediction;
				
		if (currentEntropy > entropyThreshold){
			//Bias current prediction by the information given in the fb input matrix
			
			//Choose most probable temporal group 
			int maxID = findIDOfMaxElement(inputMatrix);
			
			//Get weigths of most probable model
			SimpleMatrix weights = rsom.getNode(maxID).getVector();
			
			//Bias current prediction with weights from next temporal group
			//TODO: Multiplying or adding?
			output = prediction.elementMult(weights);
			
		} 
		
		return output;
	}
	
	private double calculateAverage(LinkedList<Double> list){
		double max = 0;
		for (Double d : list){
			max += d;
		}
		
		double avg = max / (double) list.size();
		return avg;
	}
	
	private int findIDOfMaxElement(SimpleMatrix m){
		double max = Double.NEGATIVE_INFINITY;
		int maxID = -1;
		for (int i = 0; i < m.getNumElements(); i++){
			double value = m.get(i);
			if (value > max){
				maxID = i;
				max = value;
			}
		}		

		return maxID;
		
	}
	
	public void setLearning(boolean learning){
		this.learning = learning;
	}
	
	public RSOM getRsom(){
		return rsom;
	}
	
}
