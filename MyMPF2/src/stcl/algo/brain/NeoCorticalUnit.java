package stcl.algo.brain;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.poolers.NewSequencer;
import stcl.algo.poolers.SOM;
import stcl.algo.poolers.SpatialPooler;
import stcl.algo.poolers.TemporalPooler;
import stcl.algo.predictors.Decider;
import stcl.algo.predictors.Predictor;
import stcl.algo.predictors.Predictor_VOMM;
import stcl.algo.util.Normalizer;
import dk.stcl.core.basic.containers.SomNode;

public class NeoCorticalUnit{
	
	private SpatialPooler spatialPooler;
	//private Predictor_VOMM predictor;
	private Decider decider;
	private SimpleMatrix biasMatrix;
	private SimpleMatrix predictionMatrix;
	
	private SimpleMatrix ffOutput;
	private SimpleMatrix fbOutput;
	
	private int ffInputVectorSize;
	private int spatialMapSize;
	private int temporalMapSize;
	
	private boolean needHelp;
	private double predictionEntropy;
	private double entropyThreshold; //The exponential moving average of the prediction entropy
	private double entropyDiscountingFactor;
	
	private boolean useMarkovPrediction;
	private boolean active;

	private boolean entropyThresholdFrozen;
	private boolean biasBeforePredicting;
	private boolean useBiasedInputInSequencer;
	
	private NewSequencer sequencer;
	private boolean noTemporal;
	
	
	public NeoCorticalUnit(Random rand, int ffInputLength, int spatialMapSize, int temporalMapSize, double initialPredictionLearningRate, boolean useMarkovPrediction, int markovOrder) {
		this(rand, ffInputLength, spatialMapSize, temporalMapSize, initialPredictionLearningRate, useMarkovPrediction, markovOrder, false);
	}
	/**
	 * 
	 * @param rand
	 * @param ffInputLength
	 * @param spatialMapSize
	 * @param temporalMapSize
	 * @param initialPredictionLearningRate
	 * @param useMarkovPrediction
	 * @param markovOrder
	 */
	public NeoCorticalUnit(Random rand, int ffInputLength, int spatialMapSize, int temporalMapSize, double initialPredictionLearningRate, boolean useMarkovPrediction, int markovOrder, boolean noTemporal) {
		double decay = calculateDecay(markovOrder,0.01);// 1.0 / markovOrder);
		entropyDiscountingFactor = decay; //TODO: Does this make sense?
		//TODO: All parameters should be handled in parameter file
		
		spatialPooler = new SpatialPooler(rand, ffInputLength, spatialMapSize, 0.1, Math.sqrt(spatialMapSize), 0.125); //TODO: Move all parameters out
		if (!noTemporal) {
			sequencer = new NewSequencer(markovOrder, temporalMapSize, spatialMapSize * spatialMapSize);
			this.temporalMapSize = temporalMapSize;
		} else {
			this.temporalMapSize = spatialMapSize;
		}
		decider = new Decider(markovOrder, initialPredictionLearningRate, rand, 1, 1, 0.3, spatialMapSize);
		//predictor = new Predictor_VOMM(markovOrder, initialPredictionLearningRate, rand);
		biasMatrix = new SimpleMatrix(spatialMapSize, spatialMapSize);
		biasMatrix.set(1);
		ffOutput = new SimpleMatrix(this.temporalMapSize, this.temporalMapSize);
		fbOutput = new SimpleMatrix(1, ffInputLength);
		ffInputVectorSize = ffInputLength;
		this.useMarkovPrediction = useMarkovPrediction;
		this.spatialMapSize = spatialMapSize;
		
		needHelp = false;
		entropyThreshold = 0;
		entropyThresholdFrozen = false;
		biasBeforePredicting = false;
		useBiasedInputInSequencer = false;
		this.noTemporal = noTemporal;
	}
	
	public SimpleMatrix feedForward(SimpleMatrix inputVector){
		return this.feedForward(inputVector, 0);
	}
	
	public SimpleMatrix feedForward(SimpleMatrix inputVector, double reward){
		//Test input
		if (!inputVector.isVector()) throw new IllegalArgumentException("The feed forward input to the neocortical unit has to be a vector");
		if (inputVector.numCols() != ffInputVectorSize) throw new IllegalArgumentException("The feed forward input to the neocortical unit has to be a 1 x " + ffInputVectorSize + " vector");
		
		active = true;
		
		//Spatial classification
		SimpleMatrix spatialFFOutputMatrix = spatialPooler.feedForward(inputVector);
		
		//Bias output
		SimpleMatrix biasedOutput = biasMatrix(spatialFFOutputMatrix, biasMatrix);
		
		//Predict next spatialFFOutputMatrix
		if (useMarkovPrediction){
			decider.giveExternalReward(reward);
			if (biasBeforePredicting) {
				predictionMatrix = decider.predict(biasedOutput);
			} else {
				predictionMatrix = decider.predict(spatialFFOutputMatrix);
			}
		} 		
		
		predictionEntropy = calculateEntropy(predictionMatrix);
		
		if (predictionEntropy > entropyThreshold) needHelp = true;
		if (!entropyThresholdFrozen){
			entropyThreshold = entropyDiscountingFactor * predictionEntropy + (1-entropyDiscountingFactor) * entropyThreshold;
		}
		
		ffOutput = biasedOutput;
		
		if (!noTemporal) {
			//Transform spatial output matrix to vector
			double[] spatialFFOutputDataVector;
			if (useBiasedInputInSequencer){
				spatialFFOutputDataVector = biasedOutput.getMatrix().data;		
			} else {
				spatialFFOutputDataVector = spatialFFOutputMatrix.getMatrix().data;	
			}
			SimpleMatrix temporalFFInputVector = new SimpleMatrix(1, spatialFFOutputDataVector.length);
			temporalFFInputVector.getMatrix().data = spatialFFOutputDataVector;
			
			ffOutput = sequencer.feedForward(temporalFFInputVector, spatialPooler.getSOM().getBMU().getId(), needHelp);
		} 
		
		return ffOutput;
	}
	
	
	/**
	 * 
	 * @param inputMatrix
	 * @param correlationMatrix
	 * @return
	 */
	public SimpleMatrix feedBackward(SimpleMatrix inputMatrix){
		//Test input
		//if (inputMatrix.isVector()) throw new IllegalArgumentException("The feed back input to the neocortical unit has to be a matrix");
		if (inputMatrix.numCols() != temporalMapSize || inputMatrix.numRows() != temporalMapSize) throw new IllegalArgumentException("The feed back input to the neocortical unit has to be a " + temporalMapSize + " x " + temporalMapSize + " matrix");

		if (needHelp){
			//Normalize
			SimpleMatrix normalizedInput = normalize(inputMatrix);
			
			if (!noTemporal){
				//Selection of best temporal model
				SimpleMatrix temporalPoolerFBOutput = sequencer.feedBackward(normalizedInput);
				
				//Normalize
				SimpleMatrix normalizedTemporalPoolerFBOutput = normalize(temporalPoolerFBOutput);
				
				//Transformation into matrix
				normalizedTemporalPoolerFBOutput.reshape(spatialMapSize, spatialMapSize); //TODO: Is this necessary?
				
				//Combine FB output from temporal pooler with bias and prediction (if enabled)
				biasMatrix = normalizedTemporalPoolerFBOutput;
			} else {
				biasMatrix = inputMatrix;
			}
			
			//biasMatrix = biasMatrix.plus(1, predictionMatrix);
			biasMatrix = biasMatrix.elementMult(predictionMatrix);
			
			
			biasMatrix = normalize(biasMatrix);			
			
		} else {
			biasMatrix = predictionMatrix;
		}
		
		//biasMatrix = biasMatrix.plus(0.1 / biasMatrix.getNumElements()); //Add small uniform mass
		
		SimpleMatrix biasedTemporalFBOutput = biasMatrix;
		
		//Selection of best spatial mode
		SimpleMatrix spatialPoolerFBOutputVector = spatialPooler.feedBackward(biasedTemporalFBOutput);
		
		fbOutput = spatialPoolerFBOutputVector;
		
		needHelp = false;
		active = false;
		
		return fbOutput;
	}
	
	/**
	 * Calculate the decay which will have the first input in a sequence have a t least minInfluence influence on the difference vector in the rsom
	 * @param memoryLength
	 * @param minInfluence
	 * @return
	 */
	private double calculateDecay(int memoryLength, double minInfluence){
		double decay = 1 - Math.pow(minInfluence, 1.0 / memoryLength);
		return decay;
 	}
	
	private SimpleMatrix biasMatrix(SimpleMatrix matrixToBias, SimpleMatrix biasMatrix){
		SimpleMatrix biasedMatrix = matrixToBias.elementMult(biasMatrix);
		biasedMatrix = Normalizer.normalize(biasedMatrix);
		return biasedMatrix;
	}
	
	private double calculateEntropy(SimpleMatrix m){
		double sum = 0;
		for (Double d : m.getMatrix().data){
			if (d != 0) sum += d * Math.log(d);
		}
		return -sum;
	}
	
	private SimpleMatrix normalize(SimpleMatrix m){
		return Normalizer.normalize(m);
	}
	
	public void flush(){
		biasMatrix.set(1);
		decider.flush();
		if (sequencer != null) sequencer.reset();
	}
	
	public void setLearning(boolean learning){
		spatialPooler.setLearning(learning);
		decider.setLearning(learning);
		if (sequencer != null) sequencer.setLearning(learning);
	}

	public SpatialPooler getSpatialPooler() {
		return spatialPooler;
	}

	public SimpleMatrix getFFOutput() {
		return ffOutput;
	}

	public SimpleMatrix getFBOutput() {
		return fbOutput;
	}
	
	public void sensitize(int iteration){
		spatialPooler.sensitize(iteration);
	}
	
	public int findTemporalBMUID(){
		int maxID = -1;
		double maxValue = Double.NEGATIVE_INFINITY;
		
		for (int i = 0; i < ffOutput.getNumElements(); i++){
			double v = ffOutput.get(i);
			if (v > maxValue){
				maxValue = v;
				maxID = i;
			}
		}
		
		return maxID;
	}
	
	/*
	public Predictor getPredictor(){
		return predictor;
	}
*/
	public SOM getSOM() {
		return spatialPooler.getSOM();
	}

	public void printPredictionModel() {
		decider.printModel();
		
	}

	public boolean needHelp() {
		return needHelp;
	}

	public double getEntropy() {
		return predictionEntropy;
	}
	
	public double getEntropyThreshold() {
		return entropyThreshold;
	}
	
	public boolean active(){
		return active;
	}
	
	public NewSequencer getSequencer(){
		return sequencer;
	}

	/**
	 * @param entropyThresholdFrozen the entropyThresholdFrozen to set
	 */
	public void setEntropyThresholdFrozen(boolean entropyThresholdFrozen) {
		this.entropyThresholdFrozen = entropyThresholdFrozen;
	}
	
	public void setBiasBeforePrediction(boolean flag){
		biasBeforePredicting = flag;
	}

	public void setUseBiasedInputInSequencer(boolean useBiasedInputInSequencer) {
		this.useBiasedInputInSequencer = useBiasedInputInSequencer;
	}

	/**
	 * @return the temporalMapSize
	 */
	public int getTemporalMapSize() {
		return temporalMapSize;
	}
	

}
