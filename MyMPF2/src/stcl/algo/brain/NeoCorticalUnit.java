package stcl.algo.brain;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.poolers.SOM;
import stcl.algo.poolers.SpatialPooler;
import stcl.algo.poolers.TemporalPooler;
import stcl.algo.predictors.Predictor;
import stcl.algo.predictors.Predictor_VOMM;
import stcl.algo.util.Normalizer;
import dk.stcl.core.basic.containers.SomNode;

public class NeoCorticalUnit implements NU{
	
	private SpatialPooler spatialPooler;
	private TemporalPooler temporalPooler;
	private Predictor_VOMM predictor;
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
	
	/**
	 * 
	 * @param rand
	 * @param maxIterations
	 * @param ffInputLength
	 * @param spatialMapSize
	 * @param temporalMapSize
	 * @param initialPredictionLearningRate
	 * @param useMarkovPrediction
	 * @param decayFactor
	 */
	public NeoCorticalUnit(Random rand, int ffInputLength, int spatialMapSize, int temporalMapSize, double initialPredictionLearningRate, boolean useMarkovPrediction, int markovOrder) {
		double decay = calculateDecay(markovOrder, 0.01);
		entropyDiscountingFactor = 0.3;//decay; //TODO: Does this make sense?
		//TODO: All parameters should be handled in parameter file
		spatialPooler = new SpatialPooler(rand, ffInputLength, spatialMapSize, 0.1, 2, 0.125); //TODO: Move all parameters out
		temporalPooler = new TemporalPooler(rand, spatialMapSize * spatialMapSize, temporalMapSize, 0.1, 5, 0.125, decay); //TODO: Move all parameters out
		predictor = new Predictor_VOMM(markovOrder, initialPredictionLearningRate, rand);
		biasMatrix = new SimpleMatrix(spatialMapSize, spatialMapSize);
		biasMatrix.set(1);
		ffOutput = new SimpleMatrix(temporalMapSize, temporalMapSize);
		fbOutput = new SimpleMatrix(1, ffInputLength);
		ffInputVectorSize = ffInputLength;
		this.useMarkovPrediction = useMarkovPrediction;
		this.spatialMapSize = spatialMapSize;
		this.temporalMapSize = temporalMapSize;
		needHelp = false;
		entropyThreshold = 0;
	}
	
	/**
	 * Calculate the decay which will have the first input in a sequence have a t least minInfluence influence on the difference vector in the rsom
	 * @param memoryLength
	 * @param minInfluence
	 * @return
	 */
	private double calculateDecay(int memoryLength, double minInfluence){
		double decay = 1 - Math.pow(minInfluence, 1 / memoryLength);
		return decay;
 	}
	
	public SimpleMatrix feedForward(SimpleMatrix inputVector){
		//Test input
		if (!inputVector.isVector()) throw new IllegalArgumentException("The feed forward input to the neocortical unit has to be a vector");
		if (inputVector.numCols() != ffInputVectorSize) throw new IllegalArgumentException("The feed forward input to the neocortical unit has to be a 1 x " + ffInputVectorSize + " vector");
		
		//Spatial classification
		SimpleMatrix spatialFFOutputMatrix = spatialPooler.feedForward(inputVector);
		
		//Predict next input
		if (useMarkovPrediction){
			predictionMatrix = predictor.predict(spatialFFOutputMatrix);
		} 		
		
		predictionEntropy = calculateEntropy(predictionMatrix);
		if (predictionEntropy >= entropyThreshold) needHelp = true;
		entropyThreshold = entropyDiscountingFactor * predictionEntropy + (1-entropyDiscountingFactor) * entropyThreshold;
		
		//Bias
		//TODO: Should biasing happen before prediction?
		SimpleMatrix biasedOutput = spatialFFOutputMatrix;
		biasedOutput = spatialFFOutputMatrix.plus(1, biasMatrix);
		//biasedOutput = spatialFFOutputMatrix.elementMult(biasMatrix);
		biasedOutput = Normalizer.normalize(biasedOutput);
		
		//Transform spatial output matrix to vector
		double[] spatialFFOutputDataVector = biasedOutput.getMatrix().data;		
		SimpleMatrix temporalFFInputVector = new SimpleMatrix(1, spatialFFOutputDataVector.length);
		temporalFFInputVector.getMatrix().data = spatialFFOutputDataVector;
		
		//Temporal classification
		SimpleMatrix temporalFFOutputMatrix = temporalPooler.feedForward(temporalFFInputVector);
		
		ffOutput = temporalFFOutputMatrix;
		
		return ffOutput;
	}
	
	private double calculateEntropy(SimpleMatrix m){
		double sum = 0;
		for (Double d : m.getMatrix().data){
			if (d != 0) sum += d * Math.log(d);
		}
		return -sum;
	}
	
	/**
	 * Calculates how much the prediction from last time step should influence the spatial output.
	 * Two rules are followed:
	 * 1) The lower the entropy of the prediction, the higher its influence should be
	 * 2) The higher the entropy of the spatial activation matrix, the higher the influence of the prediction should be
	 * 
	 * ad 1) Low entropy signifies that the predictor is sure of what comes next and we should listen to it
	 * ad 2) High entropy of the activation matrix signifies that the pooler doesn't know what it is looking at and needs help to decide by using the prediction
	 * @param predictionEntropy
	 * @return
	 */
	private double calculatePredictionBias(double predictionEntropy, double spatialEntropy){
		double predictionEntropy_scaled = predictionEntropy > 1 ? 1 : predictionEntropy;
		double spatialEntropy_scaled = spatialEntropy > 1 ? 1 : spatialEntropy;
		
		double predictionInfluence = 1 - predictionEntropy_scaled + spatialEntropy_scaled;
		if (predictionInfluence < 0) predictionInfluence = 0;
		if (predictionInfluence > 1) predictionInfluence = 1;
		
		return predictionEntropy;
		
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
			
			//Selection of best temporal model
			SimpleMatrix temporalPoolerFBOutput = temporalPooler.feedBackward(normalizedInput);
			
			//Normalize
			SimpleMatrix normalizedTemporalPoolerFBOutput = normalize(temporalPoolerFBOutput);
			
			//Transformation into matrix
			normalizedTemporalPoolerFBOutput.reshape(spatialMapSize, spatialMapSize); //TODO: Is this necessary?
			
			//Combine FB output from temporal pooler with bias and prediction (if enabled)
			biasMatrix = normalizedTemporalPoolerFBOutput;
			
			biasMatrix = biasMatrix.elementMult(predictionMatrix);
			//biasMatrix = biasMatrix.plus(0.5 / biasMatrix.getNumElements()); //Add small uniform mass
			
			biasMatrix = normalize(biasMatrix);			
			
		} else {
			biasMatrix = predictionMatrix;
		}
		
		SimpleMatrix biasedTemporalFBOutput = biasMatrix;
		
		//Selection of best spatial mode
		SimpleMatrix spatialPoolerFBOutputVector = spatialPooler.feedBackward(biasedTemporalFBOutput);
		
		fbOutput = spatialPoolerFBOutputVector;
		
		needHelp = false;
		
		return fbOutput;
	}
	
	private SimpleMatrix normalize(SimpleMatrix m){
		return Normalizer.normalize(m);
	}
	
	public void flush(){
		temporalPooler.flushTemporalMemory();
		biasMatrix.set(1);
		predictor.flush();
	}
	
	public void setLearning(boolean learning){
		spatialPooler.setLearning(learning);
		temporalPooler.setLearning(learning);
		predictor.setLearning(learning);
	}

	public SpatialPooler getSpatialPooler() {
		return spatialPooler;
	}

	public TemporalPooler getTemporalPooler() {
		return temporalPooler;
	}

	public SimpleMatrix getFfOutput() {
		return ffOutput;
	}

	public SimpleMatrix getFbOutput() {
		return fbOutput;
	}
	
	public void sensitize(int iteration){
		spatialPooler.sensitize(iteration);
		temporalPooler.sensitize(iteration);
	}
	
	public SomNode findTemporalBMU(){
		int maxID = -1;
		double maxValue = Double.NEGATIVE_INFINITY;
		
		for (int i = 0; i < ffOutput.getNumElements(); i++){
			double v = ffOutput.get(i);
			if (v > maxValue){
				maxValue = v;
				maxID = i;
			}
		}
		
		SomNode bmu = temporalPooler.getRSOM().getNode(maxID);
		return bmu;
	}
	
	public Predictor getPredictor(){
		return predictor;
	}

	@Override
	public SOM getSOM() {
		return spatialPooler.getSOM();
	}

	@Override
	public void printModel() {
		predictor.printModel();
		
	}

	@Override
	public boolean needHelp() {
		return needHelp;
	}

	@Override
	public double getEntropy() {
		return predictionEntropy;
	}
	
	@Override
	public double getEntropyThreshold() {
		return entropyThreshold;
	}
}
