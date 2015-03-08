package stcl.algo.brain;

import java.util.Observable;
import java.util.Observer;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import dk.stcl.core.basic.containers.SomNode;
import stcl.algo.poolers.SOM;
import stcl.algo.poolers.SpatialPooler;
import stcl.algo.poolers.TemporalPooler;
import stcl.algo.predictors.FirstOrderMM_Original;
import stcl.algo.predictors.FirstOrderPredictor;
import stcl.algo.predictors.Predictor;
import stcl.algo.predictors.Predictor_VOMM;
import stcl.algo.util.Normalizer;
import stcl.algo.util.Orthogonalizer;

public class NeoCorticalUnit4 implements NU{
	
	private SpatialPooler spatialPooler;
	private TemporalPooler temporalPooler;
	//private FirstOrderPredictor predictor;
	private Predictor_VOMM predictor;
	private SimpleMatrix biasMatrix;
	private SimpleMatrix predictionMatrix;
	
	private SimpleMatrix ffOutput;
	private SimpleMatrix fbOutput;
	
	private int ffInputVectorSize;
	private int spatialMapSize;
	private int temporalMapSize;
	
	private boolean DEBUG = false;
	
	private boolean learning;
	
	//Learning rates
	private double curPredictionLearningRate; //TODO: Find correct name for it
											  //TODO: Does the prediction learning rate change?
	
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
	public NeoCorticalUnit4(Random rand, int ffInputLength, int spatialMapSize, int temporalMapSize, double initialPredictionLearningRate, boolean useMarkovPrediction, double decayFactor, int markovOrder) {
		//TODO: All parameters should be handled in parameter file
		spatialPooler = new SpatialPooler(rand, ffInputLength, spatialMapSize, 0.1, 2, 0.125); //TODO: Move all parameters out
		temporalPooler = new TemporalPooler(rand, spatialMapSize * spatialMapSize, temporalMapSize, 0.1, 5, 0.125, 0.3); //TODO: Move all parameters out
		predictor = new Predictor_VOMM(markovOrder, initialPredictionLearningRate, rand);
		//predictor = new FirstOrderPredictor(spatialMapSize);
		biasMatrix = new SimpleMatrix(spatialMapSize, spatialMapSize);
		biasMatrix.set(1);
		ffOutput = new SimpleMatrix(temporalMapSize, temporalMapSize);
		fbOutput = new SimpleMatrix(1, ffInputLength);
		ffInputVectorSize = ffInputLength;
		this.curPredictionLearningRate = initialPredictionLearningRate;
		this.useMarkovPrediction = useMarkovPrediction;
		this.spatialMapSize = spatialMapSize;
		this.temporalMapSize = temporalMapSize;
		this.learning = true;
	}
	
	public SimpleMatrix feedForward(SimpleMatrix inputVector){
		//Test input
		if (!inputVector.isVector()) throw new IllegalArgumentException("The feed forward input to the neocortical unit has to be a vector");
		if (inputVector.numCols() != ffInputVectorSize) throw new IllegalArgumentException("The feed forward input to the neocortical unit has to be a 1 x " + ffInputVectorSize + " vector");
		
		//Spatial classification
		SimpleMatrix spatialFFOutputMatrix = spatialPooler.feedForward(inputVector);
		
		//Predict next input
		if (useMarkovPrediction){
			predictionMatrix = predictor.predict(spatialFFOutputMatrix, curPredictionLearningRate, learning);
		} 		
		
		//Bias
		//TODO: Should biasing happen before prediction?
		SimpleMatrix biasedOutput = spatialFFOutputMatrix;
		
		if (biasMatrix!= null){
			double predictionEntropy = calculateEntropy(predictionMatrix);
			double spatialEntropy = calculateEntropy(spatialFFOutputMatrix);
			double predictionInfluence = calculatePredictionBias(predictionEntropy, spatialEntropy);
			//if (predictionInfluence > 0) System.out.println("Prediction does have an influence");
			biasedOutput = spatialFFOutputMatrix.plus(1, biasMatrix);
		}
		
		
		
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

		//Normalize
		SimpleMatrix normalizedInput = normalize(inputMatrix);
		
		//Selection of best temporal model
		SimpleMatrix temporalPoolerFBOutput = temporalPooler.feedBackward(normalizedInput);
		
		//Normalize
		SimpleMatrix normalizedTemporalPoolerFBOutput = normalize(temporalPoolerFBOutput);
		
		//Transformation into matrix
		normalizedTemporalPoolerFBOutput.reshape(spatialMapSize, spatialMapSize); //TODO: Is this necessary?
		
		//Combine FB output from temporal pooler with bias and prediction (if enabled)
		SimpleMatrix biasedTemporalFBOutput = normalizedTemporalPoolerFBOutput;
		
		if (useMarkovPrediction){
			biasMatrix = biasedTemporalFBOutput;

			biasMatrix = biasMatrix.elementMult(predictionMatrix);
			//biasMatrix = biasMatrix.plus(0.5 / biasMatrix.getNumElements()); //Add small uniform mass
			
			biasMatrix = normalize(biasMatrix);

			
			biasedTemporalFBOutput = biasMatrix;
		} 		
		
		//Selection of best spatial mode
		SimpleMatrix spatialPoolerFBOutputVector = spatialPooler.feedBackward(biasedTemporalFBOutput);
		
		//Add noise
			//Happens in pooler
		
		fbOutput = spatialPoolerFBOutputVector;
		
		if (DEBUG)System.out.println("****************************** FB end *************************");
		
		return fbOutput;
	}
	
	private SimpleMatrix normalize(SimpleMatrix m){
		return Normalizer.normalize(m);
	}
	
	/**
	 * Orthgonalizes the matrix by setting all values to zero except for the highest value
	 * Only works with matrices containing non-negative values
	 * @param m
	 * @return
	 */
	private SimpleMatrix orthogonalize(SimpleMatrix m){
		
		return Orthogonalizer.orthogonalize(m);
		
		//return orthogonalization_NormDist(m);
		
		/*
		int maxID = -1;
		int id = 0;
		double max = Double.NEGATIVE_INFINITY;
		double value = 0;
		
		for (double d : m.getMatrix().data){
			value = d;
			if (d > max){
				max = d;
				maxID = id;
			}
			id++;
		}
		
		SimpleMatrix orthogonalized = new SimpleMatrix(m.numRows(), m.numCols());
		orthogonalized.set(maxID, 1);
		return orthogonalized;
		*/
	
	}
	
	private SimpleMatrix orthogonalization_AsInSomActivation(SimpleMatrix m){
		
		SimpleMatrix orthogonalized = m.elementPower(2);
		orthogonalized = orthogonalized.divide(-0.01 * Math.pow(0.5, 2));
		orthogonalized = orthogonalized.elementExp();
		return orthogonalized;
		
	}
	
	private SimpleMatrix orthogonalization_NormDist(SimpleMatrix m){
		double mean = 1;
		double stddev = 0.1; //TODO: make to parameter
		double maxInput = m.elementMaxAbs();
		SimpleMatrix o = m.minus(mean);
		o = o.elementPower(2);
		o = o.divide(-2 * Math.pow(stddev, 2));
		o = o.elementExp();
		o = o.scale(1 / stddev * Math.sqrt(2 * Math.PI));
		
		double maxValue = o.elementMaxAbs();
		if (maxInput > 0.7){
			maxValue = gaussValue(1, mean, stddev);
		}
		
		o = o.divide(maxValue);
		
		return o;
	}
	
	
	private double gaussValue(double x, double mean, double stddev){ //TODO: rename
		
		double v = 1 / (stddev * Math.sqrt(2 * Math.PI)) * Math.exp(-Math.pow((x - mean),2) / 2 * Math.pow(stddev, 2));
		return v;
	}
	
	
	
	
	public void flush(){
		temporalPooler.flushTemporalMemory();
		biasMatrix.set(1);
		predictor.flush();
	}
	
	public void setLearning(boolean learning){
		spatialPooler.setLearning(learning);
		temporalPooler.setLearning(learning);
		this.learning = learning;
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
	
	public void setDebug(boolean debug){
		DEBUG = debug;
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
}
