package stcl.algo.brain;

import java.util.Observable;
import java.util.Observer;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import dk.stcl.core.basic.containers.SomNode;
import stcl.algo.poolers.SpatialPooler;
import stcl.algo.poolers.TemporalPooler;
import stcl.algo.predictors.FirstOrderMM_Original;
import stcl.algo.predictors.FirstOrderPredictor;
import stcl.algo.util.Normalizer;
import stcl.algo.util.Orthogonalizer;

public class NeoCorticalUnit{
	
	private SpatialPooler spatialPooler;
	private TemporalPooler temporalPooler;
	private FirstOrderPredictor predictor;
	//FirstOrderMM_Original predictor;
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
	public NeoCorticalUnit(Random rand, int ffInputLength, int spatialMapSize, int temporalMapSize, double initialPredictionLearningRate, boolean useMarkovPrediction, double decayFactor) {
		//TODO: All parameters should be handled in parameter file
		spatialPooler = new SpatialPooler(rand, ffInputLength, spatialMapSize, 0.1, 2, 0.125); //TODO: Move all parameters out
		temporalPooler = new TemporalPooler(rand, spatialMapSize * spatialMapSize, temporalMapSize, 0.1, 5, 0.125, 0.3); //TODO: Move all parameters out
		predictor = new FirstOrderPredictor(spatialMapSize);
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
		SimpleMatrix spatialFFOutputMatrix = spatialPooler.feedForward(inputVector, false);
		
		//Bias output by the prediction from t-1
		SimpleMatrix spatialFFOutputMatrixBiased = spatialFFOutputMatrix;//spatialFFOutputMatrix.elementMult(biasMatrix);	
				
		//Normalize output
		Normalizer.normalize(spatialFFOutputMatrixBiased);
		
		//Orthogonalize output
		SimpleMatrix spatialFFOutputOrthogonalized =  orthogonalize(spatialFFOutputMatrixBiased);
		
		//Predict next input
		if (useMarkovPrediction){
			SimpleMatrix tmp = orthogonalize(spatialFFOutputMatrix);
			predictionMatrix = predictor.predict(tmp, curPredictionLearningRate, learning);
		} 		
		
		//Transform spatial output matrix to vector
		double[] spatialFFOutputDataVector = spatialFFOutputOrthogonalized.getMatrix().data;		
		SimpleMatrix temporalFFInputVector = new SimpleMatrix(1, spatialFFOutputDataVector.length);
		temporalFFInputVector.getMatrix().data = spatialFFOutputDataVector;
		
		//Temporal classification
		SimpleMatrix temporalFFOutputMatrix = temporalPooler.feedForward(temporalFFInputVector);
		
		ffOutput = temporalFFOutputMatrix;
		
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

		//Normalize
		SimpleMatrix normalizedInput = normalize(inputMatrix);
		
		//Selection of best temporal model
		SimpleMatrix temporalPoolerFBOutput = temporalPooler.feedBackward(normalizedInput);
		
		//Add noise
		 //Happens in pooler
		
		//Normalize
		SimpleMatrix normalizedTemporalPoolerFBOutput = normalize(temporalPoolerFBOutput);
		
		//Transformation into matrix
		normalizedTemporalPoolerFBOutput.reshape(spatialMapSize, spatialMapSize); //TODO: Is this necessary?
		
		//Combine FB output from temporal pooler with bias and prediction (if enabled)
		SimpleMatrix biasedTemporalFBOutput = normalizedTemporalPoolerFBOutput;
		
		if (useMarkovPrediction){
			biasMatrix = biasedTemporalFBOutput;

			biasMatrix = biasMatrix.elementMult(predictionMatrix);
			biasMatrix = biasMatrix.plus(0.5 / biasMatrix.getNumElements()); //Add small uniform mass
			
			SimpleMatrix biasMatrixNormalized = normalize(biasMatrix);

			
			biasedTemporalFBOutput = biasMatrixNormalized;
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
		Normalizer.normalize(m);
		return m;
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
	
	
	
	
	public void flushTemporalMemory(){
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

}
