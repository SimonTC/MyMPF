package stcl.algo.brain;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.poolers.SpatialPooler;
import stcl.algo.poolers.TemporalPooler;
import stcl.algo.predictors.FirstOrderPredictor;

public class NeoCorticalUnit {
	
	private SpatialPooler spatialPooler;
	private TemporalPooler temporalPooler;
	private FirstOrderPredictor predictor;
	private SimpleMatrix biasMatrix;
	private SimpleMatrix predictionMatrix;
	
	private SimpleMatrix ffOutput;
	private SimpleMatrix fbOutput;
	
	private int ffInputVectorSize;
	private int spatialMapSize;
	private int temporalMapSize;
	
	private boolean DEBUG = false;
	
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
	}
	
	public SimpleMatrix feedForward(SimpleMatrix inputVector){
		//Test input
		if (!inputVector.isVector()) throw new IllegalArgumentException("The feed forward input to the neocortical unit has to be a vector");
		if (inputVector.numCols() != ffInputVectorSize) throw new IllegalArgumentException("The feed forward input to the neocortical unit has to be a 1 x " + ffInputVectorSize + " vector");
		
		if (DEBUG) System.out.println("****************************** FF start *************************");
		
		if (DEBUG)System.out.println("FF input");
		if (DEBUG)inputVector.print();
		if (DEBUG)System.out.println();
		
		//Spatial classification
		SimpleMatrix spatialFFOutputMatrix = spatialPooler.feedForward(inputVector);
		
		if (DEBUG)System.out.println("Likelihood that SOM model ij is best to describe the current input");
		if (DEBUG)spatialFFOutputMatrix.print();
		if (DEBUG)System.out.println();
		
		//Bias output by the prediction from t-1
		SimpleMatrix spatialFFOutputMatrixBiased = spatialFFOutputMatrix.elementMult(biasMatrix);	
		
		if (DEBUG)System.out.println("Bias matrix");
		if (DEBUG)biasMatrix.print();
		if (DEBUG)System.out.println();
		
		//Normalize output
		double sum = spatialFFOutputMatrixBiased.elementSum();
		spatialFFOutputMatrixBiased = spatialFFOutputMatrixBiased.scale(1/sum);
		
		if (DEBUG)System.out.println("Biased Likelihood that SOM model ij is best to describe the current input");
		if (DEBUG)spatialFFOutputMatrixBiased.print();
		if (DEBUG)System.out.println();
		
		//Orthogonalize output
		SimpleMatrix spatialFFOutputMatrixOrthogonalized =  aggressiveOrthogonalization(spatialFFOutputMatrixBiased);
		
		if (DEBUG)System.out.println("Orthogonalized biased Likelihood that SOM model ij is best to describe the current input");
		if (DEBUG)spatialFFOutputMatrixOrthogonalized.print();
		if (DEBUG)System.out.println();
		
		//Predict next input
		if (useMarkovPrediction){
			SimpleMatrix tmp = aggressiveOrthogonalization(spatialFFOutputMatrix);
			predictionMatrix = predictor.predict(tmp, curPredictionLearningRate);
			//predictionMatrix = predictor.predict(spatialFFOutputMatrixOrthogonalized, curPredictionLearningRate);
			/*
			if (DEBUG)System.out.println("Likelihood that SOM model ij will be the best to describe the next input");
			if (DEBUG)predictionMatrix.print();
			if (DEBUG)System.out.println();
			*/
		} 		
		
		//Transform spatial output matrix to vector
		double[] spatialFFOutputDataVector = spatialFFOutputMatrixOrthogonalized.getMatrix().data;		
		SimpleMatrix temporalFFInputVector = new SimpleMatrix(1, spatialFFOutputDataVector.length);
		temporalFFInputVector.getMatrix().data = spatialFFOutputDataVector;
		
		//Temporal classification
		SimpleMatrix temporalFFOutputMatrix = temporalPooler.feedForward(temporalFFInputVector);
		
		ffOutput = temporalFFOutputMatrix;
		
		if (DEBUG)System.out.println("Likelihood that we currently are in sequence hk");
		if (DEBUG)ffOutput.print();
		if (DEBUG)System.out.println();
		
		if (DEBUG)System.out.println();
		if (DEBUG)System.out.println("****************************** FF end *************************");
		if (DEBUG)System.out.println();
		
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
		
		if (DEBUG)System.out.println("****************************** FB start *************************");
		
		if (DEBUG)System.out.println("FB input");
		if (DEBUG)inputMatrix.print();
		if (DEBUG)System.out.println();
		
		//Selection of best temporal model
		SimpleMatrix temporalPoolerFBOutput = temporalPooler.feedBackward(inputMatrix);
		
		//Transformation into matrix
		temporalPoolerFBOutput.reshape(spatialMapSize, spatialMapSize);

		if (DEBUG)System.out.println("Likelihood that the next input is best described by model ij given that we are in sequence k");
		if (DEBUG)temporalPoolerFBOutput.print();
		if (DEBUG)System.out.println();

		
		//Combine FB output from temporal pooler with bias and prediction (if enabled)
		SimpleMatrix biasedTemporalFBOutput = temporalPoolerFBOutput;
		
		if (useMarkovPrediction){
			biasMatrix = biasedTemporalFBOutput;
			if (DEBUG)System.out.println("Prediction matrix");
			if (DEBUG)predictionMatrix.print();
			if (DEBUG)System.out.println();
			biasMatrix = biasMatrix.elementMult(predictionMatrix);
			
			//Normalize
			if (DEBUG)System.out.println("Bias matrix, not normalized");
			if (DEBUG)biasMatrix.print();
			if (DEBUG)System.out.println();
			
			double sum = biasMatrix.elementSum();
			biasMatrix = biasMatrix.scale(1/sum);
			
			if (DEBUG)System.out.println("Bias matrix, normalized");
			if (DEBUG)biasMatrix.print();
			if (DEBUG)System.out.println();
			
			biasedTemporalFBOutput = biasMatrix;
		} 		
		
		//Selection of best spatial mode
		SimpleMatrix spatialPoolerFBOutputVector = spatialPooler.feedBackward(biasedTemporalFBOutput);
		
		if (DEBUG)System.out.println("Expected input next step");
		if (DEBUG)spatialPoolerFBOutputVector.print();
		if (DEBUG)System.out.println();
		
		fbOutput = spatialPoolerFBOutputVector;
		
		if (DEBUG)System.out.println("****************************** FB end *************************");
		
		return fbOutput;
	}
	
	/**
	 * Orthgonalizes the matrix by setting all values to zero except for the highest value
	 * Only works with matrices containing non-negative values
	 * @param m
	 * @return
	 */
	private SimpleMatrix aggressiveOrthogonalization(SimpleMatrix m){
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
	
	}
	
	public void flushTemporalMemory(){
		temporalPooler.flushTemporalMemory();
		biasMatrix.set(1);
	}
	
	public void setLearning(boolean learning){
		spatialPooler.setLearning(learning);
		temporalPooler.setLearning(learning);
		if (learning){
			curPredictionLearningRate = 1; //TODO: Do something about this. SHhuld be based on some parameter
		} else {
			curPredictionLearningRate = 0;
		}
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

}
