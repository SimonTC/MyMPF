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
	
	//Learning rates
	private double curPredictionLearningRate; //TODO: Find correct name for it
											  //TODO: Does the prediction learning rate change?
	
	private boolean useMarkovPrediction;
	
	public NeoCorticalUnit(Random rand, int maxIterations, int ffInputLength, int spatialMapSize, int temporalMapSize, double initialPredictionLearningRate, boolean useMarkovPrediction, double decayFactor) {
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
		
		//Spatial classification
		SimpleMatrix spatialFFOutputMatrix = spatialPooler.feedForward(inputVector);
		
		//Bias output by the prediction from t-1
		SimpleMatrix spatialFFOutputMatrixBiased = spatialFFOutputMatrix.elementMult(biasMatrix);	
		
		//Normalize output
		double sum = spatialFFOutputMatrixBiased.elementSum();
		spatialFFOutputMatrixBiased = spatialFFOutputMatrixBiased.scale(1/sum);
		
		//Predict next input
		if (useMarkovPrediction){
			predictionMatrix = predictor.predict(spatialFFOutputMatrixBiased, curPredictionLearningRate);
		} 		
		
		//Transform spatial output matrix to vector
		double[] spatialFFOutputDataVector = spatialFFOutputMatrixBiased.getMatrix().data;		
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
				
		//Selection of best temporal model
		SimpleMatrix temporalPoolerFBOutput = temporalPooler.feedBackward(inputMatrix);
		
		//Transformation into matrix
		temporalPoolerFBOutput.reshape(spatialMapSize, spatialMapSize);
		
		//Combine FB output from temporal pooler with bias and prediction (if enabled)
		biasMatrix = temporalPoolerFBOutput;
		
		if (useMarkovPrediction){
			biasMatrix = biasMatrix.elementMult(predictionMatrix);
			//Normalize
			double sum = biasMatrix.elementSum();
			biasMatrix = biasMatrix.scale(1/sum);
		} 
		/*
		System.out.println();
		biasMatrix.print();
		System.out.println();
		*/
		
		//Selection of best spatial mode
		SimpleMatrix spatialPoolerFBOutputVector = spatialPooler.feedBackward(biasMatrix);
		
		fbOutput = spatialPoolerFBOutputVector;
		
		return fbOutput;
	}
	
	public void flushTemporalMemory(){
		temporalPooler.flushTemporalMemory();
	}
	
	public void setLearning(boolean learning){
		spatialPooler.setLearning(learning);
		temporalPooler.setLearning(learning);
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

}
