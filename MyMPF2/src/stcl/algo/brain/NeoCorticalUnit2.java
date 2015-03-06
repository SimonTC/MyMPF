package stcl.algo.brain;

import java.util.Observable;
import java.util.Observer;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import dk.stcl.core.basic.containers.SomNode;
import stcl.algo.poolers.SOM;
import stcl.algo.poolers.Sequencer;
import stcl.algo.poolers.SpatialPooler;
import stcl.algo.poolers.TemporalPooler;
import stcl.algo.predictors.FirstOrderMM_Original;
import stcl.algo.predictors.FirstOrderPredictor;
import stcl.algo.predictors.Predictor;
import stcl.algo.predictors.Predictor_VOMM;
import stcl.algo.util.Normalizer;
import stcl.algo.util.Orthogonalizer;

public class NeoCorticalUnit2 implements NU{
	
	private Sequencer sequencer;
	
	private SpatialPooler spatialPooler;
	private Predictor predictor;

	private SimpleMatrix ffOutput;
	private SimpleMatrix fbOutput;
	private SimpleMatrix biasMatrix;
	private int ffInputVectorSize;
	private int spatialMapSize;
	private int temporalMapSize;
	
	private boolean DEBUG = false;
	private double biasFactor;
	private boolean learning;
	
	//Learning rates
	private double curPredictionLearningRate; //TODO: Find correct name for it
											  //TODO: Does the prediction learning rate change?
	
	private boolean useMarkovPrediction;
	
	/**
	 * 
	 * @param rand
	 * @param ffInputLength
	 * @param spatialMapSize
	 * @param temporalMapSize
	 * @param initialPredictionLearningRate
	 * @param markovOrder
	 * @param decayFactor
	 */
	public NeoCorticalUnit2(Random rand, int ffInputLength, int spatialMapSize, int temporalMapSize, double initialPredictionLearningRate, int markovOrder, double decayFactor, double biasFactor) {
		//TODO: All parameters should be handled in parameter file
		spatialPooler = new SpatialPooler(rand, ffInputLength, spatialMapSize, 0.1, 2, 0.125); //TODO: Move all parameters out
		sequencer = new Sequencer(markovOrder, initialPredictionLearningRate, temporalMapSize, spatialMapSize * spatialMapSize, rand, 0.1, 0.125, 2, decayFactor, biasFactor);
		predictor = new Predictor_VOMM(1, 0.1, rand);
		//predictor = new FirstOrderPredictor(spatialMapSize);
		biasMatrix = new SimpleMatrix(spatialMapSize, spatialMapSize);
		biasMatrix.set(1);
		ffOutput = new SimpleMatrix(temporalMapSize, temporalMapSize);
		fbOutput = new SimpleMatrix(1, ffInputLength);
		ffInputVectorSize = ffInputLength;
		this.curPredictionLearningRate = initialPredictionLearningRate;
		this.spatialMapSize = spatialMapSize;
		this.temporalMapSize = temporalMapSize;
		this.learning = true;
		this.biasFactor = biasFactor;
	}
	
	public SimpleMatrix feedForward(SimpleMatrix inputVector){
		//Test input
		if (!inputVector.isVector()) throw new IllegalArgumentException("The feed forward input to the neocortical unit has to be a vector");
		if (inputVector.numCols() != ffInputVectorSize) throw new IllegalArgumentException("The feed forward input to the neocortical unit has to be a 1 x " + ffInputVectorSize + " vector");
		
		//Spatial classification
		SimpleMatrix spatialFFOutputMatrix = spatialPooler.feedForward(inputVector, false);
		
		//Normalize output
		SimpleMatrix normalized = Normalizer.normalize(spatialFFOutputMatrix);
		
		//Bias
		SimpleMatrix biasedOutput = normalized;
		if (biasMatrix!= null){
			biasedOutput = normalized.plus(biasFactor, biasMatrix);
		}
		
		//Normalize
		biasedOutput = Normalizer.normalize(biasedOutput);
		
		//Transform spatial output matrix to vector
		double[] spatialFFOutputDataVector = biasedOutput.getMatrix().data;		
		SimpleMatrix temporalFFInputVector = new SimpleMatrix(1, spatialFFOutputDataVector.length);
		temporalFFInputVector.getMatrix().data = spatialFFOutputDataVector;
		
		//Temporal classification
		SimpleMatrix temporalFFOutputMatrix = sequencer.feedForward(temporalFFInputVector);
		ffOutput = temporalFFOutputMatrix;
		if (ffOutput != null){
			ffOutput = Normalizer.normalize(ffOutput);
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

		//Collect prediction matrix
		SimpleMatrix fbSequencerOutput = sequencer.feedBackward(inputMatrix);
		
		biasMatrix = fbSequencerOutput;
		
		//Transformation into matrix
		fbSequencerOutput.reshape(spatialMapSize, spatialMapSize); 
		
		//Selection of best spatial mode
		SimpleMatrix spatialPoolerFBOutputVector = spatialPooler.feedBackward(fbSequencerOutput);
		
		fbOutput = spatialPoolerFBOutputVector;
		
		return fbOutput;
	}

	
	public void setLearning(boolean learning){
		spatialPooler.setLearning(learning);
		sequencer.setLearning(learning);
		this.learning = learning;
	}

	public SpatialPooler getSpatialPooler() {
		return spatialPooler;
	}


	public SimpleMatrix getFfOutput() {
		return ffOutput;
	}

	public SimpleMatrix getFbOutput() {
		return fbOutput;
	}
	
	public void sensitize(int iteration){
		spatialPooler.sensitize(iteration);
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
		
		SomNode bmu = sequencer.getRsom().getNode(maxID);
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
