package stcl.algo.brain;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.poolers.SOM;
import stcl.algo.poolers.SpatialPooler;
import stcl.algo.poolers.TemporalPooler;
import stcl.algo.predictors.Predictor;

public interface NU {
	
	/**
	 * 
	 * @param inputVector Vector containing the values of the units visual field
	 * @return probability matrix containing the probabilities of the temporal models being active now
	 */
	public SimpleMatrix feedForward(SimpleMatrix inputVector);
	
	/**
	 * 
	 * @param inputMatrix matrix containing the probabilities of the temporal models being active at time t + 1
	 * @return Predicted input at time t + 1
	 */
	public SimpleMatrix feedBackward(SimpleMatrix inputMatrix);
	
	public SOM getSOM();
	
	public void printModel();
	
	public SpatialPooler getSpatialPooler() ;

	public TemporalPooler getTemporalPooler() ;
	
	public SimpleMatrix getFfOutput();

	public SimpleMatrix getFbOutput();
	
	public void flush();
	
	public void setLearning(boolean learning);
	
	public Predictor getPredictor();
	
	/**
	 * The needHelp flag is set by the unit to signify that it needs help from its parent.
	 * The needHelp flag is set to true when the entropy of the prediction is higher than a entropy threshold
	 * @return
	 */
	public boolean needHelp();
}
