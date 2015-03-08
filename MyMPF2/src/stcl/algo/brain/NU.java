package stcl.algo.brain;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.poolers.SOM;
import stcl.algo.poolers.SpatialPooler;
import stcl.algo.poolers.TemporalPooler;

public interface NU {
	
	public SimpleMatrix feedForward(SimpleMatrix inputVector);
	
	public SimpleMatrix feedBackward(SimpleMatrix inputMatrix);
	
	public SOM getSOM();
	
	public void printModel();
	
	public SpatialPooler getSpatialPooler() ;

	public TemporalPooler getTemporalPooler() ;
	
	public SimpleMatrix getFfOutput();

	public SimpleMatrix getFbOutput();
	
	public void flush();
}