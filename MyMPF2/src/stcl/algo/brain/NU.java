package stcl.algo.brain;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.poolers.SOM;

public interface NU {
	
	public SimpleMatrix feedForward(SimpleMatrix inputVector);
	
	public SimpleMatrix feedBackward(SimpleMatrix inputMatrix);
	
	public SOM getSOM();
	
	public void printModel();
}
