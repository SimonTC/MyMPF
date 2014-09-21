package stcl.algo.som;

import java.util.Random;
import java.util.Vector;

import org.ejml.data.MatrixIterator;
import org.ejml.simple.SimpleMatrix;

public class SOM extends SomBasics {
	
	/**
	 * Creates a new SOM where all node vector values are initialized to a random value between 0 and 1
	 * @param columns width of the map 
	 * @param rows height of the map 
	 */
	public SOM(int columns, int rows, int inputLength, Random rand) {
		super(columns, rows, inputLength, rand);
	}
	
	
	/**
	 * Finds the BMU to the given input node and updates the vectors of all the nodes
	 * @param inputNode
	 * @param learningRate
	 * @param neighborhoodRadius
	 * @return
	 */
	public SomNode step (SimpleMatrix inputVector, double learningRate, double neighborhoodRadius){
		//Find BMU
		SomNode bmu = getBMU(inputVector);
		
		//Adjust Weights
		updateWeightMatrix(bmu, inputVector, learningRate, neighborhoodRadius);	
		
		return bmu;
	}
		
	/**
	 * Returns the node which vector is least different from the vector of the input node. This method also updates the internal error matrix
	 * @param input input as a somNode
	 * @return
	 */
	public SomNode getBMU(SimpleMatrix inputVector){
		SomNode BMU = null;
		double minDiff = Double.POSITIVE_INFINITY;
		for (SomNode n : weightMap.getNodes()){
			double diff = n.squaredDifference(inputVector);
			if (diff < minDiff){
				minDiff = diff;
				BMU = n;
			}
			
			errorMatrix.set(n.getRow(), n.getCol(), diff);
		}
		return BMU;
	}
	
	public void weightAdjustment(SomNode n, SomNode bmu, SimpleMatrix inputVector, double neighborhoodRadius, double learningRate ){
		double squaredDistance = n.distanceTo(bmu);
		double squaredRadius = neighborhoodRadius * neighborhoodRadius;
		if (squaredDistance <= squaredRadius){ 
			double learningEffect = learningEffect(squaredDistance, squaredRadius);
			n.adjustValues(inputVector, learningRate, learningEffect);					
		}
	}

	@Override
	public SomNode getBMU() throws UnsupportedOperationException{
		throw new UnsupportedOperationException();
	}
	
	
	
	

}
