package stcl.som;

import java.util.Vector;

import org.ejml.data.MatrixIterator;
import org.ejml.simple.SimpleMatrix;

public class SOMMap {
	int height, width;
	Vector<SomNode> models; //The models in the map
	SimpleMatrix errorMatrix;
	
	/**
	 * 
	 * @param columns width of the map (coulumns)
	 * @param rows height of the map (rows)
	 */
	public SOMMap(int columns, int rows) {
		models = new Vector<SomNode>(columns*rows);
		width = columns;
		height = rows;
		errorMatrix = new SimpleMatrix(rows, columns);
	}
	
	public SomNode getBMU(SomNode input){
		SomNode BMU = null;
		double minDiff = Double.POSITIVE_INFINITY;
		for (SomNode n : models){
			double diff = n.squaredDifference(input);
			if (diff < minDiff){
				minDiff = diff;
				BMU = n;
			}
			
			errorMatrix.set(n.getRow(), n.getCol(), diff);
		}
		return BMU;
	}
	
	public void adjustWeights(SomNode bmu, double learningRate, double neighborhoodRadius){
		//Calculate start and end coordinates for the weight updates
		int xStart = (int) (bmu.getCol() - neighborhoodRadius - 1);
		int yStart = (int) (bmu.getRow() - neighborhoodRadius - 1);
		int xEnd = (int) (xStart + (neighborhoodRadius * 2) + 1);
		int yEnd = (int) (yStart + (neighborhoodRadius * 2) + 1);
		
		//Make sure we dont get out of bounds errors
		if (xStart < 0) xStart = 0;
		if (yStart < 0) yStart = 0;
		if (xEnd > width) xEnd = width;
		if (yEnd > height) yEnd = height;
		
		//Adjust weights
		for (int x = xStart; x < xEnd; x++){
			for (int y = yStart; y < yEnd; y++){
				SomNode n = models.elementAt(x * y);
				double squaredDistance = n.distanceTo(bmu);
				double squaredRadius = neighborhoodRadius * neighborhoodRadius;
				if (squaredDistance < squaredRadius){ 
					double learningEffect = learningEffect(squaredDistance, squaredRadius);
					n.adjustValues(bmu.getVector(), learningRate, learningEffect);					
				}
			}
		}
	}
	
	public Vector<SomNode> getModels(){
		return models;
	}
	
	public SimpleMatrix getErrorMatrix(){
		return errorMatrix;
	}
	
	public SomNode getModel(int id){
		return models.elementAt(id);
	}
	
	
	
	private double learningEffect(double squaredDistance, double squaredRadius){
		double d = Math.exp(-(squaredDistance / (2 * squaredRadius)));
		return d;
	}
	
	
	
	
	

}
