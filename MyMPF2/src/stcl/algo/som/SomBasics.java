package stcl.algo.som;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

public abstract class SomBasics {

	protected SomMap weightMap;
	protected SimpleMatrix errorMatrix;
	protected boolean learning; //If false no learning will take place during feed forward and feed back
	
	/**
	 * 
	 * @param columns number of columns in the internal weightMap
	 * @param rows number of rows in the internal weight map
	 * @param inputLength length of the value vectors
	 * @param rand
	 */
	public SomBasics(int columns, int rows, int inputLength, Random rand) {
		weightMap = new SomMap(columns, rows, inputLength, rand);
		errorMatrix = new SimpleMatrix(rows, columns);
		learning = true;
	}
	
	public abstract SomNode getBMU();
	
	public abstract SomNode getBMU(SimpleMatrix inputVector);
	
	public SimpleMatrix getErrorMatrix(){
		return errorMatrix;
	}
	
	public int getHeight(){
		return weightMap.getHeight();
	}
	
	public SomMap getWeighttMap(){
		return weightMap;
	}
	
	public SomNode getModel(int id){
		return weightMap.get(id);
	}
	
	public SomNode getModel(int row, int col){
		return weightMap.get(col, row);
	}
	
	public SomNode[] getModels(){
		return weightMap.getNodes();
	}
	
	public int getWidth(){
		return weightMap.getWidth();
	}
	
	public boolean getLearning(){
		return learning;
	}
	
	public void setLearning(boolean learning){
		this.learning = learning;
	}
	
	/**
	 * Calculates the learning effect based on distance to the learning center.
	 * The lower the distance, the higher the learning effect
	 * @param squaredDistance
	 * @param squaredRadius
	 * @return
	 */
	protected double learningEffect(double squaredDistance, double squaredRadius){
		double d = Math.exp(-(squaredDistance / (2 * squaredRadius)));
		return d;
	}
	
	public void set(SomNode n, int row, int column){
		weightMap.set(column, row, n);
	}
	
	public abstract SomNode step (SimpleMatrix inputVector, double learningRate, double neighborhoodRadius);
	
	protected void updateWeightMatrix(SomNode bmu,SimpleMatrix inputVector, double learningRate, double neighborhoodRadius){
		//Calculate start and end coordinates for the weight updates
		int bmuCol = bmu.getCol();
		int bmuRow = bmu.getRow();
		int colStart = (int) (bmuCol - neighborhoodRadius);
		int rowStart = (int) (bmuRow - neighborhoodRadius );
		int colEnd = (int) (bmuCol + neighborhoodRadius);
		int rowEnd = (int) (bmuRow + neighborhoodRadius );
		
		//Make sure we don't get out of bounds errors
		if (colStart < 0) colStart = 0;
		if (rowStart < 0) rowStart = 0;
		if (colEnd > weightMap.getWidth()) colEnd = weightMap.getWidth();
		if (rowEnd > weightMap.getHeight()) rowEnd = weightMap.getHeight();
		
		//Adjust weights
		for (int col = colStart; col < colEnd; col++){
			for (int row = rowStart; row < rowEnd; row++){
				SomNode n = weightMap.get(col, row);
				n = weightAdjustment(n, bmu, inputVector, neighborhoodRadius, learningRate);
			}
		}
	}
	
	public abstract SomNode weightAdjustment(SomNode n, SomNode bmu, SimpleMatrix inputVector, double neighborhoodRadius, double learningRate );

}
