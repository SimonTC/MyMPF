package stcl.algo.som;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

public class SomMap {
	
	private int rows, columns;
	private SomNode[] nodes; 
	
	/**
	 * Creates a new map where all node vector values are initialized to a random value between 0 and 1
	 * @param columns width of the map 
	 * @param rows height of the map 
	 */
	public SomMap(int columns, int rows, int inputLength, Random rand) {
		this.columns = columns;
		this.rows = rows;
		initializeMap(inputLength, rand);
	}
	
	/**
	 * Fills the map with nodes where the vector values are set to random values between 0 and 1
	 * @param columns
	 * @param rows
	 * @param inputLength
	 * @param rand
	 */
	private void initializeMap(int inputLength, Random rand){
		nodes = new SomNode[rows * columns];
		
		for (int row = 0; row < rows; row++){
			for (int col = 0; col < columns; col++){
				SomNode n = new SomNode(inputLength, rand, col, row);
				nodes[coordinateToIndex(row, col)] = n;
			}
		}
	}
	
	private int coordinateToIndex(int row, int col){
		return (row * columns + col);
	}
	
	public SomNode[] getNodes(){
		return this.nodes;
	}
	
	public SomNode get(int x, int y){
		return nodes[coordinateToIndex(y, x)];
	}
	
	public void set(int x, int y, SomNode n){
		nodes[coordinateToIndex(y, x)] = n;;
	}
	
	public SomNode get(int id){
		return nodes[id];
	}
	
	public int getWidth(){
		return columns;
	}
	
	public int getHeight(){
		return rows;
	}
	

}
