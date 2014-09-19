package stcl.som;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

public class SomVector {
	
	SimpleMatrix vector;
	
	/**
	 * Creates a new vector with random numbers between 0 and 1.
	 * @param length
	 * @param rand
	 */
	public SomVector(int length, Random rand) {
		vector = SimpleMatrix.random(1, length, 0, 1, rand);	
	}
	
	/**
	 * Creates new vector with the given data
	 * @param data
	 */
	public SomVector(double[] data){
		vector = new SimpleMatrix(1, data.length);
		vector.setRow(1, 1, data);
	}
	
	public SimpleMatrix getInternalVector(){
		return vector;
	}
	
	

}
