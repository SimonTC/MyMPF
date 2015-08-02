package stcl.algo.util;

import org.ejml.simple.SimpleMatrix;

public class Orthogonalizer {
	
	public static SimpleMatrix orthogonalize(SimpleMatrix m){
		SimpleMatrix orthogonalized = m.elementPower(12);
		return orthogonalized;
	}
	
	public static SimpleMatrix aggressiveOrthogonalization(SimpleMatrix m){

		int maxID = -1;
		int id = 0;
		double max = Double.NEGATIVE_INFINITY;
		
		for (double d : m.getMatrix().data){
			if (d > max){
				max = d;
				maxID = id;
			}
			id++;
		}
		
		SimpleMatrix orthogonalized = new SimpleMatrix(m.numRows(), m.numCols());
		orthogonalized.set(maxID, 1);
		return orthogonalized;		
	
	}
	



}
