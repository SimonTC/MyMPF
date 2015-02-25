package stcl.algo.util;

import org.ejml.simple.SimpleMatrix;

public class Normalizer {
	
	public static void normalize(SimpleMatrix m){
		double sum = m.elementSum();
		if (sum > 0){
			m = m.scale(1/sum);
		} 
	}
	
	public static void normalizeColumns(SimpleMatrix m){
		for (int col = 0; col < m.numCols(); col++){
			SimpleMatrix column = m.extractVector(false, col);
			double sum = column.elementSum();
			if (sum != 0) column = column.divide(sum);
			m.setColumn(col, 0, column.getMatrix().data);
		}
	}

}
