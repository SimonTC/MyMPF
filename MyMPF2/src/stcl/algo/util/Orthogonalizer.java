package stcl.algo.util;

import org.ejml.simple.SimpleMatrix;

public class Orthogonalizer {
	
	public static SimpleMatrix orthogonalize(SimpleMatrix m){
		return aggressiveOrthogonalization(m);
	}
	
	private static SimpleMatrix aggressiveOrthogonalization(SimpleMatrix m){

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
	
	private static SimpleMatrix orthogonalization_AsInSomActivation(SimpleMatrix m){
		
		SimpleMatrix orthogonalized = m.elementPower(2);
		orthogonalized = orthogonalized.divide(-0.01 * Math.pow(0.5, 2));
		orthogonalized = orthogonalized.elementExp();
		return orthogonalized;
		
	}
	
	private static SimpleMatrix orthogonalization_NormDist(SimpleMatrix m){
		double mean = 1;
		double stddev = 0.1; //TODO: make to parameter
		double maxInput = m.elementMaxAbs();
		SimpleMatrix o = m.minus(mean);
		o = o.elementPower(2);
		o = o.divide(-2 * Math.pow(stddev, 2));
		o = o.elementExp();
		o = o.scale(1 / stddev * Math.sqrt(2 * Math.PI));
		
		double maxValue = o.elementMaxAbs();
		if (maxInput > 0.7){
			maxValue = gaussValue(1, mean, stddev);
		}
		
		o = o.divide(maxValue);
		
		return o;
	}
	
	private static double gaussValue(double x, double mean, double stddev){ //TODO: rename
		
		double v = 1 / (stddev * Math.sqrt(2 * Math.PI)) * Math.exp(-Math.pow((x - mean),2) / 2 * Math.pow(stddev, 2));
		return v;
	}

}
