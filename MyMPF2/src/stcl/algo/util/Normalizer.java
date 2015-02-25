package stcl.algo.util;

import org.ejml.simple.SimpleMatrix;

public class Normalizer {
	
	public static SimpleMatrix normalize(SimpleMatrix m){
		double sum = m.elementSum();
		SimpleMatrix normalized = m.scale(1/sum);
		return normalized;
	}

}
