package stcl.algo.util;

import org.ejml.simple.SimpleMatrix;

public class MySimpleMatrix extends SimpleMatrix {
	
	@Override
	public void reshape(int numRows, int numCols) throws UnsupportedOperationException{
		System.out.println("Why????");
	}

}
