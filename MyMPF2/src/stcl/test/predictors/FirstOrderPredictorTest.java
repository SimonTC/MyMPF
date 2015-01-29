package stcl.test.predictors;

import static org.junit.Assert.*;

import org.ejml.simple.SimpleMatrix;
import org.junit.Before;
import org.junit.Test;

import stcl.algo.predictors.FirstOrderPredictor;

public class FirstOrderPredictorTest {

	@Before
	public void setUp() throws Exception {
	}

	@Test
	public void testPredict() {
		FirstOrderPredictor predictor = new FirstOrderPredictor(2);
		double[][] tmp1 = {
				{1,0},
				{0,0}};
		
		double[][] tmp2 = {
				{0,1},
				{0,0}};
		SimpleMatrix input2 = new SimpleMatrix(tmp2);
		SimpleMatrix input1 = new SimpleMatrix(tmp1);
		SimpleMatrix output1;
		SimpleMatrix output2;
		SimpleMatrix output3;
		for (int i = 0; i < 1000; i++){
			output1 = predictor.predict(input1, 1);
			//output2 = predictor.predict(input2, 1);
			//output3 = predictor.predict(input2, 1);
		}
		
		output1 = predictor.predict(input1, 0);
		output2 = predictor.predict(input2, 0);
		output3 = predictor.predict(input2, 0);
	}

}
