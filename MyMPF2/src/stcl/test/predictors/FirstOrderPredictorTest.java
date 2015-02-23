package stcl.test.predictors;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.ejml.simple.SimpleMatrix;
import org.junit.Before;
import org.junit.Test;




import stcl.algo.predictors.FirstOrderPredictor;

public class FirstOrderPredictorTest {
	double e = 0.0001;
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
		
		double[][] tmp3 = {
				{0,0},
				{1,0}};
		
		double[][] tmp4 = {
				{0,0},
				{0,1}};
		
		SimpleMatrix input1 = new SimpleMatrix(tmp1);
		SimpleMatrix input2 = new SimpleMatrix(tmp2);
		SimpleMatrix input3 = new SimpleMatrix(tmp3);
		SimpleMatrix input4 = new SimpleMatrix(tmp4);
		
		for (int i = 0; i < 1000; i++){
			predictor.predict(input1,1, true);
			predictor.predict(input2,1, true);
			predictor.predict(input3,1, true);
			predictor.predict(input4,1, true);

		}
		
		SimpleMatrix output1 = predictor.predict(input1, 0, false);
		SimpleMatrix output2 = predictor.predict(input2, 0, false);
		SimpleMatrix output3 = predictor.predict(input3, 0, false);
		SimpleMatrix output4 = predictor.predict(input4, 0, false);
		
		SimpleMatrix diff1 = output1.minus(input2);	
		SimpleMatrix diff2 = output2.minus(input3);	
		SimpleMatrix diff3 = output3.minus(input4);	
		SimpleMatrix diff4 = output4.minus(input1);	
		
		assertEquals(0, diff1.normF(), e);
		assertEquals(0, diff2.normF(), e);
		assertEquals(0, diff3.normF(), e);
		assertEquals(0, diff4.normF(), e);
	}

}
