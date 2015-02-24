package stcl.test.predictors;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.ejml.simple.SimpleMatrix;
import org.junit.Before;
import org.junit.Test;





import stcl.algo.predictors.FirstOrderPredictor;

public class FirstOrderPredictorTest {
	double e = 0.0001;
	SimpleMatrix in1, in2 , in3 , in4 ;
	FirstOrderPredictor predictor;
	@Before
	public void setUp() throws Exception {
		
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
		in1 = new SimpleMatrix(tmp1);
		in2 = new SimpleMatrix(tmp2);
		in3 = new SimpleMatrix(tmp3);
		in4 = new SimpleMatrix(tmp4);
	}

	/*
	@Test
	public void testPredict_OneOccurenceOfEach	() {
		predictor = new FirstOrderPredictor(2);
		SimpleMatrix[] sequence = {in1, in2, in3, in4};
		
		runSequence(sequence, 1000, false);
		
		SimpleMatrix output1 = predictor.predict(in1, 0, false);
		SimpleMatrix output2 = predictor.predict(in2, 0, false);
		SimpleMatrix output3 = predictor.predict(in3, 0, false);
		SimpleMatrix output4 = predictor.predict(in4, 0, false);
		
		SimpleMatrix diff1 = output1.minus(in2);	
		SimpleMatrix diff2 = output2.minus(in3);	
		SimpleMatrix diff3 = output3.minus(in4);	
		SimpleMatrix diff4 = output4.minus(in1);	
		
		assertEquals(0, diff1.normF(), e);
		assertEquals(0, diff2.normF(), e);
		assertEquals(0, diff3.normF(), e);
		assertEquals(0, diff4.normF(), e);
	}
	*/
	
	private SimpleMatrix runSequence(SimpleMatrix[] sequence, int iterations, boolean printPredictionMatrix){
		double initialLearning = 1;
		double curLearningRate = initialLearning;
		
		SimpleMatrix inputBefore = null;
		SimpleMatrix expectedTransferMatrix = new SimpleMatrix(sequence[0].getNumElements(), sequence[0].getNumElements());
		//expectedTransferMatrix.set(1);
		normalizeColumns(expectedTransferMatrix);
		for (int i = 0; i < iterations; i++){
			for (SimpleMatrix m : sequence){
				predictor.predict(m,curLearningRate, true);
				SimpleMatrix updates = calculateExpectedTransitionMatrixUpdate(inputBefore, m, curLearningRate);
				expectedTransferMatrix = updates.plus(expectedTransferMatrix);
				normalizeColumns(expectedTransferMatrix);
				SimpleMatrix actualTransferMatrix = predictor.getConditionalPredictionMatrix();
				
				boolean equal = isEqual(actualTransferMatrix, expectedTransferMatrix);
				//assertTrue(equal);
				
				inputBefore = m;
				
				if (printPredictionMatrix){
					System.out.println("Input");
					m.print();
					System.out.println();
					System.out.println("Transfer matrix");
					predictor.getConditionalPredictionMatrix().print();
					System.out.println();
					System.out.println();
				}
			}
			//curLearningRate = initialLearning * Math.exp(-(double) i / iterations);
			if ( curLearningRate < 0.01) curLearningRate = 0.01;
		}
		return expectedTransferMatrix;
	}
	
	@Test
	public void testPredict_ComplexSequence	() {
		predictor = new FirstOrderPredictor(2);
		//Sequence: 1 2 2 4 3 1 2 4 4 2
		SimpleMatrix[] sequence = {in1, in2, in2, in4, in3, in1, in2, in4, in4, in2};
		
		//SimpleMatrix[] sequence = {in1, in2, in2, in4, in3};
		
		SimpleMatrix m = runSequence(sequence, 1000, false);
		
		
		double[][] expectedTransitionProbabilities = {
				{0, 0.25, 1, 0},
				{1, 0.25, 0, 0.333333333},
				{0, 0, 0, 0.333333333},
				{0, 0.5, 0, 0.333333333}
		};
		
		SimpleMatrix expectedTransitionMatrix = new SimpleMatrix(expectedTransitionProbabilities);
		SimpleMatrix actualTransitionMatrix = predictor.getConditionalPredictionMatrix();
		actualTransitionMatrix.print();
		
		assertTrue(isEqual(actualTransitionMatrix, expectedTransitionMatrix));
		
	}
	
	private boolean isEqual(SimpleMatrix a, SimpleMatrix b){
		SimpleMatrix diff = a.minus(b);
		double dist = diff.normF();
		return (dist > 0 - e && dist < 0 + e);
	}
	
	private SimpleMatrix calculateExpectedTransitionMatrixUpdate(SimpleMatrix inputBefore, SimpleMatrix inputNow, double learningRate){
		SimpleMatrix deltaMatrix = new SimpleMatrix(inputNow.getNumElements(), inputNow.getNumElements());
		if (inputBefore != null){
			for (int k = 0; k < inputBefore.getNumElements(); k++){
				double k_now = inputNow.get(k);
				double k_before = inputBefore.get(k);
				double delta2 = Math.max(k_now - k_before, 0);
				for (int h = 0; h < inputNow.getNumElements(); h++){
					double h_now = inputNow.get(h);
					double h_before = inputBefore.get(h);
					double delta1 = Math.max(h_before - h_now, 0);
					
					
					if (k == h & h_now == 1 && h_before == 1){
						delta1 = 1;
						delta2 = 1;
					}
					
					double delta = learningRate * delta1 * delta2;
					deltaMatrix.set(k, h, delta);
				}
				
			}
			

		}
		
		return deltaMatrix;
	}
	
	private void normalizeColumns(SimpleMatrix m){
		for (int col = 0; col < m.numCols(); col++){
			SimpleMatrix column = m.extractVector(false, col);
			double sum = column.elementSum();
			if (sum != 0) column = column.divide(sum);
			m.setColumn(col, 0, column.getMatrix().data);
		}
	}

}
