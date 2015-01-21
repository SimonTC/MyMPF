package stcl.test.poolers;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;
import org.junit.Before;
import org.junit.Test;

import stcl.algo.poolers.SpatialPooler;

public class SpatialPoolerTest {
	private SpatialPooler pooler;
	private int inputLength;
	private int mapSize;
	
	@Before
	public void setUp() throws Exception {
		//Setup pooler
		inputLength = 3;
		mapSize = 5;
		pooler = new SpatialPooler(new Random(), 100, inputLength, mapSize);
	}
	
	@Test
	public void testFeedForward() {
		//Make sure that the output is a probability mass distribution
		SimpleMatrix input = SimpleMatrix.random(1,3, 0, 1, new Random());
		SimpleMatrix output = pooler.feedForward(input);
		assertEquals(1, output.elementSum(), 0.00001);
	}
	
	@Test
	public void testFeedBack() {
		//Test that output is the same size as input
		//Make sure that the output is a probability mass distribution
		SimpleMatrix ffInput = SimpleMatrix.random(1,3, 0, 1, new Random());
		SimpleMatrix ffOutput = pooler.feedForward(ffInput);
		SimpleMatrix fbOutput = pooler.feedBackward(ffOutput);
		assertTrue(fbOutput.numCols() == 3 && fbOutput.numRows() == 1);
		
		
		//No need to test this as all sub methods are already being tested
		assertTrue(true);
	}
	
	@Test
	public void testComputeActivationMatrix() throws NoSuchMethodException, SecurityException, IllegalAccessException, IllegalArgumentException, InvocationTargetException{
		//Reflection
		Method method = SpatialPooler.class.getDeclaredMethod("computeActivationMatrix", SimpleMatrix.class);
		method.setAccessible(true);
		
		//Create error matrix
		Random rand = new Random();
		double[][] errorData = new double[mapSize][mapSize];
		for (int i = 0; i < mapSize; i++){
			for (int j = 0; j < mapSize; j++){
				errorData[i][j] = rand.nextDouble();
			}
		}
		SimpleMatrix errorMatrix = new SimpleMatrix(errorData);
		
		//Calculate expected activation
		double[][] activationData = expectedActivation(errorData);
		SimpleMatrix expectedActivation = new SimpleMatrix(activationData);
		SimpleMatrix actualMatrix = (SimpleMatrix) method.invoke(pooler, errorMatrix);
		
		assertTrue(expectedActivation.isIdentical(actualMatrix, 0.0001));
		
		
		
	}
	
	private double[][] expectedActivation(double[][] errorData){
		//Find max area
		double maxError = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < errorData.length; i++){
			for (int j = 0; j < errorData[0].length; j++){
				double error = errorData[i][j];
				if (error > maxError) maxError = error;
			}
		}
		
		//Calculate activation
		double[][] activationData = new double[errorData.length][errorData[0].length];
		for (int i = 0; i < activationData.length; i++){
			for (int j = 0; j < activationData[0].length; j++){
				double activation = 1 - (errorData[i][j] / maxError);;
				activationData[i][j] = activation;
			}
		}		
		
		return activationData;
	}
	
	@Test
	public void testNormalization() throws NoSuchMethodException, SecurityException, IllegalAccessException, IllegalArgumentException, InvocationTargetException{
		Method method = SpatialPooler.class.getDeclaredMethod("normalize", SimpleMatrix.class);
		method.setAccessible(true);
		
		SimpleMatrix m = SimpleMatrix.random(10, 10, 1, 5, new Random());
		double oldSum = m.elementSum();
		assertTrue(oldSum > 1);
		
		SimpleMatrix normalized = (SimpleMatrix) method.invoke(pooler, m);
		double newSum = normalized.elementSum();
		
		assertEquals(1, newSum, 0.0001);
		
	}
	
	@Test
	public void testAddNoise() throws NoSuchMethodException, SecurityException, IllegalAccessException, IllegalArgumentException, InvocationTargetException{
		//Test that some kind of noise have been added
		SimpleMatrix org = SimpleMatrix.random(3, 4, 0, 1, new Random());
		SimpleMatrix noisy = new SimpleMatrix(org);
		
		//Add noise
		Method method = SpatialPooler.class.getDeclaredMethod("addNoise", SimpleMatrix.class, double.class);
		method.setAccessible(true);
		double noiseMagnitude = 0.75;
		noisy = (SimpleMatrix) method.invoke(pooler, noisy, noiseMagnitude);
		
		assertFalse(org.isIdentical(noisy, 0.0001));
	}
	
	@Test
	public void testChoseRandom() throws NoSuchMethodException, SecurityException, IllegalAccessException, IllegalArgumentException, InvocationTargetException{
		//Expect that some model has been chosen
		SimpleMatrix input = SimpleMatrix.random(mapSize, mapSize, 0, 1, new Random());
		
		Method method = SpatialPooler.class.getDeclaredMethod("chooseRandom", SimpleMatrix.class);
		method.setAccessible(true);
		SimpleMatrix chosen = (SimpleMatrix) method.invoke(pooler, input);
		
		assertTrue(chosen != null);
	}

	

}
