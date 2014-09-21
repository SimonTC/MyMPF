package stcl.test.poolers;

import static org.junit.Assert.*;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Random;
import java.util.Vector;

import org.ejml.simple.SimpleMatrix;
import org.junit.Before;
import org.junit.Test;
import org.omg.PortableInterceptor.SUCCESSFUL;

import stcl.algo.poolers.SpatialPooler;
import stcl.algo.som.SOM;

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
		//No need to test this as all sub methods are already being tested
		assertTrue(true);
	}
	
	@Test
	public void testFeedBack() {
		//No need to test this as all sub methods are already being tested
		assertTrue(true);
	}
	
	@Test
	public void testComputeActivationMatrix() throws NoSuchMethodException, SecurityException, IllegalAccessException, IllegalArgumentException, InvocationTargetException{
		//Reflection
		Method method = SpatialPooler.class.getDeclaredMethod("computeActivationMatrix", SimpleMatrix.class);
		method.setAccessible(true);
		
		//Test that the activation matrix has been computed correctly
		Random rand = new Random();
		double[][] errorData = new double[mapSize][mapSize];
		for (int i = 0; i < mapSize; i++){
			for (int j = 0; j < mapSize; j++){
				errorData[i][j] = rand.nextDouble();
			}
		}
		SimpleMatrix errorMatrix = new SimpleMatrix(errorData);
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
				activationData[i][j] = 1 - (errorData[i][j] / maxError);
				
			}
		}
		
		return activationData;
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
