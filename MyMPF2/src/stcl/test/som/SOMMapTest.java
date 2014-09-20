package stcl.test.som;

import static org.junit.Assert.*;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Random;
import java.util.Vector;

import org.ejml.simple.SimpleMatrix;
import org.junit.Before;
import org.junit.Test;

import stcl.algo.som.SOMMap;
import stcl.algo.som.SomNode;

public class SOMMapTest {

	@Before
	public void setUp() throws Exception {
	}

	@Test
	public void testStep(){
		//Create map
		int rows = 5;
		int columns = 5;
		int vectorLength = 2;
		Random rand = new Random();
		SOMMap map = new SOMMap(columns, rows, vectorLength, rand);
		
		//Create input
		double[] input1 = {0,1};
		double[] input2 = {1,0};
		
		//Read initial vector values
		double[][][] initialValues = new double[rows] [columns][vectorLength];
		for (int row = 0; row < rows; row++){
			for (int col = 0; col < columns; col++){
				SomNode n = map.getModel(row, col);
				for (int i = 0; i < vectorLength; i++){
					initialValues[row][col][i] = n.getVector().get(i);
				}
			}
		}
		
		//Step1
		double radius = 3;
		double learningRate = 0.7;
		double squareRadius = radius * radius;
		SomNode bmu1 = map.step(input1, learningRate, radius);
	
		
		//Read new vector values and check that they are the expected values
		double[][][] valuesAfterStep1 = new double[rows][columns] [vectorLength];
		for (int row = 0; row < rows; row++){
			for (int col = 0; col < columns; col++){
				SomNode n = map.getModel(row, col);
				for (int i = 0; i < vectorLength; i++){
					valuesAfterStep1[row][col][i] = n.getVector().get(i);
				}
			}
		}
		
		//Step 2
		SomNode bmu2 = map.step(input2, learningRate, radius);
				
		//Read new vector values and check that they are the expected values
		double[][][] valuesAfterStep2 = new double[rows][columns] [vectorLength];
		for (int row = 0; row < rows; row++){
			for (int col = 0; col < columns; col++){
				SomNode n = map.getModel(row, col);
				for (int i = 0; i < vectorLength; i++){
					valuesAfterStep2[row][col][i] = n.getVector().get(i);
				}
			}
		}
		
		//Test that not all values have changed between initial and step 1
		int counter = 0;
		for (int row = 0; row < rows; row++){
			for (int col = 0; col < columns; col++){
				for (int i = 0; i < vectorLength; i++){
					if (initialValues[row][col][i] == valuesAfterStep1[row][col][i]){
						counter++;
					}
				}
			}
		}
		assertNotEquals(counter, rows * columns * vectorLength, 0.1);
		
		//Test that not all values have changed between step 1 and step 2
		counter = 0;
		for (int row = 0; row < rows; row++){
			for (int col = 0; col < columns; col++){
				for (int i = 0; i < vectorLength; i++){
					if (valuesAfterStep2[row][col][i] == valuesAfterStep1[row][col][i]){
						counter++;
					}
				}
			}
		}
		assertNotEquals(counter, rows * columns * vectorLength, 0.1);
		
		//Test that expected change have been performed between after step 1
		for (int row = 0; row < rows; row++){
			for (int col = 0; col < columns; col++){
				SomNode n = map.getModel(row, col);
				double squaredist = n.distanceTo(bmu1);
				double learningEffect = 0;
				if (squaredist <= squareRadius){
					learningEffect = learningEffect(squaredist, squareRadius);
				}
				System.out.println();
				System.out.println("Node at (" + n.getCol() + "," + n.getRow() + ") BMU at (" + bmu1.getCol() + "," + bmu1.getRow() + ") Squared distance: " + squaredist);
				System.out.println("Learning rate: " + learningRate + " Learning effect: " + learningEffect);
				for (int i = 0; i < vectorLength; i++){
					double oldValue = initialValues[row][col][i];
					double inputValue = input1[i];
					double newValue = valuesAfterStep1[row][col][i];
					double expectedValue = expectedValue(oldValue, learningRate, learningEffect, inputValue);
					System.out.printf("%-10.5s  %-10.5s %-10.5s %-10.5s%n", oldValue, inputValue, expectedValue, newValue);
					assertEquals(expectedValue, newValue, 0.0001);
				}
			}
		}
		
	/*
		//Test that expected change have been performed
		
		double[][][] valuesAfterStep1 = new double[rows][columns] [vectorLength];
		for (int row = 0; row < rows; row++){			
			for (int col = 0; col < columns; col++){
				SomNode n = map.getModel(row, col);
				double squareDist = bmu.distanceTo(n);
				double learningEffect = learningEffect(squareDist, squareRadius);
				if (squareDist > squareRadius) learningEffect = 0;
				for (int i = 0; i < vectorLength; i++){
					double oldValue = initialValues[row][col][i];
					double expected = expectedValue(oldValue, learningRate, learningEffect, input1[i]);	
					double newValue = n.getVector().get(i);
					valuesAfterStep1[row][col][i] = newValue;
					assertEquals(expected, newValue, 0.0001);
				}			
				
			}
			
		}
		
		//Step 2
		bmu = map.step(input2, learningRate, radius);
		
		//Read new vector values and check that they are the expected values
		double[][][] valuesAfterStep2 = new double[rows][columns] [vectorLength];
		for (int row = 0; row < rows; row++){
			for (int col = 0; col < columns; col++){
				SomNode n = map.getModel(row,col);
				double squareDist = bmu.distanceTo(n);
				double learningEffect = learningEffect(squareDist, squareRadius);
				if (squareDist > squareRadius) learningEffect = 0;
				for (int i = 0; i < vectorLength; i++){
					double oldValue = valuesAfterStep1[row][col][i];
					double expected = expectedValue(oldValue, learningRate, learningEffect, input1[i]);	
					double newValue = n.getVector().get(i);
					valuesAfterStep2[row][col][i] = newValue;
					assertEquals(expected, newValue, 0.0001);
				}
				
				
			}
			
		}	
		*/
		
	}
	
	private double expectedValue(double oldValue, double learningRate, double learningEffect, double inputValue){
		double d = oldValue + learningRate * learningEffect * (inputValue - oldValue);
		return d;
	}
	
	
	
	@Test
	public void testSOMMap() {
		int height = 26;
		int width = 13;
		int vectorLength = 6;
		Random rand = new Random();
		SOMMap map = new SOMMap(width, height, vectorLength, rand);
		
		//Test that size of model is N x M
		SomNode[] models = map.getModels();
		assertTrue(models.length == height * width);
		
		//Test that Nodes with vectors have been created for all coordinates
		for (int y = 0; y < height; y++){
			for (int x = 0; x < width; x++){
				SomNode n = map.getModel(y, x);
				assertNotNull(n);
				SimpleMatrix m = n.getVector();
				assertNotNull(m);
			}
		}
	}

	@Test
	public void testGetBMU() {
		int height = 26;
		int width = 13;
		int vectorLength = 6;
		Random rand = new Random();
		SOMMap map = new SOMMap(width, height, vectorLength, rand);
		
		double[] correctValues = {0,1,1,0,1,0};
		SomNode correct = new SomNode(correctValues, 5, 6);
		map.set(correct, 5, 6);
		
		SomNode bmu = map.getBMU(correctValues);
		
		assertTrue(bmu.getVector().isIdentical(correct.getVector(), 0.000001));

	}

	
	private double learningEffect(double squaredDist, double squaredRadius){
		double d = Math.exp(- squaredDist / (2*squaredRadius));
		return d;
	}
		
	@Test
	public void testLearningEffect_RandomCoordinate() throws NoSuchMethodException, SecurityException, IllegalAccessException, IllegalArgumentException, InvocationTargetException{
		//Create map
		Random rand = new Random();
		int rows = 7;
		int cols = 7;
		int vectorSize = 2;
		SOMMap map = new SOMMap(cols, rows, vectorSize, rand);

		Method method = SOMMap.class.getDeclaredMethod("learningEffect", double.class, double.class);
		method.setAccessible(true);
		
		//Test
		double squaredDist = 4;
		double squaredRadius = 7;
		
		double expected = learningEffect(squaredDist, squaredRadius);
		double actual = (double) method.invoke(map, squaredDist, squaredRadius);
		
		assertEquals(expected, actual, 0.00001);
	}
	
	@Test
	public void testLearningEffect_Coordinate00() throws NoSuchMethodException, SecurityException, IllegalAccessException, IllegalArgumentException, InvocationTargetException{
		//Create map
		Random rand = new Random();
		int rows = 7;
		int cols = 7;
		int vectorSize = 2;
		SOMMap map = new SOMMap(cols, rows, vectorSize, rand);

		Method method = SOMMap.class.getDeclaredMethod("learningEffect", double.class, double.class);
		method.setAccessible(true);
		
		//Test
		double squaredDist = 0;
		double squaredRadius = 7;
		
		double expected = learningEffect(squaredDist, squaredRadius);
		double actual = (double) method.invoke(map, squaredDist, squaredRadius);
		
		assertEquals(expected, actual, 0.00001);
	}
	
	@Test
	public void testGetModels() {
		//Create map
		Random rand = new Random();
		int rows = 7;
		int cols = 7;
		int vectorSize = 2;
		SOMMap map = new SOMMap(cols, rows, vectorSize, rand);
		
		int size = map.getModels().length;
		
		assertTrue(size == rows * cols);

	}

	@Test
	public void testGetErrorMatrix() {
		//Create map
		Random rand = new Random();
		int rows = 7;
		int cols = 7;
		int vectorSize = 2;
		SOMMap map = new SOMMap(cols, rows, vectorSize, rand);
		for (SomNode n :map.getModels()){
			n.getVector().set(0);
		}	
				
		//Before step 1 all entries in error matrix are zero
		SimpleMatrix errorMatrix = map.getErrorMatrix();
		assertEquals(0, errorMatrix.elementSum(), 0.0000000001);
		
		//Do step 1
		double learningRate = 0.8;
		double neighborhoodRadius = 2;
		double[] correct = {0.3,0.4};
		SomNode bmu = map.step(correct, learningRate, neighborhoodRadius);
		
		//After step 1 an error matrix have been created and contains expected errors
		//Expected error for all node = (0-0.3)^2 + (0-0.4)^2 = 0.25
		SimpleMatrix m = new SimpleMatrix(rows, cols);
		m.set(0.25);
		
		errorMatrix = map.getErrorMatrix().copy();
		assertTrue(errorMatrix.isIdentical(m, 0.000001));
		
		//After step 2 a new error matrix have been created
		double[] correct2 = {0.1,0.3};
		bmu = map.step(correct2, learningRate, neighborhoodRadius);
		SimpleMatrix errorMatrix2 = map.getErrorMatrix();
		
		assertFalse(errorMatrix.isIdentical(errorMatrix2, 0.000001));
	}

}
