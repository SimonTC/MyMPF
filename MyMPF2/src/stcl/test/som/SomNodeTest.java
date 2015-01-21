package stcl.test.som;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;
import org.junit.Before;
import org.junit.Test;

import dk.stcl.som.containers.SomNode;

public class SomNodeTest {

	Random rand = new Random();
	int numColumns = 10;
	int numRows = 1;
	int col = 5;
	int row = 6;
	
	@Before
	public void setUp() throws Exception {
	}

	
	//Test constructor where vector values are initialized to random values
	@Test
	public void testConstructor_RandomInitialization() {
		
		SomNode node = new SomNode(numColumns, rand, col, row);
		
		//Test that internal vector is indeed a vector of size 1 x N
		SimpleMatrix v = node.getVector();
		assertEquals(numRows, v.numRows());
		assertEquals(numColumns, v.numCols());
		
		//Test that vector values are between 0 and 1
		double[] data = v.getMatrix().data;
		for (double d : data){
			assertTrue(d >= 0);
			assertTrue(d <=1);
		}
		
		//Test that vector values not all are equal
		boolean same = true;
		double oldD = 0;
		for (double d : data){
			if (d != oldD) same = false;
			oldD = d;
		}
		assertFalse(same);
		
		//Test that coordinate is set correctly
		assertEquals(col, node.getCol());
		assertEquals(row, node.getRow());
		
	}

	@Test
	public void testConstructor_VectorAndCoordinates() {
		SimpleMatrix vector = SimpleMatrix.random(numRows, numColumns, 0, 1, rand);
		SomNode node = new SomNode(vector, col, row);
		
		//Test that internal vector is equal to given vector
		boolean result = vector.isIdentical(node.getVector(), 0.00000001);
		assertTrue(result);
		
		//Test that coordinaes are correct
		assertEquals(col, node.getCol());
		assertEquals(row, node.getRow());

	}

	@Test
	public void testConstructor_OnlyVector() {
		SimpleMatrix vector = SimpleMatrix.random(numRows, numColumns, 0, 1, rand);
		SomNode node = new SomNode(vector, col, row);
		
		//Test that internal vector is equal to given vector
		boolean result = vector.isIdentical(node.getVector(), 0.00000001);
		assertTrue(result);
	}

	@Test
	public void testAdjustValues() {
		//Create node vector
		double[][] vectorValues = {{0.3, 0.7, 0.38}};
		SimpleMatrix nodeVector = new SimpleMatrix(vectorValues);
		
		//Create input vector
		double[][] inputValues = {{0, 0.3, 0.26}};
		SimpleMatrix inputVector = new SimpleMatrix(inputValues);
		
		//Manually calculate expected value adjustments
		double learningRate = 0.05;
		double learningEffect = 0.78;
		double[][] adjustedData = new double[1][3];
		for (int i = 0; i < adjustedData[0].length; i++){
			adjustedData[0][i] = vectorValues[0][i] + learningRate * learningEffect * (inputValues[0][i] - vectorValues[0][i]);
		}
		
		SimpleMatrix adjustedVector = new SimpleMatrix(adjustedData);
		
		//Test
		SomNode node = new SomNode(nodeVector);
		node.adjustValues(inputVector, learningRate, learningEffect);		
		boolean result = adjustedVector.isIdentical(node.getVector(), 0.00000001);
		assertTrue(result);
	}

	@Test
	public void testSquaredDifference() {
		//Create node vector
		double[][] vectorValues = {{0.3, 0.7, 0.38}};
		SimpleMatrix nodeVector = new SimpleMatrix(vectorValues);

		//Create input vector
		double[][] inputValues = {{0.5, 0.3, 0.26}};
		SimpleMatrix inputVector = new SimpleMatrix(inputValues);
		
		//Manually calculate square diff
		double squaredDiff = 0;
		for (int i = 0; i < vectorValues[0].length; i++){
			double diff = vectorValues[0][i] - inputValues[0][i];
			diff *= diff;
			squaredDiff += diff;
		}
		
		SomNode node = new SomNode(nodeVector);
		
		double result = node.squaredDifference(inputVector);
		
		assertEquals(squaredDiff, result, 0.000001);
		
		
	}

	@Test
	public void testDistanceTo_DifferentCoordinates() {
		SomNode n1 = new SomNode(2, rand, 5, 6);
		SomNode n2 = new SomNode(2, rand, 3, 2);
		
		double expected = Math.pow(5-3, 2) + Math.pow(6-2, 2);
		
		assertEquals(expected, n1.distanceTo(n2), 0.000001);
	}
	
	@Test
	public void testDistanceTo_SameCoordinates() {
		SomNode n1 = new SomNode(2, rand, 5, 6);
		SomNode n2 = new SomNode(2, rand, 5, 6);
		
		double expected = 0;
		
		assertEquals(expected, n1.distanceTo(n2), 0.000001);
	}


}
