package stcl.test.som;

import static org.junit.Assert.*;

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
	public void testSOMMap() {
		int height = 26;
		int width = 13;
		int vectorLength = 6;
		Random rand = new Random();
		SOMMap map = new SOMMap(width, height, vectorLength, rand);
		
		//Test that size of model is N x M
		Vector<SomNode> models = map.getModels();
		assertTrue(models.size() == height * width);
		
		//Test that Nodes with vectors have been created for all coordinates
		for (int y = 0; y < height; y++){
			for (int x = 0; x < width; x++){
				SomNode n = models.get(x * y);
				assertNotNull(n);
				SimpleMatrix m = n.getVector();
				assertNotNull(m);
			}
		}
	}

	@Test
	public void testGetBMU() {
		fail("Not yet implemented");
	}

	@Test
	public void testAdjustWeights() {
		fail("Not yet implemented");
	}

	@Test
	public void testGetModels() {
		fail("Not yet implemented");
	}

	@Test
	public void testGetErrorMatrix() {
		fail("Not yet implemented");
	}

	@Test
	public void testGetModel() {
		fail("Not yet implemented");
	}

}
