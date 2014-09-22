package stcl.test.poolers;

import static org.junit.Assert.*;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;
import org.junit.Before;
import org.junit.Test;

import stcl.algo.poolers.TemporalPooler;

public class TemporalPoolerTest {

	@Before
	public void setUp() throws Exception {
	}

	@Test
	public void testFeedForward() {
		//Expect output to be a mass probability distribution
		TemporalPooler pooler = new TemporalPooler(new Random(), 100, 3, 5, 0.7);
		SimpleMatrix input = SimpleMatrix.random(1,3, 0, 1, new Random());
		SimpleMatrix output = pooler.feedForward(input);
		assertEquals(1, output.elementSum(), 0.0001);
		
		assertTrue(output.numCols() == 5 && output.numRows() == 5);

	}

	@Test
	public void testTemporalPooler() {
		fail("Not yet implemented");
	}

}
