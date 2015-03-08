package stcl.test.som;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.fail;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;
import org.junit.Before;
import org.junit.Test;

import stcl.algo.poolers.RSOM;
import dk.stcl.core.basic.containers.SomMap;
import dk.stcl.core.basic.containers.SomNode;

public class RSOMTest {
	private RSOM rsom;
	int rsomSize = 5;
	int inputVectorLength = 3;
	Random rand = new Random();
	double leakyCoefficient = 0.7;
	
	@Before
	public void setUp() throws Exception {
		rsom = new RSOM(rsomSize, inputVectorLength, rand, 0.1, 0.125, 2, leakyCoefficient);
	}

	@Test
	public void testGetBMU_NoInputParameter() {		
		//Collect map
		SomMap leakyMap = rsom.getLeakyDifferencesMap();
		
		//Set all nodes to same value int the map of leaky differences
		leakyMap.reset(1);
		
		//Expect return of a non null node
		SomNode bmu = rsom.getBMU();
		assertNotNull(bmu);
		
		//Set one node's vector value to be all zero
		leakyMap.get(5).getVector().set(0);		
		
		//Expect the returned node to be the node with all values set to zero
		bmu = rsom.getBMU();
		for (double d : bmu.getVector().getMatrix().data){
			assertEquals(0, d, 0.0001);
		}
		
	}
	
	@Test (expected = UnsupportedOperationException.class)  
	public void testGetBMU_WithInputParameter() {		
		SimpleMatrix input = new SimpleMatrix();
		rsom.getBMU(input);
	}

	@Test
	public void testWeightAdjustment() {
		//Create input vector
		SimpleMatrix input1 = SimpleMatrix.random(1, inputVectorLength, 0, 1, rand);
		
		//Do a step to update the old leaky differences map. 
		double radius = 10;
		double learningRate = 0.2;
		rsom.step(input1, learningRate, radius); //Radius set to ten to be sure that all nodes are updated
		
		//Choose a node to test and collect vector values
		double[] oldWeightValues = new double[inputVectorLength];
		SomNode chosen = rsom.getWeighttMap().get(3, 4);
		double[] tmp = chosen.getVector().getMatrix().data;
		for (int i = 0; i < inputVectorLength; i++){
			oldWeightValues[i] = tmp[i];
		}
		
		//Create new input vector
		SimpleMatrix input2 = SimpleMatrix.random(1, inputVectorLength, 0, 1, rand);
		
		//step through and get bmu
		SomNode bmu = rsom.step(input2, learningRate, radius);
		
		//Collect leaky differences for the chosen node
		double[] leakyValues = new double[inputVectorLength];
		double[] tmp2 = rsom.getLeakyDifferencesMap().get(3, 4).getVector().getMatrix().data;
		for (int i = 0; i < inputVectorLength; i++){
			leakyValues[i] = tmp2[i];
		}
		
		//Calculate expected vector values
		double squaredRadius = radius * radius;
		double squaredDistance = chosen.distanceTo(bmu);		
		double learningEffect  = Math.exp(-(squaredDistance / (2 * squaredRadius)));
		
		double[] actualWeights = chosen.getVector().getMatrix().data;
		for (int i = 0; i < inputVectorLength; i++){
			double oldWeight = oldWeightValues[i];
			double leakyValue = leakyValues[i];
			double expectedWeight = oldWeight + learningRate * learningEffect * leakyValue;
			assertEquals(expectedWeight, actualWeights[i], 0.0001);
		}
		
	}

	@Test
	public void testRSOM() {
		fail("Not yet implemented");
	}
	
	@Test
	public void testUpdateLeakyDifferences(){
		fail("Not yet implemented");
	}

}
