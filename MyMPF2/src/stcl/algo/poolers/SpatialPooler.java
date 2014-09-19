package stcl.algo.poolers;

import java.util.Random;
import java.util.Vector;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.som.SOMMap;
import stcl.algo.som.SomNode;

public class SpatialPooler {
	
	SOMMap som;
	
	SimpleMatrix matrix_Error;
	SimpleMatrix matrix_Activation;
	
	private double curLearningRate;
	private double curNeighborhoodRadius;
	private double curNoiseMagnitude;
	private Random rand;
	
	
	public SpatialPooler(Random rand) {
		// TODO Auto-generated constructor stub
	}

	public SimpleMatrix feedForward(SimpleMatrix input){
		SomNode inputNode = new SomNode(input);
		
		//Find best matching unit and adjust weights of som
		SomNode bmu = som.step(inputNode, curLearningRate, curNeighborhoodRadius);
		
		//Collect error matrix
		matrix_Error = som.getErrorMatrix();
		
		//Compute ActivationMatrix
		double maxError = matrix_Error.elementMaxAbs();
		computeActivationMatrix(maxError);
		
		return matrix_Activation;
	}
	
	public SimpleMatrix feedBack(SimpleMatrix bias){
		//Transform bias matrix into vector
		double[] vector = bias.getMatrix().data;
		
		//Choose random number between 0 and 1
		double d = rand.nextDouble();
		
		//Go through bias vector until value is >= random number
		double tmp = 0;
		int id = 0;
		while (tmp < d && id < vector.length){
			tmp += vector[id++];
		}
		
		//Choose model from som
		SimpleMatrix model = som.getModel(id).getVector();
		
		//Add noise
		model = addNoise(model, curNoiseMagnitude);
		
		return model;
		
	}
	
	private void computeActivationMatrix(double maxError){
		SimpleMatrix m = matrix_Error.divide(maxError);
		matrix_Activation.set(1);
		matrix_Activation.minus(m);		
	}
	
	private SimpleMatrix addNoise(SimpleMatrix m, double noiseMagnitude){
		double noise = (rand.nextDouble() - 0.5) * 2 * noiseMagnitude;
		m = m.plus(noise);
		return m;
	}
}
