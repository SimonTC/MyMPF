package stcl.algo.poolers;

import java.util.Random;
import java.util.Vector;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.som.SOMMap;
import stcl.algo.som.SomNode;
import stcl.test.util.ExponentialDecayFunction;

public class SpatialPooler {
	
	private SOMMap som;
	
	//Matrice used 
	private SimpleMatrix matrix_Error; //Squared error of the activation in the SOM
	private SimpleMatrix matrix_Activation; //Probablitiy that model i,j in the SOM is the best model for the given input
	
	//Variables for learning
	private double curLearningRate;
	private double curNeighborhoodRadius;
	private double curNoiseMagnitude;
	private int tick;
	
	//Decay function
	//TODO: They probably have to be changed at some point. Especially noiseDecay. It should be dependant on activitty from lower levels.
	//TODO: They should probably be moved to the Neurocortical unit
	private ExponentialDecayFunction learningDecay;
	private ExponentialDecayFunction radiusDecay;
	private ExponentialDecayFunction noiseDecay;
	
	//Misc
	private Random rand;
	
	
	public SpatialPooler(Random rand, int maxIterations, int inputLength, int mapSize) {
		this.rand = rand;
		som = new SOMMap(mapSize, mapSize, inputLength, rand);
		matrix_Error = new SimpleMatrix(mapSize, mapSize);
		matrix_Activation = new SimpleMatrix(mapSize, mapSize);
		
		//TODO: change start rates to something from a parameter file / given as parameter to constructor
		curLearningRate = 1;
		curNeighborhoodRadius = 1;
		curNoiseMagnitude = 1;
		
		//TODO: Something has to be done about this
		learningDecay = new ExponentialDecayFunction(curLearningRate, 0.01, maxIterations, Math.E);
		radiusDecay = new ExponentialDecayFunction(curNeighborhoodRadius, 0.01, maxIterations, mapSize / 2);
		noiseDecay = new ExponentialDecayFunction(curNoiseMagnitude, 0.01, maxIterations, Math.E);
	}

	public void tick(){
		tick++;
		curLearningRate = learningDecay.decayValue(tick);
		curNeighborhoodRadius = radiusDecay.decayValue(tick);
		curNoiseMagnitude = noiseDecay.decayValue(tick);
	}
	
	public SimpleMatrix feedForward(SimpleMatrix input){
		SomNode inputNode = new SomNode(input);
		
		//Adjust weights of SOM
		som.step(inputNode, curLearningRate, curNeighborhoodRadius);
		
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
