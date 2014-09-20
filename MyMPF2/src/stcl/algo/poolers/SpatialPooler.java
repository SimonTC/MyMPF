package stcl.algo.poolers;

import java.util.Random;
import java.util.Vector;

import javax.swing.plaf.basic.BasicInternalFrameTitlePane.MaximizeAction;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.som.SOM;
import stcl.algo.som.SomNode;
import stcl.algo.util.ExponentialDecayFunction;

public class SpatialPooler {
	
	//Weight matrix
	protected SOM som;
	
	//Matrice used 
	protected SimpleMatrix matrix_Error; //Squared error of the activation in the SOM
	protected SimpleMatrix matrix_Activation; //Probablitiy that model i,j in the SOM is the best model for the given input
	
	//Variables for learning
	protected double curLearningRate;
	protected double curNeighborhoodRadius;
	protected double curNoiseMagnitude;
	private int tick;
	
	//Decay function
	//TODO: They probably have to be changed at some point. Especially noiseDecay. It should be dependant on activitty from lower levels.
	//TODO: They should probably be moved to the Neurocortical unit
	private ExponentialDecayFunction learningDecay;
	private ExponentialDecayFunction radiusDecay;
	private ExponentialDecayFunction noiseDecay;
	
	//Misc
	protected Random rand;
	
	
	public SpatialPooler(Random rand, int maxIterations, int inputLength, int mapSize) {
		this.rand = rand;
		som = new SOM(mapSize, mapSize, inputLength, rand);
		matrix_Error = new SimpleMatrix(mapSize, mapSize);
		matrix_Activation = new SimpleMatrix(mapSize, mapSize);
		tick = 0;
		
		//TODO: change start rates to something from a parameter file / given as parameter to constructor
		curLearningRate = 1;
		curNeighborhoodRadius = mapSize / 2;
		curNoiseMagnitude = 1;
		
		//TODO: Something has to be done about this
		learningDecay = new ExponentialDecayFunction(curLearningRate, 0.01, maxIterations);
		radiusDecay = new ExponentialDecayFunction(curNeighborhoodRadius, 0.01, curNeighborhoodRadius);
		noiseDecay = new ExponentialDecayFunction(curNoiseMagnitude, 0.01, maxIterations);
	}
	
	public void tick(){
		tick++;
		curLearningRate = learningDecay.decayValue(tick);
		curNeighborhoodRadius = radiusDecay.decayValue(tick);
		curNoiseMagnitude = noiseDecay.decayValue(tick);
	}
	
	public SimpleMatrix feedForward(SimpleMatrix input){
		//Adjust weights of SOM
		som.step(input, curLearningRate, curNeighborhoodRadius);
		
		//Collect error matrix
		matrix_Error = som.getErrorMatrix();
		
		//Compute ActivationMatrix
		double maxError = matrix_Error.elementMaxAbs();
		computeActivationMatrix(maxError, matrix_Error);
		
		return matrix_Activation;
	}
	/**
	 * 
	 * @param input
	 * @return
	 */
	public SimpleMatrix feedBack(SimpleMatrix input){
		//Transform bias matrix into vector
		double[] vector = input.getMatrix().data;
		
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
	
	protected void computeActivationMatrix(double maxError, SimpleMatrix errorMatrix){
		SimpleMatrix m = errorMatrix.divide(maxError);
		matrix_Activation.set(1);
		matrix_Activation = matrix_Activation.minus(m);		
	}
	
	protected SimpleMatrix addNoise(SimpleMatrix m, double noiseMagnitude){
		double noise = (rand.nextDouble() - 0.5) * 2 * noiseMagnitude;
		m = m.plus(noise);
		return m;
	}
	
	public SOM getSOM(){
		return som;
	}
}
