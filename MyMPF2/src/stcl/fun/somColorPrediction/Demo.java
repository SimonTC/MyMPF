package stcl.fun.somColorPrediction;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import dk.stcl.core.som.ISOM;
import stcl.algo.poolers.SpatialPooler;
import stcl.algo.predictors.CopyOfFirstOrderPredictor;
import stcl.algo.predictors.FirstOrderPredictor;

public class Demo {
	
	private double red, green, blue;
	private Random rand = new Random(1234);
	
	public static void main(String[] args){
		Demo demo = new Demo();
		demo.runExperiment(3000);
	}
	
	public void runExperiment(int iterations){
		
		SpatialPooler pooler = new SpatialPooler(rand, 3, 10, 0.1, 2, 0.125);
		//FirstOrderPredictor predictor = new FirstOrderPredictor(10);
		CopyOfFirstOrderPredictor predictor = new CopyOfFirstOrderPredictor(10);
		
		red = 0;
		green = 0;
		blue = 0;
		
		double[] predictedColor = null;
		double initialLearning = 1;
		double curLearningRate = initialLearning;
		
		
		for (int i = 1; i <= iterations; i++){
			red = updateColor(red, 0.2);
			green = updateColor(green, 0.5);
			blue = updateColor(blue, -0.1);
			
			double[] currentColor = {red, green, blue};
			
			double[][] inputData = {currentColor};
			SimpleMatrix input = new SimpleMatrix(inputData);
			
			SimpleMatrix ffOutput = pooler.feedForward(input, false);
			double sum = ffOutput.elementSum();
			ffOutput = ffOutput.divide(sum);
			SimpleMatrix predictionMatrix = predictor.predict(ffOutput, curLearningRate, true);
			
			predictedColor = nextColor(predictionMatrix, pooler);
			
			double error = distance(predictedColor, currentColor);
			
			System.out.println("Iteration: " + i + " Error " + error);
			
			curLearningRate = initialLearning * Math.exp(-(double) i / iterations);
			if ( curLearningRate < 0.01) curLearningRate = 0.01;
			
		}
	}
	
	private double updateColor(double oldColor, double bias){
		double color = oldColor + bias + (rand.nextGaussian() - 0.5) * 0.1;
		
		//Let values wrap around zero and one
		if (color > 1){
			double diff = color - 1;
			color = diff;
		} else if (color < 0){
			color = 1 + color;
		}
		
		return color;
	}
	
	private double[] nextColor (SimpleMatrix predictionMatrix, SpatialPooler pooler){
		double[] color = chooseRandom(predictionMatrix, pooler.getSOM()).getMatrix().data;
		return color;
	}
	
	private double distance(double[] prediction, double[] actual){
		double[][] pred = {prediction};
		double[][] act = {actual};
		
		SimpleMatrix predMatrix = new SimpleMatrix(pred);
		SimpleMatrix actMatrix = new SimpleMatrix(act);
		
		SimpleMatrix diff = predMatrix.minus(actMatrix);
		double dist = diff.normF();
		
		return dist;
	}
	
	private SimpleMatrix chooseRandom(SimpleMatrix input, ISOM som){
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
		
		id--; //We have to subtract to be sure we get the correct model
		
		//Choose model from som
		SimpleMatrix model = som.getNode(id).getVector();
		
		//System.out.println("Chose model: " + id);
		
		return model;
		
	}

}
