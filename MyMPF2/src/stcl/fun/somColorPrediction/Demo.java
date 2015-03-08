package stcl.fun.somColorPrediction;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import dk.stcl.core.som.ISOM;
import stcl.algo.poolers.SpatialPooler;
import stcl.algo.predictors.Predictor_VOMM;

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
		//FirstOrderMM_Original predictor = new FirstOrderMM_Original(10);
		Predictor_VOMM predictor = new Predictor_VOMM(5, 0.1, rand);
		
		red = rand.nextDouble();
		green = rand.nextDouble();
		blue = rand.nextDouble();
		
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
			
			SimpleMatrix ffOutput = pooler.feedForward(input);
			
			SimpleMatrix predictionMatrix = predictor.predict(ffOutput);
			
			predictedColor = nextColor(predictionMatrix, pooler);
			
			double error = distance(predictedColor, currentColor);
			
			System.out.println(i + " " + error);
			
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
		double dist = 0.0;
		
		for (int i = 0 ; i < prediction.length; i++){
			dist += Math.abs(prediction[i] - actual[i]);
		}
		
		dist = dist / prediction.length;
		
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
	
	private SimpleMatrix chooseMax(SimpleMatrix m, ISOM som){
		
		//Go through bias vector until value is >= random number
		double max = Double.NEGATIVE_INFINITY;
		int maxID = -1;
		for (int i = 0; i < m.getNumElements(); i++){
			double value = m.get(i);
			if (value > max){
				maxID = i;
				max = value;
			}
		}		
		
		//Choose model from som
		SimpleMatrix model = som.getNode(maxID).getVector();
		
		//System.out.println("Chose model: " + id);
		
		return model;
		
	}

}
