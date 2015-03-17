package stcl.algo.brain.biasunits;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

/**
 * This Bias unit is placed betwen neocortical units to bias the values in the feedback pass towards values that might give better rewards
 * @author Simon
 *
 */
public class BiasUnit {

	private double biasInfluence; //Should be set by parameter
	private Random rand;
	private SimpleMatrix bias;
	
	public BiasUnit(int biasMatrixSize, double biasInfluence, Random rand) {
		this.biasInfluence = biasInfluence;
		this.rand = rand;
		this.bias = new SimpleMatrix(biasMatrixSize, biasMatrixSize);
		bias.set(1);
	}
	
	/**
	 * This bias units is placed between the different layers in the brain.
	 * Biases the feed back output from one neocortical unit to the next
	 * @param input
	 * @param correlationMatrix
	 * @param noiseMagnitude
	 * @return
	 */
	public SimpleMatrix biasFBSpatialOutput(SimpleMatrix input, SimpleMatrix correlationMatrix, double noiseMagnitude){
		SimpleMatrix s = calculateS(correlationMatrix); //TODO: FInd another name
		
		double[] sData = s.getMatrix().data;
		
		double massValue = (double) 1 / (s.getNumElements());
		for (int i = 0; i < sData.length; i++){
			double delta = sData[i] * biasInfluence + massValue;
			double min = Math.min(1, delta);
			double max = Math.max(0, min);
			bias.set(i, max);
		}
		
		//Bias input
		SimpleMatrix output = input.elementMult(bias);
		
		//Add noise
		output = addNoise(output, noiseMagnitude);
		
		//Normalize
		output = normalize(output);
		
		return output;
	}
	
	
	private SimpleMatrix calculateS (SimpleMatrix correlationMatrix){
		SimpleMatrix s = correlationMatrix.plus(1); //the plus operation does add the beta value, not multiply by it as it says in the javadoc
		s = s.scale(5);
		s = s.minus(5);
		s = s.scale(-1);
		s = s.elementExp();
		s = s.plus(1);
		SimpleMatrix ones = new SimpleMatrix(s.numRows(), s.numCols());
		ones.set(1);
		s = ones.elementDiv(s);
		s = s.minus(0.5);
		return s;
	}
	
	private SimpleMatrix addNoise(SimpleMatrix m, double noiseMagnitude){
		double noise = (rand.nextDouble() - 0.5) * 2 * noiseMagnitude;
		m = m.plus(noise);
		return m;
	}
	
	private SimpleMatrix normalize(SimpleMatrix m){
		double maxValue = m.elementMaxAbs();
		m = m.divide(maxValue);
		return m;
	}

}
