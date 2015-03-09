package stcl.algo.brain;

import java.util.ArrayList;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.biasunits.BiasUnit;
import stcl.algo.brain.rewardCorrelators.RewardCorrelator;
import stcl.algo.brain.rewardCorrelators.RewardFunction;
import stcl.algo.poolers.RSOM;
import stcl.algo.util.Normalizer;

public class Brain {
	
	private ArrayList<NeoCorticalUnit> unitlist;
	
	public Brain(int numUnits, Random rand, int ffInputLength, int spatialMapSize, int temporalMapSize, int markovOrder) {
		createUnitList(numUnits, rand, ffInputLength, spatialMapSize, temporalMapSize, markovOrder);
	}
	
	private void createUnitList(int numUnits, Random rand, int ffInputLength, int spatialMapSize, int temporalMapSize, int markovOrder){
		unitlist = new ArrayList<NeoCorticalUnit>();
		NeoCorticalUnit nu1 = new NeoCorticalUnit(rand, ffInputLength, spatialMapSize, temporalMapSize, 0.1, true, markovOrder); //First one is special
		unitlist.add(nu1);
		for (int i = 0; i < numUnits - 1; i++){
			NeoCorticalUnit nu = new NeoCorticalUnit(rand, temporalMapSize * temporalMapSize, spatialMapSize, temporalMapSize, 0.1, true, markovOrder);
			unitlist.add(nu);
		}	
	}
	
	/**
	 * 
	 * @param brain
	 * @param noiseMagnitude
	 * @param input
	 * @return feed back output from the brain
	 */
	public SimpleMatrix step(SimpleMatrix inputVector){
		SimpleMatrix uniformDistribution = null;
		
		//Feed forward
		SimpleMatrix ffInput = inputVector;
		int i = 0;
		boolean cont = true;
		do {
			NeoCorticalUnit nu = unitlist.get(i);
			SimpleMatrix m = resizeToFitFFPass(ffInput, nu);
			SimpleMatrix inputToNextLayer = nu.feedForward(m);
			//System.out.println( i + " Entropy " + nu.getEntropy() + " Threshold: " + nu.getEntropyThreshold());
			cont = nu.needHelp();
			if (cont) {
				ffInput = inputToNextLayer;
			} else {
				ffInput = null;
			}
			i++;
		} while (i < unitlist.size() && cont);
		
		//Feed back
		if (uniformDistribution == null){
			RSOM rsom = unitlist.get(unitlist.size() - 1).getTemporalPooler().getRSOM();
			int rows = rsom.getHeight();
			int cols = rsom.getWidth();
			uniformDistribution = createUniformDistribution(rows, cols);
		}
		
		SimpleMatrix fbInput = uniformDistribution;
		for (int j = unitlist.size()-1; j >= 0; j--){
			NeoCorticalUnit nu = unitlist.get(j);
			SimpleMatrix m = resizeToFitFBPass(fbInput, nu);
			fbInput = nu.feedBackward(m);
		}
		
		return fbInput; //The last fb input is the output of the brain
	}
	
	private SimpleMatrix resizeToFitFBPass(SimpleMatrix matrixToResize, NU unitToFit){
		SimpleMatrix m = new SimpleMatrix(matrixToResize);
		RSOM rsom = unitToFit.getTemporalPooler().getRSOM();
		int rows = rsom.getHeight();
		int cols = rsom.getWidth();
		
		m.reshape(rows, cols);
		return m;
	}
	
	private SimpleMatrix resizeToFitFFPass(SimpleMatrix matrixToResize, NU unitToFit){
		SimpleMatrix m = new SimpleMatrix(matrixToResize);
		int rows = 1;
		int cols = unitToFit.getSOM().getInputVectorLength();
				
		m.reshape(rows, cols);
		return m;
	}
	
	private SimpleMatrix createUniformDistribution(int rows, int columns){
		SimpleMatrix m = new SimpleMatrix(rows, columns);
		m.set(1);
		m = Normalizer.normalize(m);
		return m;
	}
	
	public void setLearning(boolean learning){
		for (NeoCorticalUnit nu : unitlist) nu.setLearning(learning);
	}
	
	public void flush(){
		for (NeoCorticalUnit nu : unitlist) nu.flush();
	}
	
	public double[] collectPredictionEntropies(){
		double[] entropies = new double[unitlist.size()];
		for (int i = 0; i < unitlist.size(); i++){
			entropies[i] = unitlist.get(i).getEntropy();
		}
		return entropies;
				
	}
	
	public double[] collectSpatialFFEntropies(){
		double[] entropies = new double[unitlist.size()];
		for (int i = 0; i < unitlist.size(); i++){
			SimpleMatrix activation = unitlist.get(i).getSpatialPooler().getActivationMatrix();
			SimpleMatrix normalized = Normalizer.normalize(activation);
			entropies[i] = calculateEntropy(normalized);
		}
		return entropies;
				
	}
	
	public int[] collectBMUs(boolean spatial){
		int[] bmus = new int[unitlist.size()];
		for (int i = 0; i < unitlist.size(); i++){
			if (spatial) {
				bmus[i] = unitlist.get(i).getSOM().getBMU().getId();
			} else {
				bmus[i] = unitlist.get(i).getTemporalPooler().getRSOM().getBMU().getId();
			}
		}
		return bmus;
	}
	
	
	
	private double calculateEntropy(SimpleMatrix m){
		double sum = 0;
		for (Double d : m.getMatrix().data){
			if (d != 0) sum += d * Math.log(d);
		}
		return -sum;
	}


	
	
	

}
