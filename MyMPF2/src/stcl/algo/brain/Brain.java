package stcl.algo.brain;

import java.util.ArrayList;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.biasunits.BiasUnit;
import stcl.algo.brain.rewardCorrelators.RewardCorrelator;
import stcl.algo.brain.rewardCorrelators.RewardFunction;
import stcl.algo.poolers.RSOM;
import stcl.algo.util.Normalizer;
import stcl.algo.util.Orthogonalizer;

public class Brain {
	
	protected ArrayList<NeoCorticalUnit> unitlist;
	private SimpleMatrix uniformDistribution;
	
	public Brain(int numUnits, Random rand, int ffInputLength, int spatialMapSize, int temporalMapSize, int markovOrder) {
		createUnitList(numUnits, rand, ffInputLength, spatialMapSize, temporalMapSize, markovOrder);
		
		RSOM rsom = unitlist.get(unitlist.size() - 1).getTemporalPooler().getRSOM();
		int rows = rsom.getHeight();
		int cols = rsom.getWidth();
		uniformDistribution = createUniformDistribution(rows, cols);
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
		SimpleMatrix m = feedForward(inputVector);
		
		SimpleMatrix output = feedBackward(m);
		
		return output;
	}
	
	protected SimpleMatrix feedForward(SimpleMatrix inputVector){
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
		
		return ffInput;
	}
	
	protected SimpleMatrix feedBackward(SimpleMatrix fbInput){
		if (fbInput == null) {
			fbInput = uniformDistribution;
		}
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
	
	public ArrayList<NeoCorticalUnit> getUnitList(){
		return unitlist;
	};


	
	
	

}
