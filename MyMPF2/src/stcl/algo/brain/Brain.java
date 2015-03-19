package stcl.algo.brain;

import java.util.ArrayList;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.poolers.RSOM;
import stcl.algo.util.Normalizer;

public class Brain {
	
	protected ArrayList<NeoCorticalUnit> unitlist;
	private SimpleMatrix uniformDistribution;
	
	public Brain(int numUnits, Random rand, int ffInputLength, int spatialMapSize, int temporalMapSize, int markovOrder) {
		this(numUnits, rand, ffInputLength, spatialMapSize, temporalMapSize, markovOrder, false);
	}
	
	public Brain(int numUnits, Random rand, int ffInputLength, int spatialMapSize, int temporalMapSize, int markovOrder, boolean firstIsSpatial) {
		createUnitList(numUnits, rand, ffInputLength, spatialMapSize, temporalMapSize, markovOrder, firstIsSpatial);
		
		uniformDistribution = createUniformDistribution(temporalMapSize, temporalMapSize);
	}	
	
	private void createUnitList(int numUnits, Random rand, int ffInputLength, int spatialMapSize, int temporalMapSize, int markovOrder, boolean firstIsSpatial){
		unitlist = new ArrayList<NeoCorticalUnit>();
		NeoCorticalUnit nu = new NeoCorticalUnit(rand, ffInputLength, spatialMapSize, temporalMapSize, 0.1, true, markovOrder, firstIsSpatial); //First one is special
		unitlist.add(nu);
		for (int i = 0; i < numUnits - 1; i++){
			NeoCorticalUnit in = nu;
			nu = new NeoCorticalUnit(rand, in.getTemporalMapSize() * in.getTemporalMapSize(), spatialMapSize, temporalMapSize, 0.1, true, markovOrder);
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
		return this.step(inputVector, 0);
	}
	
	/**
	 * 
	 * @param brain
	 * @param noiseMagnitude
	 * @param input
	 * @return feed back output from the brain
	 */
	public SimpleMatrix step(SimpleMatrix inputVector, double externalReward){
		SimpleMatrix m = feedForward(inputVector);
		
		SimpleMatrix output = feedBackward(m);
		
		return output;
	}
	
	protected SimpleMatrix feedForward(SimpleMatrix inputVector){
		return this.feedForward(inputVector, 0);
	}
	
	protected SimpleMatrix feedForward(SimpleMatrix inputVector, double externalReward){
		SimpleMatrix ffInput = inputVector;
		int i = 0;
		boolean cont = true;
		do {
			NeoCorticalUnit nu = unitlist.get(i);
			SimpleMatrix m = resizeToFitFFPass(ffInput, nu);
			SimpleMatrix inputToNextLayer = nu.feedForward(m, externalReward);
			//System.out.println( i + " Entropy " + nu.getEntropy() + " Threshold: " + nu.getEntropyThreshold());
			cont = nu.needHelp();
			if (cont) {
				//Save ffinput
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
			SimpleMatrix inputToUnit = m;
			fbInput = nu.feedBackward(inputToUnit);
		}
		
		return fbInput; //The last fb input is the output of the brain
	}
	
	private SimpleMatrix resizeToFitFBPass(SimpleMatrix matrixToResize, NeoCorticalUnit unitToFit){
		SimpleMatrix m = new SimpleMatrix(matrixToResize);
		int size = unitToFit.getTemporalMapSize();
		m.reshape(size, size);
		return m;
	}
	
	private SimpleMatrix resizeToFitFFPass(SimpleMatrix matrixToResize, NeoCorticalUnit unitToFit){
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
	}
	
	public void setEntropyThresholdFrozen(boolean entropyThresholdFrozen) {
		for (NeoCorticalUnit nu : unitlist) nu.setEntropyThresholdFrozen(entropyThresholdFrozen);
	}
	
	public void setBiasBeforePrediction(boolean biasBeforePrediction) {
		for (NeoCorticalUnit nu : unitlist) nu.setBiasBeforePrediction(biasBeforePrediction);
	}
	
	public void setUseBiasedInputToSequencer(boolean flag) {
		for (NeoCorticalUnit nu : unitlist) nu.setUseBiasedInputInSequencer(flag);
	}

	

}
