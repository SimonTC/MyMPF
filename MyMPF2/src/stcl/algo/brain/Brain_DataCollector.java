package stcl.algo.brain;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.util.Normalizer;

import java.util.ArrayList;
import java.util.Random;

import javax.swing.JSpinner.NumberEditor;

/**
 * The data collector brain works like the normal brain, but does also save all information that is received and sent out by the brain during its life time
 * @author Simon
 *
 */
public class Brain_DataCollector extends Brain {
	//Inputs and outputs to the brain
	private ArrayList<SimpleMatrix> receivedInputs;
	private ArrayList<SimpleMatrix> returnedOutputs;
	
	//Inputs to and outputs from the units
	private ArrayList<SimpleMatrix[]> FFOutputs;
	private ArrayList<SimpleMatrix[]> FBOutputs;
	
	//Help status of the units
	private ArrayList<boolean[]> helpStatuses;
	
	//Active status of the units
	private ArrayList<boolean[]> activeStatuses;
	
	//Spatial and temporal BMUs in the units
	private ArrayList<int[]> spatialBMUs;
	private ArrayList<int[]> temporalBMUs;
	
	//Activations
	private ArrayList<SimpleMatrix[]> temporalActivations;
	private ArrayList<SimpleMatrix[]> spatialActivations;
	
	//Entropies in the units
	private ArrayList<double[]> predictionEntropies;
	private ArrayList<double[]> entropiesThresholds;
	
	//Misc
	private int numUnits;

	public Brain_DataCollector(int numUnits, Random rand, int ffInputLength,
			int spatialMapSize, int temporalMapSize, int markovOrder) {
		super(numUnits, rand, ffInputLength, spatialMapSize, temporalMapSize,
				markovOrder);
		setupMemories();
		this.numUnits = numUnits;
	}
	
	private void setupMemories(){
		//Inputs and outputs to the brain
		receivedInputs = new ArrayList<SimpleMatrix>();
		returnedOutputs = new ArrayList<SimpleMatrix>();
		
		//Inputs to and outputs from the units
		FFOutputs = new ArrayList<SimpleMatrix[]>();
		FBOutputs = new ArrayList<SimpleMatrix[]>();
		
		//Help status of the units
		helpStatuses = new ArrayList<boolean[]>();
		
		//Active status of the units
		activeStatuses = new ArrayList<boolean[]>();
		
		//Spatial and Temporal BMUs in the units
		spatialBMUs = new ArrayList<int[]>();
		temporalBMUs = new ArrayList<int[]>();
		
		//Activations
		temporalActivations = new ArrayList<SimpleMatrix[]>();
		spatialActivations = new ArrayList<SimpleMatrix[]>();
		
		//Entropies in the units
		predictionEntropies = new ArrayList<double[]>();
		entropiesThresholds = new ArrayList<double[]>();
	}

	@Override
	public SimpleMatrix step(SimpleMatrix inputVector) {
		
		receivedInputs.add(inputVector);
		
		//Feed forward
		SimpleMatrix m = feedForward(inputVector);
		
		//Collect Feedforward info
		activeStatuses.add(collectActiveStatus());
		helpStatuses.add(collectHelpStatus());
		predictionEntropies.add(collectPredictionEntropies());
		entropiesThresholds.add(collectEntropyThresholds());
		spatialBMUs.add(collectBMUs(true));
		//temporalBMUs.add(collectBMUs(false));
		FFOutputs.add(collectOutputs(true));
		//temporalActivations.add(collectActivations(false));
		spatialActivations.add(collectActivations(true));
		
		//Feed back
		SimpleMatrix output = feedBackward(m);
		FBOutputs.add(collectOutputs(false));
		
		//Collect feed back info
		returnedOutputs.add(output);
		
		return output;
	}
	
	private double[] collectPredictionEntropies(){
		double[] entropies = new double[numUnits];
		for (int i = 0; i < numUnits; i++){
			entropies[i] = unitlist.get(i).getEntropy();
		}
		return entropies;
				
	}
	
	private double[] collectEntropyThresholds(){
		double[] thresholds = new double[numUnits];
		for (int i = 0; i < numUnits; i++){
			thresholds[i] = unitlist.get(i).getEntropyThreshold();
		}
		return thresholds;
	}
	
	private int[] collectBMUs(boolean spatial){
		int[] bmus = new int[numUnits];
		for (int i = 0; i < numUnits; i++){
			if (spatial) {
				bmus[i] = unitlist.get(i).getSOM().getBMU().getId();
			} else {
				bmus[i] = unitlist.get(i).getTemporalPooler().getRSOM().getBMU().getId();
			}
		}
		return bmus;
	}
	
	private boolean[] collectHelpStatus(){
		boolean[] status = new boolean[numUnits];
		for (int i = 0; i < numUnits; i++){
			boolean needHelp = unitlist.get(i).needHelp();
			status[i] = needHelp;
		}
		
		return status;
	}
	
	private boolean[] collectActiveStatus(){
		boolean[] status = new boolean[numUnits];
		for (int i = 0; i < numUnits; i++){
			boolean needHelp = unitlist.get(i).active();
			status[i] = needHelp;
		}
		
		return status;
	}
	
	private SimpleMatrix[] collectActivations(boolean spatial){
		SimpleMatrix[] activations = new SimpleMatrix[numUnits];
		for (int i = 0; i < numUnits; i++){
			SimpleMatrix m;
			if (spatial){
				m = new SimpleMatrix(unitlist.get(i).getSpatialPooler().getActivationMatrix());
			} else {
				m = new SimpleMatrix(unitlist.get(i).getTemporalPooler().getActivationMatrix());
			}
			activations[i] = m;
		}
		return activations;
	}
	
	private SimpleMatrix[] collectOutputs(boolean feedForward){
		SimpleMatrix[] outputs = new SimpleMatrix[numUnits];
		for (int i = 0; i < numUnits; i++){
			SimpleMatrix m;
			if (feedForward){
				m = new SimpleMatrix(unitlist.get(i).getFFOutput());
			} else {
				m = new SimpleMatrix(unitlist.get(i).getFBOutput());
			}
			outputs[i] = m;
		}
		return outputs;
	}
	
	/**
	 * Removes all collected data
	 */
	public void flushCollectedData(){
		setupMemories();
	}

	/**
	 * @return the receivedInputs
	 */
	public ArrayList<SimpleMatrix> getReceivedInputs() {
		return receivedInputs;
	}

	/**
	 * @return the returnedOutputs
	 */
	public ArrayList<SimpleMatrix> getReturnedOutputs() {
		return returnedOutputs;
	}

	/**
	 * @return the fFOutputs
	 */
	public ArrayList<SimpleMatrix[]> getFFOutputs() {
		return FFOutputs;
	}

	/**
	 * @return the fBOutputs
	 */
	public ArrayList<SimpleMatrix[]> getFBOutputs() {
		return FBOutputs;
	}

	/**
	 * @return the helpStatuses
	 */
	public ArrayList<boolean[]> getHelpStatuses() {
		return helpStatuses;
	}

	/**
	 * @return the activeStatuses
	 */
	public ArrayList<boolean[]> getActiveStatuses() {
		return activeStatuses;
	}

	/**
	 * @return the spatialBMUs
	 */
	public ArrayList<int[]> getSpatialBMUs() {
		return spatialBMUs;
	}

	/**
	 * @return the temporalBMUs
	 */
	public ArrayList<int[]> getTemporalBMUs() {
		return temporalBMUs;
	}

	/**
	 * @return the predictionEntropies
	 */
	public ArrayList<double[]> getPredictionEntropies() {
		return predictionEntropies;
	}
	
	public int getNumUnits(){
		return numUnits;
	}

	/**
	 * @return the temporalActivations
	 */
	public ArrayList<SimpleMatrix[]> getTemporalActivations() {
		return temporalActivations;
	}

	/**
	 * @return the spatialActivations
	 */
	public ArrayList<SimpleMatrix[]> getSpatialActivations() {
		return spatialActivations;
	}

	/**
	 * @return the entropiesThresholds
	 */
	public ArrayList<double[]> getEntropiesThresholds() {
		return entropiesThresholds;
	}
}
