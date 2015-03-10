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
	
	//Entropies in the units
	private ArrayList<double[]> predictionEntropies;
	
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
		
		//Entropies in the units
		predictionEntropies = new ArrayList<double[]>();
	}

	@Override
	public SimpleMatrix step(SimpleMatrix inputVector) {
		
		receivedInputs.add(inputVector);
		
		//Feed forward
		feedForward(inputVector);
		
		//Collect Feedforward info
		activeStatuses.add(collectActiveStatus());
		helpStatuses.add(collectHelpStatus());
		predictionEntropies.add(collectPredictionEntropies());
		spatialBMUs.add(collectBMUs(true));
		temporalBMUs.add(collectBMUs(false));
		FFOutputs.add(collectOutputs(true));
		
		//Feed back
		SimpleMatrix output = feedBackward();
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
	
	private SimpleMatrix[] collectOutputs(boolean feedForward){
		SimpleMatrix[] outputs = new SimpleMatrix[numUnits];
		for (int i = 0; i < numUnits; i++){
			SimpleMatrix m;
			if (feedForward){
				m = unitlist.get(i).getFFOutput();
			} else {
				m = unitlist.get(i).getFBOutput();
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
}
