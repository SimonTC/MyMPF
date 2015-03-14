package stcl.algo.brain;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Locale;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.util.FileWriter;

/**
 * The data collector brain works like the normal brain, but does also save all information that is received and sent out by the brain during its life time
 * @author Simon
 *
 */
public class Brain_DataCollector extends Brain {
	//Inputs and outputs to the brain
	private SimpleMatrix receivedInput;
	private SimpleMatrix returnedOutput;
	
	//Inputs to and outputs from the units
	private SimpleMatrix[] FFOutputs;
	private SimpleMatrix[] FBOutputs;
	
	private SimpleMatrix[] FFInputs;
	private SimpleMatrix[] FBInputs;
	
	//Help status of the units
	private boolean[] helpStatuses;
	
	//Active status of the units
	private boolean[] activeStatuses;
	
	//Spatial and temporal BMUs in the units
	private int[] spatialBMUs;
	private int[] temporalBMUs;
	
	//Activations
	private SimpleMatrix[] temporalActivations;
	private SimpleMatrix[] spatialActivations;
	
	//Entropies in the units
	private double[] predictionEntropies;
	private double[] entropiesThresholds;
	
	//Misc
	private int numUnits;
	private boolean collectData;

	private FileWriter brainWriter;
	private FileWriter[] unitWriters;
	
	public Brain_DataCollector(int numUnits, Random rand, int ffInputLength,
			int spatialMapSize, int temporalMapSize, int markovOrder) {
		this(numUnits, rand, ffInputLength, spatialMapSize, temporalMapSize, markovOrder, "", false);
		collectData = false;
	}

	public Brain_DataCollector(int numUnits, Random rand, int ffInputLength,
			int spatialMapSize, int temporalMapSize, int markovOrder, String parentFolder, boolean append) {
		super(numUnits, rand, ffInputLength, spatialMapSize, temporalMapSize,
				markovOrder);
		
		this.numUnits = numUnits;
		collectData = false;
		if (!parentFolder.isEmpty()){
			collectData = true;
			try {
				setupWriters(parentFolder, numUnits);
				openFiles(append);
				writeHeaders();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}		
	}
	
	private void setupWriters(String parentFolder, int numUnits){
		brainWriter = new FileWriter(parentFolder + "/brain.csv");
		unitWriters = new FileWriter[numUnits];
		for (int i = 0; i < numUnits; i++){
			FileWriter f = new FileWriter(parentFolder + "/unit_" + i);
			unitWriters[i] = f;
		}		
	}
	
	public void openFiles(boolean append){
		try {
			brainWriter.openFile(append);
			for (FileWriter f : unitWriters){
				f.openFile(append);
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		
	}
	
	public void closeFiles(){
		try {
			brainWriter.closeFile();
			for (FileWriter f : unitWriters){
				f.closeFile();
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		
	}
	
	

	@Override
	public SimpleMatrix step(SimpleMatrix inputVector, double externalReward) {
		if (collectData)  receivedInput = inputVector;
		
		//Feed forward
		SimpleMatrix m = feedForward(inputVector);
		
		if (collectData){
			//Collect Feedforward info
			activeStatuses = (collectActiveStatus());
			helpStatuses = (collectHelpStatus());
			predictionEntropies = (collectPredictionEntropies());
			entropiesThresholds = (collectEntropyThresholds());
			spatialBMUs = (collectBMUs(true));
			temporalBMUs = (collectBMUs(false));
			//FFOutputs = (collectOutputs(true));
			temporalActivations = (collectActivations(false));
			spatialActivations = (collectActivations(true));
		}
		
		//Feed back
		SimpleMatrix output = feedBackward(m);
		if (collectData) FBOutputs = (collectOutputs(false));
		
		//Collect feed back info
		if (collectData) returnedOutput = output;
		
		//Print data to files
		if (collectData){
			try {
				writeData();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		return output;
	}
	
	private void writeHeaders() throws IOException{
		
		//Write header for brain file
		String header = "";
		header += writeRepeatedString("Input", 1, ";");
		header += writeRepeatedString("Output", 1, ";");
		brainWriter.writeLine(header);
		
		//Write header for unit files
		header = "";
		header += writeRepeatedString("Prediction entropy",1, ";");
		header += writeRepeatedString("Entropy threshold", 1, ";");
		
		header += writeRepeatedString("Spatial BMU", 1, ";");
		header += writeRepeatedString("Temporal BMU", 1, ";");
		
		header += writeRepeatedString("Need help", 1, ";");
		header += writeRepeatedString("Was active", 1, ";");
		
		header += writeRepeatedString("Spatial activation", 1, ";");
		header += writeRepeatedString("Temporal activation", 1, ";");
		
		header = header.substring(0, header.length() - 1); //Remove last semi-colon
		
		for (FileWriter w : unitWriters) w.writeLine(header);

	}
	
	private void writeData() throws IOException{
		//Print brain data
		brainWriter.write(writeMatrixArray(receivedInput) + ";");
		brainWriter.writeLine(writeMatrixArray(returnedOutput) + ";");
		
		//Print unit data
		for (int i = 0; i < numUnits; i++){
			FileWriter writer = unitWriters[i];
			writer.write(predictionEntropies[i] + ";");
			writer.write(entropiesThresholds[i] + ";");
			
			writer.write(spatialBMUs[i] + ";");
			writer.write(temporalBMUs[i] + ";");
			
			writer.write(helpStatuses[i] + ";");
			writer.write(activeStatuses[i] + ";");
						
			writer.write(writeMatrixArray(spatialActivations[i]) + ";");
			writer.write(writeMatrixArray(temporalActivations[i]) + ";");
			
			writer.writeLine("");
		}
	}
	
	private String writeRepeatedString(String stringToRepeat, int numberOfTimes, String delimiter){
		String s = "";
		for (int i = 1; i <= numberOfTimes; i++){
			s += stringToRepeat;
			if (numberOfTimes > 1) s+= " " + i;
			s+= delimiter;
		}
		return s;
	}
	
	/**
	 * All is pretty much taken from the Matrix.toString() metods in simpleMatrix
	 * @param m
	 * @return
	 */
	private String writeMatrixArray(SimpleMatrix m){
		int numChar = 6;
		int precision = 3;
		String format = "%"+numChar+"."+precision+"f " + "  ";
		
		ByteArrayOutputStream stream = new ByteArrayOutputStream();
		PrintStream ps = new PrintStream(stream);
		
		for (double d : m.getMatrix().data){
			ps.printf(Locale.US, format, d);
		}
		
		return stream.toString();
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
				bmus[i] = unitlist.get(i).findTemporalBMUID();
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
				m = new SimpleMatrix(unitlist.get(i).getSequencer().getSequenceProbabilities());
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
	
	public void setCollectData(boolean collectData){
		this.collectData = collectData;
	}
	
}
