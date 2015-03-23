package stcl.algo.brain;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Locale;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.nodes.Network;
import stcl.algo.brain.nodes.UnitNode;
import stcl.algo.poolers.NewSequencer;
import stcl.algo.util.FileWriter;

/**
 * The data collector brain works like the normal brain, but does also save all information that is received and sent out by the brain during its life time
 * @author Simon
 *
 */
public class Network_DataCollector extends Network {
	//Inputs and outputs to the brain
	//private SimpleMatrix receivedInput;
	//private SimpleMatrix returnedOutput;
	
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
	private boolean collectData;
	private int numUnits;

	private FileWriter[] unitWriters;
	
	public void initializeWriters(String parentFolder, boolean append){
		collectData = true;
		try {
			setupWriters(parentFolder);
			openFiles(append);
			writeHeaders();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	private void setupWriters(String parentFolder){
		ArrayList<UnitNode> unitNodes = super.getUnitNodes();
		numUnits = unitNodes.size();
		unitWriters = new FileWriter[numUnits];
		int i = 0;
		for (UnitNode n : unitNodes){
			FileWriter f = new FileWriter(parentFolder + "/node_" + n.getID());
			unitWriters[i] = f;
			i++;
		}			
	}
	
	public void openFiles(boolean append){
		try {
			for (FileWriter f : unitWriters){
				f.openFile(append);
				collectData = true;
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		
	}
	
	public void closeFiles(){
		try {
			for (FileWriter f : unitWriters){
				f.closeFile();
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		
		collectData = false;
	}
	
	

	@Override
	public void step(double reward) {
		
		//Feed forward
		this.feedForward(reward);
		
		if (collectData){
			//Collect Feedforward info
			FFInputs = collectUnitInputs(true);
			activeStatuses = (collectActiveStatus());
			helpStatuses = (collectHelpStatus());
			predictionEntropies = (collectPredictionEntropies());
			entropiesThresholds = (collectEntropyThresholds());
			spatialBMUs = (collectBMUs(true));
			temporalBMUs = (collectBMUs(false));
			FFOutputs = (collectUnitOutputs(true));
			temporalActivations = (collectActivations(false));
			spatialActivations = (collectActivations(true));
		}
		
		//Feed back
		this.feedback();
		
		//Collect feed back info
		if (collectData){
			FBInputs = collectUnitInputs(false);
			FBOutputs = (collectUnitOutputs(false));
		}
		
		//Print data to files
		if (collectData){
			try {
				writeData();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
	
	private void writeHeaders() throws IOException{

		//Write header for unit files
		String header = "";
		
		header += writeRepeatedString("Feedforward input",1, ";");
		
		header += writeRepeatedString("Prediction entropy",1, ";");
		header += writeRepeatedString("Entropy threshold", 1, ";");
		
		header += writeRepeatedString("Spatial BMU", 1, ";");
		header += writeRepeatedString("Temporal BMU", 1, ";");
		
		header += writeRepeatedString("Need help", 1, ";");
		header += writeRepeatedString("Was active", 1, ";");
		
		header += writeRepeatedString("Spatial activation", 1, ";");
		header += writeRepeatedString("FF Output", 1, ";");
		
		header += writeRepeatedString("FB Input", 1, ";");
		header += writeRepeatedString("FB Output", 1, ";");
		
		header = header.substring(0, header.length() - 1); //Remove last semi-colon
		
		for (FileWriter w : unitWriters) w.writeLine(header);

	}
	
	private void writeData() throws IOException{
		//Print brain data
	//	brainWriter.write(writeMatrixArray(receivedInput) + ";");
		//brainWriter.writeLine(writeMatrixArray(returnedOutput) + ";");
		
		//Print unit data
		for (int i = 0; i < numUnits; i++){
			FileWriter writer = unitWriters[i];
			writer.write(writeMatrixArray(FFInputs[i]) + ";");
			
			writer.write(predictionEntropies[i] + ";");
			writer.write(entropiesThresholds[i] + ";");
			
			writer.write(spatialBMUs[i] + ";");
			writer.write(temporalBMUs[i] + ";");
			
			writer.write(helpStatuses[i] + ";");
			writer.write(activeStatuses[i] + ";");
						
			writer.write(writeMatrixArray(spatialActivations[i]) + ";");
			writer.write(writeMatrixArray(FFOutputs[i]) + ";");
			
			writer.write(writeMatrixArray(FBInputs[i]) + ";");
			writer.write(writeMatrixArray(FBOutputs[i]) + ";");
			
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
			entropies[i] = super.getUnitNodes().get(i).getUnit().getEntropy();
		}
		return entropies;
				
	}
	
	private double[] collectEntropyThresholds(){
		double[] thresholds = new double[numUnits];
		for (int i = 0; i < numUnits; i++){
			thresholds[i] = super.getUnitNodes().get(i).getUnit().getEntropyThreshold();
		}
		return thresholds;
	}
	
	private int[] collectBMUs(boolean spatial){
		int[] bmus = new int[numUnits];
		for (int i = 0; i < numUnits; i++){
			if (spatial) {
				bmus[i] = super.getUnitNodes().get(i).getUnit().getSOM().getBMU().getId();
			} else {
				bmus[i] = super.getUnitNodes().get(i).getUnit().findTemporalBMUID();
			}
		}
		return bmus;
	}
	
	private boolean[] collectHelpStatus(){
		boolean[] status = new boolean[numUnits];
		for (int i = 0; i < numUnits; i++){
			boolean needHelp = super.getUnitNodes().get(i).getUnit().needHelp();
			status[i] = needHelp;
		}
		
		return status;
	}
	
	private boolean[] collectActiveStatus(){
		boolean[] status = new boolean[numUnits];
		for (int i = 0; i < numUnits; i++){
			boolean needHelp = super.getUnitNodes().get(i).getUnit().active();
			status[i] = needHelp;
		}
		
		return status;
	}
	
	private SimpleMatrix[] collectActivations(boolean spatial){
		SimpleMatrix[] activations = new SimpleMatrix[numUnits];
		for (int i = 0; i < numUnits; i++){
			SimpleMatrix m;
			if (spatial){
				m = new SimpleMatrix(super.getUnitNodes().get(i).getUnit().getSpatialPooler().getActivationMatrix());
			} else {
				NewSequencer sequencer = super.getUnitNodes().get(i).getUnit().getSequencer();
				if (sequencer != null){
					m = new SimpleMatrix(sequencer.getSequenceProbabilities());
				} else {
					m = new SimpleMatrix(1, 1);
					m.set(-1);
				}
			}
			activations[i] = m;
		}
		return activations;
	}
	
	private SimpleMatrix[] collectUnitOutputs(boolean feedForward){
		SimpleMatrix[] outputs = new SimpleMatrix[numUnits];
		for (int i = 0; i < numUnits; i++){
			SimpleMatrix m;
			if (feedForward){
				m = new SimpleMatrix(super.getUnitNodes().get(i).getUnit().getFFOutput());
			} else {
				m = new SimpleMatrix(super.getUnitNodes().get(i).getUnit().getFBOutput());
			}
			outputs[i] = m;
		}
		return outputs;
	}
	
	private SimpleMatrix[] collectUnitInputs(boolean feedforward){
		SimpleMatrix[] outputs = new SimpleMatrix[numUnits];
		for (int i = 0; i < numUnits; i++){
			SimpleMatrix m;
			if (feedforward){
				m = new SimpleMatrix(super.getUnitNodes().get(i).getUnit().getFFInput());
			} else {
				m = new SimpleMatrix(super.getUnitNodes().get(i).getUnit().getFBInput());
			}
			outputs[i] = m;
		}
		return outputs;
	}
	
	public void setCollectData(boolean collectData){
		this.collectData = collectData;
	}
	
}
