package stcl.algo.brain;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Locale;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import dk.stcl.core.basic.containers.SomMap;
import dk.stcl.core.basic.containers.SomNode;
import stcl.algo.brain.nodes.Sensor;
import stcl.algo.brain.nodes.UnitNode;
import stcl.algo.poolers.SOM;
import stcl.algo.poolers.Sequencer;
import stcl.algo.util.FileWriter;

/**
 * The data collector brain works like the normal brain, but does also save all information that is received and sent out by the brain during its life time
 * @author Simon
 *
 */
public class Network_DataCollector extends Network {
	//Inputs and outputs to the brain
	private SimpleMatrix receivedInput[];
	private SimpleMatrix returnedOutput[];
	private SimpleMatrix[] actionModels;
	
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
	
	private int[] actionVotes;
	
	//Misc
	private boolean collectData;
	private int numUnits;

	private FileWriter[] unitWriters;
	private FileWriter brainWriter;
	
	public Network_DataCollector(String networkFileName, Random rand) throws FileNotFoundException {
		super(networkFileName, rand);
	}
	
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
		brainWriter = new FileWriter(parentFolder + "/network");
		for (UnitNode n : unitNodes){
			FileWriter f = new FileWriter(parentFolder + "/node_" + n.getID());
			unitWriters[i] = f;
			i++;
		}			
	}
	
	public void openFiles(boolean append){
		try {
			brainWriter.openFile(append);
			for (FileWriter f : unitWriters){
				f.openFile(append);
			}
			collectData = true;
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
		collectData = false;
	}
	
	

	@Override
	public void step(double reward) {
		
		//Feed forward
		this.feedForward(reward);
		
		if (collectData){
			//Collect Feedforward info
			receivedInput = collectNetworkInput();
			FFInputs = collectUnitInputs(true);
			actionModels = collectActionModels();
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
			returnedOutput = collectNetworkOutput();
			FBInputs = collectUnitInputs(false);
			FBOutputs = (collectUnitOutputs(false));
			actionVotes = collectActionVotes();
		}
		
		super.resetUnitActivity();
		
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

		//Write ehader for network file
		String header = "";
		header += "Received input;";
		header += "Returned output;";
		header += "Action map;";
		header = header.substring(0, header.length() - 1); //Remove last semi-colon
		brainWriter.writeLine(header);
		
		//Write header for unit files
		header = "";
		
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
		
		header +=writeRepeatedString("Action vote", 1, ";");
		
		header = header.substring(0, header.length() - 1); //Remove last semi-colon
		
		for (FileWriter w : unitWriters) w.writeLine(header);

	}
	
	private void writeData() throws IOException{
		//Print brain data
		for (SimpleMatrix m : receivedInput) brainWriter.write(writeMatrixArray(m));
		brainWriter.write(";");
		for (SimpleMatrix m : returnedOutput) brainWriter.write(writeMatrixArray(m));
		brainWriter.write(";");
		for (SimpleMatrix m : actionModels) brainWriter.write(writeMatrixArray(m) + ",");
		brainWriter.writeLine("");
		
		
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
			
			writer.write(actionVotes[i] + ";");
			
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
	
	private SimpleMatrix[] collectActionModels(){
		return collectSomModels(super.getActionNode().getPooler().getSOM());
	}
	
	private SimpleMatrix[] collectSomModels(SOM som){
		SomNode[] nodes = som.getNodes();
		SimpleMatrix[] vectors = new SimpleMatrix[nodes.length];
		for (int i = 0; i < nodes.length; i++ ){
			SomNode n = nodes[i];
			vectors[i] = new SimpleMatrix(n.getVector());
		}
		return vectors;
	}
	
	private SimpleMatrix[] collectNetworkInput(){
		ArrayList<Sensor> sensors = super.getSensors();
		SimpleMatrix[] inputs = new SimpleMatrix[sensors.size()];
		for (int i = 0; i < sensors.size(); i++){
			inputs[i] = sensors.get(i).getFeedforwardOutput();
		}
		return inputs;
	}
	
	private SimpleMatrix[] collectNetworkOutput(){
		ArrayList<Sensor> sensors = super.getSensors();
		SimpleMatrix[] inputs = new SimpleMatrix[sensors.size()];
		for (int i = 0; i < sensors.size(); i++){
			inputs[i] = sensors.get(i).getFeedbackOutput();
		}
		return inputs;
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
	
	private int[] collectActionVotes(){
		int[] votes = new int[numUnits];
		for (int i = 0; i < numUnits; i++){
			int vote = super.getUnitNodes().get(i).getActionVote();
			votes[i] = vote;
		}
		
		return votes;
	}
	
	private SimpleMatrix[] collectActivations(boolean spatial){
		SimpleMatrix[] activations = new SimpleMatrix[numUnits];
		for (int i = 0; i < numUnits; i++){
			SimpleMatrix m;
			if (spatial){
				m = new SimpleMatrix(super.getUnitNodes().get(i).getUnit().getSpatialPooler().getActivationMatrix());
			} else {
				Sequencer sequencer = super.getUnitNodes().get(i).getUnit().getSequencer();
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
