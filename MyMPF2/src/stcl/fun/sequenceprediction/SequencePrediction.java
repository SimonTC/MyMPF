package stcl.fun.sequenceprediction;

import java.util.ArrayList;
import java.util.Random;

import javax.swing.JFrame;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.NeoCorticalUnit;
import stcl.graphics.MovingLinesGUI_Prediction;

public class SequencePrediction {
	
	private MovingLinesGUI_Prediction frame;
	private NeoCorticalUnit unit;
	private Random rand = new Random();
	private ArrayList<SimpleMatrix[]> sequences;
	private final int NUM_ITERAIONS = 80000;
	
	private SimpleMatrix uniformDistribution;
	
	private SimpleMatrix bigT;
	private SimpleMatrix smallO;
	private SimpleMatrix bigO;
	private SimpleMatrix smallV;
	private SimpleMatrix blank;
	
	int FRAMES_PER_SECOND = 10;
	int SKIP_TICKS = 1000 / FRAMES_PER_SECOND;
	
	private final boolean VISUALIZE_TRAINING = false;
	private final boolean VISUALIZE_RESULT = true;

	public SequencePrediction() {
		// TODO Auto-generated constructor stub
	}

	public static void main(String[] args) {
		SequencePrediction runner = new SequencePrediction();
		runner.run();

	}
	
	public void run(){
		setupExperiment();
		if (VISUALIZE_TRAINING) setupGraphics();
		runExperiment(VISUALIZE_TRAINING);
		
		unit.flushTemporalMemory();
		unit.setLearning(false);
		
		evaluate(1000);
		
		if (VISUALIZE_RESULT){
			unit.setLearning(false);
			unit.flushTemporalMemory();
			setupGraphics();
			runExperiment(true);
		}
	}
	
	private void runExperiment(boolean visualize){
		for (int i = 0; i < NUM_ITERAIONS; i++){
			int seqID = rand.nextInt(sequences.size());
			SimpleMatrix[] input = sequences.get(seqID);
			
			for (SimpleMatrix m : input){
				m.reshape(1, m.numCols() * m.numRows());
				SimpleMatrix ffOutput = unit.feedForward(m);
				SimpleMatrix fbOutput = unit.feedBackward(ffOutput);
				
				if (visualize){
		    		//Update graphics
		    		updateGraphics(m, i);
		    		
		    		//Sleep
					try {
						Thread.sleep(SKIP_TICKS);
					} catch (InterruptedException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}	
	    		}
			}
			//unit.flushTemporalMemory();
		}
	}
	
	private void evaluate(int iterations){
		double predictionError_Total = 0;
		double spatialError_Total = 0;
		for (int i = 0; i < iterations; i++){
			int seqID = rand.nextInt(sequences.size());
			SimpleMatrix[] input = sequences.get(seqID);
			double predictionError_Sequence = 0;
			double spatialError_Sequence = 0;
			for (int patternID = 0; patternID < input.length; patternID++){
				SimpleMatrix m = input[patternID];
				m.reshape(1, m.numCols() * m.numRows());
				SimpleMatrix ffOutput = unit.feedForward(m);
				SimpleMatrix bmuVector = unit.getSpatialPooler().getSOM().getBMU().getVector();
				double spatialError_Pattern = calculateError(m, bmuVector);
				SimpleMatrix fbOutput = unit.feedBackward(ffOutput);
				SimpleMatrix expectedOutput;
				boolean dontJudge = false;
				if (patternID == input.length - 1){
					expectedOutput = blank;
					dontJudge = true;
				} else {
					expectedOutput = input[patternID + 1];
					dontJudge = false;
				}
				double predictionError_Pattern = 0;
				if (!dontJudge){
					expectedOutput.reshape(1,m.numCols() * m.numRows() );
					predictionError_Pattern = calculateError(fbOutput, expectedOutput);
				}
				predictionError_Sequence += predictionError_Pattern;
				spatialError_Sequence += spatialError_Pattern;
				
			}
			//unit.flushTemporalMemory();
			predictionError_Total += predictionError_Sequence;
			spatialError_Total += spatialError_Sequence;
		}
		
		predictionError_Total = predictionError_Total / (double) iterations;
		spatialError_Total = spatialError_Total / (double) iterations;

		System.out.println("Average spatial error: " + spatialError_Total);
		System.out.println("Average prediction error: " + predictionError_Total);
		
	}
	
	private double calculateError(SimpleMatrix a, SimpleMatrix b){
		SimpleMatrix diff = a.minus(b);
		double error = diff.normF();
		error = error / diff.numCols();
		return error;
	}
	
	private void setupExperiment(){
		
		buildSequences();
		
		int inputLenght = blank.getMatrix().data.length;
		int spatialMapSize = 2;
		int temporalMapSize = 2;
		double initialPredictionLearningRate = 0.1;
		boolean useFirstOrderPrediction = true;
		double decay = 0.7;
		unit = new NeoCorticalUnit(rand, NUM_ITERAIONS, inputLenght, spatialMapSize, temporalMapSize, initialPredictionLearningRate, useFirstOrderPrediction, decay);
		
		
	}
	
	private void setupGraphics(){
		frame = new MovingLinesGUI_Prediction(unit);
		frame.setTitle("Visualiztion");
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		updateGraphics(blank,0); //Give a blank
		frame.pack();
		frame.setVisible(true);
		
	}
	
	private void updateGraphics(SimpleMatrix inputVector, int iteration){
		frame.updateData(inputVector, unit);
		frame.setTitle("Visualiztion - Iteration: " + iteration);
		frame.revalidate();
		frame.repaint();

	}

	private void buildSequences(){
		System.out.println("Building sequences");
		createFigures();
		
		sequences = new ArrayList<SimpleMatrix[]>();
		
		SimpleMatrix[] seq1 = {bigT, smallO, smallO, bigT, blank};
		SimpleMatrix[] seq2 = {bigO, smallO, bigO, smallO, blank};
		SimpleMatrix[] seq3 = {smallO, smallV, smallV, bigT, blank};
		SimpleMatrix[] seq4 = {bigO, smallV, bigO, bigT, blank};
		SimpleMatrix[] seq5 = {bigO, bigO, bigO, bigO, bigO, bigO, bigO, bigO, bigO, bigO};
		SimpleMatrix[] seq6 = {smallV, smallV, smallV, smallV, smallV, smallV, smallV, smallV, smallV, smallV};
		SimpleMatrix[] seq7 = {bigT, bigT, bigT, bigT, bigT, bigT, bigT, bigT, bigT, bigT};
		SimpleMatrix[] seq8 = {smallO, smallO, smallO, smallO, smallO, smallO, smallO, smallO, smallO, smallO};
		SimpleMatrix[] seq9 = {smallO, bigT, bigT, bigO};
		SimpleMatrix[] seq10 = {bigT, smallO, bigO, smallO};
		
		//sequences.add(seq1);
		//sequences.add(seq2);
		//sequences.add(seq3);
		//sequences.add(seq4);
		//sequences.add(seq5);
		//sequences.add(seq6);
		//sequences.add(seq7);
		//sequences.add(seq8);
		sequences.add(seq9);
		sequences.add(seq10);
		
	}
	
	private void createFigures(){		
		double[][] bigTData = {
				{0,0,0,0,0},
				{0,1,1,1,0},
				{0,0,1,0,0},
				{0,0,1,0,0},
				{0,0,1,0,0}};
		bigT = new SimpleMatrix(bigTData);
		bigT.reshape(1, bigT.numCols() * bigT.numRows());
		
		double[][] smallOData = {
				{0,0,0,0,0},
				{0,1,1,1,0},
				{0,1,0,1,0},
				{0,1,1,1,0},
				{0,0,0,0,0}};
		smallO = new SimpleMatrix(smallOData);
		smallO.reshape(1, smallO.numCols() * smallO.numRows());
		
		double[][] bigOData = {
				{1,1,1,1,1},
				{1,0,0,0,1},
				{1,0,0,0,1},
				{1,0,0,0,1},
				{1,1,1,1,1}};
		bigO = new SimpleMatrix(bigOData);
		bigO.reshape(1, bigO.numCols() * bigO.numRows());
		
		double[][] smallVData = {
				{0,0,0,0,0},
				{0,0,0,0,0},
				{1,0,0,0,1},
				{0,1,0,1,0},
				{0,0,1,0,0}};
		smallV = new SimpleMatrix(smallVData);
		smallV.reshape(1, smallV.numCols() * smallV.numRows());
		
		double[][] blankData = {
				{0,0,0,0,0},
				{0,0,0,0,0},
				{0,0,0,0,0},
				{0,0,0,0,0},
				{0,0,0,0,0}};
		blank = new SimpleMatrix(blankData);
		blank.reshape(1, blank.numCols() * blank.numRows());
	}

}
