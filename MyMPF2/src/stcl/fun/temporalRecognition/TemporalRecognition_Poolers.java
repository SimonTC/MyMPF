package stcl.fun.temporalRecognition;

import java.util.ArrayList;
import java.util.Random;

import javax.swing.JFrame;

import org.ejml.simple.SimpleMatrix;

import dk.stcl.core.basic.containers.SomNode;
import stcl.algo.poolers.SpatialPooler;
import stcl.algo.poolers.TemporalPooler;
import stcl.graphics.MovingLinesGUI_Prediction;

public class TemporalRecognition_Poolers {
	private ArrayList<SimpleMatrix[]> sequences;
	private SpatialPooler spatialPooler;
	private TemporalPooler temporalPooler;
	private MovingLinesGUI_Prediction frame;
	private Random rand = new Random(1234);
	
	private final int ITERATIONS = 10000;
	private final boolean VISUALIZE_TRAINING = false;
	private final boolean VISUALIZE_RESULT = false;
	private SimpleMatrix bigT;
	private SimpleMatrix smallO;
	private SimpleMatrix bigO;
	private SimpleMatrix smallV;
	private SimpleMatrix blank;
	private SimpleMatrix joker;
	
	int FRAMES_PER_SECOND = 10;
    int SKIP_TICKS = 1000 / FRAMES_PER_SECOND;
	
	public static void main(String[] args){
		TemporalRecognition_Poolers runner = new TemporalRecognition_Poolers();
		runner.run();
	}
	
	public void run(){
		//Setup experiment		
		setupExperiment(ITERATIONS, rand);
		
		//Setup graphics
		if (VISUALIZE_TRAINING) setupGraphics();
		
		//Train
		training(ITERATIONS, rand, VISUALIZE_TRAINING);
		temporalPooler.flushTemporalMemory();
		temporalPooler.setLearning(false);
		spatialPooler.setLearning(false);
		
		//Label
		int[] labels = createLabels();
		RSomLabeler labeler = new RSomLabeler();
		labeler.label(spatialPooler, temporalPooler, sequences, labels);
		
		//Evaluate
		double noise = 0.0;
		for (int i = 0; i < 100; i++){
			RsomEvaluator evaluator = new RsomEvaluator();
			double fitness = evaluator.evaluate(spatialPooler, temporalPooler, sequences, labels, joker, noise, 1000, rand);
			
			System.out.println("Fitness: " + fitness);
			noise += 0.01;
		}
		System.out.println();
		System.out.println("RSOM labels");
		temporalPooler.getRSOM().printLabelMap();
		
		/*
		//Print models
	    System.out.println("SOM models:");
	    for (SomNode n : spatialPooler.getSOM().getNodes()){
	    	SimpleMatrix vector = new SimpleMatrix(n.getVector());
	    	vector.reshape(5,5);
	    	vector.print();
	    	System.out.println(vector.elementSum());
	    	System.out.println();
	    }
	    
	    System.out.println("Rsom models:");
	    System.out.println();
	    for (SomNode n : temporalPooler.getRSOM().getNodes()){
	    	SimpleMatrix vector = new SimpleMatrix(n.getVector());
	    	vector.reshape(spatialPooler.getMapSize(), spatialPooler.getMapSize());
	    	vector.print();
	    	System.out.println(vector.elementSum());
	    	System.out.println();
	    }

		*/
		if (VISUALIZE_RESULT){
			temporalPooler.flushTemporalMemory();
			temporalPooler.setLearning(false);
			spatialPooler.setLearning(false);
			 setupGraphics();
			 training(ITERATIONS, rand, true);			 
		}		
	}
	
	private int[] createLabels(){
		int[] labels = new int[sequences.size()];
		for (int i = 0; i < labels.length; i++){
			labels[i] = i;
		}
		
		return labels;
	}
		
	private void training(int maxIterations, Random rand, boolean visualize){
	    int curSeqID = 0;
	    
	    for (int i = 0; i < maxIterations; i++){
	    	//Flush memory
	    	temporalPooler.flushTemporalMemory();
	    	
	    	//Choose sequence	    	
	    	curSeqID = rand.nextInt(sequences.size());
	    	SimpleMatrix[] curSequence = sequences.get(curSeqID);
	    	
	    	for (SimpleMatrix input : curSequence){
	    		//Spatial classification
	    		SimpleMatrix spatialFFOutputMatrix = spatialPooler.feedForward(input);
	    		
	    		//Transform spatial output matrix to vector
	    		SimpleMatrix temporalFFInputVector = new SimpleMatrix(spatialFFOutputMatrix);
	    		temporalFFInputVector.reshape(1, spatialFFOutputMatrix.getMatrix().data.length);
	    		
	    		//Temporal classification
	    		temporalPooler.feedForward(temporalFFInputVector);
	    		
	    		if (visualize){
		    		//Update graphics
		    		updateGraphics(input, curSeqID);
		    		
		    		//Sleep
					try {
						Thread.sleep(SKIP_TICKS);
					} catch (InterruptedException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}	
	    		}
	    	}
    		
    		spatialPooler.sensitize(i);
    		temporalPooler.sensitize(i);
	    }

	}
	
	private void setupGraphics(){
		frame = new MovingLinesGUI_Prediction(spatialPooler, temporalPooler);
		frame.setTitle("Visualiztion");
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		updateGraphics(blank,0); //Give a blank
		frame.pack();
		frame.setVisible(true);
		
	}
	
	private void updateGraphics(SimpleMatrix inputVector, int iteration){
		frame.updateData(inputVector, spatialPooler, temporalPooler);
		frame.setTitle("Visualiztion - Current sequence: " + iteration);
		frame.revalidate();
		frame.repaint();

	}
	
	private void setupExperiment(int maxIterations, Random rand){
		buildSequences();
		buildPoolers(maxIterations, rand);		
	}
	
	private void buildPoolers(int maxIterations, Random rand){
		
		//Spatial pooler
		int spatialInputLength = blank.getNumElements();
		int spatialMapSize = 3;
		double spatialInitialLearningRate = 0.1;
		double stddev_spatial = 2;
		double activationCodingFactor_spatial = 0.125;
		
		spatialPooler =  new SpatialPooler(rand, spatialInputLength, spatialMapSize, spatialInitialLearningRate, stddev_spatial, activationCodingFactor_spatial);
		
		//Temporal pooler
		int temporalInputLength = spatialMapSize * spatialMapSize;
		int temporalMapSize = 3;
		double decay = 0.3;
		double stdDev_temporal = 2;
		double temporalLearningRate = 0.1;
		double activationCodingFactor_Temporal = 0.125;
		temporalPooler = new TemporalPooler(rand, temporalInputLength, temporalMapSize, temporalLearningRate, stdDev_temporal, activationCodingFactor_Temporal, decay);
	}
	
	private void buildSequences(){
		createFigures();
		sequences = new ArrayList<SimpleMatrix[]>();

		SimpleMatrix[] seq1 = {bigT};
		SimpleMatrix[] seq2 = {bigO};
		SimpleMatrix[] seq3 = {smallO};
		
		SimpleMatrix[] seq4 = {smallO, bigO};
		SimpleMatrix[] seq5 = {bigT, smallO};
		SimpleMatrix[] seq6 = {bigT, bigT};
		
		SimpleMatrix[] seq7 = {bigO, smallO, smallO, smallV};
		SimpleMatrix[] seq8 = {bigT, bigO, smallV, bigT};
		SimpleMatrix[] seq9 = {smallO, bigT, bigT, bigO};
		SimpleMatrix[] seq10 = {bigT, smallO, bigO, smallO};
		SimpleMatrix[] seq11 = {bigT, bigT, bigT, bigT};
		SimpleMatrix[] seq12 = {smallO, smallO, bigO, bigO};
		
		SimpleMatrix[] seq13 = {smallO, smallO, bigO, bigO, bigT};

		
		//sequences.add(seq1);
		//sequences.add(seq2);
		//sequences.add(seq3);
		
		//sequences.add(seq4);
		//sequences.add(seq5);
		//sequences.add(seq6);
		
		sequences.add(seq7);
		sequences.add(seq8);
		
		sequences.add(seq9);
		sequences.add(seq10);
		sequences.add(seq11);
		sequences.add(seq12);
		
		//sequences.add(seq13);
		
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
		
		double[][] jokerData = {
				{1,0,0,0,1},
				{0,1,0,1,0},
				{0,0,1,0,0},
				{0,1,0,1,0},
				{1,0,0,0,1}};
		joker = new SimpleMatrix(jokerData);
		joker.reshape(1, joker.numCols() * joker.numRows());
	}
}