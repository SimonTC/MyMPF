package stcl.fun.temporalRecognition;

import java.util.ArrayList;
import java.util.Random;

import javax.swing.JFrame;

import org.ejml.simple.SimpleMatrix;

import dk.stcl.core.basic.containers.SomNode;
import dk.stcl.core.rsom.RSOM_Simple;
import dk.stcl.core.som.SOM_SemiOnline;
import stcl.algo.brain.NeoCorticalUnit;
import stcl.algo.poolers.RSOM;
import stcl.algo.poolers.SpatialPooler;
import stcl.algo.poolers.TemporalPooler;
import stcl.algo.predictors.Predictor;
import stcl.graphics.MovingLinesGUI_Prediction;

public class TemporalRecognition_RSOM2 {
	private ArrayList<SimpleMatrix[]> sequences;
	private MovingLinesGUI_Prediction frame;
	private Random rand = new Random(1234);
	
	private RSOM_Simple rsom;
	
	private final int ITERATIONS = 10000;
	private final boolean VISUALIZE_TRAINING = false;
	private final boolean VISUALIZE_RESULT = true;
	private boolean usePrediction = true;
	private boolean evaluate = true;
	private SimpleMatrix bigT;
	private SimpleMatrix smallO;
	private SimpleMatrix bigO;
	private SimpleMatrix smallV;
	private SimpleMatrix blank;
	private SimpleMatrix joker;
	private SOM_SemiOnline spatialDummy = new SOM_SemiOnline(2,5,rand,0,0,0);
	
	int FRAMES_PER_SECOND = 1;
    int SKIP_TICKS = 1000 / FRAMES_PER_SECOND;
	
	public static void main(String[] args){
		TemporalRecognition_RSOM2 runner = new TemporalRecognition_RSOM2();
		runner.run();
	}
	
	public void run(){
		//Setup experiment		
		setupExperiment(ITERATIONS, rand);
		
		//Setup graphics
		if (VISUALIZE_TRAINING) setupGraphics();
		
		//Train
		training(ITERATIONS, rand, VISUALIZE_TRAINING);
		rsom.flush();
		rsom.setLearning(false);
		
		//Label
		int[] labels = createLabels();
		RSomLabeler labeler = new RSomLabeler();
		labeler.label(rsom, sequences, labels);
		//labeler.label(rsom, sequences, labels, true);
		
		if (evaluate){
		//Evaluate
			double noise = 0.0;
			for (int i = 0; i < 100; i++){
				//nu.setDebug(true);
				RsomEvaluator evaluator = new RsomEvaluator();
				double fitness = evaluator.evaluate(rsom, sequences, labels, joker, noise, 1000, rand);			
				System.out.println("Fitness: " + fitness);
				noise += 0.01;
			}
		}
		System.out.println();
		System.out.println("RSOM labels");
		rsom.printLabelMap();
	    
	    System.out.println("Rsom models:");
	    System.out.println();
	    for (SomNode n : rsom.getNodes()){
	    	SimpleMatrix vector = new SimpleMatrix(n.getVector());
	    	vector.print();
	    	System.out.println(vector.elementSum());
	    	System.out.println();
	    }
		
		if (VISUALIZE_RESULT){
			rsom.flush();
			rsom.setLearning(false);
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
	    	rsom.flush();
	    	
	    	//Choose sequence	    	
	    	curSeqID = rand.nextInt(sequences.size());
	    	SimpleMatrix[] curSequence = sequences.get(curSeqID);
	    	
	    	for (SimpleMatrix input : curSequence){
	    		
	    		rsom.step(input);
	    		rsom.computeActivationMatrix();
	    		//System.out.println("Leaky differences:");
	    		//rsom.getLeakyDifferencesMap();

	    		
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
    		
    		rsom.sensitize(i);
	    }

	}
	
	private void setupGraphics(){
		frame = new MovingLinesGUI_Prediction(spatialDummy, rsom);
		frame.setTitle("Visualiztion");
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		updateGraphics(blank,0); //Give a blank
		frame.pack();
		frame.setVisible(true);
		
	}
	
	private void updateGraphics(SimpleMatrix inputVector, int iteration){
		frame.updateData(inputVector, spatialDummy,rsom);
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
		int spatialInputLength = joker.getNumElements();
		int spatialMapSize = 3;
		double spatialInitialLearningRate = 0.1;
		double stddev_spatial = 2;
		double activationCodingFactor_spatial = 0.125;

		
		//Temporal pooler
		int temporalInputLength = spatialMapSize * spatialMapSize;
		int temporalMapSize = 3;
		double decay = 0.3;
		double stdDev_temporal = 2;
		double temporalLearningRate = 0.1;
		double activationCodingFactor_Temporal = 0.125;
		double initialPredictionLearningRate = 0.5;
		
		rsom = new RSOM_Simple(temporalMapSize, joker.getNumElements(), rand, temporalLearningRate, activationCodingFactor_Temporal, maxIterations, decay);
		

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
		
		sequences.add(seq4);
		sequences.add(seq5);
		//sequences.add(seq6);
		
		sequences.add(seq7);
		sequences.add(seq8);
		
		sequences.add(seq9);
		sequences.add(seq10);
		//sequences.add(seq11);
		//sequences.add(seq12);
		
		//sequences.add(seq13);
		
	}
	
	private void createFigures(){		
		double[][] bigTData = {
				{1,0,0,0,0}};
		bigT = new SimpleMatrix(bigTData);
		bigT.reshape(1, bigT.numCols() * bigT.numRows());
		
		double[][] smallOData = {
				{0,1,0,0,0}};
		smallO = new SimpleMatrix(smallOData);
		smallO.reshape(1, smallO.numCols() * smallO.numRows());
		
		double[][] bigOData = {
				{0,0,1,0,0}};
		bigO = new SimpleMatrix(bigOData);
		bigO.reshape(1, bigO.numCols() * bigO.numRows());
		
		double[][] smallVData = {
				{0,0,0,1,0}};
		smallV = new SimpleMatrix(smallVData);
		smallV.reshape(1, smallV.numCols() * smallV.numRows());
		
		double[][] jokerData = {
				{0,0,0,0,1}};
		joker = new SimpleMatrix(jokerData);
		joker.reshape(1, joker.numCols() * joker.numRows());
		
		blank = joker;
	}
}
