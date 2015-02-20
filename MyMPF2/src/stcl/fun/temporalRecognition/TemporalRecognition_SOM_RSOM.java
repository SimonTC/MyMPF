package stcl.fun.temporalRecognition;

import java.util.ArrayList;
import java.util.Random;

import javax.swing.JFrame;

import org.ejml.simple.SimpleMatrix;

import dk.stcl.core.basic.containers.SomNode;
import dk.stcl.core.rsom.IRSOM;
import dk.stcl.core.rsom.RSOM_SemiOnline;
import dk.stcl.core.rsom.RSOM_Simple;
import dk.stcl.core.som.ISOM;
import dk.stcl.core.som.SOM_SemiOnline;
import dk.stcl.core.som.SOM_Simple;
import stcl.algo.poolers.SpatialPooler;
import stcl.algo.poolers.TemporalPooler;
import stcl.graphics.MovingLinesGUI_Prediction;

public class TemporalRecognition_SOM_RSOM {
	private ArrayList<SimpleMatrix[]> sequences;
	private IRSOM rsom;
	private ISOM som;
	private MovingLinesGUI_Prediction frame;
	private Random rand = new Random(1234);
	
	private final int ITERATIONS = 1000;
	private final boolean VISUALIZE_TRAINING = false;
	private final boolean VISUALIZE_RESULT = true;
	private SimpleMatrix bigT;
	private SimpleMatrix smallO;
	private SimpleMatrix bigO;
	private SimpleMatrix smallV;
	private SimpleMatrix blank;
	
	private final boolean USE_SIMPLE_SOM = false;
	int FRAMES_PER_SECOND = 10;
	
	public static void main(String[] args){
		TemporalRecognition_SOM_RSOM runner = new TemporalRecognition_SOM_RSOM();
		runner.run();
	}
	
	public void run(){
		//Setup experiment		
		setupExperiment(ITERATIONS, rand);
		
		//Setup graphics
		if (VISUALIZE_TRAINING) setupGraphics();
		
		runExperiment(ITERATIONS, rand, VISUALIZE_TRAINING);
		
		if (VISUALIZE_RESULT){
			rsom.flush();
			rsom.setLearning(false);
			som.setLearning(false);
			 setupGraphics();
			 runExperiment(ITERATIONS, rand, true);
			 
		}
		
	}
	
	private void runExperiment(int maxIterations, Random rand, boolean visualize){
	    int SKIP_TICKS = 1000 / FRAMES_PER_SECOND;
	   
	    float next_game_tick = System.currentTimeMillis();
	    int curSeqID = 0;
    	int curInputID = -1;
	    
	    for (int i = 0; i < maxIterations; i++){
	    	rsom.flush();
	    	
	    	//Choose sequence	    	
	    	curSeqID = rand.nextInt(sequences.size());
	    	SimpleMatrix[] curSequence = sequences.get(curSeqID);
	    	
	    	for (SimpleMatrix input : curSequence){
	    		//Spatial classification
	    		som.step(input);
	    		SimpleMatrix spatialFFOutputMatrix = som.computeActivationMatrix();
	    		
	    		spatialFFOutputMatrix = orthogonalize(spatialFFOutputMatrix);
	    		
	    		//Transform spatial output matrix to vector
	    		SimpleMatrix temporalFFInputVector = new SimpleMatrix(spatialFFOutputMatrix);
	    		temporalFFInputVector.reshape(1, spatialFFOutputMatrix.getMatrix().data.length);
	    		
	    		//Temporal classification
	    		rsom.step(temporalFFInputVector);
	    		rsom.computeActivationMatrix();
	    		
	    		if (visualize){
		    		//Update graphics
		    		updateGraphics(input, curSeqID);
		    		
		    		//Sleep
					next_game_tick+= SKIP_TICKS;
					try {
						Thread.sleep(SKIP_TICKS);
					} catch (InterruptedException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}	
	    		}
	    	}
	    	
    		som.sensitize(i);
    		rsom.sensitize(i);
	    }
	    
	    System.out.println("SOM models:");
	    for (SomNode n : som.getNodes()){
	    	SimpleMatrix vector = new SimpleMatrix(n.getVector());
	    	vector.reshape(5,5);
	    	vector.print();
	    	System.out.println(vector.elementSum());
	    	System.out.println();
	    }
	    
	    System.out.println("Rsom models:");
	    System.out.println();
	    for (SomNode n : rsom.getNodes()){
	    	SimpleMatrix vector = new SimpleMatrix(n.getVector());
	    	vector.reshape(som.getHeight(), som.getWidth());
	    	vector.print();
	    	System.out.println(vector.elementSum());
	    	System.out.println();
	    }
	}
	
	private SimpleMatrix orthogonalize(SimpleMatrix m) {
		double max = Double.NEGATIVE_INFINITY;
		int maxID = -1;
		for (int i = 0; i < m.getNumElements(); i++){
			double value = m.get(i);
			if (value > max){
				max = value;
				maxID = i;
			}
		}
		
		SimpleMatrix ortho = new SimpleMatrix(m.numRows(), m.numCols());
		ortho.set(maxID, 1);
		return ortho;
	}

	
	private SimpleMatrix normalize(SimpleMatrix matrix){
		double sum = matrix.elementSum();
		SimpleMatrix m = matrix.scale(1/sum);
		return m;
	}
	
	private void setupGraphics(){
		frame = new MovingLinesGUI_Prediction(som, rsom);
		frame.setTitle("Visualiztion");
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		updateGraphics(blank,0); //Give a blank
		frame.pack();
		frame.setVisible(true);
		
	}
	
	private void updateGraphics(SimpleMatrix inputVector, int iteration){
		frame.updateData(inputVector, som, rsom);
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
		double learningRate_Spatial = 0.1;
		double stddev_spatial = 2;
		double activationCodingFactor_spatial = 0.125;
		
		if (USE_SIMPLE_SOM){
			som = new SOM_Simple(spatialMapSize, spatialInputLength, rand, learningRate_Spatial, activationCodingFactor_spatial, maxIterations);
		} else {
			som = new SOM_SemiOnline(spatialMapSize, spatialInputLength, rand, learningRate_Spatial, activationCodingFactor_spatial, stddev_spatial);
		}
		
		//Temporal pooler
		int temporalInputLength = spatialMapSize * spatialMapSize;
		int temporalMapSize = 2;
		double decay = 0.3;
		double stdDev = 2;
		double temporalLearningRate = 0.1;
		double activationCodingFactor = 0.125;
		if (USE_SIMPLE_SOM){
			rsom = new RSOM_Simple(temporalMapSize, temporalInputLength, rand, temporalLearningRate, activationCodingFactor, maxIterations, decay);
		} else {
			rsom = new RSOM_SemiOnline(temporalMapSize, temporalInputLength, rand, temporalLearningRate, activationCodingFactor, stdDev, decay);
		}
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
		sequences.add(seq11);
		sequences.add(seq12);
		
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
