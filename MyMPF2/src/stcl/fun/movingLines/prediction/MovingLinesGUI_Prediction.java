package stcl.fun.movingLines.prediction;

import java.awt.Dimension;
import java.awt.GridLayout;
import java.util.ArrayList;

import javax.swing.JFrame;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.NeoCorticalUnit;
import stcl.algo.poolers.SpatialPooler;
import stcl.algo.poolers.TemporalPooler;
import stcl.fun.movingLines.MatrixPanel;
import stcl.fun.movingLines.SomPanel;
import dk.stcl.som.containers.SomNode;
import dk.stcl.som.rsom.IRSOM;
import dk.stcl.som.som.ISOM;
import dk.stcl.som.som.SOM;

public class MovingLinesGUI_Prediction extends JFrame {
	private MatrixPanel input;
	private SomPanel spatialModels;
	private MatrixPanel spatialActivation;
	private MatrixPanel predictedInput;
	private ArrayList<SomPanel> rsomModels;
	ArrayList<MatrixPanel> rsomActivations;

	
	private int singleSomModelWidth;
	
	
	public MovingLinesGUI_Prediction(NeoCorticalUnit unit) {
		SOM spatialSom = unit.getSpatialPooler().getSOM();
		
		//Create overall grid layout
		int rows = 3;
		int cols = 4;
		int gap = 2;
		setLayout(new GridLayout(rows, cols, gap, gap));
		
		//Set preffered size of GUI
		int somHeight = spatialSom.getHeight(); //This should be equal to the width also
		singleSomModelWidth =  (int) Math.sqrt(spatialSom.getInputVectorLength()); //How many cells (size x size) are there in a single SOM model
		int somModelCellSize = 5;	  //How many pixels (size x size) does a single cell in a SOM model require in height and width
		int gapBetweenSomodels = 2;
		
		int somPanelSize = somHeight * singleSomModelWidth * gapBetweenSomodels * somModelCellSize;
		
		setPreferredSize(new Dimension(cols * somPanelSize, rows * somPanelSize));

		//Add input area
		input = new MatrixPanel(new SimpleMatrix(singleSomModelWidth, singleSomModelWidth));
		input.setSize(somHeight, somHeight);
		add(input);
		
		//Add visualization of spatial som models
		spatialModels = new SomPanel(spatialSom, singleSomModelWidth, singleSomModelWidth);
		spatialModels.setSize(somPanelSize, somPanelSize);
		add(spatialModels);
		
		//Add visualization of spatial activation
		spatialActivation = new MatrixPanel(new SimpleMatrix(5, 5));
		spatialActivation.setSize(somPanelSize, somPanelSize);
		add(spatialActivation);
		
		//Add predicted next input
		predictedInput = new MatrixPanel(new SimpleMatrix(singleSomModelWidth, singleSomModelWidth));
		predictedInput.setSize(somHeight, somHeight);
		add(predictedInput);
		
		//Add rsom models and activation
		int numRsomModels = unit.getTemporalPooler().getSOM().getHeight();
		numRsomModels *= numRsomModels;
		rsomActivations = new ArrayList<MatrixPanel>();
		rsomModels = new ArrayList<SomPanel>();
		//Add models
		for (int i = 0; i < numRsomModels; i++){
			SomPanel model = new SomPanel(spatialSom, singleSomModelWidth, singleSomModelWidth);
			model.setSize(somPanelSize, somPanelSize);
			add(model);
			rsomModels.add(model);
		}
		//Add activation
		for (int i = 0; i < numRsomModels; i++){
			MatrixPanel activation = new MatrixPanel(new SimpleMatrix(1, 1));;
			activation.setSize(somPanelSize, somPanelSize);
			add(activation);
			rsomActivations.add(activation);
		}
		
	}
	
	public void updateData(SimpleMatrix inputVector, NeoCorticalUnit unit){
		
		SpatialPooler spatialPooler = unit.getSpatialPooler();
		TemporalPooler temporalPooler = unit.getTemporalPooler();
		
		//Update input area
		SimpleMatrix inputMatrix = new SimpleMatrix(inputVector);
		inputMatrix.reshape(singleSomModelWidth, singleSomModelWidth);
		input.updateData(inputMatrix);
		input.repaint();
		input.revalidate();
		
		//Update spatial activation
		SimpleMatrix activationMatrix = spatialPooler.getActivationMatrix();
		spatialActivation.updateData(activationMatrix);
		spatialActivation.repaint();
		spatialActivation.revalidate();
		
		//Get predicted next output
		SimpleMatrix expectedInput = unit.getFbOutput();
		expectedInput.reshape(singleSomModelWidth, singleSomModelWidth);
		predictedInput.updateData(expectedInput);
		predictedInput.repaint();
		predictedInput.revalidate();		
		
		//Create list of spatial models to be highlighted
		int maxID = findIDOfMaxValue(activationMatrix);
		boolean[] highlights = new boolean[activationMatrix.getMatrix().data.length];
		highlights[maxID] = true;
		
		//Update Spatial models
		SOM spatialSom = spatialPooler.getSOM();
		spatialModels.updateData(spatialSom, highlights);
		spatialModels.repaint();
		spatialModels.revalidate();		
		
		//Update RSOM activation
		SimpleMatrix temporalActivationMatrix = temporalPooler.getActivationMatrix();
		int rows = temporalActivationMatrix.numRows();
		int cols = temporalActivationMatrix.numCols();
		int id = 0;
		for (int row = 0; row < rows; row++){
			for (int col = 0; col < cols; col++){
				//Update activation
				double[][] tmp = {{temporalActivationMatrix.get(row, col)}};
				MatrixPanel panel = rsomActivations.get(id);
				panel.updateData(new SimpleMatrix(tmp));
				panel.repaint();
				panel.revalidate();
				
				//Update rsom model
				SomPanel somPanel = rsomModels.get(id);
				somPanel.updateData(spatialSom, getSpatialModelsInTemporalModel(temporalPooler, id));
				somPanel.repaint();
				somPanel.revalidate();
				id++;
			}
		}		
	}
	
	private boolean[] getSpatialModelsInTemporalModel(TemporalPooler temporalPooler, int modelID){
		//Collect Model
		SomNode model = temporalPooler.getSOM().getNode(modelID);
		
		//Create boolean vector where an item is true if weight value of that item is higher than the mean.
		SimpleMatrix weightVector = model.getVector();
		int vectorSize = weightVector.numCols() * weightVector.numRows();
		double mean = weightVector.elementSum() / (double)vectorSize;
		double threshold = 0.9; //2 * ( 1 / (temporalPooler.getHeight() * temporalPooler.getWidth())); 
		boolean[] importantModels = new boolean[vectorSize];
		
		for (int i = 0; i <vectorSize; i++){
			if (weightVector.get(i) > threshold){
				importantModels[i] = true;
			}
		}
		
		
		return importantModels;
	}
	
	/**
	 * 
	 * @param m
	 * @return id of cell with max value
	 */
	private int findIDOfMaxValue(SimpleMatrix m){
		double maxValue = Double.NEGATIVE_INFINITY;
		int maxID = -1;
		for (int row = 0; row < m.numRows(); row++){
			for (int col = 0; col < m.numCols(); col++){
				double value = m.get(row, col);
				if (value > maxValue){
					maxValue = value;
					maxID = m.getIndex(row, col);
				}
			}
		}
		return maxID;
	}

	

}
