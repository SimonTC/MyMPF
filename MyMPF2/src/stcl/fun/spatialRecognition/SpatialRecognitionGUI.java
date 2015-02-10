package stcl.fun.spatialRecognition;

import java.awt.Dimension;
import java.awt.GridLayout;
import java.awt.HeadlessException;

import javax.swing.JFrame;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.poolers.SOM;
import stcl.algo.poolers.SpatialPooler;
import stcl.graphics.MatrixPanel;
import stcl.graphics.SomPanel;

public class SpatialRecognitionGUI extends JFrame {
	private MatrixPanel input;
	private SomPanel spatialModels;
	private MatrixPanel spatialActivation;
	
	private int sizeOfSingleSomModel;
	private int sizeOfSom;
	/**
	 * 
	 * @param spatialSom
	 * @param somModelSize How many cells (size x size) are there in a single SOM model
	 * @param sizeOfSom How many models does the SOM map have (size x size)
	 * @throws HeadlessException
	 */
	public SpatialRecognitionGUI(SOM spatialSom, int somModelSize, int sizeOfSom) throws HeadlessException {
		//Create grid layout
		int panelsAcross = 1;
		int panelsDown = 3;
		int gapBetweenPanels = 2;
		setLayout(new GridLayout(panelsAcross, panelsDown, gapBetweenPanels, gapBetweenPanels));
		
		//Set preffered size of GUI
		this.sizeOfSingleSomModel = somModelSize; //How many cells (size x size) are there in a single SOM model
		int somModelCellSize = 5;	  //How many pixels (size x size) does a single cell in a SOM model require
		this.sizeOfSom = sizeOfSom; //How many models does the SOM map have (size x size)
		int gapBetweenSomodels = 2;
		
		int somSize = sizeOfSingleSomModel * sizeOfSom * gapBetweenSomodels * somModelCellSize;
		
		setPreferredSize(new Dimension(panelsDown * somSize, panelsAcross * somSize));
		
		//Add input area
		input = new MatrixPanel(new SimpleMatrix(sizeOfSingleSomModel, sizeOfSingleSomModel));
		input.setSize(sizeOfSingleSomModel, sizeOfSingleSomModel);
		add(input);
		
		//Add visualization of spatial som models
		spatialModels = new SomPanel(spatialSom, sizeOfSingleSomModel, sizeOfSingleSomModel);
		spatialModels.setSize(somSize, somSize);
		add(spatialModels);
		
		//Add visualization of spatial activation
		spatialActivation = new MatrixPanel(new SimpleMatrix(sizeOfSom, sizeOfSom));
		spatialActivation.setSize(somSize, somSize);
		add(spatialActivation);
				
	}
	
	public void updateData(SimpleMatrix inputVector, SpatialPooler spatialPooler){
		
		//Update input area
		SimpleMatrix inputMatrix = new SimpleMatrix(inputVector);
		inputMatrix.reshape(sizeOfSingleSomModel, sizeOfSingleSomModel);
		input.updateData(inputMatrix);
		input.repaint();
		input.revalidate();
		
		//Update Spatial models
		spatialModels.updateData(spatialPooler.getSOM());
		spatialModels.repaint();
		spatialModels.revalidate();
		
		//Update spatial activation
		spatialActivation.updateData(spatialPooler.getActivationMatrix());
		spatialActivation.repaint();
		spatialActivation.revalidate();
	}


}
