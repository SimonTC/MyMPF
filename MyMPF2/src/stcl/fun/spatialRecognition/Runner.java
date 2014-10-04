package stcl.fun.spatialRecognition;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.Raster;
import java.awt.image.SampleModel;
import java.io.File;
import java.util.Random;

import javax.imageio.ImageIO;
import javax.swing.JFrame;

import org.ejml.simple.SimpleMatrix;
import org.omg.CORBA.OMGVMCID;

import stcl.algo.poolers.SpatialPooler;
import stcl.algo.som.SomNode;
import stcl.graphics.MultipleMapDrawerGRAY;

public class Runner {
	
	private MultipleMapDrawerGRAY frame;
	private SpatialPooler pooler;
	private SimpleMatrix[] figureMatrices;
	
	public static void main(String[] args) {
		Runner runner = new Runner();
		runner.run();
	}
	
	public void run(){
		int FRAMES_PER_SECOND = 40;
	    int SKIP_TICKS = 1000 / FRAMES_PER_SECOND;
	   
	    float next_game_tick = System.currentTimeMillis();
	    float sleepTime = 0;
		
	    int maxIterations = 50;
	    boolean useSimpleImages = false;
		setupExperiment(maxIterations, useSimpleImages);
		
		
		
		for (int i = 0; i < maxIterations; i++){
			SimpleMatrix[] outputs = new SimpleMatrix[figureMatrices.length];
			
			for (int j = 0; j < figureMatrices.length; j++){
				//Feed forward
				SimpleMatrix out = pooler.feedForward(figureMatrices[j]);			
				
				//Collect BMU
				SomNode bmu = pooler.getSOM().getBMU(figureMatrices[j]);
				SimpleMatrix m = new SimpleMatrix(bmu.getVector());
				m.reshape(100, 100);
				outputs[j] = m;				
			}
			
			
			//Visualize maps
			updateGraphics(outputs, i);
			
			//Sleep
			next_game_tick+= SKIP_TICKS;
			sleepTime = next_game_tick - System.currentTimeMillis();
			if (sleepTime >= 0){
				try {
					Thread.sleep(SKIP_TICKS);
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
			pooler.tick();
		}
				
			
	}
	
	private void setupExperiment(int iterations, boolean simple){
		int figureRows = 0;
		int figureColumns = 0;
		
		//Create Figure matrices
		if (simple){
			figureMatrices = simpleFigures();
			figureRows = 5;
			figureColumns = 5;
		} else {
			figureMatrices = pixelArtFigures();
			figureRows = 100;
			figureColumns = 100;
			
		}

		
		//Create spatial pooler
		Random rand = new Random();
		int maxIterations = iterations;
		int inputLength = figureColumns * figureRows;
		int mapSize = 10;
		double initialLearningRate = 0.5;
		pooler = new SpatialPooler(rand, maxIterations, inputLength, mapSize, initialLearningRate, mapSize / 2, 1);
		
		//Setup graphics
		setupGraphics(figureRows, figureColumns);
		
	}
	
	private void updateGraphics(SimpleMatrix[] matrices, int iteration){
		frame.updateMaps(matrices);
		frame.setTitle("Visualiztion - Iteration: " + iteration);
		frame.revalidate();
		frame.repaint();
	}
	
	private void setupGraphics(int mapHeight, int mapWidth ) {
		int mapGuiSize = 200;
		frame = new MultipleMapDrawerGRAY(mapHeight, mapWidth, figureMatrices.length, mapGuiSize, mapGuiSize);
		frame.setTitle("Visualiztion");
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		
		SimpleMatrix[] tmp = new SimpleMatrix[figureMatrices.length];
		for (int i = 0; i < figureMatrices.length; i++){
			SimpleMatrix m = new SimpleMatrix(mapHeight, mapWidth);
			tmp[i] = m;
		}
		
		frame.updateMaps(tmp);
		frame.pack();

		frame.setVisible(true);

	}
	
	private SimpleMatrix[] simpleFigures(){
		SimpleMatrix[] matrices = new SimpleMatrix[4];
		
		double[][] bigTData = {
				{0,0,0,0,0},
				{0,1,1,1,0},
				{0,0,1,0,0},
				{0,0,1,0,0},
				{0,0,1,0,0}};
		SimpleMatrix bigT = new SimpleMatrix(bigTData);
		bigT.reshape(1, bigT.numCols() * bigT.numRows());
		matrices[0] = bigT;
		
		double[][] smallOData = {
				{0,0,0,0,0},
				{0,1,1,1,0},
				{0,1,0,1,0},
				{0,1,1,1,0},
				{0,0,0,0,0}};
		SimpleMatrix smallO = new SimpleMatrix(smallOData);
		smallO.reshape(1, smallO.numCols() * smallO.numRows());
		matrices[1] = smallO;
		
		double[][] bigOData = {
				{1,1,1,1,1},
				{1,0,0,0,1},
				{1,0,0,0,1},
				{1,0,0,0,1},
				{1,1,1,1,1}};
		SimpleMatrix bigO = new SimpleMatrix(bigOData);
		bigO.reshape(1, bigO.numCols() * bigO.numRows());
		matrices[2] = bigO;
		
		double[][] smallVData = {
				{0,0,0,0,0},
				{0,0,0,0,0},
				{1,0,0,0,1},
				{0,1,0,1,0},
				{0,0,1,0,0}};
		SimpleMatrix smallV = new SimpleMatrix(smallVData);
		smallV.reshape(1, smallV.numCols() * smallV.numRows());
		matrices[3] = smallV;
		
		return matrices;
	}
	
	private SimpleMatrix[] pixelArtFigures(){
		String parent = "images/PixelArt/";
		String marioPath = parent + "MarioBW.png";
		String nyanCatPath = parent + "NyanCatBW.png";
		String obama1Path = parent + "Obama1BW.png";
		String obama2Path = parent + "Obama2BW.png";
		
		SimpleMatrix[] matrices = new SimpleMatrix[4];
		double[][] marioData = loadNormalizedImage(marioPath);
		double[][] nyanCatData = loadNormalizedImage(nyanCatPath);
		double[][] obama1Data = loadNormalizedImage(obama1Path);
		double[][] obama2Data = loadNormalizedImage(obama2Path);
		
		SimpleMatrix mario = new SimpleMatrix(marioData);
		mario.reshape(1, 100*100);
		
		
		SimpleMatrix nyanCat = new SimpleMatrix(nyanCatData);
		nyanCat.reshape(1, 100*100);
		
		SimpleMatrix obama1 = new SimpleMatrix(obama1Data);
		obama1.reshape(1, 100*100);
		
		SimpleMatrix obama2 = new SimpleMatrix(obama2Data);
		obama2.reshape(1, 100*100);
		
		matrices[0] = mario;
		matrices[1] = nyanCat;
		matrices[2] = obama1;
		matrices[3] = obama2;
		
		
		
		return matrices;
	}
	
	private double[][] loadNormalizedImage(String imagePath){
		try 
		{
		    File image = new File(imagePath);
			BufferedImage img = ImageIO.read(image);
			//int[][] rgbValues = convertToRGB(img);
			
			Raster raster=img.getData();
			
			int w = raster.getWidth();
		    int h=raster.getHeight();
		    double normPixelValues[][]=new double[w][h];
		    
		    int counter = 0;
		    
		   for (int x=0;x<w;x++)
		    {
		        for(int y=0;y<h;y++)
		        {
		        	int rgb = img.getRGB(x, y);
		        	int grayValue = (rgb) & 0x000000FF; //Is the same as the blue = red = green rgb value
		        	normPixelValues[y][x]=   1- ((double)grayValue / 255);
		        	
		        }
		    }

		    return normPixelValues;
		    
			

		}
		catch (Exception e)
		{
		    e.printStackTrace();
		}
		return null;
	}

	private int count(double[][] array){
		int counter = 0;
		for (int x = 0; x < array[0].length; x++){
			for (int y = 0; y < array.length; y++){
				double value = array[x][y];
				if (value > 0 && value < 1){
	        		counter++;
	        	}
			}
		}
		return counter;
	}
}
