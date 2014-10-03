package stcl.fun.spatialRecognition;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.poolers.SpatialPooler;
import stcl.algo.som.SomNode;
import stcl.graphics.BWMapRenderer;

public class Runner {

	public static void main(String[] args) {
		Runner runner = new Runner();
		runner.run();
	}
	
	public void run(){
		
		//Create Figure matrices
		//SimpleMatrix[] matrices = matrices();
		
		SimpleMatrix bigT = bigT();
		
		//Create spatial pooler
		Random rand = new Random();
		int maxIterations = 10;
		int inputLength = 5*5;
		int mapSize = 50;
		SpatialPooler pooler = new SpatialPooler(rand, maxIterations, inputLength, mapSize);
		
		//Create map renderer
		BWMapRenderer renderer = new BWMapRenderer();
		
		
	}
	
	private SimpleMatrix[] matrices(){
		SimpleMatrix[] matrices = new SimpleMatrix[4];
		double[][] bigTData = {
				{0,0,0,0,0},
				{0,1,1,1,0},
				{0,0,1,0,0},
				{0,0,1,0,0},
				{0,0,1,0,0}};
		SimpleMatrix bigT = new SimpleMatrix(bigTData);
		matrices[0] = bigT;
		
		double[][] smallOData = {
				{0,0,0,0,0},
				{0,1,1,1,0},
				{0,1,0,1,0},
				{0,1,1,1,0},
				{0,0,0,0,0}};
		SimpleMatrix smallO = new SimpleMatrix(smallOData);
		matrices[1] = smallO;
		
		double[][] bigOData = {
				{1,1,1,1,1},
				{1,0,0,0,1},
				{1,0,0,0,1},
				{1,0,0,0,1},
				{1,1,1,1,1}};
		SimpleMatrix bigO = new SimpleMatrix(bigOData);
		matrices[2] = bigO;
		
		double[][] smallVData = {
				{0,0,0,0,0},
				{0,0,0,0,0},
				{1,0,0,0,1},
				{0,1,0,1,0},
				{0,0,1,0,0}};
		SimpleMatrix smallV = new SimpleMatrix(smallVData);
		matrices[3] = smallV;
		
		return matrices;
	}
	
	private SimpleMatrix bigT(){
		double[][] bigTData = {
				{0,0,0,0,0},
				{0,1,1,1,0},
				{0,0,1,0,0},
				{0,0,1,0,0},
				{0,0,1,0,0}};
		SimpleMatrix bigT = new SimpleMatrix(bigTData);
		return bigT;
	}

	
	
	
	

}
