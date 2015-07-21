package stcl.algo.poolers;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import dk.stcl.core.basic.SomBasics;
import dk.stcl.core.basic.containers.SomNode;
import dk.stcl.core.utils.SomConstants;

public abstract class Pooler {
	
	//Matrice used 
	protected SimpleMatrix activationMatrix;
	
	//Variables used for testing inputs
	protected int inputLength;
	protected int mapSize;
	
	//Misc
	private Random rand;
	
	public Pooler(Random rand, int inputLength, int mapSize){
		this.rand = rand;
		this.inputLength = inputLength;
		this.mapSize = mapSize;
		this.activationMatrix = new SimpleMatrix(mapSize, mapSize);
	}
	
	public Pooler(String initializationString, int startLine, Random rand){
		String[] lines = initializationString.split(SomConstants.LINE_SEPARATOR);
		String[] poolerInfo = lines[startLine].split(" ");
		inputLength = Integer.parseInt(poolerInfo[0]);
		mapSize = Integer.parseInt(poolerInfo[1]);
		activationMatrix = new SimpleMatrix(mapSize, mapSize);
		this.rand = rand;
	}
	
	public String toInitializationString(){
		String s = inputLength + " " + mapSize + SomConstants.LINE_SEPARATOR;
		return s;
	}
	
	protected SimpleMatrix chooseRandom(SimpleMatrix input, SomBasics map){
		//Transform matrix into vector
		double[] vector = input.getMatrix().data;
		
		//Choose random number between 0 and 1
		double d = rand.nextDouble();
		
		//Go through bias vector until value is >= random number
		double tmp = 0;
		int id = 0;
		while (tmp < d && id < vector.length){
			tmp += vector[id++];
		}
		
		id--; //We have to subtract to be sure we get the correct model
		
		//Choose model from som
		SimpleMatrix model = map.getNode(id).getVector();
		
		//System.out.println("Chose model: " + id);
		
		return model;
		
	}
	
	/**
	 * Choose the most probable model from the given SomBasics object
	 * @param input
	 * @param map
	 * @return
	 */
	protected SimpleMatrix chooseMax(SimpleMatrix input, SomBasics map){
		//Transform bias matrix into vector
		double[] vector = input.getMatrix().data;
		
		//Go through bias vector until value is >= random number
		double max = Double.NEGATIVE_INFINITY;
		int maxID = -1;
		int id = 0;
		
		for (double d : vector){
			if (d > max){
				max = d;
				maxID = id;
			}
			id++;
		}
		
		//System.out.println("Chose model: " + maxID);
		
		//Choose model from som
		SimpleMatrix model = map.getNode(maxID).getVector();
		
		return model;
		
	}
	
	public int getMapSize(){
		return this.mapSize;
	}
	
	public SimpleMatrix getActivationMatrix(){
		return this.activationMatrix;
	}
	
	public abstract void setLearning(boolean learning);
	
	public abstract void sensitize(int iteration);
	
	public abstract void printModelWeigths();
	
	public abstract SimpleMatrix feedForward(SimpleMatrix feedForwardInputVector);
	
	public abstract SimpleMatrix feedBackward(SimpleMatrix feedBackwardInputMatrix);
	
	
}