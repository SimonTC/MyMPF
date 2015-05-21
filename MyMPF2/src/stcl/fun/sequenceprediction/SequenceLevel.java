package stcl.fun.sequenceprediction;

import java.util.ArrayList;
import java.util.Random;


public class SequenceLevel{
	private int[][] blocks;
	private SequenceLevel child;
	private Random rand;
	
	public SequenceLevel(int alphabetSize, int minBlockLength, int maxBlockLength, SequenceLevel child, Random rand) {
		this.rand = rand;
		this.child = child;
		blocks = createLevelBlocks(alphabetSize, minBlockLength, maxBlockLength);
		
	}
	
	public int[] unpackBlock(int blockID){
		int[] block = blocks[blockID];
		
		if (child == null) return block;
		ArrayList<int[]> blockList = new ArrayList<int[]>();
		int totalLength = 0;
		for (int i : block){
			int[] childBlock = child.unpackBlock(i);
			blockList.add(childBlock);
			totalLength += childBlock.length;
		}
		
		int[] unpackedBlock = new int[totalLength];
		int counter = 0;
		for (int[] childBlock : blockList){
			for (int i : childBlock){
				unpackedBlock[counter] = i;
				counter++;
			}
		}
		
		return unpackedBlock;
	}
	
	
	public int[][] createLevelBlocks(int alphabetSize, int minBlockLength, int maxBlockLength){
		int numBlocks = alphabetSize;
		int[][] blocks = new int[numBlocks][];
		
		for (int blockID = 0; blockID < numBlocks; blockID++){
			int blockLength = minBlockLength + rand.nextInt(maxBlockLength - minBlockLength + 1);
			int[] block = new int[blockLength];
			for (int i = 0; i < blockLength; i++){
				block[i] = rand.nextInt(alphabetSize);
			}
			blocks[blockID] = block;
		}
		
		return blocks;		
	}
	
	public SequenceLevel getChild(){
		return child;
	}
	
	public int[][] getBlocks(){
		return blocks;
	}
}
