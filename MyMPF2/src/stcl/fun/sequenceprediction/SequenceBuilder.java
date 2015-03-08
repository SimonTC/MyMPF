package stcl.fun.sequenceprediction;

import java.util.ArrayList;
import java.util.Random;
import java.util.Stack;

public class SequenceBuilder {
	private Stack<Level> levels;
	private Random rand;
	
	/**
	 * Creates blocks of length minBlockLength >= length <= maxBlockLength
	 * @param rand
	 * @param numLevels
	 * @param alphabetSize
	 * @param minBlockLength
	 * @param maxBlockLength
	 * @return
	 */
	public int[] buildSequence(Random rand, int numLevels, int alphabetSize, int minBlockLength, int maxBlockLength ){
				
		this.rand = rand;
		levels = createLevels(numLevels, alphabetSize, minBlockLength, maxBlockLength);
		Level topLevel = levels.peek();
		int[] finalSequence = topLevel.unpackBlock(0);
		
		return finalSequence;
	}
	
	/**
	 * Creates numLevels + 1 levels. The top level is only used to call when writing the sequence 
	 * @param numLevels
	 * @param alphabetSize
	 * @param minBlockLength
	 * @param maxBlockLength
	 * @return The top level
	 */
	private Stack<Level>  createLevels(int numLevels, int alphabetSize, int minBlockLength, int maxBlockLength){
		Stack<Level> levels = new Stack<Level>();
		
		Level firstLevel = new Level(alphabetSize, minBlockLength, maxBlockLength, null);
		levels.push(firstLevel);		
		
		Level lastLevel = null;
		for (int i = 0; i < numLevels - 1; i++){
			lastLevel = new Level(alphabetSize, minBlockLength, maxBlockLength, levels.peek());
			levels.push(lastLevel);
		}
		
		Level topLevel = new Level(1, 1, 1, lastLevel);
		levels.push(topLevel);
		return levels;
	}
	
	
	private class Level{
		private int[][] blocks;
		private Level child;
		
		public Level(int alphabetSize, int minBlockLength, int maxBlockLength, Level child) {
			blocks = createLevelBlocks(alphabetSize, minBlockLength, maxBlockLength);
			this.child = child;
		}
		
		private int[] unpackBlock(int blockID){
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
		
		
		private int[][] createLevelBlocks(int alphabetSize, int minBlockLength, int maxBlockLength){
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
	}
}
