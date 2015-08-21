package stcl.tests.fun.rps.sequenceprediction;

import static org.junit.Assert.*;

import java.util.Random;
import java.util.Stack;

import org.junit.Before;
import org.junit.Test;

import stcl.fun.sequenceprediction.SequenceBuilder;

public class SequenceBuilderTest {

	@Before
	public void setUp() throws Exception {
	}

	@Test
	public void test() {
		SequenceBuilder sb = new SequenceBuilder();
		Random rand = new Random(1234);
		int numLevels = 3;
		int alphabetSize = 3;
		int minBlockLength = 2;
		int maxBlockLength = 5;
		
		int[] originalSequence = sb.buildSequence(rand, numLevels, alphabetSize, minBlockLength, maxBlockLength);
		originalSequence = copySequence(originalSequence);
		
		//Test that same sequence is returned when no changes have been made to the builder
		int[] newSequence = sb.getTopLevel().unpackBlock(0);
		
		for (int i = 0; i < newSequence.length; i++){
			assertTrue(newSequence[i] == originalSequence[i]);
		}
		
		//Test that architecture doesn't change when randomizing
		newSequence = sb.randomizeValues();
		assertTrue("Length is different", newSequence.length == originalSequence.length);
		
		//Test that sequence is different
		boolean status = false;
		for (int i = 0; i < newSequence.length; i++){
			if (newSequence[i] != originalSequence[i]){
				status = true;
				break;
			}
		}
		assertTrue("All elements have the same value", status);
		
		
	}
	
	private int[] copySequence(int[] sequence){
		int[] copy = new int[sequence.length];
		
		for (int i = 0; i < copy.length; i++){
			copy[i] = sequence[i];
		}
		
		return copy;	
	}
}
