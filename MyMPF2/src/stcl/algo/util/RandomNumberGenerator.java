package stcl.algo.util;

import java.io.Serializable;
import java.util.Random;
/**
 * Class extending the Random number generator with extra abilities
 * @author Simon
 *
 */
public class RandomNumberGenerator implements Serializable {
	private static final long serialVersionUID = 1L;
	Random rand;
	long seed;
	
	/**
	 * Constructor to create new Random number generator. It is seeded with the current time in milliseconds
	 */
	public RandomNumberGenerator(){
		seed = System.currentTimeMillis();
		rand = new Random(seed);
	}
	
	/**
	 * Constructor to create new random number generator seeded with the given seed
	 * @param seed
	 */
	public RandomNumberGenerator(long seed){
		this.seed = seed;
		rand = new Random(seed);
	}
	
	/**
	 * Creates a new random number generator with the same seed as the given random number generator
	 * @param original
	 */
	public RandomNumberGenerator(RandomNumberGenerator original){
		this.seed = original.getSeed();
		rand = new Random(seed);
	}
	
	public long getSeed(){
		return this.seed;
	}
	
	/**
	 * Returns a double in the range [-range, range]
	 * @param range
	 * @return
	 */
	public double nextDoublePosNeg(double range){
		double result = 2 * rand.nextDouble() * range - range;
		return result;
	}
	
	/**
	 * Returns a double in the range [0.0d;1.0d[
	 * @return
	 */
	public double nextDouble(){
		return rand.nextDouble();
	}
	
	public int nextInt(){
		return rand.nextInt();
	}
	
	public int nextInt(int n){
		return rand.nextInt(n);
	}
	
	/**
	 * Returns an integer in the range floor(inclusive) to ceiling (exclusive)
	 * @param floor
	 * @param ceiling
	 * @return
	 */
	public int nextInt(int floor, int ceiling){
		int diff = ceiling - floor;
		int result = rand.nextInt(diff);
		result = floor + result;
		return result;
	}
	
	public boolean nextBoolean(){
		return rand.nextBoolean();
	}
	
	
}
