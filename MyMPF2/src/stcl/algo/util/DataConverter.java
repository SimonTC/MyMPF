package stcl.algo.util;

public class DataConverter {
	
	/**
	 * 
	 * @param i integer to convert to bitstring
	 * @param length length of bit string. 0's will be added in front of the bit string if it is not long enough
	 * @return
	 */
	public static String intToBitString(int i, int length){
		String s = Integer.toBinaryString(i);
		while (s.length() < length){
			s = "0" + s;
		}
		
		return s;
	}
	
	/**
	 * 
	 * @param s bistring to convert to int
	 * @return
	 */
	public static int bitstringToInt(String s){
		return Integer.parseInt(s, 2);
	}

}
