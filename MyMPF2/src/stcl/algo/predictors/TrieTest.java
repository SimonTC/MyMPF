package stcl.algo.predictors;

import java.util.LinkedList;
import java.util.Queue;

public class TrieTest {
	
	private String data = "aaababbbbbaabccddcbaaaa";//"aaaaaccbbabcbba";
	private Trie<Character> trie;
	
	public static void main(String[] args) {
		TrieTest t = new TrieTest();
		t.run(3);

	}
	
	public void run(int maxDepth){
		trie = new Trie<Character>();
		LinkedList<Character> sequence = new LinkedList<Character>();
		String seq = "";
		//Present data
		for (int i = 0; i < data.length(); i++){
			Character c = data.charAt(i);
			seq = seq + c;
			sequence.add(c);
			if (sequence.size() > maxDepth){
				sequence.poll();
				seq = seq.substring(1);
			}
			
			
			
			trie.add(sequence);
			System.out.println("New symbol: " + c);
			System.out.println("Input: " + seq);
			System.out.println("-----------------------");
			trie.printTrie(3);
			System.out.println("-----------------------");
		}
		
		//Print data
		trie.printTrie(maxDepth);
	}

}
