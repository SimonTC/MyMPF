package stcl.test.predictors;

import java.util.LinkedList;

import stcl.algo.predictors.trie.Trie;

public class ActiveLeZi {
	
	private Trie<Character> trie;
	
	public static void main(String[] args) {
		String data = "aaaaaccbbabcbba";
		ActiveLeZi t = new ActiveLeZi();
		System.out.println();
		System.out.println("Active LeZi:");
		t.runActive(3, data);

	}
	
	public void runActive(int maxDepth, String data){
		trie = new Trie<Character>();
		LinkedList<Character> sequence = new LinkedList<Character>();
				
		for (int i = 0; i < data.length(); i++){
			Character v = data.charAt(i);
			sequence.add(v);			
			if (sequence.size() > maxDepth) sequence.removeFirst();
			
			LinkedList<Character> tmp = new LinkedList<Character>();
			for (int j = 1; j <= sequence.size(); j++){
				Character c = sequence.get(sequence.size() - j);
				tmp.addFirst(c);
				trie.add(tmp);
			}
			
		}
		trie.printTrie(3);
	}

}
