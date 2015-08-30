# MyMPF

This is an implementation of the Memory Prediction Framework proposed by Jeff Hawkins in the book 'On Intelligence'. 
It has been created as part of my master thesis 'Combining HyperNEAT and MPF to Create Adaptable Networks' written in the Spring and Summer of 2015 at the IT University of Copenhagen.

It is inspired by the implementation by David Rawlinson and Gideon Kowadlo used in the paper 'Generating Adaptive Behaviour Within a Memory-Prediction Framework' (http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0029264).

It makes use of an action node to give the network the ability to react to inputs. Mapping states to actions happens in the individual neocortical units in the network.

It is used as part of the MPF-HyperNEAT program found in my repository

#Main elements
The neocortical units used in this implementation consists of a spatial pooler, a temporal pooler, a predictor and an action decider.

The poolers are implemented using Self-Organizing Maps.

The predictor is implemented as a Variable order Markov model using the Model Averaged PPM algorithm (K. Gopalratnam and D. J. Cook, “Active LeZi: An Incremental Parsing Algorithm for Sequential Prediction.”)

The action decider is implemented using Q-learning.

The network has shown some promises in playing Rock-Paper-Scissors but more wok needs to be done if it should be used for any advanced tasks.

# Dependencies
Efficient Java Matrix Library version 0.26 (http://ejml.org/wiki/index.php?title=Main_Page)

Q_Learning (https://github.com/SimonTC/Q_learning)

MySOM
