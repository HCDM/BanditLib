Influence maximization in Social Graphs
=======================================

Repository for research project that studies [influence maximization problem.](http://www-bcf.usc.edu/~dkempe/publications/spread.pdf)

Degree Discount Algorithm
-------------------------

Degree discount heuristic was proposed first by [Chen et al.](http://snap.stanford.edu/class/cs224w-readings/chen09influence.pdf) (Algorithm 4) Its results are slightly inferior than greedy hill-climbing algorithms, however it runs several orders of magnitudes faster. It's fine-tuned for Independent Cascade model. Running time is O(k*log(n)+m), where k - number of initial targets, n - number of vertices, m - number of edges. 

Representative Nodes Algorithm
------------------------------
Representative nodes heuristic is searching for the set of k nodes such that its size is maximum. Two different definitions of set size are used. First is cumulative sum of all distances between each pair of vertices inside the set. Second is minimum distance between a pair of vertices in the set. Algorithm didn't show significant results, probably because small differences in edge weights don't make some vertices be "more influential" than others. 

General Greedy Algorithm
------------------------
General greedy heuristic is iterative method, each step it chooses node that along with already chosen nodes will bring the most propagation. Since propagation is random process for each node it runs R iteration of RanCas to calculate the average number of nodes it reaches. Running time is O(knRm). Source: [Chen et al.](http://snap.stanford.edu/class/cs224w-readings/chen09influence.pdf) (Algorithm 1)

New Greedy IC Algorithm
-----------------------
newGreedyIC heuristic reduces the amount of iterations in general greedy algorithm. Running time is O(kRm). Source: [Chen et al.](http://snap.stanford.edu/class/cs224w-readings/chen09influence.pdf) (Algorithm 2)

Single Discount
---------------
A simple degree discount heuristic where each neighbor of a newly selected seed discounts its degree by one. This can be applied to all influence cascade models. The heuristic has the same running time as for degree discount heuristic. Additional information can be found in [Chen et al.](http://snap.stanford.edu/class/cs224w-readings/chen09influence.pdf)

Degree Heuristic
----------------
As a comparison, a simple heuristic that selects the k vertices with the largest degrees. Running time is O(m). Additional information can be found in [Kempe et al.](http://www-bcf.usc.edu/~dkempe/publications/spread.pdf) and [Chen et al.](http://snap.stanford.edu/class/cs224w-readings/chen09influence.pdf)

Random Heuristic
----------------
As a baseline, heuristic takes k nodes uniformly at random. Running time is O(k). 
