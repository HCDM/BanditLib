# BanditLib
This repo contains the implementation of serveral contextual bandits algorithm, including CoLin, hLinUCB, factorUCB, GOB.Lin, LinUCB, HybridLinUCB, PTS, and UCBPMF. CoLin, hLinUCB and factorUCB are our proposed algorithms published in [1], [2], [3]. We are updating the library and current version may have bugs. You can also check a stable release here: https://github.com/huazhengwang/BanditLib/releases/tag/1.0

## Usage
Run the simulator: `python Simulation.py --alg XXX` where parameter alg represents the name of algorithm. 

Run with different parameters: `python Simulation.py --alg XXX --contextdim XX --userNum XX --Sparsity XX --NoiseScale XX --matrixNoise  XX --hiddendim XX` 
where 
`--contextdim` stands for dimension of contextual features;
`--userNum` stands for the number of users simulated in the simulator. Each user is associated with a preference parameter, which is known to the environment but not to the algorithms. In simulator, we can choose to simulate users every time we run the program or simulate users once, save it to 'Simulation_MAB_files', and keep using it.
`--Sparsity` stands for the sparsity level of the user graph. Sparsity XX means to only maintain the top XX most connected users. Sparsity should be smaller than userNum, when equal to userNum, the graph is a full connected graph'
`--hiddendim`stands for dimension of hidden features. This parameter is specifically for the algorithms that can estimate hidden features, such as hLinUCB, PTS. For all the other contextual bandit algorithms, the default setting for this parameter should be 0.

## Algorithms' details
**LinUCB**: A state-of-art contextual bandit algorithm. It select arms based on an upper confidence bound of the estimated reward with given context vectors. LinUCB assume that users/bandits' parameters are independent with each other. And LinUCB only works with the observed features and does not consider hidden features.

**CoLin**: A collaborative contextual bandit algorithm which explicitly models the underlying dependency among users/bandits. In CoLin, a weighted adjacency graph is constructed, where each node represents a contextual bandit deployed for a single user and the weight on each edge indicates the influence between a pair of users. Based on this dependency structure, the observed payoffs on each user are assumed to be determined by a mixture of neighboring users in the graph. Bandit parameters over all the users are estimated in a collaborative manner: both context and received payoffs from one user are prorogated across the whole graph in the process of online updating. CoLin establishes a bridge to share information among heterogenous users and thus reduce the sample com- plexity of preference learning. We rigorously prove that our CoLin achieves a remarkable reduction of upper regret bound with high probability, comparing to the linear regret with respect to the number of users if one simply runs independent bandits on them (LinUCB). 

**hLinUCB**: A contextual bandit algorithm with hidden feature learning, in which hidden features are explicitly introduced in our reward generation assumption, in addition to the observable contextual features. Coordinate descent with provable exploration bound is used to iteratively estimate the hidden features and unknown model parameters on the fly. At each iteration, closed form solutions exist and can be efficiently computed. Most importantly, we rigorously prove that with proper initialization the developed hLinUCB algorithm with hidden features learning can obtain a sublinear upper regret bound with high probability, and a linear regret is inevitable at the worst case if one fails to model such hidden features.

**FactorUCB**: A factorization-based bandit algorithm, in which low-rank matrix completion is performed over an incrementally constructed user-item preference matrix and where an upper confidence bound based item selection strategy is developed to balance the exploit/explore trade-off in online learning. Observable conextual features and dependency among users (e.g., social influence) are leveraged to improve the algorithm’s convergence rate and help conquer cold-start in recommendation. A high probability sublinear upper regret bound is proved in the developed algorithm, where considerable regret reduction is achieved on both user and item sides.

## Result
The results will be written under folder `./SimulationResults`, including accumulated regret and parameter estimation error. You can then plot the result, and here are one example we used in [1], where we run the simulator to compare CoLin, GOB.Lin, LinUCB, and HybridLinUCB:

Regret                                          | Parameter estimation error
------------------------------------------------| -------------
![image](SimulationResults/regret.png "regret") | ![image](SimulationResults/ParameterEstimation.png "ParameterEstimation")

## Redesign

### Configuration Files
A configuration yaml file can now be used to specify system level parameters for the simulation.
The simulator will use a config file using the following command:
```
python Simulation.py --config <config file name>.yaml
```
An example config file can be found below:
```yaml
general:
  testing_iterations: 1000
  context_dimension: 16
  pool_article_size: 10
  plot: True

user:
  number: 10
  collaborative: yes

article:
  number: 1000

reward:
  type: SocialLinear

alg:
  general:
    alpha: 0.3
    lambda_: 0.1
    parameters:
      Theta: True
      CoTheta: False
      W: False
      V: False
  specific:
     CoLinUCB:
       parameters:
         Theta: False
         CoTheta: True
```
Each section defines parameters for different modules in the system. In the alg section, two sub-headers are present: `specific` and `general`. `general` defines parameters which will be used in every algorithm. `specific` defines the algorithms which should be run and any parameters specific to that algorithm. The `config.yaml` files contains all possible options and explanations for each of the fields

### Adding New Algorithms
A new algorithm can be defined by extending the BaseAlg class and implementing the following methods:
```python
class ExampleAlgorithm(BaseAlg):

	def decide(self, pool_articles, userID, k = 1):
		articles = []
		return articles


	def updateParameters(self, articlePicked, click, userID):
```
`def decide(self, pool\_articles, userID, k = 1)}` : returns a list of `k` article items found in `pool_articles` which the algorithm predicts will give the highest reward to the user with id of `userID`. This entails making some prediction of the users valuation of an article, but that is not required initially.

`def updateParameters(self, articlePicked, click, userID):` updates the algorithm’s understanding of the preference of `userID` user given the reward, `click`, provided by the system for the article picked. The system notifies the algorithm of the true reward with this method. In this way, the algorithm can learn more about the user through online learning.

To define the default parameters for a new algorithm, a dictionary function can be defined in `util_functions.py`. It will take the form of:
```python
def create<Example>Dict(specific, general, W, system_params):
	base_dict = {
		'alpha': 0.3,
		'lambda_': 0.1,
		'parameters': {
			'Theta': True,
		}
	}
	return createSpecificAlgDict(specific, general, W, system_params, base_dict)
```
Where variables like `alpha` and `lambda_` are variables needed in the Example Algorithm. The parameters sections defines which variables should be tracked, such as `Theta`, `CoTheta`, `W`, and `V`.

## References
[1]: Qingyun Wu, Huazheng Wang, Quanquan Gu and Hongning Wang. Contextual Bandits in A Collaborative Environment. The 39th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR'2016), p529-538, 2016.

[2]: Huazheng Wang, Qingyun Wu and Hongning Wang. Learning Hidden Features for Contextual Bandits. The 25th ACM International Conference on Information and Knowledge Management (CIKM 2016), p1633-1642, 2016.

[3]: Huazheng Wang, Qingyun Wu and Hongning Wang. Factorization Bandits for Interactive Recommendation. The Thirty-First AAAI Conference on Artificial Intelligence (AAAI 2017). (to appear)

