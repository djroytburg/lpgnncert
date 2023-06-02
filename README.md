# Simplifying  Transferable   Adversarial Attacks on  Graph Representation Learning (STAA)
 Official implementation of [ Simplifying  Transferable   Adversarial Attacks on  Graph Representation Learning]


Built based on [GCA](https://github.com/CRIPAC-DIG/GCA), [DeepRobust](https://deeprobust.readthedocs.io/en/latest/#), [Viking](https://github.com/virresh/viking), [CLGA](https://github.com/RinneSz/CLGA)

## Requirements
Tested on pytorch 1.7.1 and torch_geometric 1.6.3.

## Usage
1.To produce poisoned graphs with CLGA
```
python CLGA.py --dataset Cora --num_epochs 3000 --device cuda:0
```
It will automatically save three poisoned adjacency matrices in ./poisoned_adj which have 1%/5%/10% edges perturbed respectively. You may reduce the number of epochs for a faster training.

2.To produce poisoned graphs with baseline attack methods
```
python baseline_attacks.py --dataset Cora --method dice --rate 0.10 --device cuda:0
```
It will save one poisoned adjacency matrix in ./poisoned_adj.

3.To train the graph contrastive model for node classification with the poisoned graph
```
python train_GCA.py --dataset Cora --perturb --attack_method CLGA --attack_rate 0.10 --device cuda:0
```
It will load and train on the corresponding poisoned adjacency matrix specified by dataset, attack_method and attack_rate.

For link prediction, run train_LP.py.


python main.py --dataset Cora --lp_model metamodel --exp_type clean --device cuda:0 --seed 1

python main.py --dataset Cora --attack_method nodeembeddingattack --exp_type poisoning --device cuda:0 --attack_rate 0.05 --seed 2 --lp_model deepwalk  --attack_goal='availability'

python main.py --dataset CiteSeer --attack_method aalp --exp_type poisoning --device cuda:0 --attack_rate 0.1 --seed 2 --lp_model deepwalk  --attack_goal='availability'

python main.py --dataset Cora --attack_method aalp --exp_type evasion --device cuda:0 --attack_rate 0.1 --seed 1 --lp_model gcn --attack_goal='availability'
