# MNIST Population Based Training Optimization

## Dataset: MNIST
Handwritten digits from 0 to 9

## Model: LeNet
Implemented in Tensorflow, tested with TF 1.11 

## Hyperparameters:
lr - Learning Rate

## Use
### Requirements:
- Python 3 (module load cray-python)
- Set PYTHONPATH to the parent directory of crayai 
- Active exlcusive allocation (salloc -N n --exclusive)

### Example:
How to run on a CS system with the Urika launcher

```
$ export PYTHONPATH=/path/to/parent/dir/of/crayai:$PYTHONPATH
$ salloc -N 8 --exclusive
$ module load openmpi/3.0.0
$ module load analytics
$ python genetic.py --nodes=8 --urika --generations 10 --gens_per_epoch 2 --pop_size 8
``` 

### Debugging:
Set the verbose flag on evaluator to True. This will dump all output from run_training calls so you can detect whats going wrong. 
