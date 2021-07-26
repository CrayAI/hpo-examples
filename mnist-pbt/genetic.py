"""
Population-based training (PBT) example with genetic search
"""
import argparse

from crayai import hpo

argparser = argparse.ArgumentParser()
argparser.add_argument('--generations', type=int, default=20)
argparser.add_argument('--num_demes', type=int, default=2)
argparser.add_argument('--pop_size', type=int, default=32)
argparser.add_argument('--mutation_rate', type=float, default=0.40)
argparser.add_argument('--crossover_rate', type=float, default=0.33)
argparser.add_argument('--checkpoint', type=str, default='checkpoints')
args = argparser.parse_args()

params = hpo.Params([['--lr', 0.01, (0.0001, 1.0)]])

evaluator = hpo.Evaluator('python3 source/mnist.py --FoM --epochs=1 ' +
                          '--load_checkpoint=@checkpoint/model.h5 ' +
                          '--save_checkpoint=@checkpoint/model.h5',
                          checkpoint  = args.checkpoint,
                          workload_manager='slurm',
                          verbose=False)

optimizer = hpo.GeneticOptimizer(evaluator,
                                  generations=args.generations,
                                  num_demes=args.num_demes,
                                  pop_size=args.pop_size,
                                  mutation_rate=args.mutation_rate,
                                  crossover_rate=args.crossover_rate,
                                  verbose=True)

optimizer.optimize(params)
