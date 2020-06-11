import numpy as np
import csv
import argparse
import pprint
import yaml
from gridworld import GridWorld
from RLearningAgent import RLearningAgent
from ast import literal_eval

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple Gridworld Experiment')
    parser.add_argument('-c', '--config', help='Config File', default=None)
    parser.add_argument('-f', '--csv', help='CSV File', default="results.csv")
    args = parser.parse_args()

    # 0: baseline
    # 1: hand-crafted shaping (Ng et al.)
    # 2: temporal-logic-based shaping
    # -1: shielding

    if not args.config:
        config = {
            'shaping': 2,
            'grid_size': 6,
            'goal': '[(5, 5)]',
            'goal_known': '[(5, 5)]'
        }
    else:
        with open(args.config, 'r') as f:
            config = yaml.load(f)

    pprint.pprint(config)
    size = config.get('grid_size', 6)
    grid = np.zeros([size, size])
    
    # add wall
    #grid[2, 2:] = 1
    print (grid)
    
    initial_pos = literal_eval(config.get('initial_pos', None)) if config.get('initial_pos', None) else None
    goal = literal_eval(config.get('goal', [(5, 5)]))
    goal_known = literal_eval(config.get('goal_known', [(5, 5)]))
    gw = GridWorld(grid, initial_pos, goal, goal_known)
    

    with open(args.csv, "w", newline='') as csvfile:
        fieldnames = ['Number of steps', 'Average reward']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        def eval_fn(num_steps):
            writer.writerow({'Number of steps': num_steps, 'Average reward': gw.get_avg_reward()})

        gw.reset()
        gw.reset_pos()
        agent = RLearningAgent(gw)
        agent.r_learning(eval_fn, config.get('shaping', 0))
