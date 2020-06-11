import pprint

import argparse

import numpy as np
import yaml
import random
import os
import csv
import json

import train
import gridworld


def generate_random_grid(base, num_event_cells, period_range, bound, mode='linear', stack=True, event_region=None):
    min_period, max_period = period_range
    free_spaces = np.argwhere(base == 0) if event_region is None else event_region
    cells = []
    for n in range(num_event_cells):
        obj = gridworld.Object(x=free_spaces[n, 1], y=free_spaces[n, 0], period=random.randint(min_period, max_period),
                               bound=bound)
        cells.append(obj)

    gw = gridworld.GridWorld(base, cells, person=None, viewable_distance=0, mode=mode, stack=stack)
    return gw


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Shaping Experiment')
    parser.add_argument('-c', '--config', help='Config File', default=None)
    parser.add_argument('-f', '--csv', help='CSV File', default="results.csv")
    args = parser.parse_args()
    if not args.config:
        config = {
            'mode': 'linear',
            'bound': 1,
            'average_reward_learning_rate': 0.0001,
            'eval_period': 500,
            'exploration_sched_timesteps': 10000,
            'replay_buffer_size': 300000,
            'perfect_knowledge': False,
        }
    else:
        with open(args.config, 'r') as f:
            config = yaml.load(f)

    # Print config
    pprint.pprint(config)

    # Visibility
    invisibility_file = 'iset.json'
    invisibility_dict = {}
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), invisibility_file), 'r') as f:
        invisibility = json.load(f)

        for s1, s2 in invisibility.items():
            invisibility_dict[int(s1)] = s2

    with open(args.csv, 'w', newline='') as csvfile:
        fieldnames = ['TYPE', 'ADT', 'DPS']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        img = np.zeros([15, 15])
        ## four big rooms
        img[0:6, 7] = 1
        img[9:15, 7] = 1
        img[5][0:5] = 1
        img[5][6] = 1
        img[5][10:15] = 1
        img[5][8] = 1
        img[9][0:5] = 1
        img[9][6] = 1
        img[9][10:15] = 1
        img[9][8] = 1

        print(img)

        if config.get('perfect_knowledge', True):
            event_region = np.array([(r, c) for r in range(10, 15) for c in range(8, 15)])
            gw = generate_random_grid(img, 35, (10, 20), config.get('bound', 1), mode=config.get('mode', 'linear'),
                stack=True, event_region=event_region)
        else:
            event_region = np.array([(r, c) for r in range(6, 15) for c in range(8, 15)])
            gw = generate_random_grid(img, 40, (10, 20), config.get('bound', 1), mode=config.get('mode', 'linear'),
                stack=True, event_region=event_region)

        gw.invisibility = invisibility_dict

        spec = np.array([(r, c) for r in range(10, 15) for c in range(8, 15)])

        csvfile.flush()

        # RL
        gw.reset()

        eval_period = config.get('eval_period', 20000)

        np.set_printoptions(precision=3, suppress=True, linewidth=150)

        def sliding_window_eval_fn(env, policy, q_func, vizgrid, num_iters):
            adt = env.gw.get_adt()
            #dps = env.gw.get_dps()
            dps = (env.gw.num_detections - env.gw.prev_num_detections) / eval_period
            print("ADT: ", adt, "\tDPS: ", dps)
            writer.writerow({'TYPE': str(num_iters),
                             'ADT': adt,
                             'DPS': dps})
            csvfile.flush()


        writer.writerow({'TYPE': 'Shaping'})
        csvfile.flush()
        print("Shaping")
        train.run(config, gw, spec, "shaping", eval_period, sliding_window_eval_fn)
        gw.reset()
        
        writer.writerow({'TYPE': 'Baseline'})
        csvfile.flush()
        print("Baseline")
        train.run(config, gw, None, None, eval_period, sliding_window_eval_fn)
        gw.reset()

        writer.writerow({'TYPE': 'Shielding'})
        csvfile.flush()
        print("Shielding")
        train.run(config, gw, spec, "shielding", eval_period, sliding_window_eval_fn)
        gw.reset()
