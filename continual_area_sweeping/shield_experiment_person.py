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

def generate_random_grid(base, num_event_cells, period_range, bound, mode='linear', stack=True, event_region=None, extra_event_region=[]):
    min_period, max_period = period_range
    free_spaces = np.argwhere(base == 0) if event_region is None else event_region
    np.random.shuffle(free_spaces)
    cells = []
    for n in range(num_event_cells):
        obj = gridworld.Object(x=free_spaces[n, 1], y=free_spaces[n, 0], period=random.randint(min_period, max_period),
                               bound=bound)
        cells.append(obj)

    pos = (free_spaces[num_event_cells, 1], free_spaces[num_event_cells, 0])
    person = None
    if mode == "person":
        person = gridworld.Person((free_spaces[num_event_cells, 1], free_spaces[num_event_cells, 0]))
        cells = [gridworld.Object(x=free_spaces[n, 1], y=free_spaces[n, 0], period=random.randint(min_period, max_period),
                                 bound=bound) for n in range(len(free_spaces))]

    gw = gridworld.GridWorld(base, cells, person=person, initialpos=pos, viewable_distance=0, mode=mode,
                             stack=stack, extra_event_region=extra_event_region)
    return gw


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Shaping Experiment')
    parser.add_argument('-c', '--config', help='Config File', default=None)
    parser.add_argument('-f', '--csv', help='CSV File', default="results.csv")
    args = parser.parse_args()
    if not args.config:
        config = {
            'mode': 'person',
            'bound': 1,
            'average_reward_learning_rate': 0.0001,
            'eval_period': 1000,
            'exploration_sched_timesteps': 10000,
            'strategy_file': 'Example1_Perm_readable.json',
            'replay_buffer_size': 100000,
            'perfect_knowledge': False
        }
    else:
        with open(args.config, 'r') as f:
            config = yaml.load(f)

    # Print config
    pprint.pprint(config)

    # Sheild
    strategy_file = config.get("strategy_file", None)
    w_dict = None
    if strategy_file:
        w_dict = {}
        following_region = [[] for x in range(225)]
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), strategy_file), 'r') as f:
            strategy = json.load(f)
            for num, state in strategy.items():
                successors = []
                for successor in state["Successors"]:
                    succ_state = strategy[str(successor)]["State"]
                    successors.append((succ_state["s"], succ_state["st"]))
                w_dict[(state["State"]["s"], state["State"]["st"])] = successors
                if state["State"]["st"] < 225:
                    following_region[state["State"]["st"]].append(np.unravel_index(state["State"]["s"], [15, 15]))


    # Visibility
    invisibility_file = 'iset.json'
    invisibility_dict = {}
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), invisibility_file), 'r') as f:
        invisibility = json.load(f)

        for s1, s2 in invisibility.items():
            invisibility_dict[int(s1)] = s2

    with open(args.csv, 'w', newline='') as csvfile:
        fieldnames = ['TYPE', 'ADT', 'DPS', 'TOTALDETECTIONS', 'TOTALSTEPS', 'TOTALEVENTS', 'NUMVISIBLE']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        img = np.zeros([15, 15])
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

        print (img)

        if config.get('perfect_knowledge', True):
            gw = generate_random_grid(img, 188, (10, 20), config.get('bound', 1), mode=config.get('mode', 'linear'),
                                   stack=True)
        else:
            gw = generate_random_grid(img, 188, (10, 20), config.get('bound', 1), mode=config.get('mode', 'linear'),
                                   stack=True, extra_event_region = [(r, c) for r in range(6, 9) for c in range(0, 15)])

        strategy = (w_dict, following_region)
        gw.invisibility = invisibility_dict

        # RL
        gw.reset()

        eval_period = config.get('eval_period', 20000)

        np.set_printoptions(precision=3, suppress=True, linewidth=150)

        def sliding_window_eval_fn(env, policy, q_func, vizgrid, num_iters):
            adt = env.gw.get_adt()
            #dps = env.gw.get_dps()
            dps = (env.gw.num_detections - env.gw.prev_num_detections) / eval_period
            print(num_iters, "ADT: ", adt, "\tDPS: ", dps, "\tDetections: ", env.gw.num_detections, \
                  "\tTotal Timesteps: ", env.gw.timestep, "\tTotal Events: ", env.gw.total_num_events,
                  "\tVisible: ", env.gw.person.viewable_counts)
            writer.writerow({'TYPE': str(num_iters),
                             'ADT': adt,
                             'DPS': dps,
                             'TOTALDETECTIONS': env.gw.num_detections,
                             'TOTALSTEPS': env.gw.timestep,
                             'TOTALEVENTS': env.gw.total_num_events,
                             'NUMVISIBLE': env.gw.person.viewable_counts})
            csvfile.flush()


        writer.writerow({'TYPE': 'Shaping'})
        csvfile.flush()
        print("Shaping")
        train.run(config, gw, strategy, "shaping", eval_period, sliding_window_eval_fn)
        gw.reset()

        writer.writerow({'TYPE': 'Baseline'})
        csvfile.flush()
        print("Baseline")
        train.run(config, gw, strategy, None, eval_period, sliding_window_eval_fn)
        gw.reset()

        writer.writerow({'TYPE': 'Shielding'})
        csvfile.flush()
        print("Shielding")
        train.run(config, gw, strategy, "shielding", eval_period, sliding_window_eval_fn)
        gw.reset()

