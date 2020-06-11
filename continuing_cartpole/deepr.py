import os
import tempfile
import csv

import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np
import yaml

import baselines.common.tf_util as U
from baselines.common.tf_util import load_variables, save_variables
from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines.common import set_global_seeds
from baselines.common.models import mlp

from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from utils import ObservationInput
from build_graph import build_train

from baselines.common.tf_util import get_session
from models import build_q_func

class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params
        self.initial_state = None

    @staticmethod
    def load_act(path):
        with open(path, "rb") as f:
            model_data, act_params = cloudpickle.load(f)
        act = deepr.build_act(**act_params)
        sess = tf.Session()
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            load_variables(os.path.join(td, "model"))

        return ActWrapper(act, act_params)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def step(self, observation, **kwargs):
        # DQN doesn't use RNNs so we ignore states and masks
        kwargs.pop('S', None)
        kwargs.pop('M', None)
        return self._act([observation], np.zeros(2), **kwargs), None, None, None

    def save_act(self, path=None):
        """Save model to a pickle located at `path`"""
        if path is None:
            path = os.path.join(logger.get_dir(), "model.pkl")

        with tempfile.TemporaryDirectory() as td:
            save_variables(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            cloudpickle.dump((model_data, self._act_params), f)

    def save(self, path):
        save_variables(path)


def load_act(path):
    """Load act function that was returned by learn function.

    Parameters
    ----------
    path: str
        path to the act function pickle

    Returns
    -------
    act: ActWrapper
        function that takes a batch of observations
        and returns actions.
    """
    return ActWrapper.load_act(path)

def get_mask(env, method_type):
    
    if method_type == "baseline":
        action_mask = np.zeros(env.action_space.n)
    elif method_type == "shielding":
        action_mask = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            if not check_allowed_action(env, action):
                action_mask[action] = -np.inf
    elif method_type == "shaping":
        action_mask = np.full(env.action_space.n, -get_state_distance(env))
        for action in range(env.action_space.n):
            if check_allowed_action(env, action):
                action_mask[action] += 1

    return action_mask

def get_state_distance(env):
    x, xdot, theta, thetadot = env.state
    distance = 0
    x_p = x + env.tau * xdot
    if x_p < -env.known_x_threshold:
        distance = (-env.known_x_threshold - x_p) * 100
    elif x_p > env.known_x_threshold:
        distance = (x_p - env.known_x_threshold) * 100

    return distance

def check_allowed_action(env, action):
    
    x, xdot, theta, thetadot = env.state
    allowed = True
    if (x + env.tau * xdot < -env.known_x_threshold) and xdot < 0 and env.actions[action] < 0:
        allowed = False
    elif (x + env.tau * xdot > env.known_x_threshold) and xdot > 0 and env.actions[action] > 0:
        allowed = False

    return allowed


def learn(env,
          network,
          seed=None,
          lr=5e-4,
          total_timesteps=100000,
          buffer_size=100000,
          exploration_fraction=0.1,
          exploration_final_eps=0.1,
          train_freq=1,
          batch_size=64,
          print_freq=1,
          eval_freq=2500,
          checkpoint_freq=10000,
          checkpoint_path=None,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          param_noise=False,
          callback=None,
          load_path=None,
          csv_path="results.csv",
          method_type="baseline",
          **network_kwargs
            ):
    """Train a deepr model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    network: string or a function
        neural network to use as a q function approximator. If string, has to be one of the names of registered models in baselines.common.models
        (mlp, cnn, conv_only). If a function, should take an observation tensor and return a latent variable tensor, which
        will be mapped to the Q function heads (see build_q_func in baselines.deepr.models for details on that)
    seed: int or None
        prng seed. The runs with the same seed "should" give the same results. If None, no seeding is used.
    lr: float
        learning rate for adam optimizer
    total_timesteps: int
        number of env steps to optimizer for
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
    batch_size: int
        size of a batch sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to total_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    param_noise: bool
        whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.
    load_path: str
        path to load the model from. (default: None)
    **network_kwargs
        additional keyword arguments to pass to the network builder.

    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepr/categorical.py for details on the act function.
    """
    # Create all the functions necessary to train the model

    sess = get_session()
    set_global_seeds(seed)

    #q_func = build_q_func(network, **network_kwargs)
    q_func = build_q_func(mlp(num_layers=4, num_hidden=64), **network_kwargs)
    #q_func = build_q_func(mlp(num_layers=2, num_hidden=64, activation=tf.nn.relu), **network_kwargs)

    # capture the shape outside the closure so that the env object is not serialized
    # by cloudpickle when serializing make_obs_ph

    observation_space = env.observation_space
    def make_obs_ph(name):
        return ObservationInput(observation_space, name=name)

    act, train, update_target, debug = build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,
        param_noise=param_noise
    )

    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': env.action_space.n,
    }

    act = ActWrapper(act, act_params)

    # Create the replay buffer
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = total_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                 #initial_p=1.0,
                                 initial_p=exploration_final_eps,
                                 final_p=exploration_final_eps)

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()

    eval_rewards = [0.0]
    saved_mean_reward = None
    obs = env.reset()
    reset = True

    with tempfile.TemporaryDirectory() as td:
        td = checkpoint_path or td

        model_file = os.path.join(td, "model")
        model_saved = False

        if tf.train.latest_checkpoint(td) is not None:
            load_variables(model_file)
            logger.log('Loaded model from {}'.format(model_file))
            model_saved = True
        elif load_path is not None:
            load_variables(load_path)
            logger.log('Loaded model from {}'.format(load_path))

        csvfile = open(csv_path, 'w', newline='')
        fieldnames = ['STEPS', 'REWARD']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for t in range(total_timesteps+1):
            if callback is not None:
                if callback(locals(), globals()):
                    break
            # Take action and update exploration to the newest value
            kwargs = {}
            if not param_noise:
                #update_eps = exploration.value(t)
                update_eps = exploration_final_eps
                update_param_noise_threshold = 0.
            else:
                update_eps = 0.
                # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                # for detailed explanation.
                update_param_noise_threshold = -np.log(1. - exploration.value(t) + exploration.value(t) / float(env.action_space.n))
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                kwargs['update_param_noise_scale'] = True

            
            action_mask = get_mask(env, method_type)
            a = act(np.array(obs)[None], unused_actions_neginf_mask=action_mask, update_eps=update_eps, **kwargs)[0]
            
            env_action = a
            reset = False
            new_obs, rew, done, _ = env.step(env_action)

            eval_rewards[-1] += rew

            action_mask_p = get_mask(env, method_type)
            # Shaping
            if method_type == 'shaping':
                
                ## look-ahead shaping
                ap = act(np.array(new_obs)[None], unused_actions_neginf_mask=action_mask_p, stochastic=False)[0]
                f = action_mask_p[ap] - action_mask[a]
                rew = rew + f

            # Store transition in the replay buffer.
            #replay_buffer.add(obs, a, rew, new_obs, float(done), action_mask_p)
            if method_type != 'shaping':
                replay_buffer.add(obs, a, rew, new_obs, float(done), np.zeros(env.action_space.n))
            else:
                replay_buffer.add(obs, a, rew, new_obs, float(done), action_mask_p)
            obs = new_obs

            if t % eval_freq == 0:
                eval_rewards.append(0.0)

            if t > learning_starts and t % train_freq == 0:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if prioritized_replay:
                    experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones, masks_tp1 = replay_buffer.sample(batch_size)
                    weights, batch_idxes = np.ones_like(rewards), None
                td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights, masks_tp1)
                if prioritized_replay:
                    new_priorities = np.abs(td_errors) + prioritized_replay_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)

            if t > learning_starts and t % target_network_update_freq == 0:
                # Update target network periodically.
                update_target()

            mean_eval_reward = round(np.mean(eval_rewards[-1-print_freq:-1]), 1) 
            num_evals = len(eval_rewards)
            if t > 0 and t % eval_freq == 0 and print_freq is not None and t % (print_freq*eval_freq) == 0:
            #if done and print_freq is not None and len(eval_rewards) % print_freq == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("evals", num_evals)
                logger.record_tabular("average reward in this eval", mean_eval_reward / (eval_freq))
                logger.record_tabular("total reward in this eval", mean_eval_reward)
                logger.dump_tabular()

                writer.writerow({"STEPS": t, "REWARD": mean_eval_reward / (eval_freq)})
                csvfile.flush()

            if (checkpoint_freq is not None and t > learning_starts and
                    num_evals > 100 and t % checkpoint_freq == 0):
                if saved_mean_reward is None or mean_eval_reward > saved_mean_reward:
                    if print_freq is not None:
                        logger.log("Saving model due to mean reward increase: {} -> {}".format(
                                   saved_mean_reward, mean_eval_reward))
                    save_variables(model_file)
                    model_saved = True
                    saved_mean_reward = mean_eval_reward
        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            load_variables(model_file)

    return act
