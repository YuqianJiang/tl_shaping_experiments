import numpy as np
import tensorflow as tf

import baselines.common.tf_util as U
from baselines.common.schedules import LinearSchedule
from build_graph import build_train
from replay_buffer import ReplayBuffer
from utils import PlaceholderTfInput
from models import build_q_func
import models

import gridworld
from env import GridWorldEnv


def region_distance(pos, region, gw):
    def manhattan_dist(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    min_dist = gw.fw.distance(pos, (region[0][1], region[0][0])) #region is row/column
    min_cell = region[0]
    for cell in region[1:]:
        dist = gw.fw.distance(pos, (cell[1], cell[0]))
        if dist < min_dist:
            min_dist = dist
            min_cell = cell

    return min_dist

def get_mask_region_shaping(spec, gw, pos):
    dist_curr = region_distance(pos, spec, gw)
    phi_mask = np.full(len(gw.actions), -dist_curr)
    for action in range(len(gw.actions)):
        target = gw.get_target(action, pos)
        if not gw.check_target(target, pos):
            continue
        dist_next = region_distance(target, spec, gw)
        if dist_next < dist_curr:
            phi_mask[action] += 1
        elif dist_curr == 0 and dist_next == 0:
            phi_mask[action] += 1

    return phi_mask

def get_mask_region_shielding(spec, gw, pos):
    shield_neginf_mask = np.full(len(gw.actions), -np.inf)
    dist_curr = region_distance(pos, spec, gw)
    for action in range(len(shield_neginf_mask)):
        target = gw.get_target(action, pos)
        if not gw.check_target(target, pos):
            continue
        dist_next = region_distance(target, spec, gw)
        if dist_next < dist_curr:
            shield_neginf_mask[action] = 0
        elif dist_curr == 0 and dist_next == 0:
            shield_neginf_mask[action] = 0

    return shield_neginf_mask

def get_mask_region_all(gw, method_type, spec):
    mask_list = []
    for y in range(gw.grid.shape[0]):
        for x in range(gw.grid.shape[1]):
            mask_list.append(get_mask_region_pos(gw, (x, y), method_type, spec))

    mask = np.asarray(mask_list)
    return mask

def get_mask_region_pos(gw, method_type, spec, pos):
    if not method_type:
        mask = np.zeros(len(gw.actions))
    elif method_type == "shielding":
        mask = get_mask_region_shielding(spec, gw, pos)
    elif method_type == "shaping":
        mask = get_mask_region_shaping(spec, gw, pos)
    return mask

def get_mask_person_shaping(spec, gw, pos, person_pos):

    if gw.person_viewable(pos, person_pos):  # the person is visible now
        spec_dict, following_region = spec
        ind_person = grid_to_array(person_pos, gw.grid.shape)
        dist_curr = region_distance(pos, following_region[ind_person], gw)
        phi_mask = np.full(len(gw.actions), -dist_curr)
        for action in range(len(gw.actions)):
            target = gw.get_target(action, pos)
            if not gw.check_target(target, pos):
                continue
            dist_next = region_distance(target, following_region[ind_person], gw)
            if dist_next < dist_curr:
                phi_mask[action] += 1
            if dist_next == 0:
                phi_mask[action] = 0
    else:
        phi_mask = np.full(len(gw.actions), -6)
    return phi_mask

def get_mask_person_shielding(spec, gw, pos, person_pos):
    spec_dict, following_region = spec
    shield_neginf_mask = np.full(len(gw.actions), -np.inf)
    ind_person = grid_to_array(person_pos, gw.grid.shape)

    if gw.person_viewable(pos, person_pos):  # but the person is visible now
        for action in range(len(gw.actions)):
            target = gw.get_target(action, pos)
            ind_robot_next = target[1] * gw.grid.shape[1] + target[0]
            if (ind_robot_next, ind_person) in spec_dict:
                shield_neginf_mask[action] = 0
    else:
        print("Lost the person while shielding!")
    return shield_neginf_mask

def get_mask_person_pos(gw, method_type, spec, pos, person_pos):
    if not method_type:
        mask = np.zeros(len(gw.actions))
    elif method_type == "shielding":
        mask = get_mask_person_shielding(spec, gw, pos, person_pos)
    elif method_type == "shaping":
        mask = get_mask_person_shaping(spec, gw, pos, person_pos)
    return mask


def run(config, gw, spec=None, method_type=None, eval_period=1, eval_fn=lambda: None):
    tf.reset_default_graph()
    with U.make_session() as sess:

        tensorboard = config.get('tensorboard', '/tmp/tensorboard/run')
        output = config.get('weightsoutput', '/tmp/out.weights')

        # Create the environment
        env = GridWorldEnv(gw)
        print("Set up gridworld!")

        s_shape = env.state.shape

        q_func = build_q_func(models.original_dqn_conv, hiddens=[64])

        # Create all the functions necessary to train the model
        act, train, update_target, debug = build_train(
            scope="explorerl",
            make_obs_ph=lambda name: PlaceholderTfInput(tf.placeholder(tf.float32, (None,) + s_shape, name=name)),
            q_func=q_func,
            num_actions=env.action_space.n,
            avg_reward_learning_rate=config.get('average_reward_learning_rate', 0.0001),
            optimizer=tf.train.AdamOptimizer(learning_rate=2.5e-4),
            gamma=0.99
        )
        # Create the replay buffer
        replay_buffer = ReplayBuffer(config.get('replay_buffer_size', 100000))
        exploration = LinearSchedule(schedule_timesteps=config.get('exploration_sched_timesteps', 100000),
                                     # const epsilon
                                     initial_p=config.get('exploration_sched_finalp', 0.1), final_p=config.get('exploration_sched_finalp', 0.1))

        def policy():
            env.state = env.construct_state()
            if env.gw.mode == "person":
                action_mask = get_mask_person_pos(gw, method_type, spec, gw.pos, env.gw.person.pos)
            else:
                action_mask = get_mask_region_pos(gw, method_type, spec, gw.pos)

            action = act(env.state[None], action_mask, stochastic=False)[0]
            a = env.gw.check_action_target(action)

            a = (a[1], a[0])
            # Action is in r,c return x,y
            return a[1], a[0]

        # States to evaluate
        eval_states = None

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()

        obs = env.reset()

        visitation_counts = np.zeros(env.gw.grid.shape)

        for t in range(config.get('exploration_sched_timesteps', 100000)):

            visitation_counts[env.gw.pos[1], env.gw.pos[0]] += 1
            if env.gw.grid[env.gw.pos[1], env.gw.pos[0]] != 0:
                print (env.gw.pos[1], env.gw.pos[0], "!!!!!!!!!!")

            if gw.mode == "person":
                action_mask = get_mask_person_pos(env.gw, method_type, spec, env.gw.pos, env.gw.person.pos)
            else:
                action_mask = get_mask_region_pos(env.gw, method_type, spec, env.gw.pos)
            a = act(obs[None], action_mask, update_eps=exploration.value(t))[0]

            target = env.gw.get_target(a) # x, y

            s = grid_to_array(env.gw.pos, env.gw.grid.shape)

            target = (target[1], target[0])

            new_obs, rew, done, info = env.step(target) # row, column

            action_mask_p = np.zeros(len(gw.actions))
            if method_type:
                if gw.mode == "person":
                    action_mask_p = get_mask_person_pos(env.gw, method_type, spec, env.gw.pos, env.gw.person.pos)
                else:
                    action_mask_p = get_mask_region_pos(env.gw, method_type, spec, env.gw.pos)

                if method_type == "shaping":
                    # shape the reward
                    ap = act(new_obs[None], action_mask_p, stochastic=False)[0]
                    f = action_mask_p[ap] - action_mask[a]
                    rew = rew + f

            # Store transition in the replay buffer.
            replay_buffer.add(obs, a, rew, new_obs, float(done), action_mask_p)
            obs = new_obs

            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            if t > 900:
                if eval_states is None:
                    eval_states, _, _, _, _, _ = replay_buffer.sample(128)

                obses_t, actions, rewards, obses_tp1, dones, masks_tp1 = replay_buffer.sample(64)

                rhist, td_err, q_t_sel, q_t_sel_target = train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards), masks_tp1)
                train_summ = tf.Summary()
                train_summ.value.add(tag='td_err', simple_value=td_err)
                train_summ.value.add(tag='q_t_sel', simple_value=q_t_sel)
                train_summ.value.add(tag='q_t_sel_target', simple_value=q_t_sel_target)

            if (t + 1) % 1000 == 0:
                update_target()
                U.save_variables(output, sess=sess)
                print("Saved@ %d!" % t)


            # Visualize
            if (t + 1) % eval_period == 0:
                np.set_printoptions(precision=4, suppress=True)
                print(visitation_counts / np.sum(visitation_counts))
                events_grid = np.zeros(env.gw.grid.shape)
                for idx, o in enumerate(env.gw.event_region):
                    events_grid[o.y, o.x] = env.gw.num_events[idx]
                visitation_counts = np.zeros(env.gw.grid.shape)
                eval_fn(env, policy, debug['q_values'], None, t+1)
                env.reset_stats()

def grid_to_array(pos, shape):
    return pos[1] * shape[1] + pos[0]

