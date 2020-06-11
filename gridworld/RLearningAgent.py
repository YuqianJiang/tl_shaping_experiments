from gridworld import GridWorld
import numpy as np
import random

class RLearningAgent:
    def __init__(self, gw):
        self.gw = gw
        self.actions = gw.actions
        self.R = np.zeros([self.gw.grid.shape[0]*self.gw.grid.shape[1], len(self.actions)])

        self.phi = np.zeros([self.gw.grid.shape[0]*self.gw.grid.shape[1], len(self.gw.actions)])
        for y in range(self.gw.grid.shape[0]):
            for x in range(self.gw.grid.shape[1]):
                dist_s = self.gw.min_goal_manhattan_dist((x, y))
                for i, a in enumerate(self.gw.actions):
                    self.phi[y * self.gw.grid.shape[1] + x, i] = 1 if self.check_allowed((x, y), a) else -1

    def greedy_action_selection(self, r_s):
        return r_s.argmax()

    def random_action_selection(self):
        a = random.randint(0, len(self.gw.actions)-1)
        return a

    def get_r(self, s):
        return np.copy(self.R[s, :])

    def get_biased_r(self, s):
        r_s = np.copy(self.R[s, :])
        
        for a, action in enumerate(self.gw.actions):
            r_s[a] += self.phi[s, a]

        return r_s

    def get_shielded_r(self, s, pos):
        r_s = np.copy(self.R[s, :])
        for a, action in enumerate(self.gw.actions):
            if not self.check_allowed(pos, action):
                r_s[a] = -np.inf
            
        return r_s

    def check_allowed(self, pos, action):
        if action[0] + action[1] == 1:
            return True
        else:
            return False

    def state_shaping(self, dist):
        return -dist

    def r_learning(self, eval_fn, shaping=0):
        epsilon = 0.1
        alpha = 0.1
        beta = 0.1
        rho = 0

        reset_trajectory_period = 100

        for it in range(1000):

            self.gw.reset()

            for traj in range(reset_trajectory_period):

                s = self.pos_to_index(self.gw.pos)

                if shaping == -1:
                    r_s = self.get_shielded_r(s, self.gw.pos)
                elif shaping == 2:
                    r_s = self.get_biased_r(s)
                else:
                    r_s = self.get_r(s)
                
                if random.random() <= epsilon:
                    a = self.random_action_selection()
                else:
                    a = self.greedy_action_selection(r_s)
                next_pos = self.pos_act(self.gw.pos, self.gw.actions[a])

                dist_s = self.gw.min_goal_manhattan_dist(self.gw.pos)

                time, reward = self.gw.move(next_pos)

                sp = self.pos_to_index(self.gw.pos)
                dist_sp = self.gw.min_goal_manhattan_dist(self.gw.pos)

                if shaping == 1:
                    phi_s = self.state_shaping(dist_s)
                    phi_sp = self.state_shaping(dist_sp)
                    f = phi_sp - phi_s
                    td_error = reward + f - rho + np.max(self.R[sp, :]) - self.R[s, a]
                elif shaping == 2:
                    r_sp = self.get_biased_r(sp)
                    ap = self.greedy_action_selection(r_sp)
                    f = self.phi[sp, ap] - self.phi[s, a]
                    td_error = reward + f - rho + self.R[sp, ap] - self.R[s, a]
                else:
                    td_error = reward - rho + np.max(self.R[sp, :]) - self.R[s, a]

                self.R[s, a] = self.R[s, a] + beta * td_error
                rho = rho + alpha * td_error

            eval_fn(it*reset_trajectory_period)

    def pos_act(self, pos, action):
        return (pos[0] + action[0], pos[1] + action[1])

    def pos_to_index(self, pos):
        return pos[1] * self.gw.grid.shape[1] + pos[0]