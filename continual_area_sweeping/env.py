import gym
from gym import spaces
import numpy as np
import tensorflow as tf

class GridWorldEnv(gym.Env):
    metadata = {'render.modes': [None]}

    def __init__(self, gw):
        self.gw = gw
        self.prev_dps = 0
        self.num_actions = 0

        self.state = self.construct_state()

    def step(self, action):
        # Action is row, column, but we need x, y
        time, changeseen = self.gw.move((action[1], action[0]))

        #reward = 10*(changeseen[-1] / time)
        #reward = 10*(sum(changeseen) / time)
        reward = 10*(sum(changeseen))
        #reward = 10*(self.gw.get_dps()*(self.num_actions+1) - self.prev_dps*self.num_actions)

        self.prev_dps = self.gw.get_dps()
        self.num_actions += 1

        if self.gw.mode == "person":
            self.gw.person.viewable_counts += int(self.gw.person_viewable())
            self.person_step()
        state_prime = self.construct_state()

        self.state = state_prime
        return state_prime, reward, False, {'unreachable_map': self.gw.grid}

    def person_step(self):
        if self.gw.person is not None:
            self.gw.update_person()

    def construct_state(self):
        # Pos is in x, y so turn it into r, c
        position = (self.gw.pos[1], self.gw.pos[0])
        # Position grid
        posgrid = np.zeros(self.gw.grid.shape)
        posgrid[position] = 1

        # Person position grid
        person_posgrid = None
        if self.gw.person is not None:
            person_position = (self.gw.person.pos[1], self.gw.person.pos[0])
            person_posgrid = np.zeros(self.gw.grid.shape)
            if self.gw.person_viewable():
                person_posgrid[person_position] = 1
            #person_posgrid[person_position] = 1

        decaygrid = np.zeros(self.gw.grid.shape)
        for idx, cell in enumerate(self.gw.event_region):
            if self.gw.last_seen[idx] == 0:
                decaygrid[cell.y, cell.x] = 0
            else:
                tsteps_elapsed = (self.gw.timestep - self.gw.last_seen[idx])
                decaygrid[cell.y, cell.x] = np.exp(-0.05*tsteps_elapsed)
                self.tsteps_elapsed = max(getattr(self, 'tsteps_elapsed', 0), tsteps_elapsed)

        statecomponents = [self.gw.grid.astype(np.float32), posgrid, decaygrid]
        if self.gw.person is not None:
            statecomponents.append(person_posgrid)
        return np.stack(statecomponents, axis=2)

    def reset(self, pos=None):
        #self.gw.reset(pos=(7, 7))
        print ("resetting")
        self.gw.reset()
        self.state = self.construct_state()
        self.num_actions = 0
        self.prev_dps = 0
        return self.state

    def reset_stats(self):
        self.gw.reset_stats()

    def render(self, mode=None, close=False):
        pass

    @property
    def action_space(self):
        #return spaces.Discrete(4)
        return spaces.Discrete(len(self.gw.actions))
