import numpy as np
import random

class FloydWarshall:

    def __init__(self, grid):
        self.grid = grid
        self.dists, self.nexts = self.__floyd_warshall(grid)

    def distance(self, *indices):
        (x1, y1), (x2, y2) = indices
        return self.dists[self.rc_to_ind(y1, x1), self.rc_to_ind(y2, x2)]

    def path(self, *indices):
        (x1, y1), (x2, y2) = indices
        if np.isnan(self.nexts[self.rc_to_ind(y1, x1), self.rc_to_ind(y2, x2)]):
            return []

        path = [(x1, y1)]
        u = self.rc_to_ind(y1, x1)
        v = self.rc_to_ind(y2, x2)
        while u != v:
            u = int(self.nexts[u, v])
            r, c = self.ind_to_rc(u)
            path.append((c, r))

        return path

    def rc_to_ind(self, r, c):
        return int(r*self.grid.shape[1] + c)

    def ind_to_rc(self, ind):
        return int(ind // self.grid.shape[1]), int(ind % self.grid.shape[1])

    def __floyd_warshall(self, grid):
        h, w = grid.shape
        n = h*w
        dists = np.ones((n, n))*np.inf
        nexts = np.ones((n, n))*np.nan

        for r in range(h):
            for c in range(w):
                cur = self.rc_to_ind(r, c)
                dists[cur, cur] = 0

                if r > 0 and grid[r-1, c] == 0:
                    nxt = self.rc_to_ind(r-1, c)
                    dists[cur, nxt] = 1
                    nexts[cur, nxt] = nxt
                if r < h-1 and grid[r+1, c] == 0:
                    nxt = self.rc_to_ind(r+1, c)
                    dists[cur, nxt] = 1
                    nexts[cur, nxt] = nxt
                if c > 0 and grid[r, c-1] == 0:
                    nxt = self.rc_to_ind(r, c-1)
                    dists[cur, nxt] = 1
                    nexts[cur, nxt] = nxt
                if c < w-1 and grid[r, c+1] == 0:
                    nxt = self.rc_to_ind(r, c+1)
                    dists[cur, nxt] = 1
                    nexts[cur, nxt] = nxt

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dists[i, j] > dists[i, k] + dists[k, j]:
                        dists[i, j] = dists[i, k] + dists[k, j]
                        nexts[i, j] = nexts[i, k]

        return dists, nexts

class GridWorld:

    def __init__(self, grid, initial_pos=None, goal=[(5, 5)], goal_known=[(5, 5)]):
        self.initial_pos = initial_pos
        self.grid = grid
        self.reset_pos()
        self.goal = goal
        self.goal_known = goal_known
        self.timestep = 0
        self.total_reward = 0
        self.actions = [(-1, 0), (0, -1), (1, 0), (0, 1)]

        self.fw = FloydWarshall(self.grid)

    def move(self, action):

        if (action[0] < 0 or action[0] >= self.grid.shape[1] or action[1] < 0 or action[1] >= self.grid.shape[0]):
            return 1, 0

        if self.grid[action[1], action[0]] != 0:
            return 1, 0

        dist = self.fw.distance(self.pos, action)

        if self.grid[action[1], action[0]] != 0:
            raise RuntimeError("Invalid Action!")

        # Process cycle
        path = self.fw.path(self.pos, action)

        # If action is to stay there, just put that in the path
        if dist == 0:
            path = [action]
        else:
            path = path[1:]

        reward = 0
        for x, y in path:
            self.timestep += 1
            self.pos = (x,y)
            #reward -=1

            if (x,y) in self.goal:
                if (x,y) == self.goal[-1]:
                    reward += 100
                    self.reset_pos()
                    break
                else:
                    reward += 100

        self.total_reward += reward

        return len(path), reward

    def reset_pos(self):
        if self.initial_pos:
            self.pos = self.initial_pos
        else:
            pos = (random.randint(0, self.grid.shape[1]-1), random.randint(0, self.grid.shape[0]-1))
            while self.grid[pos[1], pos[0]] != 0:
                pos = (random.randint(0, self.grid.shape[1]-1), random.randint(0, self.grid.shape[0]-1))
            self.pos = pos

    def reset(self):
        self.reset_pos()
        self.timestep = 0
        self.total_reward = 0

    def get_avg_reward(self):
        return self.total_reward/(self.timestep+1)

    def is_cell_empty(self, pos):
        if not self.is_cell_in_grid(pos):
            return False
        else:
            return self.grid[pos[1], pos[0]] == 0

    def is_cell_in_grid(self, pos):
        if (pos[0] < 0 or pos[0] >= self.grid.shape[1] or pos[1] < 0 or pos[1] >= self.grid.shape[0]):
            return False
        return True

    def manhattan_dist(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def min_goal_manhattan_dist(self, pos):
        min_dist = self.manhattan_dist(pos, self.goal_known[0])
        for g in self.goal_known[1:]:
            dist = self.manhattan_dist(pos, g)
            if dist < min_dist:
                min_dist = dist

        return min_dist

