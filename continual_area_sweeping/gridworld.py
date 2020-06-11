import numpy as np
import collections
import random
import copy


Object = collections.namedtuple('Object', ['x', 'y', 'period', 'bound'])

class Person:
    def __init__(self, pos):
        self.pos = pos

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

    def __init__(self, grid, event_region, person=None, initialpos=(0, 0), viewable_distance=0, mode='linear', stack=False, extra_event_region=[]):
        self.pos = initialpos
        self.grid = grid
        self.event_region = event_region
        self.viewable_distance = viewable_distance
        self.mode = mode
        self.stack = stack
        self.last_updated = [0] * len(event_region)
        self.last_seen = [0] * len(event_region)
        self.num_events = [0] * len(event_region)
        self.last_visited = np.zeros((grid.shape[0], grid.shape[1]))
        self.timestep = 0
        self.total_detection_time = 0
        self.total_num_events = 0
        self.num_detections = 0
        self.prev_num_detections = 0
        self.person = person
        self.extra_event_region = extra_event_region

        self.fw = FloydWarshall(self.grid)

        self.actions = [(0, -3), (-1, -2), (0, -2), (1, -2), (-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1), \
                        (-3, 0), (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), (3, 0), \
                        (-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1), (-1, 2), (0, 2), (1, 2), (0, 3)]

    def update_events(self):
        for idx, o in enumerate(self.event_region):
            should_update = False

            if self.mode == 'linear' and random.random() <= 1 / o.period:
                should_update = True

            if self.mode == 'periodic' and self.timestep % o.period == 0:
                should_update = True

            if self.mode == "person" and o.x == self.person.pos[0] and o.y == self.person.pos[1] and random.random() <= 0.2:
                should_update = True

            if self.mode == "person" and (o.y, o.x) in self.extra_event_region and random.random() <= 1 / o.period:
                should_update = True

            if should_update:
                self.last_updated[idx] = self.timestep

                #if self.num_events[idx] < o.bound:
                self.total_num_events += 1

                if self.stack:
                    self.num_events[idx] += 1
                    self.num_events[idx] = min(o.bound, self.num_events[idx])
                else:
                    self.num_events[idx] = 1
            if self.last_updated[idx] < self.timestep:
                if random.random() <= 0.2:
                    self.num_events[idx] = 0


    def update_person(self):

        self.person.prev_pos = self.person.pos
        while self.person.pos == self.person.goal:
            self.reset_person()
        self.person.index += 1
        self.person.pos = self.person.path[self.person.index]



    def move(self, action):

        self.timestep += 1
        self.update_events()

        if not self.check_target(action):
            return 1, [0]

        dist = self.fw.distance(self.pos, action)

        # Process cycle
        changeseen = []
        path = self.fw.path(self.pos, action)

        # If action is to stay there, just put that in the path
        if dist == 0:
            path = [action]
        else:
            path = path[1:]

        for x, y in path:
            self.last_visited[y, x] = self.timestep

            observed = False
            for idx, o in enumerate(self.event_region):

                if self.last_seen[idx] < self.last_updated[idx] and self.num_events[idx] > 0 and self.__object_viewable((x, y), o):
                    self.last_seen[idx] = self.timestep
                    changeseen.append(self.num_events[idx])
                    observed = True
                    self.num_events[idx] = 0

                if self.last_seen[idx] < self.last_updated[idx]:
                    self.total_detection_time += self.num_events[idx]

            if not observed:
                changeseen.append(0)

        self.num_detections += sum(changeseen)
        self.pos = action
        return len(path), changeseen

    def __object_viewable(self, robotxy, object):
        x, y = robotxy
        within_distance = np.linalg.norm([object.x - x, object.y - y]) <= self.viewable_distance
        #return within_distance

        ind_robot = robotxy[1] * self.grid.shape[1] + robotxy[0]
        ind_object = object.y * self.grid.shape[1] + object.x
        viewable = ind_object not in self.invisibility[ind_robot]
        #return viewable

        return within_distance and viewable

    def person_viewable(self, pos=None, person_pos=None):
        if pos is None:
            pos = self.pos
        if person_pos is None:
            person_pos = self.person.pos
        ind_robot = pos[1] * self.grid.shape[1] + pos[0]
        ind_person = person_pos[1] * self.grid.shape[1] + person_pos[0]

        return ind_person not in self.invisibility[ind_robot]

    def reset(self, pos=None):
        free_spaces = np.argwhere(self.grid == 0)
        np.random.shuffle(free_spaces)
        pos = (free_spaces[0, 1], free_spaces[0, 0])
        self.pos = pos
        self.timestep = 0
        for idx, obj in enumerate(self.event_region):
            self.last_seen[idx] = 0
            self.last_updated[idx] = 0
            self.num_events[idx] = 0
        self.last_visited = np.zeros_like(self.last_visited)
        self.total_detection_time = 0
        self.total_num_events = 0
        self.num_detections = 0
        self.prev_num_detections = 0

        if self.mode == 'person':
            self.person.prev_pos = pos
            self.person.pos = pos
            self.reset_person()
            self.person.viewable_counts = 0

    def reset_person(self):
        goals = [[r, c] for r in range(6, 9) for c in range(0, 15)] + [[r, c] for r in range(0, 5) for c in range(0, 7)]
        np.random.shuffle(goals)
        self.person.goal = goals[0][1], goals[0][0]
        self.person.path = self.fw.path(self.person.pos, self.person.goal)
        self.person.index = 0

    def reset_to_state(self, posx, posy, lastseens):
        self.pos = (posx, posy)
        for idx, o in enumerate(self.event_region):
            self.last_seen[idx] = self.timestep - lastseens[idx]

    def reset_stats(self):
        self.prev_num_detections = self.num_detections
        if self.mode == "person":
            self.person.viewable_counts = 0

    def get_adt(self):
        return self.total_detection_time/self.total_num_events

    def get_dps(self):
        return self.num_detections/(self.timestep+1)

    def check_target(self, target, pos=None):
        if not pos:
            pos = self.pos
        # target is x, y
        if 0 <= target[1] < self.grid.shape[0] and 0 <= target[0] < self.grid.shape[1]:
            if self.grid[target[1]][target[0]] == 0 and self.fw.distance(pos, target) <= 3:
                return True
        return False

    def get_target(self, action, pos=None):
        if not pos:
            pos = self.pos
        # target is x, y
        target = (pos[0] + self.actions[action][0], pos[1] + self.actions[action][1])
        return target
