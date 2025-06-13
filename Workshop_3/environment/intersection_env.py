import gym
from gym import spaces
import numpy as np

class IntersectionEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.num_intersections = 2
        self.num_directions = 4  # 0=N, 1=E, 2=S, 3=W

        self.crossing_duration = 3
        self.vehicle_pass_rate = 2
        self.right_turn_limit = 3

        self.state_size = self.num_intersections * self.num_directions * 4
        self.action_space = spaces.MultiDiscrete([self.num_directions] * self.num_intersections)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_size,), dtype=np.float32)

        self.buffer_A_to_B = np.zeros(self.num_directions, dtype=int)
        self.buffer_B_to_A = np.zeros(self.num_directions, dtype=int)
        self.reset()

    def reset(self):
        self.queues = np.random.randint(0, 5, size=(self.num_intersections, self.num_directions))
        self.ped_requests = np.random.randint(0, 2, size=(self.num_intersections, self.num_directions))
        self.ped_timers = np.zeros_like(self.ped_requests)
        self.signals = np.zeros_like(self.queues)
        self.signal_timer = np.zeros_like(self.queues)

        # vehicle transfer buffer: [A→B], [B→A]
        self.buffer_A_to_B = np.zeros(self.num_directions, dtype=int)
        self.buffer_B_to_A = np.zeros(self.num_directions, dtype=int)


        return self._get_state()

    def step(self, actions):
        rewards = []

        for inter_id, action_dir in enumerate(actions):
            reward = self._process_intersection(inter_id, action_dir)
            rewards.append(reward)

        self._update_signal_timers(actions)
        self._update_pedestrian_timers()
        self._transfer_vehicles()

        return self._get_state(), sum(rewards), False, {}

    def _process_intersection(self, inter_id, action_dir):
        vehicle_queue = self.queues[inter_id][action_dir]
        ped_waiting = self.ped_requests[inter_id][action_dir]
        ped_crossing = self.ped_timers[inter_id][action_dir] > 0

        self.signals[inter_id] = 0
        self.signals[inter_id][action_dir] = 1

        reward = 0
        can_move = not ped_crossing

        if can_move:
            moved = min(vehicle_queue, self.vehicle_pass_rate)
            self.queues[inter_id][action_dir] -= moved
            self._buffer_vehicle(inter_id, action_dir, moved)
            reward += moved * 2.0
        else:
            reward -= 1.0

        if ped_waiting and not ped_crossing:
            self.ped_timers[inter_id][action_dir] = self.crossing_duration
            self.ped_requests[inter_id][action_dir] = 0
            reward += 3.0

        for dir_id in range(self.num_directions):
            if dir_id != action_dir and self.ped_timers[inter_id][dir_id] > 0:
                reward -= 2.0

        return reward

    def _buffer_vehicle(self, inter_id, exit_dir, count):
        """
        Determine where to send vehicles from this intersection.
        """
        if count == 0:
            return

        if inter_id == 0:
            if exit_dir == 0:    # A[N] → B[S]
                self.buffer_A_to_B[2] += count
            elif exit_dir == 1:  # A[E] (right turn) → B[N]
                self.buffer_A_to_B[0] += count
        elif inter_id == 1:
            if exit_dir == 2:    # B[S] → A[N]
                self.buffer_B_to_A[0] += count
            elif exit_dir == 3:  # B[W] (right turn) → A[S]
                self.buffer_B_to_A[2] += count

    def _transfer_vehicles(self):
        self.queues[1] += self.buffer_A_to_B.astype(int)
        self.queues[0] += self.buffer_B_to_A.astype(int)
        self.buffer_A_to_B[:] = 0
        self.buffer_B_to_A[:] = 0

    def _update_signal_timers(self, actions):
        for inter_id, active_dir in enumerate(actions):
            for dir_id in range(self.num_directions):
                if dir_id == active_dir:
                    self.signal_timer[inter_id][dir_id] += 1
                else:
                    self.signal_timer[inter_id][dir_id] = 0

    def _update_pedestrian_timers(self):
        for i in range(self.num_intersections):
            for d in range(self.num_directions):
                if self.ped_timers[i][d] > 0:
                    self.ped_timers[i][d] -= 1

    def _get_state(self):
        norm_queues = self.queues.flatten() / 10.0
        ped_waiting = self.ped_requests.flatten()
        ped_crossing = (self.ped_timers > 0).astype(float).flatten()
        signal_status = self.signals.flatten()
        return np.concatenate([norm_queues, ped_waiting, ped_crossing, signal_status])
