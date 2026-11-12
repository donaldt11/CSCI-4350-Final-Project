class WarehouseEnvironment:
    def __init__(self):
        self.grid_size = 6 #set size of grid

        #add obstacles to grid
        self.obstacles = {
            (0, 3), (1, 3),
            (3, 1), (3, 2)
        }

        #define inital states
        self.start = (0, 0)
        self.package = (1, 4)
        self.delivery = (5, 0)

        self.max_steps = 150
        self.reset()

    #reset environment to initial state
    def reset(self):
        self.robot_pos = self.start
        self.carrying = False
        self.steps = 0
        return self.get_state()

    def get_state(self):
        return (
            self.robot_pos[0],
            self.robot_pos[1],
            int(self.carrying)
        )

    #define actions for robot
    def step(self, action):
        # 0=up, 1=down, 2=left,
        # 3=right, 4=pickup, 5=drop
        movements = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1)
        }

        reward = -1
        done = False

        if action in movements:
            dr, dc = movements[action]
            row, col = self.robot_pos
            new_pos = (row + dr, col + dc)

            valid = (
                0 <= new_pos[0] < self.grid_size
                and 0 <= new_pos[1] < self.grid_size
                and new_pos not in self.obstacles
            )

            if valid:
                self.robot_pos = new_pos
            else:
                reward = -5

        elif action == 4:
            if self.robot_pos == self.package and not self.carrying:
                self.carrying = True
                reward = 10
            else:
                reward = -5

        elif action == 5:
            if self.robot_pos == self.delivery and self.carrying:
                reward = 100
                done = True
            else:
                reward = -5

        else:
            raise ValueError("Invalid action")

        self.steps += 1
        done = done or self.steps >= self.max_steps

        return self.get_state(), reward, done
