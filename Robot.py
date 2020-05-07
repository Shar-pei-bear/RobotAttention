from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import sdeint
import pygame


class PointRobot:
    def __init__(self, x0=None, t0=0, step=0.01, size=80, num_rows=8, num_cols=8, height=0, width=0, filename='robot.jpeg'):
        # set initial time and state

        if x0 is None:
            self.x = np.array([0, 0, 0, 0])
            self.x0 = np.array([0, 0, 0, 0])
        else:
            self.x = np.array(x0)
            self.x0 = np.array(x0)

        self.t = t0
        self.t0 = t0

        self.width = width
        self.height = height

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.size = size

        self.trajectory = []
        self.step = step

        self.A = np.array([[0.9909, 0, 0.008611, 0],
                           [0, 0.9909, 0, 0.008611],
                           [-1.722, 0, 0.7326, 0],
                           [0, -1.722, 0, 0.7326]])

        self.B = np.array([[0.009056, 0], [0, 0.009056], [1.722, 0], [0, 1.722]])

        self.image = pygame.image.load(filename).convert()
        self.image = pygame.transform.scale(self.image, (size, size))
        self.rect = self.image.get_rect()

        self.state2pixel()

    def update(self, goal):

        u = np.array(goal) + np.random.randn(2)*self.step*100
        self.x = self.A.dot(self.x) + self.B.dot(u)
        self.t = self.t + self.step

        # if the robot reaches the boundaries, it will bounce back with reverse velocity

        if self.x[0] < 0:
            self.x[0] = 0
            self.x[2] = -self.x[2]
        elif self.x[0] > (self.num_cols - 1):
            self.x[0] = self.num_cols - 1
            self.x[2] = -self.x[2]

        if self.x[1] < 0:
            self.x[1] = 0
            self.x[3] = -self.x[3]
        elif self.x[1] > (self.num_rows - 1):
            self.x[1] = self.num_rows - 1
            self.x[3] = -self.x[3]

        self.state2pixel()

    def state2pixel(self):

        pixel_x = 1 + self.x[0]*(self.width - 2 - self.size) / (self.num_cols - 1)
        pixel_y = 1 + self.x[1] * (self.height - 2 - self.size) / (self.num_rows - 1)

        delta_x = pixel_x - self.rect.left
        delta_y = pixel_y - self.rect.top

        self.rect = self.rect.move((delta_x, delta_y))

    def run(self, interval, goal):
        self.trajectory = []
        self.trajectory.append([self.x[0], self.x[1], self.t])
        tf = self.t + interval
        while self.t < tf:
            self.update(goal)
            self.trajectory.append([self.x[0], self.x[1], self.t])

        self.trajectory = np.array(self.trajectory)

    def reset(self):
        self.trajectory = []
        self.t = self.t0
        self.x = self.x0

    def discrete_state(self):
        return int(self.x[0]), int(self.x[1])

def main():

    robot1 = PointRobot(x0=[0, 0, 0, 0])
    robot1.run(1, [1, 1])
    robot1.run(1, [2, 2])
    plt.plot(robot1.trajectory[:, 2], robot1.trajectory[:, 0], 'b', label='x(t)')
    plt.plot(robot1.trajectory[:, 2], robot1.trajectory[:, 1], 'g', label='y(t)')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
