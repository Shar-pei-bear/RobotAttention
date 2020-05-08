from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import pygame
import sdeint


class Cat:
    def __init__(self, x0=None, t0=0, step=0.01, size=80, image_size=40, num_rows=8, num_cols=8, height=0, width=0,
                 obstacles=None, filename='cat1.jpeg'):
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
        self.image_size = image_size

        self.trajectory = []
        self.step = step

        self.image = pygame.image.load(filename).convert()
        self.image = pygame.transform.scale(self.image, (image_size, image_size))
        self.rect = self.image.get_rect()

        self.state2pixel()
        self.obstacles = obstacles
        self.caught = False

    def update(self):
        self.x = self.x + np.array([self.x[2], self.x[3], 100*np.random.randn(), 100*np.random.randn()])*self.step
        self.t = self.t + self.step
        self.check_wall()
        self.check_obstacles()
        self.state2pixel()

        # if the robot reaches the boundaries, it will bounce back with reverse velocity

    def check_wall(self):
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

        # if the robot hits obstacles, it will come to a halt

    def check_obstacles(self):
        x, y = self.discrete_state()
        if (x, y) in self.obstacles:
            # determine which side of the grid the object hit
            temp = np.abs([self.x[0] + 0.5 - x, self.x[0] - 0.5 - x, self.x[1] + 0.5 - y, self.x[1] - 0.5 - y])
            side_index = np.argmin(temp)
            if side_index == 0 and self.x[2] > 0:
                self.x[0] = self.x[0] - self.x[2]*self.step
                self.x[2] = 0
            elif side_index == 1 and self.x[2] < 0:
                self.x[0] = self.x[0] - self.x[2]*self.step
                self.x[2] = 0
            elif side_index == 2 and self.x[3] > 0:
                self.x[1] = self.x[1] - self.x[3]*self.step
                self.x[3] = 0
            elif side_index == 3 and self.x[3] < 0:
                self.x[1] = self.x[1] - self.x[3]*self.step
                self.x[3] = 0

    def state2pixel(self):

        pixel_x = 1 + self.size/2 - self.image_size/2 + self.x[0]*(self.width - 2 - self.size) / (self.num_cols - 1)
        pixel_y = 1 + self.size/2 - self.image_size/2 + self.x[1]*(self.height - 2 - self.size) / (self.num_rows - 1)

        delta_x = pixel_x - self.rect.left
        delta_y = pixel_y - self.rect.top

        self.rect = self.rect.move((delta_x, delta_y))

    def run(self, interval):
        self.trajectory = []
        self.trajectory.append([self.x[0], self.x[1], self.t])
        tf = self.t + interval
        while self.t < tf:
            self.update()
            self.trajectory.append([self.x[0], self.x[1], self.t])

        self.trajectory = np.array(self.trajectory)

    def reset(self):
        self.trajectory = []
        self.t = self.t0
        self.x = self.x0

    def discrete_state(self):
        return np.rint(self.x[0]).astype(int), np.rint(self.x[1]).astype(int)

    def discrete_state_policy(self):
        return np.rint(self.x[1]).astype(int), np.rint(self.x[0]).astype(int)

    
def main():
    pygame.init()
    pygame.display.set_mode()

    cat1 = Cat(x0=[0, 0, 0, 0])
    cat1.run(1)
    cat1.run(1)

    plt.plot(cat1.trajectory[:, 2], cat1.trajectory[:, 0], 'b', label='x(t)')
    plt.plot(cat1.trajectory[:, 2], cat1.trajectory[:, 1], 'b', label='x(t)')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
