import  sys

import pygame
import pygame.locals as pgl

from Target import *
from Robot import *

import pickle


class GridWorldGui:
    def __init__(self, x0=None, t0=0, step=0.01, num_rows=5, num_cols=5, size=80, image_size=40,
                 obstacles=None, forbidden_zone=None):

        # compute the appropriate height and width (with room for cell borders)
        self.height = num_rows * size + num_rows + 1
        self.width = num_cols * size + num_cols + 1
        self.size = size
        self.num_states = num_rows * num_cols
        self.num_rows = num_rows
        self.num_cols = num_cols

        self.left_edge = []
        self.top_edge = []
        self.right_edge = []
        self.bottom_edge = []

        policy_name = "policy_semi_1.pkl"
        policyact_name = "policy_sto.pkl"
        with open(policy_name, "rb") as f1:
            self.policy = pickle.load(f1)

        with open(policyact_name, "rb") as f2:
            self.policy_act = pickle.load(f2)

        self.obstacles = obstacles
        self.forbidden_zone = forbidden_zone
        # initialize pygame ( SDL extensions )
        pygame.init()
        pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('GridWorld')

        # initialize robot and target
        if x0 is None:
            self.cat1 = Cat(x0, t0, step, size, image_size, num_rows, num_cols, self.height, self.width, self.obstacles,
                            filename='cat1.jpeg')
            self.cat2 = Cat(x0, t0, step, size, image_size, num_rows, num_cols, self.height, self.width, self.obstacles,
                            filename='cat2.jpeg')
            self.robot = PointRobot(x0, t0, step, size, image_size, num_rows, num_cols, self.height, self.width,
                                    self.obstacles, filename='robot.jpeg')
        else:
            self.cat1 = Cat(x0[0:2] + [0, 0], t0, step, size, image_size, num_rows, num_cols, self.height, self.width,
                            self.obstacles, filename='cat1.jpeg')
            self.cat2 = Cat(x0[2:4] + [0, 0], t0, step, size, image_size, num_rows, num_cols, self.height, self.width,
                            self.obstacles, filename='cat2.jpeg')
            self.robot = PointRobot(x0[4:6] + [0, 0], t0, step, size, image_size, num_rows, num_cols, self.height,
                                    self.width, self.obstacles, filename='robot.jpeg')

        self.screen = pygame.display.get_surface()
        self.surface = pygame.Surface(self.screen.get_size())
        self.bg = pygame.Surface(self.screen.get_size())
        self.bg_rendered = False  # optimize background render

        self.background()
        # self.surface.blit(self.cat1.image, self.cat1.rect)
        # self.surface.blit(self.cat2.image, self.cat2.rect)
        # self.surface.blit(self.robot.image, self.robot.rect)
        self.screen.blit(self.surface, (0, 0))

        pygame.display.update()
        # pygame.time.delay(5000)

    def indx2coord(self, i, j, center=False):
        # the +1 indexing business is to ensure that the grid cells
        # have borders of width 1px

        if center:
            return i * (self.size + 1) + 1 + self.size / 2, \
                   j * (self.size + 1) + 1 + self.size / 2
        else:
            return i * (self.size + 1) + 1, j * (self.size + 1) + 1

    def coords(self, s):
        return s // self.num_cols, s % self.num_cols  # the coordinate for state s.

    def background(self):

        if self.bg_rendered:
            self.surface.blit(self.bg, (0, 0))
        else:
            self.bg.fill((0, 0, 0))
            for s in range(self.num_states):
                i, j = self.coords(s)
                x, y = self.indx2coord(i, j, False)
                coords = pygame.Rect(x, y, self.size, self.size)
                pygame.draw.rect(self.bg, (255, 255, 255), coords)

                # Draw Wall in black color.
            for (i, j) in self.obstacles:
                x, y = self.indx2coord(i, j, False)
                # coords = pygame.Rect(y-self.size/2, x - self.size/2, self.size, self.size)
                coords = pygame.Rect(x, y, self.size, self.size)
                pygame.draw.rect(self.bg, (255, 0, 0), coords)  # the obstacles are in color red

            for (i, j) in self.forbidden_zone:
                x, y = self.indx2coord(i, j, False)
                # coords = pygame.Rect(y-self.size/2, x - self.size/2, self.size, self.size)
                coords = pygame.Rect(x, y, self.size, self.size)
                pygame.draw.rect(self.bg, (0, 0, 0), coords)  # the forbidden zoom are in color black

        self.bg_rendered = True  # don't render again unless flag is set

        self.surface.blit(self.bg, (0, 0))

    def run(self):
        loop_index = 0
        goal = []
        while 1:
            for event in pygame.event.get():
                if event.type in (pygame.QUIT, pygame.KEYDOWN):
                    sys.exit()

            self.cat1.run(0.01)
            self.cat2.run(0.01)

            if loop_index % 10 == 0:
                loop_index = 0
                # The goal for the robot is updated every 10 iteration of simulation. The current goal is generated
                # is generated randomly, Haoxiang you can insert your algorithm here.
                robot_st = self.robot.discrete_state()
                cat1_st = self.cat1.discrete_state()
                cat2_st = self.cat2.discrete_state()

                joint_st = (robot_st, cat1_st, cat2_st)
                if robot_st in self.obstacles:
                    x, y = robot_st
                    temp = np.abs([self.robot.x[0] + 0.5 - x, self.robot.x[0] - 0.5 - x,
                                   self.robot.x[1] + 0.5 - y, self.robot.x[1] - 0.5 - y])
                    side_index = np.argmin(temp)
                    if side_index == 0:
                        exact_action = 'N'
                    elif side_index == 1:
                        exact_action = 'S'
                    elif side_index == 2:
                        exact_action = 'W'
                    else:
                        exact_action = 'E'
                else:
                    # choose specific policy based the current progress here
                    if self.cat1.caught and self.cat2.caught:
                        pass
                    elif self.cat1.caught and not self.cat2.caught:
                        pass
                    elif self.cat2.caught and not self.cat1.caught:
                        pass
                    else:
                        pass

                    target = self.policy[joint_st][0]

                    if target == 1:
                        f_state = (robot_st, cat1_st)
                    elif target == 2:
                        f_state = (robot_st, cat2_st)
                    action_dict = self.policy_act[f_state]
                    exact_action = randomchoose(action_dict)

                if exact_action == "N":
                    goal = np.array(robot_st) + [-1, 0]
                elif exact_action == "S":
                    goal = np.array(robot_st) + [1, 0]
                elif exact_action == "W":
                    goal = np.array(robot_st) + [0, -1]
                elif exact_action == "E":
                    goal = np.array(robot_st) + [0, 1]


                # goal = [np.random.random_integers(low=0, high=7), np.random.random_integers(low=0, high=7)]

            loop_index = loop_index + 1
            self.robot.run(0.01, goal)

            if self.cat1.discrete_state() == self.robot.discrete_state():
                self.cat1.caught = True

            if self.cat2.discrete_state() == self.robot.discrete_state():
                self.cat2.caught = True

            self.background()
            if not self.cat1.caught:
                self.surface.blit(self.cat1.image, self.cat1.rect)

            if not self.cat2.caught:
                self.surface.blit(self.cat2.image, self.cat2.rect)
            self.surface.blit(self.robot.image, self.robot.rect)
            self.screen.blit(self.surface, (0, 0))
            pygame.display.update()
            pygame.time.delay(100)

def randomchoose(dic):
    keylist = []
    valuelist = []
    for key in dic.keys():
        keylist.append(key)
        valuelist.append(dic[key])
    choice_index = np.random.choice(len(keylist), 1, p = valuelist)[0]
    choice = keylist[choice_index]
    return choice

def main():
    obstacles = [(1, 1), (1, 3), (3, 1), (3, 3)]
    forbidden_zone = [(2, 2)]
    sim = GridWorldGui(x0=[0, 0.5, 0.5, 4, 4, 4], obstacles=obstacles, forbidden_zone=forbidden_zone)
    # sim = GridWorldGui()
    sim.run()

if __name__ == "__main__":
    main()
