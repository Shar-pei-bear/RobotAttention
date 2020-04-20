import  sys

import pygame
import pygame.locals as pgl

from Target import *
from Robot import *


class GridWorldGui:
    def __init__(self, x0=None, t0=0, step=0.01, num_rows=8, num_cols=8, size=80):
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

        for x in range(self.num_states):
            # note that edges are not disjoint, so we cannot use elif
            if x % self.num_cols == 0:
                self.left_edge.append(x)
            if 0 <= x < self.num_cols:
                self.top_edge.append(x)
            if x % self.num_cols == self.num_cols - 1:
                self.right_edge.append(x)
            if (self.num_rows - 1) * self.num_cols <= x <= self.num_states:
                self.bottom_edge.append(x)
        self.edges = self.left_edge + self.top_edge + self.right_edge + self.bottom_edge

        # initialize pygame ( SDL extensions )
        pygame.init()
        pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('GridWorld')
        self.screen = pygame.display.get_surface()
        self.surface = pygame.Surface(self.screen.get_size())
        self.bg = pygame.Surface(self.screen.get_size())
        self.bg_rendered = False  # optimize background render

        self.background()
        self.screen.blit(self.surface, (0, 0))
        pygame.display.flip()
        pygame.time.delay(5000)
        if x0 is None:
            cat1 = Cat(x0, t0, step)
            cat2 = Cat(x0, t0, step)
            robot = PointRobot(x0, t0, step)
        else:
            cat1 = Cat(x0[0:2], t0, step)
            cat2 = Cat(x0[2:4], t0, step)
            robot = PointRobot(x0[4:6], t0, step)

    def indx2coord(self, s, center=False):
        # the +1 indexing business is to ensure that the grid cells
        # have borders of width 1px
        i, j = self.coords(s)
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
                x, y = self.indx2coord(s, False)
                coords = pygame.Rect(y, x, self.size, self.size)
                pygame.draw.rect(self.bg, (250, 250, 250), coords)

                # Draw Wall in black color.
            for s in self.edges:
                (x, y) = self.indx2coord(s)
                # coords = pygame.Rect(y-self.size/2, x - self.size/2, self.size, self.size)
                coords = pygame.Rect(y, x, self.size, self.size)
                pygame.draw.rect(self.bg, (192, 192, 192), coords)  # the obstacles are in color grey

        self.bg_rendered = True  # don't render again unless flag is set
        self.surface.blit(self.bg, (0, 0))


def main():
    sim = GridWorldGui()


if __name__ == "__main__":
    main()
