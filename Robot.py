from object import *
import matplotlib
import matplotlib.pyplot as plt

class PointRobot(Object):
    def __init__(self, x0=None, t0=0, step=0.01, size=80, image_size=40, num_rows=8, num_cols=8, height=0, width=0,
                 obstacles=None, filename='robot.jpeg'):
        # set initial time and state
        super().__init__(x0, t0, step, size, image_size, num_rows, num_cols, height, width, obstacles, filename)

        self.A = np.array([[0.9909, 0, 0.008611, 0],
                           [0, 0.9909, 0, 0.008611],
                           [-1.722, 0, 0.7326, 0],
                           [0, -1.722, 0, 0.7326]])

        self.B = np.array([[0.009056, 0], [0, 0.009056], [1.722, 0], [0, 1.722]])

    def update(self, goal):

        u = np.array(goal) + np.random.randn(2)*self.step*0
        self.x = self.A.dot(self.x) + self.B.dot(u)
        self.t = self.t + self.step
        self.check_wall()
        self.check_obstacles()
        self.state2pixel()

    def run(self, interval, goal):
        self.trajectory = []
        self.trajectory.append([self.x[0], self.x[1], self.t])
        tf = self.t + interval
        while self.t < tf:
            self.update(goal)
            self.trajectory.append([self.x[0], self.x[1], self.t])

        self.trajectory = np.array(self.trajectory)


def main():
    pygame.init()
    pygame.display.set_mode()

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
