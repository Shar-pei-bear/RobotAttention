from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import sdeint


class PointRobot:
    def __init__(self, x0=None, t0=0, step=0.01):
        # set initial time and state

        if x0 is None:
            self.x = np.array([0, 0, 0, 0])
            self.x0 = np.array([0, 0, 0, 0])
        else:
            self.x = np.array(x0)
            self.x0 = np.array(x0)

        self.t = t0
        self.t0 = t0

        self.trajectory = []
        self.step = step

        self.A = np.array([[0.9909, 0, 0.008611, 0],
                           [0, 0.9909, 0, 0.008611],
                           [-1.722, 0, 0.7326, 0],
                           [0, -1.722, 0, 0.7326]])

        self.B = np.array([[0.009056, 0], [0, 0.009056], [1.722, 0], [0, 1.722]])

    def update(self, goal):

        u = np.array(goal) + np.random.randn(2)*self.step*100
        print(self.B.dot(u))
        self.x = self.A.dot(self.x) + self.B.dot(u)
        self.t = self.t + self.step

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
