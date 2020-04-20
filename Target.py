from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import sdeint


class Cat:
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

    def update(self):
        self.x = self.x + np.array([self.x[2], self.x[3], 10*np.random.randn(), 10*np.random.randn()])*self.step
        self.t = self.t + self.step

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

def main():

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
