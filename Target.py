from object import *
import matplotlib
import matplotlib.pyplot as plt


class Cat(Object):
    def update(self):
        self.x = self.x + np.array([self.x[2], self.x[3], 100*np.random.randn(), 100*np.random.randn()])*self.step
        self.t = self.t + self.step
        self.check_wall()
        self.check_obstacles()
        self.state2pixel()

    def run(self, interval):
        self.trajectory = []
        self.trajectory.append([self.x[0], self.x[1], self.t])
        tf = self.t + interval
        while self.t < tf:
            self.update()
            self.trajectory.append([self.x[0], self.x[1], self.t])

        self.trajectory = np.array(self.trajectory)

def main():
    pygame.init()
    pygame.display.set_mode()

    cat1 = Cat(x0=[0, 0, 0, 0], obstacles=[(1, 1), (1, 3), (3, 1), (3, 3)])
    cat1.run(100)
    # cat1.run(1)

    plt.plot(cat1.trajectory[:, 2], cat1.trajectory[:, 0], 'b', label='x(t)')
    plt.plot(cat1.trajectory[:, 2], cat1.trajectory[:, 1], 'b', label='x(t)')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
