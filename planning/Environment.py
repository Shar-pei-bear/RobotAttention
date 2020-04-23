import numpy as np

class Env:
    ##Assuming the obstacles here is the same with boundary, when agent is trying to enter obstacle or hit the wall, he will stay at the current state
    def __init__(self, height, width, envrand):
        ##Height and width is the size of the gridworld, envrand is the stochastic in this environment
        self.height = height
        self.width = width
        self.obstacles = []   #this is like wall
        self.sinkobs = []  # This is sink state
        self.envrand = envrand
        self.statespace = self.calstatesp()

    def calstatesp(self):
        ## This is used to calculate the statespace
        statespace = []
        for h in range(self.height):
            for w in range(self.width):
                statespace.append((h,w))
        return statespace

    def addobstacle(self, state):
        ## This is used to add obstacles to the environment
        self.statespace.remove(state)
        self.obstacles.append(state)


    def addsinkobs(self, state):
        self.sinkobs.append(state)

class agent:
    def __init__(self,env, autopolicy = False):
        self.statespace = env.statespace.copy()
        self.obstacles = env.obstacles.copy()
        self.sinkobs = env.sinkobs.copy()
        self.rand = env.envrand
        self.action = {'N':(-1,0),'S':(1,0),'W':(0,-1),'E':(0,1)}
        self.P = self.calP()   # In the form of P[state][action][state_]
        if autopolicy:
            self.policy = self.getpolicy()
            #print(self.policy)
            self.stTransit = self.getstatetransit()    # In the form of P[state][state_]
        ## Once the policy is given, you should call getstatetransit() to calculate the self.stTransit
    def calP(self):    #This function is used to calculate the transition probability of agent
        P = {}
        for state in self.statespace :   ## enumerate all state in the statespace
            P[state] = {}
            for action in self.action.keys():
                P[state][action] = {}
                P[state][action][state] = 0
                state_ = tuple(np.array(state) + np.array(self.action[action]))   ## The possible next state
                if state_ in self.statespace and state_ not in self.obstacles:
                    P[state][action][state_] = 1 - self.rand * 2
                    if self.action[action][0] == 0:
                        tempaction_1 = (1, 0)
                        tempaction_2 = (-1, 0)    ##the neighbour action of the selected action
                        tempstate_1 = tuple(np.array(state) + np.array(tempaction_1))
                        tempstate_2 = tuple(np.array(state) + np.array(tempaction_2))
                        if tempstate_1 in self.statespace and tempstate_1 not in self.obstacles:
                            P[state][action][tempstate_1] = self.rand
                        else:
                            P[state][action][state] += self.rand
                        if tempstate_2 in self.statespace and tempstate_2 not in self.obstacles:
                            P[state][action][tempstate_2] = self.rand
                        else:
                            P[state][action][state] += self.rand
                    else:
                        tempaction_1 = (0, 1)
                        tempaction_2 = (0, -1)  ##the neighbour action of the selected action
                        tempstate_1 = tuple(np.array(state) + np.array(tempaction_1))
                        tempstate_2 = tuple(np.array(state) + np.array(tempaction_2))
                        if tempstate_1 in self.statespace and tempstate_1 not in self.obstacles:
                            P[state][action][tempstate_1] = self.rand
                        else:
                            P[state][action][state] += self.rand
                        if tempstate_2 in self.statespace and tempstate_2 not in self.obstacles:
                            P[state][action][tempstate_2] = self.rand
                        else:
                            P[state][action][state] += self.rand
                ## These two part's logic are the same, if 'if' part is correct, the same for 'else'
                else:
                    P[state][action][state] += 1 - self.rand * 2
                    if self.action[action][0] == 0:
                        tempaction_1 = (1, 0)
                        tempaction_2 = (-1, 0)    ##the neighbour action of the selected action
                        tempstate_1 = tuple(np.array(state) + np.array(tempaction_1))
                        tempstate_2 = tuple(np.array(state) + np.array(tempaction_2))
                        if tempstate_1 in self.statespace and tempstate_1 not in self.obstacles:
                            P[state][action][tempstate_1] = self.rand
                        else:
                            P[state][action][state] += self.rand
                        if tempstate_2 in self.statespace and tempstate_2 not in self.obstacles:
                            P[state][action][tempstate_2] = self.rand
                        else:
                            P[state][action][state] += self.rand
                    else:
                        tempaction_1 = (0, 1)
                        tempaction_2 = (0, -1)  ##the neighbour action of the selected action
                        tempstate_1 = tuple(np.array(state) + np.array(tempaction_1))
                        tempstate_2 = tuple(np.array(state) + np.array(tempaction_2))
                        if tempstate_1 in self.statespace and tempstate_1 not in self.obstacles:
                            P[state][action][tempstate_1] = self.rand
                        else:
                            P[state][action][state] += self.rand
                        if tempstate_2 in self.statespace and tempstate_2 not in self.obstacles:
                            P[state][action][tempstate_2] = self.rand
                        else:
                            P[state][action][state] += self.rand
        return P

    def calP_diag(self):
        ## This is used to calculate the diagonal stochastic, develop later.
        pass

    def getpolicy(self):
        policy = {}
        for state in self.statespace:
            if state not in self.obstacles:
                policy[state] = {}
                for action in self.action.keys():
                    policy[state][action] = 0.25
        return policy

    def setpolicy(self, policy):
        self.policy = policy

    def getstatetransit(self):
        stP = {}
        for state in self.statespace:
            if state not in stP:
                stP[state] = {}
            for action in self.policy[state].keys():
                for state_ in self.P[state][action].keys():
                    if state_ not in stP[state]:
                        stP[state][state_] = 0
                    stP[state][state_] += self.policy[state][action] * self.P[state][action][state_]
        return stP

    def getstatetransitsingle(self, state, policy):
        stP = {}
        for action in policy.keys():
            for state_ in self.P[state][action].keys():
                if state_ not in stP:
                    stP[state_] = 0
                stP[state_] += policy[action] * self.P[state][action][state_]
        return stP

if __name__ == '__main__':
    env = Env(4,4,0.15)
    print(env.statespace)
    print(len(env.statespace))
    obstacles = [(1, 1), (1, 3)]
    sinkobs = [(2,3)]
    for obs in obstacles:
        env.addobstacle(obs)
    cat1 = agent(env, autopolicy = True)
    for sink in sinkobs:
        env.addsinkobs(sink)
    cat2 = agent(env, autopolicy = True)
    #print (env.obstacles)
    #print (env.sinkobs)
    print (cat1.obstacles)
    print (cat1.sinkobs)
    print (cat2.obstacles)
    print (cat2.sinkobs)
    print(len(cat1.statespace))
    for st in cat1.statespace:
        print(st,cat1.stTransit[st])
        print(st, cat1.P[st]["N"])