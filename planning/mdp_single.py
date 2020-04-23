import numpy as np
from itertools import product
import pickle
import Environment

class MDP:
    def __init__(self,  disthre = 0, C = 1, weight = [1, 0], gamma = 0.85, epsilon = 0.001):
        self.statespace = []     #Calculated via calstatespace
        self.actionspace = {}
        self.invertedindex = {}  #Calculated via calactionspace()
        self.agents = []         #Add agent via addplayer()   You have to first add agent then do other things
        self.gamma = gamma
        self.epsilon = epsilon
        self.disthre = disthre
        self.termst = []
        self.penalty = []
        #self.reward = self.getReward(rewardFile)  # Expect to read from file
        self.cost = C #The information acquire cost for 1 object
        self.weight = weight #in the form of [reward, cost], the weight of reward and cost
        self.transitMatrix = {}   #Call self.cal_Kstep_Matrix to get the k step transition matrix, in the form of P[policy_index][]
        self.statetransPro = {}
    def addplayer(self, agent):
        # Add one player into this MDP
        self.agents.append(agent)

    def calstatesp(self):
        # Calculate the statespace
        temp = []
        for agent in self.agents:
            temp.append(agent.statespace)
        statespace = list(product(*temp))
        self.statespace = statespace

    def selectactionsp(self, mode):
        if mode == 1:  #4 connected
            self.actionspace = {'N':(-1,0),'S':(1,0),'W':(0,-1),'E':(0,1)}
        elif mode == 2:    #8 connected, not implmented yet
            self.actionspace = {'N':(-1,0),'S':(1,0),'W':(0,-1),'E':(0,1), 'NE':(-1,1), 'NW':(-1,-1), 'SW':(1,-1), 'SE':(1,1)}

    def manhatDis(self, st1, st2, thre):
        # Judge if two states are in the manhattan distance threshold, if in the threshold, return True,
        # else return False
        dis = abs(st1[0] - st2[0]) + abs(st1[1] - st2[1])
        if dis <= thre:
            return True
        else:
            return False

    def terminalSt(self):
        terminalSt = []
        for st in self.statespace:
            st0 = st[0]
            st1 = st[1]
            if self.manhatDis(st0,st1,self.disthre):
                print(st)
                terminalSt.append(st)
        return terminalSt

    def setpenalty(self):
        pen = [(2,3)]

        return pen

    def initialReward(self):
        R = {s:0 for s in self.statespace}
        for s in self.statespace:
            if s[0] in self.penalty:
                R[s] = -20
        return R

    def initialValue(self):
        self.termst = self.terminalSt()
        self.penalty = self.setpenalty()
        V = {}
        for s in self.statespace:
            if s in self.termst:
                if s[0] not in self.penalty:
                    V[s] = 100
                else:
                    V[s] = 80
            else:
                V[s] = 0
        return V

    def state_transit_pro(self):
        Pro = {}
        for st in self.statespace:
            if st not in self.termst:
                Pro[st] = {}
                st0 = st[0]
                st1 = st[1]
                for act in self.actionspace.keys():
                    Pro[st][act] = {}
                    for st0_ in self.agents[0].P[st0][act].keys():
                        for st1_ in self.agents[1].stTransit[st1].keys():
                            tempst_ = (st0_, st1_)
                            if tempst_ not in Pro[st][act]:
                                Pro[st][act][tempst_] = 0
                            Pro[st][act][tempst_] += self.agents[0].P[st0][act][st0_] * self.agents[1].stTransit[st1][st1_]
            else:
                Pro[st] = {}
                for act in self.actionspace.keys():
                    Pro[st][act] = {}
                    Pro[st][act][st] = 1.0
        return Pro

    def sensor_reward(self, action):
        return 0
    def valueIteration(self, stochastic = True):
        policy = {s : {} for s in self.statespace}
        V1 = self.initialValue()
        R = self.initialReward()
        # print(V1)
        iter_count = 0
        if not stochastic:
            while True:
                iter_count += 1
                print(iter_count)
                V = V1.copy()
                delta = 0
                for s in self.statespace:
                    if s not in self.termst:
                        maxV = -100
                        maxaction = None
                        for action in self.actionspace.keys():
                            tempV = 0.0
                            P = self.statetransPro[s][action]
                            tempV = weight[0] * R[s] + weight[1] * self.sensor_reward(action) + self.gamma * sum(P[s] * V[s] for s in P.keys())
                            if tempV >= maxV:
                                maxV = tempV
                                maxaction = action
                        # print(s, maxV)
                        V1[s] = maxV
                        policy[s][maxaction] = 1.0
                        delta = max(delta, abs(V1[s] - V[s]))
                    else:
                        V1[s] = R[s] * weight[0] + weight[1] * 0
                        for act in self.actionspace.keys():
                            policy[s][act] = 0.001
                if delta < self.epsilon * (1 - self.gamma) / self.gamma:
                    return V, policy
        else:
            while True:
                V = V1.copy()
                iter_count += 1
                print(iter_count)
                delta = 0
                for s in self.statespace:
                    if s not in self.termst:
                        tempV = {}
                        for action in self.actionspace.keys():
                            P = self.statetransPro[s][action]
                            tempV[action]  = R[s] * self.weight[0] + self.gamma * sum(P[s] * V[s] for s in P.keys())
                        tempV_s = sum(np.exp(val) for val in tempV.values())
                        V1[s] = np.log(tempV_s)
                        for action in self.actionspace.keys():
                            policy[s][action] = np.exp(tempV[action])/tempV_s
                        delta = max(delta, abs(V1[s] - V[s]))
                    else:
                        if s[0] in self.penalty:
                            V1[s] = 80
                        else:
                            V1[s] = 100
                        for act in self.actionspace.keys():
                            policy[s][act] = 0.001
                if delta < self.epsilon * (1 - self.gamma) / self.gamma:
                    return V, policy
    def policy_evaluation(self, policy, stochastic = False):
        V1 = self.initialValue()
        R = self.initialReward()
        itercount = 1
        if not stochastic:
            while True:
                print(itercount)
                itercount += 1
                V = V1.copy()  # Last step
                V_list = []
                for state in self.statespace:
                    V_list.append(V[state])
                delta = 0
                for s in self.statespace:
                    if s not in self.termst:
                        tempV = {}
                        for action in self.actionspace.keys():
                            P = self.statetransPro[s][action]
                            tempV[action]  = policy[s][action]*(R[s] * self.weight[0] + self.gamma * sum(P[s] * V[s] for s in P.keys()))
                        V1[s] = sum(tempV.values())
                        delta = max(delta, abs(V1[s] - V[s]))
                    else:
                        if s[0] in self.penalty:
                            V1[s] = 80
                        else:
                            V1[s] = 100
                if delta < self.epsilon * (1 - self.gamma) / self.gamma:
                    return V
        else:
            pass
def test():
    env = Environment.Env(4, 4, 0.15)
    obstacles = [(1, 1), (1, 3)]
    for obs in obstacles:
        env.addobstacle(obs)
    agent1 = Environment.agent(env)
    cat1 = Environment.agent(env,autopolicy = True)
    mdp = MDP()
    mdp.addplayer(agent1)
    mdp.addplayer(cat1)
    mdp.calstatesp()
    mdp.selectactionsp(1)
    mdp.termst = mdp.terminalSt()
    print("mdp termst",mdp.termst)
    mdp.statetransPro = mdp.state_transit_pro()
    print(mdp.manhatDis((0,0),(0,0),0))
    print ("Pro:",mdp.statetransPro[((1, 2),(1, 2))])
    # V, policy = mdp.valueIteration(stochastic=True)
    # filename = ".\Revision0\V_sto.pkl"
    # file = open(filename, "wb")
    # pickle.dump(V, file)
    # file.close()
    # filename = ".\Revision0\policy_sto.pkl"
    # file = open(filename, "wb")
    # pickle.dump(policy, file)
    # file.close()
    with open(".\Revision0\policy_sto.pkl", "rb") as f:
        policy = pickle.load(f)
    V = mdp.policy_evaluation(policy)
    filename = ".\Revision0\evaluateV_sto.pkl"
    file = open(filename, "wb")
    pickle.dump(V, file)
    file.close()
if __name__ == '__main__':
    test()