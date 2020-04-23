"""
Author Haoxiang Ma

This is used to calculate the semi-mdp part
"""
import numpy as np
from itertools import product
import pickle
import Environment

class MDP:
    def __init__(self, disthre = 0, C = 1, weight = [0.7, 0.3], gamma = 0.85, epsilon = 0.001):
        #Initialize the parameters in mdp
        self.gridworldspace = [] #Calculated via calgridworldsp
        self.statespace = []     #Calculated via calstatespace
        self.policyspace = []    #Get via addpolicy()
        self.policyindex = []    #Get via addpolicy()
        self.actionspace = []    #Calculated via calactionspace()
        self.actionspace_single = {'N':(-1,0),'S':(1,0),'W':(0,-1),'E':(0,1)}
        self.invertedindex = {}  #Calculated via calactionspace()
        self.T = 1               #You can customize this T by setT()
        self.agents = []         #Add agent via addplayer()   You have to first add agent then do other things
        self.gamma = gamma
        self.epsilon = epsilon
        self.disthre = disthre
        self.reward = self.initialReward()  # Expect to read from file
        self.termst = []
        self.penalty = []
        self.cost = C #The information acquire cost for 1 object
        self.weight = weight #in the form of [reward, cost], the weight of reward and cost
        self.transitMatrix = {}   #Call self.cal_Kstep_Matrix to get the k step transition matrix, in the form of P[policy_index][]


    def addplayer(self, agent):
        # Add one player into this MDP
        self.agents.append(agent)

    def setT(self, t):
        self.T = t

    def addpolicy(self, policy):
        # Here,we add the index of the policy into policyindex list
        # and add the policy into the dict
        self.policyspace.append(policy)
        index = len(self.policyspace)
        self.policyindex.append(index)


    def setdisthre(self, thre):
        # Set the distance threshold
        self.disthre = thre

    def calgridworldsp(self):
        # Calculate the statespace
        temp = []
        for agent in self.agents:
            temp.append(agent.statespace)
        statespace = list(product(*temp))
        self.gridworldspace = statespace

    def calstatespace(self):
        # This will return the statespace of (gridworld X policyindex)
        index_total = 0
        invertexindex = {}
        statespace = []
        for index in self.policyindex:
            for st in self.gridworldspace:
                invertexindex[(st, index)] = index_total
                statespace.append((st, index))
                index_total += 1
        self.statespace = statespace
        self.invertedindex = invertexindex
        #return statespace, invertexindex

    def calactionspace(self):
        #If we can plan every step, then it is (policy, 1)
        self.actionspace = []
        for index in self.policyindex:
            for i in range(self.T):
                self.actionspace.append((index, i+1))

    def manhatDis(self, st1, st2, thre):
        # Judge if two states are in the manhattan distance threshold, if in the threshold, return True,
        # else return False
        dis = abs(st1[0] - st2[0]) + abs(st1[1] - st2[1])
        if dis <= thre:
            return True
        else:
            return False

    # def initialReward(self):
    #     R = []
    #     for s in self.statespace:
    #         if s[0][0] in self.penalty and s[0] in self.termst:
    #             if self.manhatDis(s[0][0], s[0][1], self.disthre) and self.manhatDis(s[0][0], s[0][2], self.disthre):
    #                 R.append(180)
    #             else:
    #                 R.append(80)
    #         elif s[0][0] in self.penalty and s[0] not in self.termst:
    #             R.append(-20)
    #         elif s[0] in self.termst and s[0][0] not in self.penalty:
    #             if self.manhatDis(s[0][0], s[0][1], self.disthre) and self.manhatDis(s[0][0], s[0][2], self.disthre):
    #                 R.append(200)
    #             else:
    #                 R.append(100)
    #         else:
    #             R.append(0)
    #     return R
    def initialReward(self):
        R = []
        for s in self.statespace:
            if s[0][0] in self.penalty:
                R.append(-20)
            else:
                R.append(0)
        return R

    def expectedReward_opt(self, R, state, action):
        # Calculate the expected reward when using policy 'pi' for time length 't'
        # Right now, we consider the reward every step is 0, later may change
        # Used in semi-mdp
        policy_index = action[0]
        t = action[1]
        stindex = self.invertedindex[state]
        epreward = R[stindex]
        if t == 1:
            return epreward
        else:
            for i in range(t-1):
                epreward += np.power(self.gamma,(i+1)) * np.dot(R,self.transitMatrix[policy_index][i+1][stindex])
        return epreward

    def expected_sensor_reward(self, policy):
        t = policy[1]
        reward = 0
        if t == 1:
            return reward
        else:
            for i in range(t-1):
                reward += np.power(self.gamma,(i+1)) * 3 # 3 is sensor reward
            return reward

    def expectedCost(self, t, k = 2):
        # Calculate the expected cost when we look into 'k' agent for time 't'
        # First you have to see all agents, then pay attention to some specific
        # Parameter may change later
        epcost = len(self.agents) * self.cost
        if t == 1:
            return epcost
        else:
            for i in range(t - 1):
                epcost += np.power(self.gamma,(i+1)) * k * self.cost
        return epcost

    def transitionMatrix(self, policy_index):
        # This function is used to calculate the transition matrix in 1 step
        length = len(self.statespace)
        transMat = np.zeros((length, length))
        stTranPro_cat1 = self.agents[1].stTransit
        stTranPro_cat2 = self.agents[2].stTransit
        # print (self.termst)
        # Calculate the transition matrix for 1 step
        for i in range(length):
            if self.statespace[i][0] not in self.termst:
                stpolicy = self.aggregatestate(policy_index, self.statespace[i][0])
                stP = self.agents[0].getstatetransitsingle(self.statespace[i][0][0], stpolicy)
                stP_1 = stTranPro_cat1[self.statespace[i][0][1]]
                stP_2 = stTranPro_cat2[self.statespace[i][0][2]]
                for st_ct in stP.keys():
                    for st_cat1 in stP_1.keys():
                        for st_cat2 in stP_2.keys():
                            new_st = ((st_ct, st_cat1, st_cat2), policy_index)
                            index_new = self.invertedindex[new_st]
                            transMat[i][index_new] += stP[st_ct] * stP_1[st_cat1] * stP_2[st_cat2]
            else:
                new_st = (self.statespace[i][0],policy_index)
                index_new = self.invertedindex[new_st]
                transMat[i][index_new] = 1.0
        return transMat

    def cal_Kstep_Matrix(self):
    # This is used to calculate K step transit matrix
        for index in self.policyindex:
            self.transitMatrix[index] = {}
        if self.T == 1:
            for p_index in self.policyindex:
                self.transitMatrix[p_index][1] = self.transitionMatrix(p_index)

        else:
            for i in range(self.T):
                if i == 0:
                    for p_index in self.policyindex:
                        self.transitMatrix[p_index][1] = self.transitionMatrix(p_index)
                else:
                    for p_index in self.policyindex:
                        self.transitMatrix[p_index][i+1] = np.dot(self.transitMatrix[p_index][i],self.transitMatrix[p_index][1])

    def aggregatestate(self, policy_index, state):
        if policy_index == 1:
            tempst = (state[0], state[1])
        else:
            tempst = (state[0], state[2])
        policy =  self.policyspace[policy_index - 1]
        stpolicy = policy[tempst]
        return stpolicy   #stpolicy in the form of P[action] = ??

    def expectedState(self, state, action):
        # Return the possible state
        transitM = self.transitMatrix[action[0]][action[1]]
        P = transitM[self.invertedindex[state]]
        return P  #P is P[i] = ??

    def terminalSt(self):
        terminalst = []
        for state in self.statespace:
            s = state[0]
            st0 = s[0]
            st1 = s[1]
            st2 = s[2]
            if self.manhatDis(st0,st1,self.disthre) or self.manhatDis(st0,st2,self.disthre):
                terminalst.append(s)
        return terminalst

    def setpenalty(self):
        pen = [(2,3)]
        return pen

    def initialValue(self):
        self.termst = self.terminalSt()
        self.penalty = self.setpenalty()
        with open(".\Vision8\evaluateV_sto.pkl", "rb") as f:
            V_single = pickle.load(f)
        V = {s : 0 for s in self.statespace}
        for s in self.termst:
            st0 = s[0]
            st1 = s[1]
            st2 = s[2]
            if st0 in self.penalty:
                if self.manhatDis(st0, st1, self.disthre) and self.manhatDis(st0, st2, self.disthre):
                    V[(s,1)] = 180 * self.weight[0]
                    V[(s,2)] = 180 * self.weight[0]
                else:
                    if self.manhatDis(st0, st1, self.disthre):
                        s_ = (st0, st2)
                        V[(s, 1)] = (100 + V_single[s_]) * self.weight[0]
                        V[(s, 2)] = (100 + V_single[s_]) * self.weight[0]
                    else:
                        s_ = (st0, st1)
                        V[(s, 1)] = (100 + V_single[s_]) * self.weight[0]
                        V[(s, 2)] = (100 + V_single[s_]) * self.weight[0]
            else:
                if self.manhatDis(st0, st1, self.disthre) and self.manhatDis(st0, st2, self.disthre):
                    V[(s,1)] = 200 * self.weight[0]
                    V[(s,2)] = 200 * self.weight[0]
                else:
                    if self.manhatDis(st0, st1, self.disthre):
                        s_ = (st0, st2)
                        V[(s, 1)] = (100 + V_single[s_]) * self.weight[0]
                        V[(s, 2)] = (100 + V_single[s_]) * self.weight[0]
                    else:
                        s_ = (st0, st1)
                        V[(s, 1)] = (100 + V_single[s_]) * self.weight[0]
                        V[(s, 2)] = (100 + V_single[s_]) * self.weight[0]
        return V

    def initialValue_sensor(self):
        V = {s:0 for s in self.statespace}
        return V
    def valueFunc_semi(self, stochastic = False):
        policy = {s: 0 for s in self.statespace}
        V1 = self.initialValue()
        R = self.initialReward()
        itercount = 1
        if not stochastic:
            while True:
                print(itercount)
                itercount += 1
                V = V1.copy()   #Last step
                V_list = []
                for state in self.statespace:
                    V_list.append(V[state])
                delta = 0
                for s in self.statespace:
                    if s[0] not in self.termst:
                        maxV = -100
                        maxaction = None
                        for action in self.actionspace:
                            P = self.expectedState(s, action)
                            # tempV = self.expectedReward_opt(R, s, action)* self.weight[0] + self.expectedCost(action[1]) * self.weight[1] + np.power(self.gamma, action[1])* sum(V[self.statespace[i]] * P[i] for i in range(len(P)))
                            # tempV = self.expectedReward_opt(R, s, action) * self.weight[0] + self.expectedCost(action[1]) * self.weight[1] + np.power(self.gamma, action[1])* np.dot(V_list,P)
                            tempV = self.expectedReward_opt(R, s, action) * self.weight[0] + self.expected_sensor_reward(action) * self.weight[1] + np.power(self.gamma, action[1])* np.dot(V_list,P)
                            # Expected reward + Expected Cost + Expected Value
                            if tempV > maxV:
                                maxV = tempV
                                maxaction = action
                        policy[s] = maxaction
                        V1[s] = maxV
                        delta = max(delta, abs(V1[s] - V[s]))
                    else:
                        if self.manhatDis(s[0][0],s[0][1],self.disthre) and self.manhatDis(s[0][0],s[0][2],self.disthre):
                            policy[s] = "End"
                        elif self.manhatDis(s[0][0],s[0][1],self.disthre):
                            policy[s] = (2, 1)
                        else:
                            policy[s] = (1, 1)

                if delta < self.epsilon * (1-self.gamma) /self.gamma:
                    return V, policy
        else:
            pass

    def valueFunc_single(self, stochastic = True):
        pass

    def policy_evaluation(self, policy, stochastic = False):
        V1 = self.initialValue()
        # V1 = self.initialValue_sensor()
        R = self.initialReward()
        with open(".\Vision8\evaluateV_sto.pkl", "rb") as f:
            V_single = pickle.load(f)
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
                    if s[0] not in self.termst:
                        action = policy[s]
                        P = self.expectedState(s, action)
                            # tempV = self.expectedReward_opt(R, s, action)* self.weight[0] + self.expectedCost(action[1]) * self.weight[1] + np.power(self.gamma, action[1])* sum(V[self.statespace[i]] * P[i] for i in range(len(P)))
                            # tempV = self.expectedReward_opt(R, s, action) * self.weight[0] + self.expectedCost(action[1]) * self.weight[1] + np.power(self.gamma, action[1])* np.dot(V_list,P)
                        V1[s] = self.expectedReward_opt(R, s, action) + np.power(self.gamma, action[1]) * np.dot(V_list, P)
                        # V1[s] = self.expected_sensor_reward(action) + np.power(self.gamma, action[1]) * np.dot(V_list, P)
                            # Expected reward + Expected Cost + Expected Value
                        delta = max(delta, abs(V1[s] - V[s]))
                    else:
                        if s[0][0] not in self.penalty:
                            if self.manhatDis(s[0][0], s[0][1], self.disthre) and self.manhatDis(s[0][0], s[0][2], self.disthre):
                                V1[s] = 200   # This is for catching reward
                            else:
                                if self.manhatDis(s[0][0], s[0][1], self.disthre):
                                    s_= (s[0][0], s[0][2])
                                    V1[s] = 100 + V_single[s_]
                                else:
                                    s_= (s[0][0], s[0][1])
                                    V1[s] = 100 + V_single[s_]
                        else:
                            if self.manhatDis(s[0][0], s[0][1], self.disthre) and self.manhatDis(s[0][0], s[0][2], self.disthre):
                                V1[s] = 180   # This is for catching reward
                            else:
                                if self.manhatDis(s[0][0], s[0][1], self.disthre):
                                    s_= (s[0][0], s[0][2])
                                    V1[s] = 100 + V_single[s_]
                                else:
                                    s_= (s[0][0], s[0][1])
                                    V1[s] = 100 + V_single[s_]

                        # V1[s] = 0    #This is for sensor
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
    cat2 = Environment.agent(env,autopolicy = True)
    with open(".\Vision8\policy_sto.pkl", "rb") as f:
        policy = pickle.load(f)
    mdp = MDP()

    mdp.addplayer(agent1)
    mdp.addplayer(cat1)
    mdp.addplayer(cat2)
    #Test calculate gridworld
    mdp.calgridworldsp()
    # for st in mdp.gridworldspace:
    #     print (st)
    assert len(mdp.gridworldspace) == 2744, 'gridworldspace section is not correct'

    # Test calculate statespace
    mdp.addpolicy(policy)
    mdp.addpolicy(policy)
    mdp.calstatespace()
    assert len(mdp.statespace) == 5488, 'statespace section is not correct'

    # Test calculate actionspace
    mdp.calactionspace()
    assert len(mdp.actionspace) == 2, 'actionspace 2 section is not correct'

    mdp.setT(1)
    mdp.calactionspace()
    assert len(mdp.actionspace) == 2, 'actionspace 8 section is not correct'
    mdp.termst = mdp.terminalSt()
    mdp.penalty = mdp.setpenalty()
    mdp.cal_Kstep_Matrix()
    # stindex = mdp.invertedindex[(((1, 2), (1, 2), (2, 3)), 1)]
    # stindex2 = mdp.invertedindex[(((1, 2), (1, 2), (2, 3)), 2)]
    # P = mdp.transitMatrix[2][2][stindex]
    # if abs(sum(P) -1) > 0.01:
    #     print(123)
    # print(P[stindex])
    # print(P[stindex2])
    V, policy = mdp.valueFunc_semi(stochastic=False)
    filename = ".\Vision8\V_semi_1.pkl"
    file = open(filename, "wb")
    pickle.dump(V, file)
    file.close()
    filename = ".\Vision8\policy_semi_1.pkl"
    file = open(filename, "wb")
    pickle.dump(policy, file)
    file.close()
    # with open(".\Vision4\policy_semi_1.pkl", "rb") as f:
    #     policy = pickle.load(f)
    # V = mdp.policy_evaluation(policy)
    # filename = ".\Vision4\evaluateVcatch_1.pkl"
    # file = open(filename, "wb")
    # pickle.dump(V, file)
    # file.close()
if __name__ == '__main__':
    test()