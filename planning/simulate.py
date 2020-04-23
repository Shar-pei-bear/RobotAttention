import Environment
import numpy as np
import pickle
import mdp_semi

def manhatDist(st1, st2, thre = 0):
    dis = abs(st1[0] - st2[0]) + abs(st1[1] - st2[1])
    if dis <= thre:
        return True
    else:
        return False

def aggregatestate(state, policy_index):
    if policy_index == 1:
        tempst = (state[0], state[1])
    else:
        tempst = (state[0], state[2])
    return tempst

def randomchoose(dict):
    keylist = []
    valuelist = []
    for key in dict.keys():
        keylist.append(key)
        valuelist.append(dict[key])
    choice_index = np.random.choice(len(keylist), 1, p = valuelist)[0]
    choice = keylist[choice_index]
    return choice

def simulate_3(agent, cat1, cat2, state, policy, policy_act):
    #Assuming the agent pay attention to the first cat in initial state
    agent_state = state[0]
    cat1_state = state[1]
    cat2_state = state[2]
    state_semi = (state, 1)
    policy_st = policy[state_semi]
    stepcount = 0
    penaltycount = 0
    sensorreward = 0
    traj = []
    traj.append(state)
    while True:
        if agent_state == (2, 3):
            penaltycount += 1
        policy_index = policy_st[0]
        policy_duration = policy_st[1]
        tempst = aggregatestate(state, policy_index)
        temppolicy = policy_act[tempst]
        action = randomchoose(temppolicy)
        agent_nextstate = randomchoose(agent.P[agent_state][action])
        cat1_nextstate = randomchoose(cat1.stTransit[cat1_state])
        cat2_nextstate = randomchoose(cat2.stTransit[cat2_state])
        policy_duration -= 1
        stepcount += 1
        state = (agent_nextstate, cat1_nextstate, cat2_nextstate)
        traj.append(state)
        if manhatDist(agent_nextstate,cat1_nextstate) and manhatDist(agent_nextstate,cat2_nextstate):
            return stepcount, traj, sensorreward, penaltycount
        elif manhatDist(agent_nextstate,cat1_nextstate):
            step_next, traj_2, penaltycount_2 = simulate_2(agent, cat2, agent_nextstate, cat2_nextstate, policy_act)
            traj.extend(traj_2)
            total_step = step_next + stepcount
            penaltycount += penaltycount_2
            return total_step, traj, sensorreward, penaltycount
        elif manhatDist(agent_nextstate,cat2_nextstate):
            step_next, traj_2, penaltycount_2 = simulate_2(agent, cat1, agent_nextstate, cat1_nextstate, policy_act)
            traj.extend(traj_2)
            total_step = step_next + stepcount
            penaltycount += penaltycount_2
            return total_step, traj, sensorreward, penaltycount
        else:
            if policy_duration == 0:
                new_state_semi = (state, policy_index)
                policy_st = policy[new_state_semi]
                agent_state = agent_nextstate
                cat1_state = cat1_nextstate
                cat2_state = cat2_nextstate
            else:
                policy_st = (policy_index, policy_duration)
                agent_state = agent_nextstate
                cat1_state = cat1_nextstate
                cat2_state = cat2_nextstate
                sensorreward += 1

def simulate_2(agent, cat, agent_state, cat_state, policy_act):
    stepcount = 0
    traj = []
    # cost = 0
    penaltycount = 0
    state = (agent_state, cat_state)
    traj.append(state)
    while True:
        if agent_state == (2, 3):
            penaltycount += 1
        temppolicy = policy_act[state]
        action = randomchoose(temppolicy)
        # cost += 2
        # print(agent.P[agent_state][action])
        agent_nextstate = randomchoose(agent.P[agent_state][action])
        cat_nextstate = randomchoose(cat.stTransit[cat_state])
        stepcount += 1
        state = (agent_nextstate, cat_nextstate)
        traj.append(state)
        if manhatDist(agent_nextstate, cat_nextstate):
            return stepcount, traj, penaltycount
        else:
            agent_state = agent_nextstate
            cat_state = cat_nextstate



def test():
    env = Environment.Env(4, 4, 0.15)
    obstacles = [(1, 1), (1, 3)]
    for obs in obstacles:
        env.addobstacle(obs)
    agent1 = Environment.agent(env)
    cat1 = Environment.agent(env, autopolicy=True)
    cat2 = Environment.agent(env, autopolicy=True)
    #Choose Policy
    with open(".\Revision1\policy_sto.pkl", "rb") as f:
        policy_act = pickle.load(f)

    with open(".\Revision1\policy_semi_2.pkl", "rb") as f:
        policy = pickle.load(f)

    agent_ini = (1,0)
    cat1_ini = (3,2)
    cat2_ini = (0,3)
    state = (agent_ini, cat1_ini, cat2_ini)
    steplist = []
    trajlist = []
    sensorrewardlist = []
    penaltylist = []
    # for i in range(1000):
    #     stepcount, traj, penaltycount = simulate_2(agent1, cat1, agent_ini, cat1_ini, policy_act)
    #     steplist.append(stepcount)
    #     trajlist.append(trajlist)
    #     # costlist.append(cost)
    #     penaltylist.append(penaltycount)
    for i in range(1000):
        stepcount, traj, sensorreward, penaltycount = simulate_3(agent1, cat1, cat2, state, policy, policy_act)
        steplist.append(stepcount)
        trajlist.append(trajlist)
        sensorrewardlist.append(sensorreward)
        penaltylist.append(penaltycount)
    filename = ".\Revision1\steplist2.pkl"
    file = open(filename, "wb")
    pickle.dump(steplist, file)
    file.close()
    filename = ".\Revision1\\trajlist2.pkl"
    file = open(filename, "wb")
    pickle.dump(trajlist, file)
    file.close()
    print(sum(steplist)/1000)
    print(sum(sensorrewardlist)/1000)
    print(sum(penaltylist)/1000)
if __name__ == "__main__":
    test()