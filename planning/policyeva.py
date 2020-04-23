import pickle

with open(".\Vision4\evaluateVcatch_1.pkl", "rb") as f:
    V1 = pickle.load(f)
with open(".\Vision4\evaluateVcatch_2.pkl", "rb") as f:
    V2 = pickle.load(f)
with open(".\Vision4\evaluateVcatch_3.pkl", "rb") as f:
    V3 = pickle.load(f)
with open(".\Vision4\evaluateVcatch_4.pkl", "rb") as f:
    V4 = pickle.load(f)
with open(".\Vision3\evaluateVsensor_1.pkl", "rb") as f:
    V5 = pickle.load(f)
with open(".\Vision3\evaluateVsensor_2.pkl", "rb") as f:
    V6 = pickle.load(f)
with open(".\Vision3\evaluateVsensor_3.pkl", "rb") as f:
    V7 = pickle.load(f)
with open(".\Vision3\evaluateVsensor_4.pkl", "rb") as f:
    V8 = pickle.load(f)

state = (((1,0),(3,2),(0,3)),1)


print(V1[state])
print(V2[state])
print(V3[state])
print(V4[state])
print(V5[state])
print(V6[state])
print(V7[state])
print(V8[state])