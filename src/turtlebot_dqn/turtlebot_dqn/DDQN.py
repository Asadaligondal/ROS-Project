
import sim as vrep
import numpy as np
import sys
import time
import math
import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from collections import deque
import time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard
from numpy import asarray
from numpy import savetxt

from collections import deque
import time
import random
import os
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.models import model_from_json
REPLAY_MEMORY_SIZE = 20000
MIN_REPLAY_MEMORY_SIZE = 400
MINIBATCH_SIZE = 64
UPDATE_TARGET_EVERY = 7 # it is to check the condition for updating the target_model after every 5 steps
DISCOUNT = 0.99
n=20
actions = [[2,0], [0,2], [4,4]]
act = [0, 1, 2]

vrep.simxFinish(-1) # just in case, close all opened connections
clientID=vrep.simxStart('127.0.0.1',19997,True,True,5000,5)

err_code,l_motor_handle = vrep.simxGetObjectHandle(clientID,"leftMotor", vrep.simx_opmode_blocking)
err_code,r_motor_handle = vrep.simxGetObjectHandle(clientID,"rightMotor", vrep.simx_opmode_blocking)
err_code,ps_handle = vrep.simxGetObjectHandle(clientID,"Proximity_sensor", vrep.simx_opmode_blocking)
err_code,BR = vrep.simxGetObjectHandle(clientID,"Cuboid", vrep.simx_opmode_blocking)
err_code, goal_handle = vrep.simxGetObjectHandle(clientID,"Goal0", vrep.simx_opmode_blocking)
err_code, Walls = vrep.simxGetObjectHandle(clientID,"Walls", vrep.simx_opmode_blocking)

class DQNAgent:
    def __init__(self): # when object is made it is the first function which is called.

        #main model of dqn, it gets traned or fit every single step
        self.model = self.create_model()
        #then we have an equelent target model. we .predict with this model on every single step, remember the Q dash value in q-learning
        self.target_model = self.create_model()
        # the process of updating of the weigts of target network happens after a fixed steps that we can set.
        self.target_model.set_weights(self.model.get_weights()) # we are updating the weigths of target nn just by copying the main nn.
        # now in the following we create a replay memeory of like 50,000 steps then we take randome batchs of it to train the main nn
        self.replay_memory = deque(maxlen = REPLAY_MEMORY_SIZE)# it makes our model to stabilize, rather then fitting to a single step at a time, we take
        #randome batch and train on it
        self.target_update_counter = 0
        self.curiosity_model = self.curiosity_model()
        self.param =[]
        
    def create_model(self):
        model = Sequential()
        
        model.add(Dense(200, input_shape = (200,)))
        model.add(Activation("relu"))

        model.add(Dense(64))
        model.add(Activation("relu"))

        model.add(Dense(64))
        model.add(Activation("relu"))

        model.add(Dense(3, activation="linear"))
        model.compile(loss="mse",optimizer=Adam(lr=0.002), metrics=['accuracy'])
        return model
    
    def curiosity_model(self):
        
        model = Sequential()
        
        model.add(Dense(201, input_shape = (201,)))
        model.add(Activation("relu"))
        
        model.add(Dense(64))
        model.add(Activation("relu"))
        
        model.add(Dense(200, activation="linear"))
        
        model.compile(loss="mse",optimizer=Adam(lr=0.002), metrics=['accuracy'])
        
        return model
    
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    
    def get_qs(self, state):
        return self.model.predict(np.array([state]))#[-1] # model.predict return a list of elements
    
    
    
    def train(self, terminal_state):
        if len(self.replay_memory)< MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE) # randome samling from replay memeory
        current_states = np.array([transition[0] for transition in minibatch]) # /255,, dont need that in my code, this gives currnt state so...
        actions = np.array([transition[1] for transition in minibatch])
        length = len(current_states)
        pair_ =[]
        
        for i in range(length):
            state_ = current_states[i]
            state_ = np.array(state_).reshape(200,)
            state_ = state_.tolist()
            if actions[i].tolist() == [ 2, 0]:
                action = 0
            elif actions[i].tolist() == [0, 2]:
                action = 1
            elif actions[i].tolist() == [ 4, 4]:
                action = 2
            state_.append(action)
            pair_.append(state_)
        current_qs_list = self.model.predict(current_states)
        new_current_states = np.array([transition[3] for transition in minibatch])# /255
        Predicted_Values = self.curiosity_model.predict(pair_)
        future_qs_list = self.target_model.predict(new_current_states)
        
        X= [] # features,, in other words these are the states from the environment,, i dont know maybe the sensor readings
        Y= [] # this is for the actions, like foward, left or right
        SUM = 0
        
        for i, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[i]) #
                
                Intrensic_Reward = Predicted_Values[i] - new_current_state
               
                max_future_q = np.max(future_qs_list[i]) 
                Intrensic_Reward = sum(Intrensic_Reward)/len(Intrensic_Reward)
                if Intrensic_Reward <0:
                    Intrensic_Reward = Intrensic_Reward*-1
                SUM =SUM + Intrensic_Reward
                new_q = Intrensic_Reward + reward + DISCOUNT * max_future_q #this comes from the main equation of DQN...
            else:
                new_q = reward # when episdoe is done we dont have any futre q so the q value will be simply equal to the reward
            current_qs = current_qs_list[i]
            
            if action == [ 2, 0]:
                action = 0
            elif action  == [0, 2]:
                action = 1
            elif action == [ 4, 4]:
                action = 2
                
            current_qs[action] = new_q
            X.append(current_state) # inputs to nn
            Y.append(current_qs) # outputs of nn
            
        self.param.append(SUM)
        self.curiosity_model.fit(np.array(pair_), np.array(new_current_states), batch_size = MINIBATCH_SIZE, verbose=0, shuffle=False if terminal_state else None)
        self.model.fit(np.array(X), np.array(Y), batch_size = MINIBATCH_SIZE, verbose=0, shuffle=False if terminal_state else None) # callbacks=[self.tensorboard]
        
        if terminal_state:
            self.target_update_counter +=1
        if self.target_update_counter> UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
        return self.model




def getdisp():
    err_code,Goal= vrep.simxGetObjectPosition(clientID, goal_handle, -1, vrep.simx_opmode_streaming)
    #print(Goal)
    #time.sleep(1)
    err_code,resultingState= vrep.simxGetObjectPosition(clientID, BR, -1, vrep.simx_opmode_streaming)
    x = Goal[0]-resultingState[0]
    y = Goal[1]-resultingState[1]
    z = np.sqrt(x*x+y*y)
    return z


def getB():
    err_code,resultingState= vrep.simxGetObjectPosition(clientID, BR, -1, vrep.simx_opmode_streaming)
    err_code, Angles=vrep.simxGetObjectOrientation(clientID,BR, -1,vrep.simx_opmode_streaming)
    xangle = math.degrees(Angles[2])
    err_code,Goal= vrep.simxGetObjectPosition(clientID, goal_handle, -1, vrep.simx_opmode_streaming)
    theta = (Goal[1] - resultingState[1])/(Goal[0] - resultingState[0]+0.000000001)
    Theta = math.degrees(math.atan(theta))
        
    if Theta<0:
        if xangle<0:
            xangle = xangle*-1
        B = ((180+Theta) - xangle)
    else:    
        B = Theta - xangle
       # print(B)
    return int(B)

def distance():
    err_code,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=vrep.simxReadProximitySensor(clientID,ps_handle,vrep.simx_opmode_streaming)
    sensor_val = np.linalg.norm(detectedPoint)
    
    return (100*sensor_val)

def collision():

    sensor_val = distance()
    
    if int(sensor_val)<28 and int(sensor_val)>0:
        return True
    
#    B = getB()
#    if B <-90 or B> 90:
#        
#        return True
    
    else:
        return False
    
def RLagent(action, agentPosition):
    B = getB()
    if action == [0,2] or action == [2,0]:
        
        while True:
            err_code=vrep.simxStartSimulation(clientID,vrep.simx_opmode_oneshot)
            err_code = vrep.simxSetJointTargetVelocity(clientID,l_motor_handle,action[0],vrep.simx_opmode_streaming)
            err_code = vrep.simxSetJointTargetVelocity(clientID,r_motor_handle,action[1],vrep.simx_opmode_streaming)
            if getB()>B+5:
                err_code=vrep.simxPauseSimulation(clientID,vrep.simx_opmode_oneshot)
                break
            elif getB()<B-5:
                err_code=vrep.simxPauseSimulation(clientID,vrep.simx_opmode_oneshot)
                break
            elif collision()==True:
                break
    if action ==[4,4]:
        err_code = vrep.simxSetJointTargetVelocity(clientID,l_motor_handle,action[0],vrep.simx_opmode_streaming)
        err_code = vrep.simxSetJointTargetVelocity(clientID,r_motor_handle,action[1],vrep.simx_opmode_streaming)
        time.sleep(0.13)
    err_code=vrep.simxPauseSimulation(clientID,vrep.simx_opmode_oneshot)
    resultingState = int(getB())
    reward = reward_(resultingState)
    return resultingState, reward, collision(), reachedGoal()
    

def reward_(angle):
    if collision():
        reward = -100
        return reward
    
    #if reachedGoal():
     #   reward = 100
     #   return reward
    else:
        reward = 0.0#(0.1 + 1/getdisp())
    return reward

def reachedGoal():
    if 0.1< getdisp() <0.46:
            print("Reached the goal")
            return True
        
agent = DQNAgent()
err_code,Goal= vrep.simxGetObjectPosition(clientID, goal_handle, -1, vrep.simx_opmode_streaming)
if __name__ == '__main__':
    totalIntrensic = []
    steps = []
    totalRewards = []
    EPS = 1.0
    t=0
    allowed_steps = 400
    b=0
    numGames = 1000
    speed = [[2,-2], [-2,2]]
    coordinates = [[-0.86849, -1.8082, 0.068001], [1.1565, -1.758, 0.068001], [1.1065, -0.40820 , 0.068001], [0.10651 , -0.083200 , 0.068001]]
    ind = [0,1,2,3]
    
    
    for i in range(numGames):
        episode_intrensic = 0
        Ind = np.random.choice(ind)
#        vrep.simxSetObjectPosition(clientID, BR, -1 , coordinates[Ind] ,vrep.simx_opmode_oneshot)
        #time.sleep(3)
        #err_code=vrep.simxStartSimulation(clientID,vrep.simx_opmode_oneshot)
        
        current_state = getB()
        r = -1
        episode_reward = 0
        done = False
        f = False
        k = 0
        c = False
        act1 = [0,1]
        GoalDone =False
        randind = np.random.choice(act1)
        actio = speed[randind]
        err_code = vrep.simxSetJointTargetVelocity(clientID,l_motor_handle,actio[0],vrep.simx_opmode_streaming)
        err_code = vrep.simxSetJointTargetVelocity(clientID,r_motor_handle,actio[1],vrep.simx_opmode_streaming)
        #err_code=vrep.simxStartSimulation(clientID,vrep.simx_opmode_oneshot)
        time.sleep(0.1)
        err_code = vrep.simxSetObjectPosition(clientID, BR, -1 , coordinates[Ind] ,vrep.simx_opmode_oneshot)
        
        while not done:
            err_code=vrep.simxStartSimulation(clientID,vrep.simx_opmode_oneshot)
            time.sleep(0.2)
            if f == True:
                err_code,string_signalValue=vrep.simxGetStringSignal(clientID,"Data",vrep.simx_opmode_streaming)    
                string_unpackedData=vrep.simxUnpackFloats(string_signalValue)
                if np.shape(string_unpackedData) != (0,):
                    c = True
                    STATE = np.array(string_unpackedData).reshape(200,3)
                    onerow = []
                    for i in range(200):
                        x,y,z = STATE[i,:]
                        d = np.sqrt(x*x+y*y+z*z)
                        onerow.append(d)
#                    onerow.append(getdisp())
                    STATE = np.array(onerow).reshape(200,)
                    state = np.array(onerow).reshape(200,)
                    state = state.tolist()
                    A=agent.model.predict(np.array([STATE]))
                    #print(A[0])
                    rand = np.random.random()
                    randind = np.random.choice(act)
                    A = agent.get_qs(STATE)
                    if r< numGames-100:
                        if rand< (1-EPS):
                            action = np.argmax(agent.get_qs(STATE))
                            action = actions[action]
                            if action == [2, 0]:
                                print("RIGHT")
                            elif action == [0, 2]:
                                print("LEFT")
                            else:
                                print("FORWARD")
                        
                        else:
                            randind = np.random.choice(act)
                            action= actions [randind]
                    else:
                        action = np.argmax(agent.get_qs(STATE))
                        action = actions[action]
                    new_state, reward, done,GoalDone = RLagent(action, current_state)
                    episode_reward +=reward
                    current_state = new_state
                   # print(action)
                    if action == [ 2, 0]:
                        state.append(0)
                    elif action == [0, 2]:
                        state.append(1)
                    elif action == [ 4, 4]:
                        state.append(2)
                    state = np.array(state).reshape(201,)
                    Predicted_Value = agent.curiosity_model.predict(np.array([state]))
#                   
                    Predicted_Value = Predicted_Value[0]
                    if action == [2, 0]:
                        print("RIGHT")
                    elif action == [0, 2]:
                        print("LEFT")
            r+=1
            if f == True:
                err_code,string_signalValue=vrep.simxGetStringSignal(clientID,"Data",vrep.simx_opmode_streaming)    
                string_unpackedData=vrep.simxUnpackFloats(string_signalValue)
                if np.shape(string_unpackedData) != (0,) and c == True:
                    STATE_ = np.array(string_unpackedData).reshape(200,3)
                    onerow = []
                    for i in range(200):
                        x,y,z = STATE_[i,:]
                        d = np.sqrt(x*x+y*y+z*z)
                        onerow.append(d)
#                    onerow.append(getdisp())
#                    STATE_ = np.array(onerow).reshape(75,)
                    actual_state = np.array(onerow).reshape(200,)
                    STATE_ = np.array(onerow).reshape(200,)
                    Intrensic_Reward = Predicted_Value - actual_state
                    Intrensic_Reward = Intrensic_Reward
                    
                    Intrensic_Reward = sum(Intrensic_Reward)/len(Intrensic_Reward)
                    if Intrensic_Reward <0:
                        Intrensic_Reward = Intrensic_Reward*-1
                        
                    agent.update_replay_memory((STATE, action, reward, STATE_, done))
                    episode_intrensic +=Intrensic_Reward
            dqn = agent.train(True)
            if r == allowed_steps:
                #vrep.simxStopSimulation(clientID,vrep.simx_opmode_oneshot)
                err_code=vrep.simxPauseSimulation(clientID,vrep.simx_opmode_oneshot)
                break
            
            
            if done:
                print("collided", t, "times")
#                vrep.simxStopSimulation(clientID,vrep.simx_opmode_oneshot)
                err_code=vrep.simxPauseSimulation(clientID,vrep.simx_opmode_oneshot)
                time.sleep(0.02)
                b= b+1
                t+=1
                break
            
            
            if GoalDone:
                print("Reached the Goal")
#                vrep.simxStopSimulation(clientID,vrep.simx_opmode_oneshot)
                err_code=vrep.simxPauseSimulation(clientID,vrep.simx_opmode_oneshot)
                time.sleep(0.08)
                t+=1
                break
            
            if k ==2:
                f = True
                k = 0
            k+=1
            
        if EPS - 1 / 2000 > 0:
            EPS -= 1 / 2000
        else:
            EPS = 0
        totalIntrensic.append(episode_intrensic)
        totalRewards.append(episode_reward)
        steps.append(r)
        param =  agent.param
        
    savetxt('totalIntrensic.csv', totalIntrensic, delimiter=',')
    savetxt('Param.csv', param, delimiter=',')
    savetxt('totalsimrewardslidar.csv', totalRewards, delimiter=',')
    savetxt('simulationstepslidar.csv', steps, delimiter=',')
    plt.plot(totalRewards)
    plt.show()
    plt.plot(steps)
    plt.show()
    plt.plot(param)
    plt.show()
    plt.plot(totalIntrensic)
    plt.show()
    model_json = dqn.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    dqn.save_weights("DQNLIDARSIMv1.h5")
    print("Saved model to disk")

    