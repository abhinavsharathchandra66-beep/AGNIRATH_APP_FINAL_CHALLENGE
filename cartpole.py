import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()

dt = 0.02
class PID_controller:
    def __init__(self,kp,ki,kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.I = 0

    def compute (self,error,dt):
         
         self.I  = self.I + error*dt
         de = (error - self.prev_error)/dt
         self.prev_error = error
         u = self.kp*error + self.ki*self.I + self.kd*de
         return u
pid = PID_controller(kp =73, ki = 13, kd =7)
while True:
    error = observation[2]
    u = pid.compute(error, dt)
    action = 1 if u > 0 else 0 
    observation, reward, done, _, _ = env.step(action)
    if done:
        observation, info = env.reset()
       












