__author__ = 'Juxi Leitner, juxi@idsia.ch'

from scipy import random
from scipy.integrate import ode
from math import sqrt, cos, sin
from numpy import array
import pylab as p
from pybrain.rl.environments.environment import Environment


##def dXdt(t, X):
##    """ Return the dxdt's """
##    x, vx, y, vy, theta, omega = X
##    ul, ur = 0.0, 0.0
##
##    nu = 0.08
##    mR = (1.5 * 0.5);
###           if(prob->max_noise != 0.0) {
###               noise_x = (2 * prob->max_noise * drng()) - prob->max_noise;
###               noise_y = (2 * prob->max_noise * drng()) - prob->max_noise;
###       //      std::cout << std::endl << "Noise Test:" << noise_x << ", " << noise_y << std::endl << std::endl;
##
##    noise_x, noise_y = 0.0, 0.0
##    return array([ vx,
##                   2 * nu * vy + 3 * nu * nu * x + (ul + ur) * cos(theta) + noise_x,
##                   vy,
##                   -2 * nu * vx + (ul + ur) * sin(theta) + noise_y, 
##                   omega, 
##                   (ul - ur) * 1/mR ])  
##

class DockingEnvironment(Environment):
    """
    Simulates an ocean going ship with substantial inertia in both forward
    motion and rotation, plus noise.
    Docking problem, aimed to be used with ANNs to develop a robust controller for spacecraft docking!

    State space (continuous):
        h       heading of ship in degrees (North=0)
        hdot    angular velocity of heading in degrees/minute
        v       velocity of ship in knots
    Action space (continuous):
        rudder  angle of rudder
        thrust  propulsion of ship forward
    """

    # some (more or less) physical constants
    dt = 0.1        # simulated time (in seconds) per step
    vicinity_distance    = 0.1
    vicinity_speed       = 0.1
    vicinity_orientation = 0.2
#    mass = 1000.   # mass of ship in unclear units
#    I = 1000.      # rotational inertia of ship in unclear units

    def __init__(self):
        # initialize the environment (randomly)
        self.delay = False
        self.state = None
        self.start_cnd = [-2.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#        self.start_cnd = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        self.solver = None
        self.reset()

    def dX(self, t, X):
        """ Return the dxdt's """
        x, vx, y, vy, theta, omega = X
        ul, ur = self.action[0], self.action[1]

#        print 'X', self.t, self.action
        nu = 0.08
        mR = (1.5 * 0.5);
    #           if(prob->max_noise != 0.0) {
    #               noise_x = (2 * prob->max_noise * drng()) - prob->max_noise;
    #               noise_y = (2 * prob->max_noise * drng()) - prob->max_noise;
    #       //      std::cout << std::endl << "Noise Test:" << noise_x << ", " << noise_y << std::endl << std::endl;

        noise_x, noise_y = 0.0, 0.0
        return array([ vx,
                       2 * nu * vy + 3 * nu * nu * x + (ul + ur) * cos(theta) + noise_x,
                       vy,
                       -2 * nu * vx + (ul + ur) * sin(theta) + noise_y, 
                       omega, 
                       (ul - ur) * 1/mR ])  


    def reset(self):
        """ re-initializes the environment, setting the ship to
            a ...
        """
        self.t = 0
        # two outputs: the thrusters, u_r and u_l and stop neuron
        self.action = [0.0, 0.0, 0.0]
        #                x,  vx, y,  vy, theta, omega
        # self.state = [2.0, 0.0, 2.0, 0.0, 0.0, 0.0]
        self.state = self.start_cnd
        x, vx, y, vy, theta, omega = self.state
#        print x, self.state
        self.init_distance = self.getDistance()
        
        self.solver = ode(self.dX)
        self.solver.set_integrator('dopri5')        
        self.solver.set_initial_value(self.state, self.t)
        
        
        
##        while solver.successful() and result < 4.0 and t[-1]<100.0:
##             t.append(t[-1]+dt)
##             solver.integrate(t[-1])
##             f.append(solver.y)
##             result = f[-1][0]        


    def step(self):
        """ integrate state using odeint from numpy """
        self.solver.integrate(self.t)
        self.state = self.solver.y


    def getDistance(self):
        """ auxiliary access to distance from the target position """
        return sqrt(self.state[0] * self.state[0] + self.state[2] * self.state[2])


    def getSpeed(self):
        return sqrt(self.state[1] * self.state[1] + self.state[3] * self.state[3])


    def getOrientation(self):
        return self.state[4]


    def getState(self):
        """ returns the state one step (dt) ahead in the future. stores the state in
            self.sensors because it is needed for the next calculation.
        """
        return self.state


    def getSensors(self):
        return self.getState()


    def performAction(self, action):
        """ stores the desired action for the next time step.
        """
        self.action = action
        self.t += self.dt        
        self.step()

#       dxdt[0] = vx;
#       dxdt[1] = 2 * nu * vy + 3 * nu * nu * x + (ul + ur) * cos(theta) + noise_x
#       dxdt[2] = vy;
#       dxdt[3] = -2 * nu * vx + (ul + ur) * sin(theta) + noise_y
#       dxdt[4] = omega;
#       dxdt[5] = (ul - ur) * 1/mR


#    @threaded()
#   def updateRenderer(self):
#       self.updateDone = False
#       if not self.updateLock.acquire(False): return
#
#       # Listen for clients
#       self.server.listen()
#       if self.server.clients > 0:
#           # If there are clients send them the new data
#           self.server.send(self.sensors)
#       sleep(0.02)
#       self.updateLock.release()
#       self.updateDone = True

    @property
    def indim(self):
        return len(self.action)

    @property
    def outdim(self):
        return len(self.state)


