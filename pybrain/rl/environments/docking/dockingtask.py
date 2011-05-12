__author__ = 'Juxi Leitner, juxi@idsia.ch'

from pybrain.rl.environments import EpisodicTask
from docking import DockingEnvironment
from math import fabs

class DockingTask(EpisodicTask):

    __name__ = "Docking Task"
    
    logBuffer = None
    logging   = False
    logfileName = None
    timeNeuron_Threshold = 0.99

    """ The task of ... """
    def __init__(self, env = None, maxtime = 25, timestep = 0.1, logging = False):
        """
        :key env: (optional) an instance of a DockingEnvironment (or a subclass thereof)
        :key maxtime: (optional) maximum number per task (default: 25)
        """
        if env == None:
            env = DockingEnvironment()
        EpisodicTask.__init__(self, env)
        self.maxTime = maxtime
        self.dt = timestep
        self.env.dt = self.dt
        self.t = 0
        self.logging = logging   
        self.logfileName = 'logging_' + str() + '.txt' 

        # actions:               u_l,           u_r
        self.actor_limits = [(-0.1, +0.1), (-0.1, +0.1), (0.0, 1.0)]

        self.lastFitness = 0.0 
        self.bestFitness = 0.0        

    def reset(self):
        EpisodicTask.reset(self)
        self.t = 0
        self.lastFitness = 0.0
        self.bestFitness = 0.0
        self.appendLog() # write first line!

    def performAction(self, action):
        EpisodicTask.performAction(self, action)
        self.t += self.dt
        self.appendLog()

    def isFinished(self):
        #  TODO query self action!? time neuron that is
        if self.t + self.dt >= self.maxTime or self.env.action[2] >= self.timeNeuron_Threshold :
            # maximal timesteps
#            self.appendLog("docking::objfun_: return value:  %f\n", average);
            self.writeLog()
            return True
        # stop neuron!
        return False
        
    def getTotalReward(self):
        """ Return the accumulated reward since the start of the episode """
        return self.lastFitness
#        return self.bestFitness

    def getReward(self):
#   def evaluateFitness(self):
        """getReward, returns the fitness at the current time"""
        fitness = 0.0
        distance = self.env.getDistance()
        speed    = self.env.getSpeed()
        theta    = self.env.getOrientation()

        ## implementation 101
        timeBonus = (self.maxTime - self.t)/self.maxTime
        alpha = 1.0/((1+distance)*(1+fabs(theta))*(speed+1));
        if distance < 0.5*self.env.init_distance :
            if(distance   < self.env.vicinity_distance and
               abs(theta) < self.env.vicinity_orientation and
               speed      < self.env.vicinity_speed ):
                fitness = 1 + timeBonus;    
            else:
                fitness = alpha;
        else: fitness = 0
        self.lastFitness = fitness
        if fitness > self.bestFitness : 
            self.bestFitness = fitness 

        return fitness


    def clearLog(self):
        """clear the current logbuffer"""
        self.logBuffer = None
	
    def appendLog(self):
        """append the current state to the logbuffer"""
        if self.logBuffer == None :
            self.logBuffer = "Some header\nhere\n\n"
            self.logBuffer += "\tx\ty\ttheta : ul\tur\tt-neur\n";
        
        self.logBuffer += '%2.1f: %2.6f\t %2.6f\t %2.6f : ' % \
	           ( self.t, self.env.state[0], self.env.state[2], self.env.state[4] )
        self.logBuffer += '%1.3f\t %1.3f \t%1.2f \t' % \
               ( self.env.action[0], self.env.action[1], self.env.action[2] )
        self.logBuffer += 'Dst/Theta/Speed: \t%f\t%f\t%f \tF: %.2f \n' % \
               ( self.env.getDistance(), self.env.getOrientation(), self.env.getDistance(), self.getReward() )

            
    def writeLog(self):
        """write the state of the current task into a logfile"""
        if self.logBuffer != None and self.logging :
            f = open(self.logfileName, 'w')
            self.logBuffer += "Final Fitness: %f\n"  % self.getTotalReward()
            self.logBuffer += "\n"
            f.write(self.logBuffer)
            f.close()
        
#    def setMaxLength(self, n):
#        self.N = n

