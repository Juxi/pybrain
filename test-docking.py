#!/usr/bin/env python
"""
Trying for docking:
"""
__author__ = 'Juxi Leitner, juxi@idsia.ch'

from scipy import array
from pybrain.optimization import * # ?!?!
from pybrain.structure.parametercontainer import ParameterContainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import LinearLayer

from pybrain.rl.environments.docking import DockingEnvironment
from pybrain.rl.environments.docking import DockingTask

from pybrain.rl.agents import OptimizationAgent
from pybrain.rl.experiments import EpisodicExperiment

# ------------------------
# ---- Read Params ----
# ------------------------



# ------------------------
# ---- Initialization ----
# ------------------------
# all other black-box optimizers can be user in the same way.
# Try it yourself, change the following line to use e.g. GA, CMAES, MemeticSearch or NelderMead
algo = GA
#algo = CMAES

# a very similar interface can be used to optimize the parameters of a Module
# (here a neural network controller) on an EpisodicTask
task = DockingTask(maxtime = 20, logging = False)
#task = MySimpleT(maxtime = 20, logging = False)#False)

ann  = buildNetwork(task.outdim, 10, task.indim, hiddenclass=LinearLayer, bias=True)


l = algo(task, ann, 
         populationSize = 5,
         #crossoverRate = ,
         elitism = True,
         _eliteSize = 1,
         storeAllEvaluations = True,
         storeAllEvaluated = False) # , maxEvaluations = 
l.minimize = False       # we need to minimize our cost function
plotting = False

#l.populationSize = 5
# l.eliteSize = 1

#agent = OptimizationAgent(ann, l)
#exp = EpisodicExperiment(task, agent)

print 'Task: ', task.__name__, '(in=', task.indim, 'out=' , task.outdim, ')'
print 'Algorithm:', algo.__name__
# l = algo(f, maxEvaluations = 20)
# b) desiredValue #l = algo(desiredEvaluation = 10) #print l.learn(), ': fitness below 10 (we minimize the function).'
# c) maximal number of learning steps #l = algo(f, maxLearningSteps = 25)
# Finally you can set storage settings and then access all evaluations made
# during learning, e.g. for plotting: #l = algo(f, x0, storeAllEvaluations = True, 
#  storeAllEvaluated = True, maxEvaluations = 150)

#print l.populationSize

print 'Finished Initialization!'


# exp.doEpisodes(100)
# 
# print 'Episodes learned from:', len(l._allEvaluations)
# n, fit = l._bestFound()
# print 'Best fitness found:', fit
# print 'with this network:'
# print n
# print 'containing these parameters:'
# print fListToString(n.params, 4)
# sys.exit(1) 

# -----------------------
# ----   Learning    ----
# -----------------------
print 'Start Learning!'


task.logging = False
#### print l.learn(5)
while True :
    task.logging = False
    print l.learn(10), 'after', l.numEvaluations/l.populationSize/11, 'epochs' #'generations.'

    if (l.numEvaluations/l.populationSize/11) % 1 == 0:
        print 'in mod'
        task.logging = True
        task.logBuffer = None
        task.logfileName = "loggingBestIndividual_gen%05d.dat" % (l.numEvaluations/l.populationSize/11)
        print l._oneEvaluation(l.bestEvaluable)

    if l.bestEvaluation > 1.0 :
        break


task.logging = True
task.logBuffer = None
task.logfileName = "loggingBestIndividual.dat"
print l._oneEvaluation(l.bestEvaluable)

# print l.currentpop
#for indiv in l.currentpop :
#    task.logBuffer = None
#    l._oneEvaluation(indiv)

#print 'testing: ', l._bestFound() #same return value!
#for indiv in l.currentpop :
#    task.logBuffer = None
#    l._oneEvaluation(indiv)



if plotting:
    
    print 'Plotting...'
    try:
        pass
        import pylab
        pylab.plot(l._allEvaluations)
    #    pylab.semilogy()
        pylab.show()
    except ImportError, e:
        print 'No plotting:', e

