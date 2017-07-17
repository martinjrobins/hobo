#!/usr/bin/env python2
from __future__ import division
from __future__ import print_function
import os
import sys
import numpy as np
import myokit
import myokit.pacing as pacing
from myokit.lib import fit

# Get optimization method from arguments
method = ''
if len(sys.argv) == 2:
    method = sys.argv[1]
    if method not in ['cmaes', 'pso', 'snes', 'xnes']:
        print('Unknown method "' + str(method) + '"')
        print('Choose one of: ' + ', '.join(methods))
        sys.exit(1)

#
# Fit the Aslanidi model to real data
# 

# Load data
real = myokit.DataLog.load_csv('real-data.csv').npview()

# Load model
model = myokit.load_model('aslanidi-2009-ikr.mmt')

# Pre-pace
s = myokit.Simulation(model)
s.set_protocol(pacing.constant(-80))
s.run(10000, log=myokit.LOG_NONE)
model.set_state(s.state())

# Create step protocol
protocol = myokit.Protocol()
protocol.add_step(-80, 250)
protocol.add_step(-120, 50)
protocol.add_step(-80, 200)
protocol.add_step(40, 1000)
protocol.add_step(-120, 500)
protocol.add_step(-80, 1000)
protocol.add_step(-30, 3500)
protocol.add_step(-120, 500)
protocol.add_step(-80, 1000)
duration = protocol.characteristic_time()

# Create capacitance filter
cap_duration = 1.5
dt = 0.1
fcap = np.ones(duration / dt, dtype=int)
for event in protocol:
    i1 = int(event.start() / dt)
    i2 = i1 + int(cap_duration / dt)
    fcap[i1:i2] = 0

# Change RHS of membrane.V
model.get('membrane.V').set_rhs('if(engine.time < 3000 or engine.time >= 6500,'
    + ' engine.pace, '
    + ' - 30'
    + ' + 54 * sin(0.007 * (engine.time - 2500))'
    + ' + 26 * sin(0.037 * (engine.time - 2500))'
    + ' + 10 * sin(0.190 * (engine.time - 2500))'
    + ')')

# Define parameters
parameters = [
    'ikr.p1',
    'ikr.p2',
    'ikr.p3',
    'ikr.p4',
    'ikr.p5',
    'ikr.p6',
    'ikr.p7',
    'ikr.p8',
    ]
boundaries = {
    'ikr.p1' : [1, 1e4],        # ms        900
    'ikr.p2' : [0, 100],        # mV        5.0
    'ikr.p3' : [1, 1e4],        # ms        100
    'ikr.p4' : [0, 100],        # mV        0.085
    'ikr.p5' : [0, 100],        # mV        12.25
    'ikr.p6' : [-10, 0],        # mV        -5.4
    'ikr.p7' : [0, 100],        # mV        20.4
    'ikr.p8' : [5e-3, 1],       # mS/uF     0.04
    }
bounds = [boundaries[x] for x in parameters]

# define score function (sum of squares)
simulation = myokit.Simulation(model, protocol)
def score(p):
    simulation.reset()
    for i, name in enumerate(parameters):
        simulation.set_constant(name, p[i])
    try:
        data = simulation.run(duration, log=['ikr.IKr'], log_interval=0.1)
    except myokit.SimulationError:
        return float('inf')
    e = fcap * (np.asarray(data['ikr.IKr'])[fcap] - real['current'])
    return np.sum(e**2)
target = 0

# Get hint
hint = [model.get(x).eval() for x in parameters]
hint[-1] = 0.06
print('Initial solution:')
for k,v in enumerate(hint):
    print(myokit.strfloat(v))
print('Hint score: ' + str(score(hint)))

# Run
if method == 'pso':
    print('Running PSO')
    with np.errstate(all='ignore'): # Tell numpy not to issue warnings
        x, f = fit.pso(score, bounds, n=96, parallel=True, target=target,
                hints=[hint], max_iter=500, verbose=True)
elif method == 'xnes':
    print('Running xNES')
    with np.errstate(all='ignore'):
        x, f = fit.xnes(score, bounds, parallel=True, target=target,
                hint=hint, max_iter=500, verbose=True)
elif method == 'cmaes':
    print('Running CMA-ES')
    with np.errstate(all='ignore'):
        x, f = fit.cmaes(score, bounds, parallel=True, target=target,
                hint=hint, verbose=True)
elif method == 'snes':
    print('Running SNES')
    with np.errstate(all='ignore'):
        x, f = fit.snes(score, bounds, parallel=True, target=target,
                hint=hint, max_iter=500, verbose=True)
else:
    print('Unknown method: "' + str(method) + '"')
    sys.exit(1)

# Show final score
print('Final score: ' + str(f))
    
# Show solution
print('Current solution:')
for k, v in enumerate(x):
    print(myokit.strfloat(v))

# Store solution
with open('last-solution.txt', 'w') as f:
    for k, v in enumerate(x):
        f.write(myokit.strfloat(v) + '\n')

