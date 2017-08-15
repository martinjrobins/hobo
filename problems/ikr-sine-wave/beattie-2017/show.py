#!/usr/bin/env python2
from __future__ import division
from __future__ import print_function
import os
import sys
import numpy as np
import matplotlib.pyplot as pl
sys.path.append(os.path.abspath('../myokit/'))
import myokit
import myokit.pacing as pacing

model_file = os.path.realpath(os.path.join('..', 'models', 
    'beattie-2017-ikr.mmt'))
data_file = os.path.realpath(os.path.join('..', 'sine-wave-data',
    'sine-wave.csv'))

#
# Show the real data VS Kylie's model
#

# Select solution file
solution = 'last-solution.txt'
if len(sys.argv) > 1:
    if len(sys.argv) > 2:
        print('Syntax: show <filename>')
        sys.exit(1)
    solution = sys.argv[1]
    if not os.path.isfile(solution):
        print('File not found: ' + solution)
        sys.exit(1)

# Load data
real = myokit.DataLog.load_csv(data_file).npview()

# Load model
model = myokit.load_model(model_file)

# Create model for pre-pacing
model_pre = model.clone()

# Create pre-pacing protocol
protocol_pre = myokit.pacing.constant(-80)

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
fcap = np.ones(int(duration / dt), dtype=int)
for event in protocol:
    i1 = int(event.start() / dt)
    i2 = i1 + int(cap_duration / dt)
    if i1 > 0: # Skip first one
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
    'ikr.p9',
    ]

# Read last solution
with open(solution, 'r') as f:
    print('Showing solution from: ' + solution)
    lines = f.readlines()
    if len(lines) < len(parameters):
        print('Unable to read last solution, expected ' + str(len(parameters))
            + ' lines, only found ' + str(len(lines)))
        sys.exit(1)
    try:
        solution = [float(x.strip()) for x in lines[:len(parameters)]]
    except ValueError:
        print('Unable to read last solution')
        raise

# define score function (sum of squares)
simulation = myokit.Simulation(model, protocol)
simulation_pre = myokit.Simulation(model_pre, protocol_pre)
def score(p):
    # Simulate
    for i, name in enumerate(parameters):
        simulation.set_constant(name, p[i])
        simulation_pre.set_constant(name, p[i])
    try:
        simulation_pre.reset()
        simulation_pre.pre(10000)
        simulation.reset()
        simulation.set_state(simulation_pre.state())        
        data = simulation.run(duration, log=['ikr.IKr'], log_interval=0.1)
    except myokit.SimulationError:
        return float('inf')
    # Calculate error and return
    e = fcap * (np.asarray(data['ikr.IKr']) - real['current'])
    return np.sum(e**2)

# Run simulation
log_vars = ['engine.time', 'ikr.IKr', 'membrane.V']
simulation.reset()
simulation_pre.reset()
for p, v in zip(parameters, solution):
    simulation.set_constant(p, v)
    simulation_pre.set_constant(p, v)
simulation_pre.pre(10000)
simulation.set_state(simulation_pre.state())
d = simulation.run(duration, log=log_vars, log_interval=0.1)

# Show solution
print('Parameters:')
for k, v in enumerate(solution):
    print(myokit.strfloat(v))

print('Score: ' + str(score(solution)))

# Show plots
pl.figure()
pl.subplot(2,1,1)
pl.xlabel('Time [ms]')
pl.ylabel('V [mV]')
pl.plot(d.time(), d['membrane.V'])
pl.plot(real['time'], real['voltage'], label='Vm')
pl.subplot(2,1,2)
pl.xlabel('Time [ms]')
pl.plot(real['time'], real['current'], lw=1, label='real')
pl.plot(real['time'], d['ikr.IKr'], lw=1, label=model.name())
pl.plot(real['time'], fcap, label='filter')
pl.legend(loc='lower right')
pl.show()
#'''
