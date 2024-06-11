import numpy as np
import pandas as pd

# author: Shailaja Akella, shailaja.akella93@gmail.com

def simulate(num_units, num_trials, duration, step_size, weights, inp_curr, delay = 5, params=None):

    """
    The Izhikevich model can be represented through a 2 - D system of differential equations:

    C * dvdt = k(v - vr)(v - vt) - u + I
    dudt = a * (b * (v - vr) - u)

    Outputs
    u: Represents membrane recovery variable
    v: Represents membrane potential of the neuron

    Input parameters
    a: Time scale of the recovery variable
    ub: Sensitivity of u to the fluctuations in v
    c: After - spike reset value of v
    d: After - spike reset value of u
    I: steady input current value supplied to the neuron
    k: Related  to neuron's rheobase and input resistance

    with conditions:
        if v >= vpeak(35mV), then v = c and u = u + d

    for RS neurons:
        (a, b) = (0.03, -2)
        (c, d) = (-50, 100)

    :param num_units: total number of units in the network
    :param num_trials: total number of trials to simulate
    :param duration: duration of each trial in seconds
    :param step_size: sampling frequency
    :param weights: 2d matrix summarizing strength of connection between all neurons
                    columns are causal variables, rows are effect variables
    :param inp_curr: steady input current supplied to each unit
    :param delay: time taken for input from causal neuron to arrive at the effect neuron
    :param params: a, b, c, d, k, C
    :return: dataframe of spike times, index represents neuron number
    """

    df = pd.DataFrame(index=np.arange(num_units))
    df['spike_times'] = [[] for _ in range(num_units)]  # Initialize 'spike_times' column

    if not params:
        params = {n_unit: {'a': 0.03, 'b': -2, 'c': -50, 'd': 100, 'k': 0.7, 'C': 100} for n_unit in range(num_units)}

    for trial in range(num_trials):
        times = update_trial(num_units, weights, duration, step_size, inp_curr, params, delay)
        for n_unit in range(num_units):
            df.loc[n_unit, 'spike_times'].append(times[n_unit])

    return df


def update_trial(num_units, weights, duration, step_size, inp_curr, params, delay):
    """
    Evaluates trial wise updates
    """

    vr = -60
    vt = -40
    vpeak = 35
    I_max = 350

    T = int(duration / step_size)
    step_size = 1
    V = vr * np.ones((num_units, T))
    U = np.zeros_like(V)
    # I = np.array([inp_curr[i] * np.random.rand(T) for i in range(num_units)])
    I = np.array([inp_curr[i] + 70 * np.random.randn(T) for i in range(num_units)])
    R = np.zeros((num_units, T))
    totI = I[0]

    spike_times = {n_unit: [] for n_unit in range(num_units)}

    for t in range(T - 1):
        units_fired = []
        for n_unit in range(num_units):
            v, u, fired = izhikevich_cell(V[n_unit, t], U[n_unit, t],
                                          totI[n_unit], step_size, vpeak, vr, vt, params[n_unit])
            if fired:
                V[n_unit, t], V[n_unit, t + 1], U[n_unit, t + 1] = vpeak, params[n_unit]['c'], u + params[n_unit]['d']
                units_fired.append(n_unit)
                spike_times[n_unit].append(t * step_size)
            else:
                V[n_unit, t + 1], U[n_unit, t + 1] = v, u

        units_fired = np.array(units_fired)
        if len(units_fired) > 0:
            t_end = np.min([t + delay, T])
            R = R + np.hstack((np.zeros((num_units, t_end)),
                               np.sum(weights[:, units_fired]*I_max, axis=1)[:, np.newaxis] * np.ones(
                                   (num_units, min([delay + 1, T - t_end]))),
                               np.zeros((num_units, max(0, T - (t + 2 * delay + 1))))))

        totI = I[:, t + 1] + R[:, t + 1]

    return spike_times


def izhikevich_cell(v, u, totI, step_size, vpeak, vr, vt, params):

    """
    Model each Izhikevich cell
    """

    a = params['a']
    b = params['b']
    k = params['k']
    C = params['C']

    v = v + step_size * (k * (v - vr) * (v - vt) - u + totI) / C
    u = u + step_size * a * (b * (v - vr) - u)
    fired = 0
    if v >= vpeak:
        fired = 1

    return v, u, fired
