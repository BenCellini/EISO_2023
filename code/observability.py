
import numpy as np
from scipy.optimize import fsolve


class ObservabilityMatrix:
    def __init__(self, system, t, x, u):
        """ Construct a sliding observability matrix for a state trajectory.

            Inputs
                system:             system ode time step function
                                    takes initial condition vector, time vector, & inputs as columns in array
                                    system_ode.simulate(x0, time, input)
                t:                  time vector
                x:                  state trajectory, where columns are the states
                u:                  inputs, where columns are each input
        """
        self.system = system  # system object, used to simulate measurements
        self.t = t  # nominal time vector
        self.x = x  # nominal state trajectory
        self.u = u  # nominal input used to generate state trajectory
        self.dt = np.mean(np.diff(self.t))  # sampling time
        self.N = len(self.t)  # # of data points

        # To store data
        self.w = []
        self.O_all = []
        self.deltay_all = []
        self.O_time = []

    def sliding_O(self, time_resolution=None, simulation_time=None, eps=0.001, polar_mode=False):
        """ Calculate the observability matrix O for every point along a nominal state trajectory.

            Inputs
                simulation_time:    simulation time for each calculation of O
                time_resolution:    how often to calculate O along the nominal state trajectory.
                eps:                amount to perturb initial state
                measurement_type:   how to compute the measurement of the simulated system
                                    'linear', 'divide_first_two_states', or 'multiply_first_two_states'

            Outputs
                allO:               a list of the O's calculated at set times along the nominal state trajectory
                O_time:             the times O was computed
        """

        # Set simulation_time to fill space between time_resolution
        if simulation_time is None:
            simulation_time = time_resolution

        # The size of the simulation time
        simulation_index = np.round(simulation_time / self.dt, decimals=0).astype(int)

        # Set time_resolution to every point, if not specified
        if time_resolution is None:
            time_resolution = self.dt

        # If time_resolution is a vector, then use the entries as the indices to calculate O
        if not np.isscalar(time_resolution) or (time_resolution == 0):  # time_resolution is a vector
            O_index = time_resolution
        else:  # evenly space the indices to calculate O
            # Resolution on nominal trajectory to calculate O, measured in indices
            time_resolution_index = np.round(time_resolution / self.dt, decimals=0).astype(int)

            # All the indices to calculate O
            O_index = np.arange(0, self.N - simulation_index, time_resolution_index)  # indices to compute O

        O_time = O_index * np.array(self.dt)  # times to compute O
        n_point = len(O_index)  # # of times to calculate O

        # Calculate O for each window on nominal trajectory
        O_all = []  # where to store the O's
        deltay_all = []  # where to store the O's
        for n in range(n_point):  # each point on trajectory
            # Start simulation at point along nominal trajectory
            x0 = np.squeeze(self.x[O_index[n], :])  # get state on trajectory & set it as the initial condition

            # Get the range to pull out time & input data for simulation
            win = np.arange(O_index[n], O_index[n] + simulation_index, 1)  # index range

            # Remove part of window if it is past the edge of the nominal trajectory
            withinN = win < self.N
            win = win[withinN]

            # Pull out time & control inputs in window
            twin = self.t[win]  # time in window
            twin = twin - twin[0]  # start at 0
            uwin = self.u[win, :]  # inputs in window

            # Calculate O for window
            O, deltay = self.calculate_O(x0, twin, uwin, eps=eps, polar_mode=polar_mode)

            O_all.append(O)  # store in list
            deltay_all.append(deltay)  # store in list

        # Store data in object
        self.O_all = O_all
        self.deltay_all = deltay_all
        self.O_time = O_time

        return O_all, O_time, deltay_all

    def calculate_O(self, x0, twin, uwin, eps=0.001, polar_mode=False):
        """ Numerically calculates the observability matrix O for a given system & input.

            Inputs
                x0:                 initial state
                twin:               simulation time
                twin:               simulation inputs
                eps:                amount to perturb initial state
                polar_mode:         if system object simualtor has polar mode, can set it here

            Outputs
                O:                  numerically calculated observability matrix
                deltay:             the difference in perturbed measurements at each time step
                                    (basically un-normalized O stored in a 3D array)
        """

        # Calculate O
        self.w = len(twin)  # of points in time window
        delta = eps * np.eye(self.system.n)  # perturbation amount for each state
        deltay = np.zeros((self.system.p, self.system.n, self.w))  # preallocate deltay
        for k in range(self.system.n):  # each state
            # Perturb initial condition in both directions
            x0plus = x0 + delta[:, k]
            x0minus = x0 - delta[:, k]

            # Simulate measurements from perturbed initial conditions
            if polar_mode:
                _, yplus = self.system.simulate(x0plus, twin, uwin, polar_mode=polar_mode)
                _, yminus = self.system.simulate(x0minus, twin, uwin, polar_mode=polar_mode)
            else:
                _, yplus = self.system.simulate(x0plus, twin, uwin)
                _, yminus = self.system.simulate(x0minus, twin, uwin)

            # Calculate the numerical Jacobian
            # deltay[:, k, :] = np.array(yplus - yminus).T
            dy = special_subtract(yplus, yminus, use_angle=self.system.angular_outputs)
            deltay[:, k, :] = np.array(dy).T

        # Construct O by stacking the 3rd dimension of deltay along the 1st dimension, O is a (p*m x n) matrix
        O = np.zeros((1, self.system.n))  # make place holder 1st row
        for k in range(deltay.shape[2]):  # along 3rd dimension
            O = np.vstack((O, deltay[:, :, k]))  # stack
        O = O[1:, :]  # remove 1st row
        O = O / (2 * eps)  # normalize by 2 times the perturbation amount

        return O, deltay


def special_subtract(Y1, Y2, use_angle=None):
    """ Subtract columns of Y1 - Y2
        Use angular subtraction  for the column indices specified in use_angle
    """

    Y1 = np.atleast_2d(Y1)
    Y2 = np.atleast_2d(Y2)

    # Columns to use angular subtraction
    angle_flag = np.zeros(Y1.shape[1])
    if use_angle is not None:
        angle_flag[use_angle] = True

    # Do the subtraction
    Y1m2 = np.zeros_like(Y1)
    for c in range(Y1.shape[1]):  # each column
        y1m2 = Y1[:, c] - Y2[:, c]
        if angle_flag[c]:  # if angular subtraction
            y1m2 = np.arccos(np.cos(y1m2))  # simplified formula from https://en.wikipedia.org/wiki/Great-circle_distance

        Y1m2[:, c] = y1m2  # store

    return Y1m2
