# @title ## Load Dependencies and Functions{ display-mode: "form", run: "auto" }

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors
from glob import glob
import scipy.ndimage
import scipy.signal
from scipy.optimize import curve_fit

import time
import plotly.express as px

import plotly.graph_objects as go

# @title ## The Data Object { display-mode: "form", run: "auto" }


"""GigaAnalysis - Data Type

This holds the data class and the functions that will manipulate them.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, get_window, find_peaks


class Data():
    """
The Data Class

Data object holds the data in the measurements. It works as a simple
wrapper of a two column numpy array. The point is that operations apply
to the y values and interpretation happens to compare the cosponsoring
data points.

The initialisation is documented in the __init__ method.

Attributes:
    values (np array): Two column numpy array with the x and y data in
    x (np array): The x data in a 1D numpy array
    y (np array): The y data in a 1D numpy array
    both (two np arrays): The x data then the y data in a tuple

"""

    def __init__(self, values, split_y=None, strip_sort=False,
                 interp_full=0.):
        """
The __init__ method to produce a incidence of the Data class
Args:
    values (np array): A two column numpy array with the x data in
        the first column and the y data in the second. If a second
        no array is given then the first corresponds to the x data.
    split_y (np array default:None): A 1D numpy array containing the
        y data. If default all the data should be contained in
        first array.
"""
        if type(values) in [pd.core.frame.DataFrame,
                            pd.core.series.Series]:
            values = values.values

        if split_y is not None:
            if type(split_y) in [pd.core.frame.DataFrame,
                                 pd.core.series.Series]:
                split_y = split_y.values
            elif type(values) is not np.ndarray:
                raise ValueError(
                    "If x and y data are split both need to " + \
                    "be a 1D numpy array.\n" + \
                    "x is not a numpy array.")
            elif type(split_y) is not np.ndarray:
                raise ValueError(
                    "If x and y data are split both need to " + \
                    "be a 1D numpy array.\n" + \
                    "y is not a numpy array.")
            elif values.ndim != 1 or split_y.ndim != 1:
                raise ValueError(
                    "If x and y data are split both need to " + \
                    "be a 1D numpy array.\n" + \
                    "x or y is not 1D.")
            elif values.size != split_y.size:
                raise ValueError(
                    "If x and y data are split both need to " + \
                    "be the same size.")
            values = np.concatenate([values[:, None],
                                     split_y[:, None]], axis=1)

        if type(values) is not np.ndarray:
            raise TypeError('values is not a numpy array. \n' + \
                            'Needs to be a two column numpy array.')
        elif len(values.shape) != 2 or values.shape[1] != 2:
            raise ValueError('values dose not have two columns. \n' + \
                             'Needs to be a two column numpy array.')

        if strip_sort:
            values = values[~np.isnan(values).any(axis=1)]
            values = values[np.argsort(values[:, 0]), :]

        self.values = values.astype(float)  # All the data
        self.x = values[:, 0]  # The x data
        self.y = values[:, 1]  # The y data
        self.both = values[:, 0], values[:, 1]  # A tuple of the data

        if interp_full != 0.:
            self.to_even(interp_full)

    def __str__(self):
        return np.array2string(self.values)

    def __repr__(self):
        return 'GA Data object:\n{}'.format(self.values)

    def _repr_html_(self):
        return 'GA Data object:\n{}'.format(self.values)

    def __dir__(self):
        return ['values', 'x', 'y', 'both',
                'y_from_x', 'x_cut',
                'interp_full', 'interp_number', 'interp_range',
                'plot']

    def __len__(self):
        return self.x.size

    def __mul__(self, other):
        """
The Data class can be multiplied and this just effects the y values,
the x values stay the same.
This can be multiplied to a float, int, numpy array with 1 value,
or a numpy array with same length, or a other Data object
with the same length.
"""
        if type(other) in [float, int, np.float_, np.int_]:
            return Data(self.x, self.y * other)
        elif type(other) is np.ndarray:
            if other.size == 1:
                return Data(self.x, self.y * float(other))
            elif self.x.shape == other.shape:
                return (Data(self.x, self.y * other))
            else:
                raise ValueError('Numpy array to multiply to data ' \
                                 'is wrong shape.')
        elif type(other) == type(self):
            if np.array_equal(self.x, other.x):
                return (Data(self.x, self.y * other.y))
            else:
                raise ValueError('The two Data class need to have the same ' \
                                 'x values to be multiplied.')
        else:
            raise TypeError('Cannot multiple Data class with this type')

    __rmul__ = __mul__

    def __truediv__(self, other):
        """
The Data class can be divided and this just effects the y values,
the x values stay the same.
This can be divided to a float, int, numpy array with 1 value,
or a numpy array with same length, or a other Data object
with the same length.
"""
        if type(other) in [float, int, np.float_, np.int_]:
            return Data(self.x, self.y / other)
        elif type(other) is np.ndarray:
            if other.size == 1:
                return Data(self.x, self.y / float(other))
            elif self.x.shape == other.shape:
                return (Data(self.x, self.y / other))
            else:
                raise ValueError('Numpy array to divide to data ' \
                                 'is wrong shape.')
        elif type(other) == type(self):
            if np.array_equal(self.x, other.x):
                return (Data(self.x, self.y / other.y))
            else:
                raise ValueError('The two Data class need to have the same ' \
                                 'x values to be divided.')
        else:
            raise TypeError('Cannot divide Data class with this type')

    def __rtruediv__(self, other):
        """
The Data class can be divided and this just effects the y values,
the x values stay the same.
This can be divided to a float, int, numpy array with 1 value,
or a numpy array with same length, or a other Data object
with the same length.
"""
        if type(other) in [float, int, np.float_, np.int_]:
            return Data(self.x, other / self.y)
        elif type(other) is np.ndarray:
            if other.size == 1:
                return Data(self.x, float(other) / self.y)
            elif self.x.shape == other.shape:
                return (Data(self.x, other / self.y))
            else:
                raise ValueError('Numpy array to divide to data ' \
                                 'is wrong shape.')
        elif type(other) == type(self):
            if np.array_equal(self.x, other.x):
                return (Data(self.x, other.y / self.y))
            else:
                raise TypeError('The two Data class need to have the same ' \
                                'x values to be divided.')
        else:
            raise ValueError('Cannot divide Data class with this type')

    def __add__(self, other):
        """
The Data class can be added and this just effects the y values, the
x values stay the same.
This can be added to a float, int, numpy array with 1 value,
or a numpy array with same length, or a other Data object
with the same length.
"""
        if type(other) in [float, int, np.float_, np.int_]:
            return (Data(self.x, self.y + other))
        elif type(other) is np.ndarray:
            if other.size == 1:
                return (Data(self.x, self.y + other))
            elif self.x.shape == other.shape:
                return (Data(self.x, self.y + other))
            else:
                raise ValueError('Numpy array to add to data ' \
                                 'is wrong shape.')
        elif type(other) == type(self):
            if np.array_equal(self.x, other.x):
                return (Data(self.x, self.y + other.y))
            else:
                raise ValueError('The two Data class need to have the same ' \
                                 'x values to be added.')
        else:
            raise TypeError('Cannot add Data class with this type')

    __radd__ = __add__

    def __sub__(self, other):
        """
The Data class can be subtracted and this just effects the y values,
the x values stay the same.
This can be subtracted to a float, int, numpy array with 1 value,
or a numpy array with same length, or a other Data object
with the same length.
"""
        if type(other) in [float, int, np.float_, np.int_]:
            return (Data(self.x, self.y - other))
        elif type(other) is np.ndarray:
            if other.size == 1:
                return (Data(self.x, self.y - other))
            elif self.x.shape == other.shape:
                return (Data(self.x, self.y - other))
            else:
                raise ValueError('Numpy array to subtract to data ' \
                                 'is wrong shape.')
        elif type(other) == type(self):
            if np.array_equal(self.x, other.x):
                return (Data(self.x, self.y - other.y))
            else:
                raise ValueError('The two Data class need to have the same ' \
                                 ' x values to be subtracted.')
        else:
            raise ValueError('Cannot subtract Data class with this type')

    def __rsub__(self, other):
        """
The Data class can be subtracted and this just effects the y values,
the x values stay the same.
This can be subtracted to a float, int, numpy array with 1 value,
or a numpy array with same length, or a other Data object
with the same length.
"""
        if type(other) in [float, int, np.float_, np.int_]:
            return (Data(self.x, other - self.y))
        elif type(other) is np.ndarray:
            if other.size == 1:
                return (Data(self.x, other - self.y))
            elif self.x.shape == other.shape:
                return (Data(self.x, other - self.y))
            else:
                raise ValueError('Numpy array to subtract to data ' \
                                 'is wrong shape.')
        elif type(other) == type(self):
            if np.array_equal(self.x, other.x):
                return (Data(self.x, other.y - self.y))
            else:
                raise ValueError('The two Data class need to have the same ' \
                                 'x values to be subtracted.')
        else:
            raise TypeError('Cannot subtract Data class with this type')

    def __abs__(self):
        """
The abs function takes the absolute value of the y values.
"""
        return Data(self.x, abs(self.y))

    def __pow__(self, power):
        """
Takes the power of the y values and leaves the x-values unchanged.
"""
        return Data(self.x, pow(self.y, power))

    def __eq__(self, other):
        """
The Data class is only equal to other data classes with the same data.
"""
        if type(other) != type(self):
            return False
        else:
            return np.array_equal(self.values, other.values)

    def __iter__(self):
        """
The iteration happens on the values, like if was numpy array.
"""
        return iter(self.values)

    def y_from_x(self, x_val):
        """
This function gives the y value for a certain x value or
set of x values.
Args:
    x_val (float): X values to interpolate y values from
Returns:
    y values corresponding to the requested x values in nd array
"""
        y_val = interp1d(self.x, self.y, bounds_error=False,
                         fill_value=(self.y.min(), self.y.max()))(x_val)
        if y_val.size != 1:
            return y_val
        else:
            return float(y_val)

    def x_cut(self, x_min, x_max):
        """
This cuts the data to a region between x_min and x_max
Args:
    x_min (float): The minimal x value to cut the data
    x_max (float): The maximal x value to cut the data
Returns:
    An data object with the values cut to the given x range
"""
        if x_min > x_max:
            raise ValueError('x_min should be smaller than x_max')
        return Data(self.values[
                    np.searchsorted(self.x, x_min, side='left'):
                    np.searchsorted(self.x, x_max, side='right'), :])

    def interp_range(self, min_x, max_x, step_size, **kwargs):
        '''
This evenly interpolates the data points between a min
and max x value. This is used so that the different
sweeps can be combined with the same x-axis.
It uses scipy.interpolate.interp1d and **kwargs can be pass to it.
Args:
    data_set (Data): The data set to be interpolated
    min_x (float): The minimum x value in the interpolation
    max_y (float): The maximum x value in the interpolation
    step_size (float): The step size between each point
Returns:
    A new data set with evenly interpolated points.
'''
        if np.min(self.x) > min_x:
            raise ValueError('min_x value to interpolate is below data')
        if np.max(self.x) < max_x:
            raise ValueError('max_x value to interpolate is above data')
        x_vals = np.arange(min_x, max_x, step_size)
        return Data(x_vals,
                    interp1d(self.x, self.y, **kwargs)(x_vals))

    def to_range(self, min_x, max_x, step_size, **kwargs):
        '''
This evenly interpolates the data points between a min
and max x value. This is used so that the different
data objects can be combined with the same x-axis. This changes
the object.
It uses scipy.interpolate.interp1d and **kwargs can be pass to it.
Args:
    data_set (Data): The data set to be interpolated
    min_x (float): The minimum x value in the interpolation
    max_y (float): The maximum x value in the interpolation
    step_size (float): The step size between each point
'''
        if np.min(self.x) > min_x:
            raise ValueError('min_x value to interpolate is below data')
        if np.max(self.x) < max_x:
            raise ValueError('max_x value to interpolate is above data')
        x_vals = np.arange(min_x, max_x, step_size)
        y_vals = interp1d(self.x, self.y, **kwargs)(x_vals)
        self.values = np.concatenate((x_vals[:, None], y_vals[:, None]),
                                     axis=1)
        self.x = x_vals
        self.y = y_vals
        self.both = x_vals, y_vals

    def interp_full(self, step_size, **kwargs):
        """
This interpolates the data to give an even spacing. This is useful
for combining different data sets.
It uses scipy.interpolate.interp1d and **kwargs can be pass to it.
Args:
    step_size (float): The spacing of the data along x.
Return:
    A Data class with the interpolated data.
"""
        x_start = np.ceil(self.x.min() / step_size) * step_size
        x_stop = np.floor(self.x.max() / step_size) * step_size
        x_vals = np.linspace(x_start, x_stop,
                             int(round((x_stop - x_start) / step_size)) + 1)
        y_vals = interp1d(self.x, self.y, **kwargs)(x_vals)
        return Data(x_vals, y_vals)

    def interp_number(self, point_number, **kwargs):
        """
This interpolates the data to give an even spacing. This is useful
for saving data of different types together
It uses scipy.interpolate.interp1d and **kwargs can be pass to it.
Args:
    point_number (int): The spacing of the data along x.
Return:
    A Data class with the interpolated data.
"""

        x_vals = np.linspace(self.x.min(), self.x.max(),
                             int(point_number))
        y_vals = interp1d(self.x, self.y, **kwargs)(x_vals)
        return Data(x_vals, y_vals)

    def to_even(self, step_size, **kwargs):
        """
This interpolates the data to give an even spacing, and changes
the data file.
It uses scipy.interpolate.interp1d and **kwargs can be pass to it.
Args:
    step_size (float): The spacing of the data along x.
"""
        x_start = np.ceil(self.x.min() / step_size) * step_size
        x_stop = np.floor(self.x.max() / step_size) * step_size
        x_vals = np.linspace(x_start, x_stop,
                             int(round((x_stop - x_start) / step_size)) + 1)
        y_vals = interp1d(self.x, self.y, **kwargs)(x_vals)
        self.values = np.concatenate((x_vals[:, None], y_vals[:, None]),
                                     axis=1)
        self.x = x_vals
        self.y = y_vals
        self.both = x_vals, y_vals

    def sort(self):
        """
This sorts the data set in x and returns the new array.
Returns:
    A Data class with the sorted data.
"""
        return Data(self.values[np.argsort(self.x), :])

    def strip_nan(self):
        """
This removes any row which has a nan value in.
Returns:
    Data class without any nan in.
"""
        return Data(self.values[~np.isnan(self.values).any(axis=1)])

    def min_x(self):
        """
This provides the lowest value of x
Returns:
    A float of the minimum x value
"""
        return np.min(self.x)

    def max_x(self):
        """
This provides the lowest value of x
Returns:
    A float of the minimum x value
"""
        return np.max(self.x)

    def spacing_x(self):
        """
Provides the average separation of the x values
(max_x - min_x)/num_points
Returns:
    A float of the average spacing in x
"""
        return (self.max_x() - self.min_x()) / len(self)

    def apply_x(self, function):
        """
This takes a function and applies it to the x values.
Args:
    function (func): THe function to apply to the x values
Returns:
    Data class with new x values
"""
        return Data(function(self.x), self.y)

    def apply_y(self, function):
        """
This takes a function and applies it to the y values.
Args:
    function (func): THe function to apply to the y values
Returns:
    Data class with new x values
"""
        return Data(self.x, function(self.y))

    def plot(self, *args, axis=None, **kwargs):
        """
Simple plotting function that runs
matplotlib.pyplot.plot(self.x, self.y, *args, **kwargs)
Added a axis keyword which operates so that if given
axis.plot(self.x, self.y, *args, **kwargs)
"""
        if axis == None:
            plt.plot(self.x, self.y, *args, **kwargs)
        else:
            axis.plot(self.x, self.y, *args, **kwargs)

    # def remove_copies(self):
    #     """
    #     Removes copies of consecutive data points in the Data object
    #     """
    #     lengt = self.x.size
    #     locs = zip(np.arange(lengt)[:-1],np.arange(lengt)[1:])
    #     copies = []
    #     for s1,s2 in locs:
    #         if_copy_y = self.y[s1] == self.y[s2]
    #         if_copy_x = self.x[s1] == self.x[s2]
    #         copies.append(if_copy_y & if_copy_x)

    #     return Data(self.values[copies, :])


def sum_data(data_list):
    """
Preforms the sum of the y data a set of Data class objects.
Args:
    data_list (list of Data): List of Data objects to sum together.
Returns:
    A Data object which is the sum of the y values of the original
        data sets.
"""
    total = data_list[0]
    for data_set in data_list[1:]:
        total += data_set.y
    return total


def mean(data_list):
    """
Preforms the mean of the y data a set of Data class objects.
Args:
    data_list (list of Data): List of Data objects to combine together.
Returns:
    A Data object which is the average of the y values of the original
        data sets.
"""
    return sum_data(data_list) / len(data_list)


# @title ## Constants { display-mode: "form", run: "auto" }

"""GigaAnalysis - Constants - :mod:`gigaanalysis.const`
----------------------------------------------------------

Here is contained a collection of functions with when called return values 
of physical constants. They always return floats and all have one optional 
parameter 'unit' which default is 'SI' for the International System of Units 
values for these parameters.
The module :mod:`scipy.constants` contains many more than what is listed 
here, but I included these for the different units.
"""


def __pick_unit(unit, units_dict):
    """Takes value from dictionary and returns value

    Parameters
    ----------
    unit: str
        The unit chosen
    units_dcit: dict
        The dictionary with the units and the values

    Returns
    -------
    const_value: float
        The value that is requested
    """
    if unit in units_dict.keys():
        return units_dict[unit]
    else:
        if len(units_dict) == 1:
            raise ValueError("unit must be '{}'".format(list(units_dict)[0]))
        else:
            unit_list = ["'{}',".format(x) for x in units_dict.keys()]
            unit_list[-1] = "or {}.".format(unit_list[-1][:-1])
            raise ValueError("unit must be {}".format(" ".join(unit_list)))


def amu(unit='SI'):
    """`Unified Atomic mass unit
    <https://en.wikipedia.org/wiki/Dalton_(unit)>`_ or Dalton

    =====  =====
    Unit   Value
    =====  =====
    'SI'   1.66053906660e-27 kg
    'CGS'  1.66053906660e-24
    =====  =====

    Parameters
    ----------
    unit : str, optional
        The unit system to give the value in.

    Returns
    -------
    Value of the Atomic mass unit : float
    """
    return __pick_unit(unit, {
        'SI': 1.66053906660e-27,
        'CGS': 1.66053906660e-24,
    })


def Na(unit='SI'):
    """`Avogadro constant
    <https://en.wikipedia.org/wiki/Avogadro_constant>`_

    =====  =====
    Unit   Value
    =====  =====
    'SI'   6.02214076e+23 1/mol
    =====  =====

    Parameters
    ----------
    unit : str, optional
        The unit system to give the value in.

    Returns
    -------
    Value of the Avogadro constant : float
    """
    return __pick_unit(unit, {
        'SI': 6.02214076e+23,
    })


def kb(unit='SI'):
    """`Boltzmann constant
    <http://en.wikipedia.org/wiki/Boltzmann_constant>`_

    =====  =====
    Unit   Value
    =====  =====
    'SI'   1.380649e-23 J/K
    'eV'   8.617333262145e-5 eV/K
    'CGS'  1.380649e-16 erg/K/
    =====  =====

    Parameters
    ----------
    unit : str, optional
        The unit system to give the value in.

    Returns
    -------
    Value of the Boltzmann constant : float
    """
    return __pick_unit(unit, {
        'SI': 1.380649e-23,
        'eV': 8.617333262145e-5,
        'CGS': 1.380649e-16,
    })


def muB(unit='SI'):
    """`Bohr magneton
    <https://en.wikipedia.org/wiki/Bohr_magneton>`_

    =====  =====
    Unit   Value
    =====  =====
    'SI'   9.274009994e-24 J/T
    'eV'   5.7883818012e-5 eV/T
    'CGS'  9.274009994e-21 erg/T
    =====  =====

    Parameters
    ----------
    unit : str, optional
        The unit system to give the value in.

    Returns
    -------
    Value of the Bohr magneton : float
    """
    return __pick_unit(unit, {
        'SI': 9.274009994e-24,
        'eV': 8.617333262145e-5,
        'CGS': 1.380649e-16,
    })


def a0(unit='SI'):
    """`Bohr radius
    <https://en.wikipedia.org/wiki/Bohr_radius>`_

    ======  ======
    Unit    Value
    ======  ======
    'SI'    5.29177210903e-11 m
    'CGS'   5.29177210903e-9 cm
    ======  ======

    Parameters
    ----------
    unit : str, optional
        The unit system to give the value in.

    Returns
    -------
    Value of the Bohr radius : float
    """
    return __pick_unit(unit, {
        'SI': 5.29177210903e-11,
        'CGS': 5.29177210903e-9,
    })


def me(unit='SI'):
    """`Electron rest mass
    <https://en.wikipedia.org/wiki/Electron_rest_mass>`_

    ======  =====
    Unit    Value
    ======  =====
    'SI'    9.1093837015e-31 kg
    'CGS'   9.1093837015e-29 g
    'MeVc'  5.1099895000e-1 MeV/c^2
    'uamu'  5.48579909065e-4 Da
    ======  =====

    Parameters
    ----------
    unit : str, optional
        The unit system to give the value in.

    Returns
    -------
    Value of the Bohr magneton : float
    """
    return __pick_unit(unit, {
        'SI': 9.1093837015e-31,
        'CGS': 9.1093837015e-29,
        'MeVc': 5.1099895000e-1,
        'uamu': 5.48579909065e-4,
    })


def qe(unit='SI'):
    """`Elementary charge
    <https://en.wikipedia.org/wiki/Elementary_charge>`_

    =====  =====
    Unit   Value
    =====  =====
    'SI'   1.602176634e-19 C
    'CGS'  1.602176634e-20 statC
    =====  =====

    Parameters
    ----------
    unit : str, optional
        The unit system to give the value in.

    Returns
    -------
    Value of the Elementary charge : float
    """
    return __pick_unit(unit, {
        'SI': 1.602176634e-19,
        'CGS': 1.602176634e-20,
    })


def alpha(unit='SI'):
    """`Fine-structure constant
    <https://en.wikipedia.org/wiki/Fine-structure_constant>`_

    =====  =====
    Unit   Value
    =====  =====
    'SI'   7.2973525693e-3
    =====  =====

    Parameters
    ----------
    unit : str, optional
        The unit system to give the value in.

    Returns
    -------
    Value of the Fine-structure constant : float
    """
    return __pick_unit(unit, {
        'SI': 7.2973525693e-3,
    })


def R(unit='SI'):
    """`Gas Constant
    <https://en.wikipedia.org/wiki/Gas_constant>`_

    =====  =====
    Unit   Value
    =====  =====
    'SI'   8.31446261815324 J/K/mol
    'eV'   5.189479388046824e+19 eV/K/mol
    'CGS'  8.31446261815324e+7 erg/K/mol
    =====  =====

    Parameters
    ----------
    unit : str, optional
        The unit system to give the value in.

    Returns
    -------
    Value of the Gas Constant : float
    """
    return __pick_unit(unit, {
        'SI': 8.31446261815324,
        'eV': 5.189479388046824e+19,
        'CGS': 8.31446261815324e+7
    })


def G(unit='SI'):
    """`Gravitational constant
    <https://en.wikipedia.org/wiki/Gravitational_constant>`_

    =====  =====
    Unit   Value
    =====  =====
    'SI'   6.67430e-11 m^3/kg/s^2
    'CGS'  6.67430e-8 dyn cm^2/g^2
    =====  =====

    Parameters
    ----------
    unit : str, optional
        The unit system to give the value in.

    Returns
    -------
    Value of the Gravitational constant : float
    """
    return __pick_unit(unit, {
        'SI': 6.67430e-11,
        'CGS': 6.67430e-8,
    })


def muN(unit='SI'):
    """`Nuclear magneton
    <https://en.wikipedia.org/wiki/Nuclear_magneton>`_

    =====  =====
    Unit   Value
    =====  =====
    'SI'   5.050783699e-27 J/T
    'eV'   3.1524512550e-8 eV/T
    'CGS'  5.050783699e-24 erg/T
    =====  =====

    Parameters
    ----------
    unit : str, optional
        The unit system to give the value in.

    Returns
    -------
    Value of the Nuclear magneton : float
    """
    return __pick_unit(unit, {
        'SI': 5.050783699e-27,
        'eV': 3.1524512550e-8,
        'CGS': 5.050783699e-24,
    })


def mp(unit='SI'):
    """`Proton rest mass
    <https://en.wikipedia.org/wiki/Proton>`_

    ======  =====
    Unit    Value
    ======  =====
    'SI'    1.67262192369e-27 kg
    'CGS'   1.67262192369e-25 g
    'MeVc'  9.3827208816e+2 MeV/c^2
    'uamu'  1.007276466621e+0 Da
    ======  =====

    Parameters
    ----------
    unit : str, optional
        The unit system to give the value in.

    Returns
    -------
    Value of the Nuclear magneton : float
    """
    return __pick_unit(unit, {
        'SI': 1.67262192369e-27,
        'CGS': 1.67262192369e-25,
        'MeVc': 9.3827208816e+2,
        'uamu': 1.007276466621e+0,
    })


def h(unit='SI'):
    """`Planck constant
    <https://en.wikipedia.org/wiki/Planck_constant>`_

    =====  =====
    Unit   Value
    =====  =====
    'SI'   6.62607015e-34 J s
    'eV'   4.135667696e-15 eV s
    'CGS'  6.62607015e-27 erg s
    =====  =====

    Parameters
    ----------
    unit : str, optional
        The unit system to give the value in.

    Returns
    -------
    Value of the Planck constant : float
    """
    return __pick_unit(unit, {
        'SI': 6.62607015e-34,
        'eV': 4.135667696e-15,
        'CGS': 6.62607015e-27,
    })


def hbar(unit='SI'):
    """`Reduced Planck constant
    <https://en.wikipedia.org/wiki/Planck_constant>`_

    =====  =====
    Unit   Value
    =====  =====
    'SI'   1.054571817e-34 J s
    'eV'   6.582119569e-16 eV s
    'CGS'  1.054571817e-27 erg s
    =====  =====

    Parameters
    ----------
    unit : str, optional
        The unit system to give the value in.

    Returns
    -------
    Value of the Reduced Planck constant : float
    """
    return __pick_unit(unit, {
        'SI': 1.054571817e-34,
        'eV': 6.582119569e-16,
        'CGS': 1.054571817e-27,
    })


def c(unit='SI'):
    """`Speed of light
    <https://en.wikipedia.org/wiki/Speed_of_light>`_

    =====  =====
    Unit   Value
    =====  =====
    'SI'   2.99792458e+8 m/s
    'CGS'  2.99792458e+10 cm/s
    =====  =====

    Parameters
    ----------
    unit : str, optional
        The unit system to give the value in.

    Returns
    -------
    Value of the speed of light : float
    """
    return __pick_unit(unit, {
        'SI': 2.99792458e8,
        'CGS': 2.99792458e10,
    })


def mu0(unit='SI'):
    """`Vacuum permeability
    <https://en.wikipedia.org/wiki/Vacuum_permeability>`_

    =====  =====
    Unit   Value
    =====  =====
    'SI'   1.25663706212e-6 H/m
    'eV'   7.8433116265e+12 eV/A^2
    =====  =====

    Parameters
    ----------
    unit : str, optional
        The unit system to give the value in.

    Returns
    -------
    Value of the Vacuum permeability : float
    """
    return __pick_unit(unit, {
        'SI': 1.25663706212e-6,
        'eV': 7.8433116265e+12,
    })


def ep0(unit='SI'):
    """`Vacuum permittivity
    <https://en.wikipedia.org/wiki/Vacuum_permittivity>`_

    =====  =====
    Unit   Value
    =====  =====
    'SI'   8.8541878128e-12 F/m
    'eV'   1.4185972826e-30 C^2/eV
    =====  =====

    Parameters
    ----------
    unit : str, optional
        The unit system to give the value in.

    Returns
    -------
    Value of the Vacuum permittivity : float
    """
    return __pick_unit(unit, {
        'SI': 8.8541878128e-12,
        'eV': 1.4185972826e-30,
    })


# @title ## Quantum Oscillation Object { display-mode: "form", run: "auto" }

"""Giga Analysis - Quantum Oscillations

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, get_window, find_peaks


def invert_x(data_set):
    """
    This inverts the x data and then reinterpolates the y points so the x
    data is evenly spread.
    Args:
        data_set (Data): The data object to invert x
    Returns:
        A data object with inverted x and evenly spaced x
    """
    if not np.all(data_set.x[:-1] <= data_set.x[1:]):
        raise ValueError('Array to invert not sorted!')
    interp = interp1d(*data_set.both, bounds_error=False,
                      fill_value=(data_set.y[0], data_set.y[1])
                      )  # Needs fill_value for floating point errors
    new_x = np.linspace(1. / data_set.x.max(), 1. / data_set.x.min(),
                        len(data_set))
    return Data(new_x, interp(1 / new_x))


def loess(data_set, x_window, polyorder):
    """
This performs a Local regression smooth on the data and outputs a new
data object. It uses scipy.signal.savgol_filter
Args:
    data_set (Data): The data set to be smoothed
    x_range (float): The window length to use for the smoothing
    polyorder (int): The order of polynomial to fit
Returns:
    A data object of smoothed data.
"""
    if (np.max(data_set.x) - np.min(data_set.x)) < x_window:
        raise ValueError(
            "The loess window is longer than the given data range")
    spacing = data_set.x[1] - data_set.x[0]
    if not np.isclose(spacing, np.diff(data_set.x)).all():
        raise ValueError('The data needs to be evenly spaced to smooth')
    smooth_data = savgol_filter(data_set.y,
                                2 * int(round(0.5 * x_window / spacing)) + 1,
                                polyorder=polyorder)
    return Data(data_set.x, smooth_data)


def poly_reg(data_set, polyorder):
    """
This performs a polynomial fit to the data to smooth it and outputs the
smoothed data. It uses numpy.polyfit
Args:
    data_set (Data): The data set to be smoothed
    polyorder (int): The order of the polynomial to fit
Returns:
    A data set of the fitted values
"""
    fit = np.polyfit(*data_set.both, deg=polyorder)
    y_vals = 0.
    for n, p in enumerate(fit):
        y_vals += p * np.power(data_set.x, polyorder - n)
    return Data(data_set.x, y_vals)


def FFT(data_set, n=6553600, window='hanning', freq_cut=0):
    """
This performs an FFT on the data set and outputs another.
Args:
    data_set (Data): The data to perform the FFT on
    n (int default:65536): The number of points to FFT, extra points
        will be added with zero pading
    window (str default:'hanning'): The windowing function to use the
        list is given in scipy.signal.get_window
    freq_cut (float default:0): If given the frequencies higher than this
        are not included with the FFT
Returns:
    A data set of the resulting FFT
"""
    spacing = data_set.x[1] - data_set.x[0]
    if not np.isclose(spacing,
                      np.diff(data_set.x)).all():
        raise ValueError('The data needs to be evenly spaced to smooth')
    fft_vals = np.abs(np.fft.rfft(data_set.y * get_window(window,
                                                          len(data_set)),
                                  n=n))
    fft_freqs = np.fft.rfftfreq(n, d=spacing)
    freq_arg = None
    if freq_cut > 0:
        freq_arg = np.searchsorted(fft_freqs, freq_cut)
    return Data(fft_freqs[0:freq_arg], fft_vals[0:freq_arg])


def get_peaks(data_set, n_peaks=4, **kwargs):
    """
Finds the four largest peaks in the data set and output a two column
numpy array with the frequencies and amplitudes.
Args:
    data_set (Data): The FFT data to find the peaks in
    n_peaks (int): The number of peaks to find
    **kwargs will be passed to scipy.singal.find_peaks
Returns:
    A two column numpy array with frequencies and amplitudes
"""
    peak_args = find_peaks(data_set.y, **kwargs)[0]
    if len(peak_args) < n_peaks:
        print('Few peaks were found, try reducing restrictions.')
    peak_args = peak_args[data_set.y[peak_args].argsort()
                ][:-(n_peaks + 1):-1]
    return np.concatenate([data_set.x[peak_args, None],
                           data_set.y[peak_args, None]], axis=1)


def peak_height(data_set, position, x_range, x_value=False):
    """
This takes a data set around a positions and in a certain range takes
the largest value and outputs that values hight. This is useful for
extracting peak heights from an FFT.
Args:
    data_set (Data): The data object to get the peak height from
    position (float): The central location of the peak in x
    x_range (float): The range in x to look for the peak
    x_value (bool, default:False): If true x value also produced
Returns:
    If x_value is false a float is returned with the hight of the peak
    If x_value is true a np array with x and y values of peak
"""
    trimmed = data_set.x_cut(position - x_range / 2, position + x_range / 2)
    peak_arg = np.argmax(trimmed.y)
    if x_value:
        return np.array([trimmed.x[peak_arg], trimmed.y[peak_arg]])
    else:
        return trimmed.y[peak_arg]


def counting_freq(start_field, end_field, number_peaks):
    """
This can give you the frequency from counting peaks.
Performs n*B1*B2/(B2-B1)
Args:
    start_field (float): The lowest field in the counting range
    end_field (float): The highest field in the counting range
    number_peaks (float): The number of peaks in the range
Returns
    The frequency in Tesla as a float
"""
    return number_peaks * start_field * end_field / (end_field - start_field)


def counting_freq(start_field, end_field, frequency):
    """
This provides the number of peaks you expect to count for a frequency.
Performs Freq*(B2-B1)/(B1*B2)
Args:
    start_field (float): The lowest field in the counting range
    end_field (float): The highest field in the counting range
    frequency (float): The frequency in Tesla
Returns
    The number of peaks as a float
"""
    return frequency * (end_field - start_field) / (start_field * end_field)


class QO():
    """
This class is designed to keep all the information for one sweep together
The data given needs to be a ga.Data class or objects that can be passed
to make one.

The first set of attributes are the same as the parameters passed to the
class in initialisation. The remaining are mostly ga.Data objects that
are produced in the steps of analysis the quantum oscillations.

Attributes:
    raw (ga.Data): Original data passed to the class
    min_field (float): The minimum field to be considered
    max_field (float): The maximum field to be considered
    step_size (float): The spacing between the field points to
        be interpolated
    interp (ga.Data): The sweep cut to the field range and interpolated
        evenly in field
    sub (ga.Data): The sweep with the background subtracted
    invert (ga.Data): The background subtracted sweep evenly interpolated
        in inverse field
    fft (ga.Data): The flourier transform of the inverse
"""

    def __init__(self, data, min_field, max_field, subtract_func,
                 step_size=None, fft_cut=0, n=65536):
        if type(data) != Data:
            try:
                data = Data(data)
            except:
                raise TypeError('Not given data class!\n' \
                                'Was given {}'.format(type(data)))

        self.raw = data
        self.min_field = min_field
        self.max_field = max_field
        self.n = n

        if np.min(self.raw.x) > min_field:
            raise ValueError(
                "max_field value to interpolate is below data")
        if np.max(self.raw.x) < max_field:
            raise ValueError(
                "max_field value to interpolate is above data")

        if step_size == None:
            self.step_size = np.abs(np.average(np.diff(data.x))) / 4
        else:
            self.step_size = step_size

        self.interp = self.raw.interp_range(min_field, max_field,
                                            self.step_size)
        self.sub = subtract_func(self.interp)
        self.invert = invert_x(self.sub)
        self.fft = FFT(self.invert, freq_cut=fft_cut, n=n)

    def __dir__(self):
        return ['raw', 'min_field', 'max_field', 'step_size',
                'interp', 'sub', 'invert', 'fft', 'peaks',
                'peak_hight', 'FFT_again']

    def __len__(self):
        return self.interp.x.size

    def _repr_html_(self):
        return print('Quantum Oscillation object:\n' \
                     'Field Range {:.2f} to  {:.2f} \n'.format(
            self.min_field, self.max_field))

    def peaks(self, n_peaks=4, **kwargs):
        """
Calls ga.get_peaks on the FTT
Finds the four largest peaks in the data set and output a two column
numpy array with the frequencies and amplitudes.
Args:
    n_peaks (int): The number of peaks to find
    **kwargs will be passed to scipy.singal.find_peaks
Returns:
    A two column numpy array with frequencies and amplitudes
"""
        return get_peaks(self.fft, n_peaks, **kwargs)

    def peak_hight(self, position, x_range, x_value=False):
        """
Calls ga.peak_hight on the FFT
This takes a data set around a positions and in a certain range takes
the largest value and outputs that values hight. This is useful for
extracting peak heights from an FFT.
Args:
    position (float): The central location of the peak in x
    x_range (float): The range in x to look for the peak
    x_value (bool, default:False): If true x value also produced
Returns:
    If x_value is false a float is returned with the hight of the peak
    If x_value is true a np array with x and y values of peak
"""
        return peak_height(self.fft, position, x_range, x_value=False)

    def FFT_again(self, n=65536, window='hanning', freq_cut=0):
        """
Recalculates the FTT and returns it, also saved to self.fft
This is so the extra settings can be used. Makes use of ga.FFT
Args:
    n (int default:65536): The number of points to FFT, extra points
        will be added with zero pading
    window (str default:'hanning'): The windowing function to use the
        list is given in scipy.signal.get_window
    freq_cut (float default:0): If given the frequencies higher than this
        are not included with the FFT
"""
        self.fft = FFT(self.invert, n=n, window=window, freq_cut=freq_cut)
        return self.fft

    def to_csv(self, file_name, sep=','):
        """
This saves the data in a csv file. It includes the interpolated,
subtracted, inverse signals as well as the FFT. The FFT is
interpolated to be the same length as the interpolated data.
Args:
    file_name (str): The file name to save the data
    sep (str default:','): The character to delimitate the data
"""
        if file_name[-4:] not in ['.csv', '.txt', '.dat']:
            file_name += '.csv'

        output_data = np.concatenate([
            self.interp.values,
            self.sub.values,
            self.invert.values,
            self.fft.interp_number(len(self)).values,
        ], axis=1)
        header_line = 'Field_Interp{0:s}Interp_Signal{0:s}' \
                      'Field_Sub{0:s}Sub_Signal{0:s}' \
                      'Inverse_Field{0:s}Inverse_Signal{0:s}' \
                      'FFT_freq{0:s}FFT_amp'.format(sep)
        np.savetxt(file_name, output_data,
                   delimiter=sep, comments='',
                   header=header_line)


class QO_loess(QO):
    """
This class is designed to keep all the information for one sweep together
The data given needs to be a ga.Data class or objects that can be passed
to make one.

This class is a subclass of ga.QO
This class is using LOESS background fitting to perform the subtraction.

The first set of attributes are the same as the parameters passed to the
class in initialisation. The remaining are mostly ga.Data objects that
are produced in the steps of analysis the quantum oscillations.

Attributes:
    raw (ga.Data): Original data passed to the class
    min_field (float): The minimum field to be considered
    max_field (float): The maximum field to be considered
    loess_win (float): The window length to be passed to ga.loess
    loess_poly (float): The polynomial order to be
        passed to ga.loess_poly
    step_size (float): The spacing between the field points to
        be interpolated
    interp (ga.Data): The sweep cut to the field range and interpolated
        evenly in field
    sub (ga.Data): The sweep with the background subtracted
    invert (ga.Data): The background subtracted sweep evenly interpolated
        in inverse field
    fft (ga.Data): The flourier transform of the inverse
"""

    def __init__(self, data, min_field, max_field, loess_win, loess_poly,
                 step_size=None, fft_cut=0, n=65536):
        def bg_sub(interp):
            return interp - loess(interp, loess_win, loess_poly)

        QO.__init__(self, data, min_field, max_field, bg_sub,
                    step_size=step_size, fft_cut=fft_cut, n=n)

        self.loess_win = loess_win
        self.loess_poly = loess_poly
        self.n = n

    def __dir__(self):
        return [*QO.__dir__(self), 'loess_poly', 'loess_win']

    def _repr_html_(self):
        return print('Quantum Oscillation object:\n' \
                     'LOESS Background Subtraction\n' \
                     'Field Range {:.2f} to  {:.2f} \n' \
                     'LOESS polynomial {:.2f}\n' \
                     'LOESS window {:.2f}\n'.format(
            self.min_field, self.max_field,
            self.loess_poly, self.loess_win))


class QO_loess_av(QO_loess):
    """
This class is designed to keep all the information for one sweep together
The data given needs to be a ga.Data class or objects that can be passed
to make one.

This class is a subclass of ga.QO_loess
This class is using LOESS background fitting to perform the subtraction.

The first set of attributes are the same as the parameters passed to the
class in initialisation. The remaining are mostly ga.Data objects that
are produced in the steps of analysis the quantum oscillations.

Attributes:
    raw ([ga.Data]): Original data passed to the class in the from of a list
        of ga.Data objects
    min_field (float): The minimum field to be considered
    max_field (float): The maximum field to be considered
    loess_win (float): The window length to be passed to ga.loess
    loess_poly (float): The polynomial order to be
        passed to ga.loess_poly
    step_size (float): The spacing between the field points to
        be interpolated
    interp (ga.Data): The sweep cut to the field range and interpolated
        evenly in field
    sub (ga.Data): The sweep with the background subtracted
    invert (ga.Data): The background subtracted sweep evenly interpolated
        in inverse field
    fft (ga.Data): The flourier transform of the inverse
"""

    def __init__(self, data_list, min_field, max_field, loess_win,
                 loess_poly, step_size=None, fft_cut=0, n=6553600):
        if type(data_list) != list:
            raise TypeError('Not given a list of ga.Data class\n' \
                            'Was given {}'.format(type(data_list)))
        self.raw = []
        for data in data_list:
            if type(data) != Data:
                try:
                    self.raw.append(Data(data))
                except:
                    raise TypeError('Not given data class in list!\n' \
                                    'Was given {}'.format(type(data)))
            else:
                self.raw.append(data)

        self.min_field = min_field
        self.max_field = max_field
        self.loess_win = loess_win
        self.loess_poly = loess_poly
        self.n = n

        if step_size == None:
            self.step_size = np.min(
                [np.abs(np.average(np.diff(data.x))) / 4 for data in self.raw])
        else:
            self.step_size = step_size

        interp_list = []
        for data in self.raw:
            interp_list.append(data.interp_range(min_field, max_field,
                                                 self.step_size))

        def bg_sub(interp):
            return interp - loess(interp, loess_win, loess_poly)

        sub_list = []
        for interp in interp_list:
            sub_list.append(bg_sub(interp))

        self.interp = mean(interp_list)
        self.sub = mean(sub_list)
        self.invert = invert_x(self.sub)
        self.fft = FFT(self.invert, freq_cut=fft_cut, n=n)


# @title ## Data Fitting Object { display-mode: "form", run: "auto" }
"""GigaAnalysis - Fitting

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as sp_curve_fit


class Fit_result():
    """This class is to hold the results of the fits on data objects.

    Parameters
    ----------
    func : function
        The function used in the fitting.
    popt : numpy.ndarray
        The optimum values for the parameters.
    pcov : numpy.ndarray
        The estimated covariance.
    results : gigaanalysis.data.Data
        The optimal values obtained from the fit, will be
        none if `full`=`False` when performing the fit.
    residuals : gigaanalysis.data.Data
        The residuals of the fit, will be none
        if `full`=`False` when performing the fit.

    Attributes
    ----------
    func : function
        The function used in the fitting.
    popt : numpy.ndarray
        The optimum values for the parameters.
    pcov : numpy.ndarray
        The estimated covariance.
    results : gigaanalysis.data.Data
        The optimal values obtained from the fit, will be
        none if `full`=`False` when performing the fit.
    residuals : gigaanalysis.data.Data
        The residuals of the fit, will be none
        if `full`=`False` when performing the fit.

    """

    def __init__(self, func, popt, pcov, results, residuals):
        """The __init__ method to produce the fit_result class
        The point of this class is to store the results from a gigaanalysis
        fit so the arguments are the same as the attributes.
        """
        self.func = func
        self.popt = popt
        self.pcov = pcov
        self.results = results
        self.residuals = residuals

    def __str__(self):
        return np.array2string(self.popt)

    def __repr__(self):
        return 'GA fit results:{}'.format(self.popt)

    def _repr_html_(self):
        return 'GA fit results:{}'.format(self.popt)

    def __dir__(self):
        return ['func', 'popt', 'pcov', 'results', 'residuals', 'predict']

    def __len__(self):
        return self.popt.size

    def predict(self, x_vals):
        """This takes a value or an array of x_values and calculates the
        predicated y_vales.

        Parameters
        ----------
        x_vals : numpy.ndarray
            An array of x_vales.

        Returns
        -------
        y_vals : gigaanalysis.data.Data
            An Data object with the predicted y_values.

        """
        return Data(x_vals, self.func(x_vals, *self.popt))


def curve_fit(data_set, func, p0=None, full=True, **kwargs):
    """This is an implementation of :func:`scipy.optimize.curve_fit`
    for acting on :class:`gigaanalysis.data.Data` objects. This performs
    a least squares fit to the data of a function.

    Parameters
    ----------
    data_set : gigaanalysis.data.Data
        The data to perform the fit on.
    func : function
        The model function to fit. It must take the x values as
        the first argument and the parameters to fit as separate remaining
        arguments.
    p0 : numpy.ndarray, optional
        Initial guess for the parameters. Is passed to
        :func:`scipy.optimize.curve_fit` included so it can be addressed
        positionally. If `None` unity will be used for every parameter.
    full : bool, optional
        If `True`, `fit_result` will include residuals, and if `False`
        they will not be calculated and only results included.
    kwargs:
        Keyword arguments are passed to :func:`scipy.optimize.curve_fit`.

    Returns
    -------
    fit_result : gigaanalysis.fit.Fit_result
        A gigaanalysis Fit_result object containing the results

    """
    popt, pcov = sp_curve_fit(func, data_set.x, data_set.y, p0=p0, **kwargs)
    if full:
        results = Data(data_set.x, func(data_set.x, *popt))
        residuals = data_set - results
    else:
        results, residuals = None, None
    return Fit_result(func, popt, pcov, results, residuals)


def any_poly(x_data, *p_vals, as_Data=False):
    """The point of this function is to generate the values expected from a
    linear fit. It is designed to take the values obtained from
    :func:`numpy.polyfit`.
    For a set of p_vals of length n+1 ``y_data = p_vals[0]*x_data**n +
    p_vals[0]*x_data**(n-1) + ... + p_vals[n]``

    Parameters
    ----------
    x_data :  numpy.ndarray
        The values to compute the y values of.
    p_vals : float
        These are a series of floats that are the coefficients of the
        polynomial starting with with the highest power.
    as_Data : bool, optional
        If False returns a :class:`numpy.ndarray` which is the default
        behaviour. If True returns a :class:`gigaanalysis.data.Data` object
        with the x values given and the cosponsoring y values.

    Returns
    -------
    results : numpy.ndarray or gigaanalysis.data.Data
        The values expected from a polynomial with the
        specified coefficients.

    """
    results = x_data * 0
    for n, p in enumerate(p_vals[::-1]):
        results += p * np.power(x_data, n)
    if as_Data:
        return ga.Data(x_data, results)
    else:
        return results


def poly_fit(data_set, order, full=True):
    """This function fits a polynomial of a certain order to a given
    data set. It uses :func:`numpy.polyfit` for the fitting. The function
    which is to produce the data is :func:`gigaanalysis.fit.any_poly`.

    Parameters
    ----------
    data_set : gigaanalysis.data.Data
        The data set to perform the fit on.
    order : int
        The order of the polynomial.
    full : bool, optional
        If True fit_result will include residuals if False they will
        not be calculated and only results included.

    Returns
    -------
    fit_result : gigaanalysis.fit.Fit_result
        A gigaanalysis Fit_result object containing the results the
        fit parameters are the coefficients of the polynomial. Follows the
        form of :func:`gigaanalysis.fit.any_poly`.

    """
    popt, pcov = np.polyfit(data_set.x, data_set.y, order, cov=True)
    func = any_poly
    if full:
        results = Data(data_set.x, func(data_set.x, *popt))
        residuals = data_set - results
    else:
        results, residuals = None, None
    return Fit_result(func, popt, pcov, results, residuals)


def make_sin(x_data, amp, wl, phase, offset, as_Data=False):
    """This function generates sinusoidal signals
    The form of the equation is
    ``amp*np.sin(x_data*np.pi*2./wl + phase*np.pi/180.) + offset``

    Parameters
    ----------
    x_data :  numpy.ndarray
        The values to compute the y values of.
    amp : float
        Amplitude of the sin wave.
    wl : float
        Wavelength of the sin wave units the same as `x_data`.
    phase : float
        Phase shift of the sin wave in deg
    offset : float
        Shift of the y values
    as_Data : bool, optional
        If False returns a :class:`numpy.ndarray` which is the default
        behaviour. If True returns a :class:`gigaanalysis.data.Data` object
        with the x values given and the cosponsoring y values.

    Returns
    -------
    results : numpy.ndarray or gigaanalysis.data.Data
        The values expected from the sinusoidal with the given parameters

    """
    results = amp * np.sin(x_data * np.pi * 2. / wl + phase * np.pi / 180.) + offset
    if as_Data:
        return ga.Data(x_data, results)
    else:
        return results


def sin_fit(data_set, full=True):
    """This function fits a polynomial of a certain order to a given
    data set. It uses :func:`numpy.polyfit` for the fitting. The function
    which is to produce the data is :func:`gigaanalysis.fit.any_poly`.

    Parameters
    ----------
    data_set : gigaanalysis.data.Data
        The data set to perform the fit on.
    full : bool, optional
        If True fit_result will include residuals if False they will
        not be calculated and only results included.

    Returns
    -------
    fit_result : gigaanalysis.fit.Fit_result
        A gigaanalysis Fit_result object containing the results the
        fit parameters are the coefficients of the polynomial. Follows the
        form of :func:`gigaanalysis.fit.any_poly`.

    """
    popt, pcov = np.polyfit(data_set.x, data_set.y, order, cov=True)
    func = any_poly
    if full:
        results = Data(data_set.x, func(data_set.x, *popt))
        residuals = data_set - results
    else:
        results, residuals = None, None
    return Fit_result(func, popt, pcov, results, residuals)


# @title ## Functions for Notebook { display-mode: "form", run: "auto" }

def remove_copies(data):
    # if_copy_y = data.y[1:] != data.y[:-1]
    if_copy_x = data.x[1:] != data.x[:-1]
    if_copy = if_copy_x  # if_copy_y &
    if_copy = if_copy[1:] & if_copy[:-1]
    if_copy = np.concatenate([[True], if_copy, [True]])
    return Data(data.values[if_copy, :])


def sort_func(filepath):
    name = filepath.split('/')[-1]
    angle = name.split('_')[1].split('d')[0]
    return float(angle)


def sort_func_temp(filepath):
    name = filepath.split('/')[-1]
    temp = name.split('_')[0].split('K')[0]
    return float(temp)


def wait_for_change(widget, value):
    future = asyncio.Future()

    def getvalue(change):
        # make the new value available
        future.set_result(change.new)
        widget.unobserve(getvalue, value)

    widget.observe(getvalue, value)
    return future


def load_that_data(path_to_folder, remove_copies_from_data=False, sort_by='Temperature'):
    dset = {}

    meta_df = pd.DataFrame(
        columns=['Min Field (T)', 'Max Field (T)', 'Num Points', 'Direction', 'Temp (K)', 'Angle', 'Sweep'])

    data_locs = glob(
        path_to_folder + "*.csv"
    )

    if sort_by == 'Temperature':

        data_locs.sort(key=sort_func_temp)
    elif sort_by == 'Angle':

        data_locs.sort(key=sort_func)
    else:

        print('Use correct sort mechanism you goofball!')
    for n, file in enumerate(data_locs):
        field, volt = pd.read_csv(
            file, sep=',').values.T
        filename = file.split('/')[-1]
        temp_of_run = filename.split('_')[0]
        angle = filename.split('_')[1]
        sweep = filename.split('_')[2]
        direc = filename.split('_')[3].split('.')[0]

        meta_data = {}  # Making meta data to save
        meta_data['Max Field (T)'] = np.max(field)
        meta_data['Min Field (T)'] = np.min(field)
        meta_data['Num Points'] = len(field)
        meta_data['Direction'] = direc
        meta_data['Temp (K)'] = temp_of_run
        meta_data['Angle'] = angle
        key = f"s{n + 1:03d}"
        meta_data['Sweep'] = key

        if remove_copies_from_data == True:
            dset[key] = remove_copies(Data(field, volt, strip_sort=True))
        else:
            dset[key] = Data(field, volt, strip_sort=True)
        meta_df.loc[key] = meta_data

    return dset, meta_df


def get_them_groups(meta_df, group):
    groups = meta_df.groupby(group, sort=False)
    d = {}
    for name, g in groups:
        d[str(name)] = list(g['Sweep'])

    return d


def loess_errything(data_dict, min_field, max_field, loess_window, loess_poly, fft_cut, n=65536, step_size=None):
    loesses = {}

    for key, dat in data_dict.items():
        loesses[key] = QO_loess(dat,
                                min_field, max_field, loess_window, loess_poly,
                                step_size=step_size, fft_cut=fft_cut, n=n)

    return loesses


def get_freq_spacing(fft):
    return np.max(fft.x) / len(fft.x)


def find_peaks_in_region(fft, approx_freqs, interval_size=200):
    peak_lst, freq_list, field_lst = [], [], []

    spacing = get_freq_spacing(fft)
    half_interval = interval_size / 2
    num_inds = int(round(half_interval / spacing, 0))

    for f in approx_freqs:

        f_ind = int(round(f / spacing, 0))
        if f_ind - num_inds < 0:
            start = 0
        else:
            start = f_ind - num_inds
        if f_ind + num_inds > len(fft.x):
            end = len(fft.x) - 1
        else:
            end = f_ind + num_inds

        interval = np.arange(start, end, dtype=np.int)

        fft_amp = fft.y[interval]
        fft_freq = fft.x[interval]
        maxim = np.max(fft_amp)
        max_f = fft_freq[np.argmax(fft_amp)]

        peak_lst.append(maxim)
        freq_list.append(max_f)

    return peak_lst, freq_list


def get_dominant_frequencies(loess_dict, groups_dict, approx_freqs, check=False, interval_size=100):
    avg_dict, std_dict = {}, {}
    freq_dict, freq_std_dict = {}, {}
    for temp, relevant_keys in groups_dict.items():
        fft_amps, fft_freqs = [], []
        for c, key in enumerate(relevant_keys):
            qos = loess_dict[key]
            fft_dat = qos.fft
            lst, f_lst = find_peaks_in_region(fft_dat, approx_freqs, interval_size=interval_size)
            fft_amps.append(lst)
            fft_freqs.append(f_lst)

        if check:
            print(fft_amps)

        arr = np.array(fft_amps)
        arr2 = np.array(fft_freqs)

        avg_dict[temp] = np.average(arr, axis=0)
        std_dict[temp] = np.std(arr, axis=0)
        freq_dict[temp] = np.average(arr2, axis=0)
        freq_std_dict[temp] = np.std(arr2, axis=0)

    return avg_dict, std_dict, freq_dict


def lifshitz_kosevich(temps, e_mass, amp, field=20.0):  # 31.876923076923077

    kb = 1.380649e-23
    me = 9.1093837015e-31
    hbar = 1.054571817e-34
    qe = 1.602176634e-19

    chi = 2 * np.pi * np.pi * kb * temps * me * e_mass / (hbar * qe * field)

    r_lk = amp * chi / np.sinh(chi)

    return r_lk


def lk_field_val(min_field, max_field):
    denom = 1 / min_field + 1 / max_field
    return 2 / denom

