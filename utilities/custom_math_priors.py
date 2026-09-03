"""
FROM GWCOSMO

This module collects analytical and numerical probability density functions.
"""

import copy as _copy
import json as _json

import numpy as _np
from scipy.interpolate import interp1d as _interp1d
from scipy.special import erf as erf
from scipy.special import logsumexp as _logsumexp
from scipy.integrate import cumulative_trapezoid as _cumulative_trapezoid
import math
import torch
# from torch.special import erf
torch.set_num_threads(1)  # to avoid core-hogging

def high_pass_filter(mass, mmin, delta_m):
    """
    This function return the value of the window function defined as Eqs B6 and B7 of https://arxiv.org/pdf/2010.14533.pdf

    Parameters
    ----------
    mass: np.array or float
        array of x or masses values
    mmin: float or np.array (in this case len(mmin) == len(mass))
        minimum value of window function
    delta_m: float or np.array (in this case len(delta_m) == len(mass))
        width of the window function

    Returns
    -------
    Values of the window function
    """

    if not isinstance(mass, torch.Tensor):
        mass = torch.as_tensor([mass])  # TODO check if this needs type casting?

    to_ret = torch.ones_like(mass, device=mass.device, dtype=mass.dtype)
    if delta_m == 0:
        return to_ret

    mprime = mass - mmin

    # Defines the different regions of the window function ad in Eq. B6 of  https://arxiv.org/pdf/2010.14533.pdf
    select_window = (mass > mmin) & (mass < (delta_m + mmin))
    select_one = mass >= (delta_m + mmin)
    select_zero = mass <= mmin

    effe_prime = torch.ones_like(mass)

    # Definethe f function as in Eq. B7 of https://arxiv.org/pdf/2010.14533.pdf
    effe_prime[select_window] = torch.exp(
        torch.nan_to_num(
            (delta_m / mprime[select_window]) + (delta_m / (mprime[select_window] - delta_m))
        )
    )
    to_ret = 1.0 / (effe_prime + 1)
    to_ret[select_zero] = 0.0
    to_ret[select_one] = 1.0
    return to_ret


def low_pass_filter(mass, mmax, delta_m):
    """
    Parameters
    ----------
    mass: np.array or float
        array of x or masses values
    mmax: float or np.array (in this case len(mmin) == len(mass))
        maximum value of window function
    delta_m: float or np.array (in this case len(delta_m) == len(mass))
        width of the window function

    Returns
    -------
    Values of the window function
    """

    if not isinstance(mass, torch.Tensor):
        mass = torch.Tensor([mass])

    to_ret = torch.ones_like(mass, device = mass.device, dtype = mass.dtype)
    if delta_m == 0:
        return to_ret

    mprime = mmax - mass

    # Defines the different regions of thw window function ad in Eq. B6 of  https://arxiv.org/pdf/2010.14533.pdf
    select_window = (mass < mmax) & (mass > (mmax - delta_m))
    select_one = mass <= (mmax - delta_m)
    select_zero = mass >= mmax

    effe_prime = torch.ones_like(mass, device=mass.device, dtype=mass.dtype)

    # Definethe f function as in Eq. B7 of https://arxiv.org/pdf/2010.14533.pdf
    effe_prime[select_window] = torch.exp(
        torch.nan_to_num(
            (delta_m / mprime[select_window]) + (delta_m / (mprime[select_window] - delta_m))
        )
    )
    to_ret = 1.0 / (effe_prime + 1)
    to_ret[select_zero] = 0.0
    to_ret[select_one] = 1.0
    return to_ret


def notch_filter(mass, notch_right, right_smooth, notch_left, left_smooth, A):
    """
    This function returns a notch filter based on the one defined in eq. (4) of https://arxiv.org/pdf/2111.03498.pdf, but using the

    Parameters
    ----------
    mass: np.array or float
        array of x or masses values in solar masses
    high_pass_min,low_pass_max: float
        maximum value of window function (maximum credible mass of the spectrum)
    right_smooth, right_smooth: float
        width of the high or low pass window function
    A: float
        Fraction of the signal to substract

    Returns
    -------
    Values of the notch filter
    """

    filter = 1 - A * high_pass_filter(mass, notch_left, left_smooth) * low_pass_filter(
        mass, notch_right, right_smooth
    )
    return filter


def get_PL_norm(alpha, minv, maxv):
    """
    This function returns the powerlaw normalization factor

    Parameters
    ----------
    alpha: float
        Powerlaw slope
    min_pl: float
        lower cutoff
    max_pl: float
        low_pass_max cutoff
    """

    # Get the PL norm as in Eq. 24 on the tex document
    if alpha == -1:
        return torch.log(torch.tensor(maxv / minv))
    else:
        return torch.as_tensor((maxv**( alpha + 1) - minv**( alpha + 1)) / (alpha + 1))


def get_gaussian_norm(mu, sigma, min_g, max_g):
    """
    This function returns the gaussian normalization factor

    Parameters
    ----------
    mu: float
        mu of the gaussian
    sigma: float
        standard deviation of the gaussian
    min_pl: float
        lower cutoff
    max_pl: float
        low_pass_max cutoff
    """

    # Get the gaussian norm as in Eq. 28 on the tex document
    max_point = (max_g - mu) / (sigma * math.sqrt(2.0))
    min_point = (min_g - mu) / (sigma * math.sqrt(2.0))
    # Ensure inputs are tensors for torch.erf
    max_point_t = torch.as_tensor(max_point)
    min_point_t = torch.as_tensor(min_point)
    return 0.5 * torch.erf(max_point_t) - 0.5 * torch.erf(min_point_t)


class torch_evaluatable_interpolator:
    def __init__(self, xp, fp, fill_value=[0.,0.]) -> None:
        self.xp = xp
        self.fp = fp
        self.fill = fill_value
        self.device=self.xp.device
        self.dtype=self.xp.dtype

    def interp(self, x, xp, fp):
        idx = torch.searchsorted(xp, x, side='right')  # where our sample points lie on the xp grid
        idx[idx  > len(xp)-1] = len(xp) - 1
        x1 = xp[idx]
        x0 = xp[idx-1]
        y1 = fp[:,idx]
        y0 = fp[:,idx-1]

        interps = (y0 * (x1 - x)[None,:] + y1 * (x - x0)[None,:]) / (x1 - x0)[None,:]
        return interps

    def __call__(self,x):
        out = self.interp(x, self.xp, torch.atleast_2d(self.fp))
        if self.fill[1] is not None:
            out[:,x>self.xp[-1]] = self.fill[1]
        out[:,x < self.xp[0]] = self.fill[0]
        return torch.squeeze(out)
 
    def set_device(self, device):
        self.xp = self.xp.to(device)
        self.fp = self.fp.to(device)
        self.device=device

    def set_dtype(self, dtype):
        self.xp = self.xp.type(dtype)
        self.fp = self.fp.type(dtype)
        self.dtype=dtype


class SmoothedProb(object):
    """
    Class for smoothing the low part of a PDF. The smoothing follows Eq. B7 of
    2010.14533.

    Parameters
    ----------
    origin_prob: class
        Original prior class to smooth from this module
    high_pass_min: float
        minimum cut-off. Below this, the window is 0.
    high_pass_smooth: float
        smooth factor. The smoothing acts between high_pass_min and high_pass_min+high_pass_smooth
    """

    def __init__(self, origin_prob, high_pass_min, high_pass_smooth, device="cpu"):

        self.origin_prob = _copy.deepcopy(origin_prob)
        self.origin_prob.set_device(device)
        self.high_pass_smooth = high_pass_smooth
        self.high_pass_min = high_pass_min
        self.maximum = self.origin_prob.maximum
        self.minimum = self.origin_prob.minimum
        self.device = device
        self.dtype = torch.get_default_dtype()

        # Find the values of the integrals in the region of the window function before and after the smoothing
        int_array = torch.linspace(high_pass_min, high_pass_min + high_pass_smooth,1000, device = self.device, dtype=self.dtype)
        integral_before = torch.trapezoid(self.origin_prob.prob(int_array),int_array)
        integral_now = torch.trapezoid(self.prob(int_array),int_array)

        self.integral_before = integral_before
        self.integral_now = integral_now
        # Renormalize the the smoother function.
        self.norm = 1 - integral_before + integral_now - self.origin_prob.cdf(high_pass_min)

        x_eval = torch.logspace(
            torch.log10(torch.tensor(high_pass_min)), torch.log10(torch.tensor(high_pass_min + high_pass_smooth)), 1000, device=self.device
        )
    
        cdf_numeric = torch.cumulative_trapezoid(self.prob(x_eval), x=x_eval)    
        self.cached_cdf_window = torch_evaluatable_interpolator(x_eval[:-1:], cdf_numeric, fill_value=[0., None])
    

    def set_device(self, device):
        self.cached_cdf_window.set_device(device)
        self.origin_prob.set_device(device)
        self.integral_before = self.integral_before.to(device)
        self.integral_now = self.integral_now.to(device)
        self.device=device

    def set_dtype(self, dtype):
        self.integral_before = self.integral_before.type(dtype)
        self.integral_now = self.integral_now.type(dtype)
        self.norm = self.norm.type(dtype)
        self.cached_cdf_window.set_dtype(dtype)
        self.origin_prob.set_dtype(dtype)

        self.dtype = dtype

    def prob(self, x):
        """
        Returns the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        return torch.exp(self.log_prob(x))

    def log_prob(self, x):
        """
        Returns the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        # Return the window function
        window = high_pass_filter(x, self.high_pass_min, self.high_pass_smooth)
        if hasattr(self, "norm"):
            prob_ret = self.origin_prob.log_prob(x) + torch.log(window) - torch.log(self.norm)
        else:
            prob_ret = self.origin_prob.log_prob(x) + torch.log(window)

        return prob_ret

    def log_conditioned_prob(self, x, a, b):
        """
        Returns the conditional probability between two new boundaries [a,b]

        Parameters
        ----------
        x: np.array
            Value at which compute the probability
        a: np.array
            New lower boundary
        b: np.array
            New low_pass_max boundary
        """

        to_ret = self.log_prob(x)
        # Find the new normalization in the new interval
        new_norm = self.cdf(b) - self.cdf(a)
        # Apply the new normalization and put to zero all the values above/below the interval
        wok = torch.where(new_norm > 0)[0]
        if to_ret.dim() > 0:
            to_ret[wok] -= torch.log(new_norm[wok])
        elif wok:
            to_ret -= torch.log(new_norm)
        wnull = torch.where(new_norm <= 0)[0]
        if len(wnull) > 0:
            to_ret[wnull] = -torch.inf

        to_ret[(x < a) | (x > b)] = -torch.inf
        return to_ret

    def cdf(self, x):
        """
        Returns the cumulative density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the cumulative
        """

        to_ret = torch.ones_like(x, device=self.device)
        to_ret[x < self.high_pass_min] = 0.0
        to_ret[(x >= self.high_pass_min) & (x <= (self.high_pass_min + self.high_pass_smooth))] = (
            self.cached_cdf_window(
                x[(x >= self.high_pass_min) & (x <= (self.high_pass_min + self.high_pass_smooth))]
            )
        )
        to_ret[x >= (self.high_pass_min + self.high_pass_smooth)] = (
            self.integral_now
            + self.origin_prob.cdf(x[x >= (self.high_pass_min + self.high_pass_smooth)])
            - self.origin_prob.cdf(self.high_pass_min + self.high_pass_smooth)
        ) / self.norm

        return to_ret

    def conditioned_prob(self, x, a, b):
        """
        Returns the conditional probability between two new boundaries [a,b]

        Parameters
        ----------
        x: np.array
            Value at which compute the probability
        a: np.array
            New lower boundary
        b: np.array
            New low_pass_max boundary
        """

        return torch.exp(self.log_conditioned_prob(x, a, b))


class SmoothedDipProb(object):
    """
    Class for low pass and high pass smoothing of a PDF, and adding a dip. The smoothing follows Eq. B7 of
    2010.14533.

    Parameters
    ----------
    origin_prob: class
        Original probability
    right_smooth: float
        high pass window
    left_smooth: float
        low pass window
    notch_lower: float
        Where to find the start of the dip
    notch_upper: float
        Where to find the end of the dip
    notch_lower_smooth: float
        The window size for the left part of the smooth
    notch_upper_smooth: float
        The window size for the right part of the smooth
    A: float
        The fraction of pdf to suppress in the dip.
    """

    def __init__(
        self,
        origin_prob,
        right_smooth,
        left_smooth,
        A,
        notch_lower,
        notch_lower_smooth,
        notch_upper,
        notch_upper_smooth,
        device= "cpu"
    ):

        self.origin_prob = _copy.deepcopy(origin_prob)
        self.origin_prob.set_device(device)
        self.right_smooth = right_smooth
        self.high_pass_min = self.origin_prob.minimum
        self.left_smooth = left_smooth
        self.low_pass_max = self.origin_prob.maximum
        self.A = A
        self.notch_lower = notch_lower
        self.notch_lower_smooth = notch_lower_smooth
        self.notch_upper = notch_upper
        self.notch_upper_smooth = notch_upper_smooth
        self.maximum = self.origin_prob.maximum
        self.minimum = self.origin_prob.minimum
        self.device = device
        self.dtype = torch.get_default_dtype()


        # Find the values of the integrals in the region of the window function before and after the smoothing
        int_points = 1500

        # high pass
        int_array = torch.linspace(self.minimum, self.minimum + self.right_smooth, int_points, device = self.device, dtype=self.dtype)
        self.integral_before_1 = torch.trapezoid(self.origin_prob.prob(int_array), int_array)
        self.integral_now_1 = torch.trapezoid(self.prob(int_array), int_array)

        # notch filter
        int_array = torch.linspace(self.notch_lower, notch_upper, int_points, device = self.device, dtype=self.dtype)
        self.integral_before_2 = torch.trapezoid(self.origin_prob.prob(int_array), int_array)
        self.integral_now_2 = torch.trapezoid(self.prob(int_array), int_array)

        # low pass
        int_array = torch.linspace(self.maximum - self.left_smooth, self.maximum, int_points, device = self.device, dtype=self.dtype)
        self.integral_before_3 = torch.trapezoid(self.origin_prob.prob(int_array), int_array)
        self.integral_now_3 = torch.trapezoid(self.prob(int_array), int_array)

        # compute norm
        self.integral_before = (
            self.integral_before_1 + self.integral_before_2 + self.integral_before_3
        )
        self.integral_now = self.integral_now_1 + self.integral_now_2 + self.integral_now_3

        self.norm = 1 - self.integral_before + self.integral_now

        # create and compute cdf ranges

        self.x_eval_1 = torch.linspace(self.minimum, self.minimum + self.right_smooth, int_points, device = self.device)
        self.cdf_numeric_1 = torch.cumsum(
            self.prob((self.x_eval_1[:-1:] + self.x_eval_1[1::]) * 0.5)
        , dim = 0) * (self.x_eval_1[1::] - self.x_eval_1[:-1:])

        self.x_eval_2 = torch.linspace(self.notch_lower, notch_upper, int_points, device = self.device)
        self.cdf_numeric_2 = (
            self.integral_now_1
            + self.origin_prob.cdf(torch.as_tensor([self.notch_lower], device=self.device, dtype=self.dtype))
            - self.integral_before_1
        ) / (self.norm) + torch.cumsum(
            self.prob((self.x_eval_2[:-1:] + self.x_eval_2[1::]) * 0.5), dim =0 
        ) * (
            self.x_eval_2[1::] - self.x_eval_2[:-1:]
        )

        self.x_eval_3 = torch.linspace(self.maximum - self.left_smooth, self.maximum, int_points, device = self.device)
        self.cdf_numeric_3 = (
            self.integral_now_1
            + self.integral_now_2
            + self.origin_prob.cdf(torch.as_tensor([self.maximum - self.left_smooth], device=self.device, dtype=self.dtype))
            - self.integral_before_1
            - self.integral_before_2
        ) / self.norm + torch.cumsum(self.prob((self.x_eval_3[:-1:] + self.x_eval_3[1::]) * 0.5),dim=0) * (
            self.x_eval_3[1::] - self.x_eval_3[:-1:]
        )
        self.cached_cdf_window_1 = torch_evaluatable_interpolator(
            self.x_eval_1[:-1:],
            self.cdf_numeric_1,
            fill_value=[0., None]
        )
        self.cached_cdf_window_2 = torch_evaluatable_interpolator(
            self.x_eval_2[:-1:],
            self.cdf_numeric_2,
            fill_value=[0., None]
        )
        self.cached_cdf_window_3 = torch_evaluatable_interpolator(
            self.x_eval_3[:-1:],
            self.cdf_numeric_3,
            fill_value=[0., None]
        )

    def set_device(self, device):
        self.cached_cdf_window_1.set_device(device)
        self.cached_cdf_window_2.set_device(device)
        self.cached_cdf_window_3.set_device(device)
        self.origin_prob.set_device(device)
        self.device=device

    def set_dtype(self, dtype):
        self.integral_before = self.integral_before.type(dtype)
        self.integral_now = self.integral_now.type(dtype)
        self.integral_before_1 = self.integral_before_1.type(dtype)
        self.integral_now_1 = self.integral_now_1.type(dtype)
        self.integral_before_2 = self.integral_before_2.type(dtype)
        self.integral_now_2 = self.integral_now_2.type(dtype)
        self.integral_before_3 = self.integral_before_3.type(dtype)
        self.integral_now_3 = self.integral_now_3.type(dtype)
        self.norm = self.norm.type(dtype)
        self.cached_cdf_window_1.set_dtype(dtype)
        self.cached_cdf_window_2.set_dtype(dtype)
        self.cached_cdf_window_3.set_dtype(dtype)
        self.origin_prob.set_dtype(dtype)
        self.dtype = dtype


    def prob(self, x):
        """
        Returns the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        return torch.exp(self.log_prob(x))

    def log_prob(self, x):
        """
        Returns the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        # Return the window function
        window_high_pass = high_pass_filter(x, self.high_pass_min, self.right_smooth)
        window_low_pass = low_pass_filter(x, self.low_pass_max, self.left_smooth)
        notch = notch_filter(
            x,
            self.notch_upper,
            self.notch_upper_smooth,
            self.notch_lower,
            self.notch_lower_smooth,
            self.A,
        )

        if hasattr(self, "norm"):
            prob_ret = (
                self.origin_prob.log_prob(x)
                + torch.log(window_high_pass)
                + torch.log(window_low_pass)
                + torch.log(notch)
                - torch.log(self.norm)
            )
        else:
            prob_ret = (
                self.origin_prob.log_prob(x)
                + torch.log(window_high_pass)
                + torch.log(window_low_pass)
                + torch.log(notch)
            )
        return prob_ret

    def log_conditioned_prob(self, x, a, b):
        """
        Returns the conditional probability between two new boundaries [a,b]

        Parameters
        ----------
        x: np.array
            Value at which compute the probability
        a: np.array
            New lower boundary
        b: np.array
            New low_pass_max boundary
        """

        to_ret = self.log_prob(x)
        # Find the new normalization in the new interval
        new_norm = self.cdf(b) - self.cdf(a)
        # Apply the new normalization and put to zero all the values above/below the interval
        wok = torch.where(new_norm > 0)[0]
        if to_ret.dim() > 0:
            to_ret[wok] -= torch.log(new_norm[wok])
        elif wok:
            to_ret -= torch.log(new_norm)
        wnull = torch.where(new_norm <= 0)[0]
        if len(wnull) > 0:
            to_ret[wnull] = -torch.inf

        to_ret[(x < a) | (x > b)] = -torch.inf


        return to_ret

    def cdf(self, x):
        """
        Returns the cumulative density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the cumulative
        """

        to_ret = torch.ones_like(x)
        to_ret[x < self.high_pass_min] = 0.0
        to_ret[(x >= self.high_pass_min) & (x < (self.high_pass_min + self.right_smooth))] = (
            self.cached_cdf_window_1(
                x[(x >= self.high_pass_min) & (x < (self.high_pass_min + self.right_smooth))]
            )
        )
        to_ret[(x >= (self.high_pass_min + self.right_smooth)) & (x < self.notch_lower)] = (
            self.integral_now_1
            - self.integral_before_1
            + self.origin_prob.cdf(
                x[(x >= (self.high_pass_min + self.right_smooth)) & (x < self.notch_lower)]
            )
        ) / self.norm
        to_ret[(x >= self.notch_lower) & (x < self.notch_upper)] = self.cached_cdf_window_2(
            x[(x >= self.notch_lower) & (x < self.notch_upper)]
        )
        to_ret[(x >= self.notch_upper) & (x < self.low_pass_max - self.left_smooth)] = (
            self.integral_now_1
            - self.integral_before_1
            + self.integral_now_2
            - self.integral_before_2
            + self.origin_prob.cdf(
                x[(x >= self.notch_upper) & (x < (self.low_pass_max - self.left_smooth))]
            )
        ) / self.norm
        to_ret[x >= (self.low_pass_max - self.left_smooth)] = self.cached_cdf_window_3(
            x[x >= (self.low_pass_max - self.left_smooth)]
        )
        return to_ret

    def conditioned_prob(self, x, a, b):
        """
        Returns the conditional probability between two new boundaries [a,b]

        Parameters
        ----------
        x: np.array
            Value at which compute the probability
        a: np.array
            New lower boundary
        b: np.array
            New low_pass_max boundary
        """

        return torch.exp(self.log_conditioned_prob(x, a, b))


class PowerLaw_math(object):
    """
    Class for a powerlaw probability :math:`p(x) \\propto x^{\\alpha}` defined in
    [a,b]

    Parameters
    ----------
    alpha: float
        Powerlaw slope
    min_pl: float
        lower cutoff
    max_pl: float
        low_pass_max cutoff
    """

    def __init__(self, alpha, min_pl, max_pl, device="cpu"):

        self.minimum = min_pl
        self.maximum = max_pl
        self.min_pl = min_pl
        self.max_pl = max_pl
        self.alpha = alpha
        self.device = device

        self.dtype = torch.get_default_dtype()

        # Get the PL norm and as Eq. 24 on the paper
        self.norm = get_PL_norm(alpha, min_pl, max_pl)

    def set_device(self, device):
        self.device=device

    def set_dtype(self, dtype):
        self.dtype = dtype

    def prob(self, x):
        """
        Returns the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        return torch.exp(self.log_prob(x))

    def log_prob(self, x):
        """
        Returns the logarithm of the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        to_ret = self.alpha * torch.log(x) - torch.log(self.norm)
        to_ret[(x < self.min_pl) | (x > self.max_pl)] = -torch.inf

        return to_ret

    def log_conditioned_prob(self, x, a, b):
        """
        Returns the conditional probability between two new boundaries [a,b]

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        a: np.array or float
            New lower boundary
        b: np.array or float
            New low_pass_max boundary
        """

        norms = get_PL_norm(self.alpha, a, b)
        to_ret = self.alpha * torch.log(x) - torch.log(norms)
        to_ret[(x < a) | (x > b)] = -torch.inf

        return to_ret

    def cdf(self, x):
        """
        Returns the cumulative density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the cumulative
        """

        # Define the cumulative density function, see  Eq. 24 to see the integral form

        if self.alpha == -1:
            to_ret = torch.log(x / self.min_pl) / self.norm
        else:
            to_ret = (
                (x**(self.alpha + 1) - self.min_pl**(self.alpha + 1)) 
                / (self.alpha + 1)
            ) / self.norm

        to_ret *= x >= self.min_pl

        if hasattr(x, "__len__"):
            to_ret[x > self.max_pl] = 1.0
        else:
            if x > self.max_pl:
                to_ret = 1.0

        return to_ret

    def conditioned_prob(self, x, a, b):
        """
        Returns the conditional probability between two new boundaries [a,b]

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        a: np.array or float
            New lower boundary
        b: np.array or float
            New low_pass_max boundary
        """

        return torch.exp(self.log_conditioned_prob(x, a, b))


class Truncated_Gaussian_math(object):
    """
    Class for a truncated gaussian in
    [a,b]

    Parameters
    ----------
    mu: float
        mean of the gaussian
    sigma: float
        standard deviation of the gaussian
    min_g: float
        lower cutoff
    max_g: float
        low_pass_max cutoff
    """

    def __init__(self, mu, sigma, min_g, max_g, device="cpu"):

        self.minimum = min_g
        self.maximum = max_g
        self.max_g = max_g
        self.min_g = min_g
        self.mu = mu
        self.sigma = sigma
        self.device = device

        self.dtype = torch.get_default_dtype()

        # Find the gaussian normalization as in Eq. 28 in the tex document
        self.norm = get_gaussian_norm(mu, sigma, min_g, max_g)

    def set_device(self, device):
        self.device=device

    def set_dtype(self, dtype):
        self.dtype = dtype

    def prob(self, x):
        """
        Returns the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        return torch.exp(self.log_prob(x))

    def log_prob(self, x):
        """
        Returns the logarithm of the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        log_sigma = torch.log(torch.tensor(self.sigma, device=self.device, dtype=self.dtype))
        log_norm = torch.log(self.norm.to(self.device, dtype=self.dtype))
        log_2pi = torch.log(torch.tensor(2 * torch.pi, device=self.device, dtype=self.dtype))

        to_ret = (
            -log_sigma
            - 0.5 * log_2pi
            - 0.5 * (((x - self.mu) / self.sigma) ** 2.0)
            - log_norm
        )
        to_ret[(x < self.min_g) | (x > self.max_g)] = -torch.inf

        return to_ret

    def log_conditioned_prob(self, x, a, b):
        """
        Returns the conditional probability between two new boundaries [a,b]

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        a: np.array or float
            New lower boundary
        b: np.array or float
            New low_pass_max boundary
        """

        norms = get_gaussian_norm(self.mu, self.sigma, a, b)
        log_sigma = torch.log(torch.tensor(self.sigma, device=self.device, dtype=self.dtype))
        log_norm = torch.log(self.norm.to(self.device, dtype=self.dtype))
        log_2pi = torch.log(torch.tensor(2 * torch.pi, device=self.device, dtype=self.dtype))

        to_ret = (
            -log_sigma
            - 0.5 * log_2pi
            - 0.5 * (((x - self.mu) / self.sigma) ** 2.0)
            - log_norm
        )
        to_ret[(x < a) | (x > b)] = -torch.inf

        return to_ret

    def cdf(self, x):
        """
        Returns the cumulative density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the cumulative
        """

        # Define the cumulative density function as in Eq. 28 on the paper to see the integral form

        sqrt2 = torch.sqrt(torch.tensor(2.0, device=self.device, dtype=self.dtype))
        max_point = (x - self.mu) / (self.sigma * sqrt2)
        min_point = (self.min_g - self.mu) / (self.sigma * sqrt2)
        max_point = torch.as_tensor(max_point, dtype = self.dtype, device = self.device)
        min_point = torch.as_tensor(min_point, dtype = self.dtype, device = self.device)

        to_ret = (0.5 * torch.erf(max_point) - 0.5 * torch.erf(min_point)) / self.norm

        to_ret *= x >= self.min_g

        if hasattr(x, "__len__"):
            to_ret[x > self.max_g] = 1.0
        else:
            if x > self.max_g:
                to_ret = 1.0

        return to_ret

    def conditioned_prob(self, x, a, b):
        """
        Returns the conditional probability between two new boundaries [a,b]

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        a: np.array or float
            New lower boundary
        b: np.array or float
            New low_pass_max boundary
        """

        return torch.exp(self.log_conditioned_prob(x, a, b))


class PowerLawGaussian_math(object):
    """
    Class for a powerlaw probability plus gausian peak
    :math:`p(x) \\propto (1-\\lambda)x^{\\alpha}+\\lambda \\mathcal{N}(\\mu,\\sigma)`. Each component is defined in
    a different interval

    Parameters
    ----------
    alpha: float
        Powerlaw slope
    min_pl: float
        lower cutoff
    max_pl: float
        low_pass_max cutoff
    lambda_g: float
        fraction of prob coming from gaussian peak
    mu_g: float
        mean for the gaussian
    sigma_g: float
        standard deviation for the gaussian
    min_g: float
        minimum for the gaussian component
    max_g: float
        maximim for the gaussian component
    """

    def __init__(self, alpha, min_pl, max_pl, lambda_g, mu_g, sigma_g, min_g, max_g, device="cpu"):

        self.minimum = min([min_pl, min_g])
        self.maximum = max([max_pl, max_g])

        self.lambda_g = lambda_g

        self.device = device
        self.dtype = torch.get_default_dtype()

        self.pl = PowerLaw_math(alpha, min_pl, max_pl, device=device)
        self.gg = Truncated_Gaussian_math(mu_g, sigma_g, min_g, max_g, device=device)

    def set_device(self, device):
        self.pl.set_device(device)
        self.gg.set_device(device)
        self.device=device

    def set_dtype(self, dtype):
        self.pl.set_dtype(dtype)
        self.gg.set_dtype(dtype)

        self.dtype=dtype
    def prob(self, x):
        """
        Returns the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        # Define the PDF as in Eq. 36-37-38 on on the tex document
        return torch.exp(self.log_prob(x))

    def cdf(self, x):
        """
        Returns the cumulative density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the cumulative
        """

        return (1 - self.lambda_g) * self.pl.cdf(x) + self.lambda_g * self.gg.cdf(x)

    def conditioned_prob(self, x, a, b):
        """
        Returns the conditional probability between two new boundaries [a,b]

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        a: np.array or float
            New lower boundary
        b: np.array or float
            New low_pass_max boundary
        """

        return torch.exp(self.log_conditioned_prob(x, a, b))

    def log_prob(self, x):
        """
        Returns the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        # Define the PDF as in Eq. 36-37-38 on on the tex document
        lambda_g = torch.tensor(self.lambda_g, device=self.device, dtype=self.dtype)
        return torch.logaddexp(
            torch.log1p(-lambda_g) + self.pl.log_prob(x),
            torch.log(lambda_g) + self.gg.log_prob(x),
        )

    def log_conditioned_prob(self, x, a, b):
        """
        Returns the conditional probability between two new boundaries [a,b]

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        a: np.array or float
            New lower boundary
        b: np.array or float
            New low_pass_max boundary
        """

        lambda_g = torch.tensor(self.lambda_g, device=self.device, dtype=self.dtype)
        return torch.logaddexp(
            torch.log1p(-lambda_g) + self.pl.log_conditioned_prob(x, a, b),
            torch.log(lambda_g) + self.gg.log_conditioned_prob(x, a, b),
        )


class PowerLawDoubleGaussian_math(object):
    """
    Class for a powerlaw probability plus gausian peak
    :math:`p(x) \\propto (1-\\lambda)x^{\\alpha}+\\lambda \\lambda_1 \\mathcal{N}(\\mu_1,\\sigma_1)+\\lambda (1-\\lambda_1) \\mathcal{N}(\\mu_2,\\sigma_2)`.
    Each component is defined ina different interval

    Parameters
    ----------
    alpha: float
        Powerlaw slope
    min_pl: float
        lower cutoff
    max_pl: float
        low_pass_max cutoff
    lambda_g: float
        fraction of prob coming in both gaussian peaks
    lambda_g_0: float
        fraction of prob in lower gaussian peak
    mu_g_0: float
        mean for the lower gaussian peak
    sigma_g_0: float
        standard deviation for the gaussian # Define the PDF as in Eq. 37 on on the tex document
    mu_g_1: float
        mean for the higher gaussian peak
    sigma_g_1: float
        standard deviation for the lower gaussian peak
    min_g: float
        minimum for the gaussian components
    max_g: float
        maximum for the gaussian components
    """

    def __init__(
        self,
        alpha,
        min_pl,
        max_pl,
        lambda_g,
        lambda_g_0,
        mu_g_0,
        sigma_g_0,
        mu_g_1,
        sigma_g_1,
        min_g,
        max_g, 
        device="cpu"
    ):

        self.minimum = _np.min([min_pl, min_g])
        self.maximum = _np.max([max_pl, max_g])

        self.lambda_g = lambda_g
        self.lambda_g_0 = lambda_g_0

        self.device = device
        self.dtype = torch.get_default_dtype()

        self.pl = PowerLaw_math(alpha, min_pl, max_pl, device=device)
        self.gg_low = Truncated_Gaussian_math(mu_g_0, sigma_g_0, min_g, max_g, device=device)
        self.gg_high = Truncated_Gaussian_math(mu_g_1, sigma_g_1, min_g, max_g, device=device)

    def set_device(self, device):
        self.gg_low.set_device(device)
        self.gg_high.set_device(device)
        self.device=device

    def set_dtype(self, dtype):
        self.pl.set_dtype(dtype)
        self.gg_low.set_dtype(dtype)
        self.gg_high.set_dtype(dtype)

        self.dtype=dtype

    def prob(self, x):
        """
        Returns the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        return torch.exp(self.log_prob(x))

    def log_prob(self, x):
        """
        Returns the log probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        # Define the PDF as in Eq. 44-45-46 on the tex document

        lambda_g = torch.tensor(self.lambda_g, device=self.device, dtype=self.dtype)
        lambda_g_0 = torch.tensor(self.lambda_g_0, device=self.device, dtype=self.dtype)

        pl_part = torch.log1p(-lambda_g) + self.pl.log_prob(x)
        g_low = self.gg_low.log_prob(x) + torch.log(lambda_g) + torch.log(lambda_g_0)
        g_high = self.gg_high.log_prob(x) + torch.log(lambda_g) + torch.log1p(-lambda_g_0)

        return torch.logsumexp(torch.stack([pl_part, g_low, g_high]), dim=0)

    def log_conditioned_prob(self, x, a, b):
        """
        Returns the log conditional probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        # Define the PDF as in Eq.  44-45-46  on the tex document

        lambda_g = torch.tensor(self.lambda_g, device=self.device, dtype=self.dtype)
        lambda_g_0 = torch.tensor(self.lambda_g_0, device=self.device, dtype=self.dtype)

        pl_part = torch.log1p(-lambda_g) + self.pl.log_conditioned_prob(x, a, b)
        g_low = (
            self.gg_low.log_conditioned_prob(x, a, b)
            + torch.log(lambda_g)
            + torch.log(lambda_g_0)
        )
        g_high = (
            self.gg_high.log_conditioned_prob(x, a, b)
            + torch.log(lambda_g)
            + torch.log1p(-lambda_g_0)
        )

        return torch.logsumexp(torch.stack([pl_part, g_low, g_high]), dim=0)

    def cdf(self, x):
        """
        Returns the cumulative density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the cumulative
        """

        lambda_g = torch.tensor(self.lambda_g, device=self.device, dtype=self.dtype)
        lambda_g_0 = torch.tensor(self.lambda_g_0, device=self.device, dtype=self.dtype)

        pl_part = (1 - lambda_g) * self.pl.cdf(x)
        g_part = self.gg_low.cdf(x) * lambda_g * lambda_g_0 + self.gg_high.cdf(
            x
        ) * lambda_g * (1 - lambda_g_0)
        return pl_part + g_part

    def conditioned_prob(self, x, a, b):
        """
        Returns the conditional probability between two new boundaries [a,b]

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        a: np.array or float
            New lower boundary
        b: np.array or float
            New low_pass_max boundary
        """

        return torch.exp(self.log_conditioned_prob(x, a, b))


class BrokenPowerLaw_math(object):
    """
    Class for a broken powerlaw probability
    :math:`p(x) \\propto x^{\\alpha}` if :math:`min<x<b(max-min)`, :math:`p(x) \\propto x^{\\beta}` if :math:`b(max-min)<x<max`.

    Parameters
    ----------
    alpha_1: float
        Powerlaw slope for first component
    alpha_2: float
        Powerlaw slope for second component
    min_pl: float
        lower cutoff
    max_pl: float
        low_pass_max cutoff

    """

    def __init__(self, alpha_1, alpha_2, min_pl, max_pl, b, device="cpu"):

        self.minimum = min_pl
        self.maximum = max_pl

        self.min_pl = min_pl
        self.max_pl = max_pl

        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2

        # Define the breaking point
        self.break_point = min_pl + b * (max_pl - min_pl)
        self.b = b

        self.device = device
        self.dtype = torch.get_default_dtype()

        # Initialize the single powerlaws
        self.pl1 = PowerLaw_math(alpha_1, min_pl, self.break_point, device=self.device)
        self.pl2 = PowerLaw_math(alpha_2, self.break_point, max_pl, device=self.device)

        # Define the broken powerlaw as in Eq. 39-40-41 on the tex document
        break_point_tensor = torch.as_tensor([self.break_point], device=self.device, dtype=self.dtype)
        self.new_norm=(1+self.pl1.prob(break_point_tensor)/self.pl2.prob(break_point_tensor))

    def set_device(self,device):
        self.pl1.set_device(device)
        self.pl2.set_device(device)
        self.new_norm = self.new_norm.to(device)
        self.device=device

    def set_dtype(self, dtype):
        self.pl1.set_dtype(dtype)
        self.pl2.set_dtype(dtype)
        self.new_norm = self.new_norm.type(dtype)
        self.dtype=dtype

    def prob(self, x):
        """
        Returns the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        # Define the PDF as in Eq. 39-40-41 on the tex document
        return torch.exp(self.log_prob(x))

    def log_prob(self, x):
        """
        Returns the log probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        # Define the PDF as in Eq. 39-40-41 on the tex document
        break_point_tensor = torch.as_tensor([self.break_point], device=self.device, dtype=self.dtype)
        to_ret = torch.logaddexp(
            self.pl1.log_prob(x),
            self.pl2.log_prob(x)
            + self.pl1.log_prob(break_point_tensor)
            - self.pl2.log_prob(break_point_tensor),
        ) - torch.log(self.new_norm)
        return to_ret

    def log_conditioned_prob(self, x, a, b):
        """
        Returns the log conditional probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        # Define the PDF as in Eq. 39-40-41 on the tex document
        break_point_tensor = torch.as_tensor([self.break_point], device=self.device, dtype=self.dtype)
        to_ret = torch.logaddexp(
            self.pl1.log_conditioned_prob(x, a, b),
            self.pl2.log_conditioned_prob(x, a, b)
            + self.pl1.log_prob(break_point_tensor)
            - self.pl2.log_prob(break_point_tensor),
        ) - torch.log(self.new_norm)

        return to_ret

    def cdf(self, x):
        """
        Returns the cumulative density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the cumulative
        """
        break_point_tensor = torch.as_tensor([self.break_point], device=self.device, dtype=self.dtype)
        return (
            self.pl1.cdf(x)
            + self.pl2.cdf(x)
            * (
                self.pl1.prob(break_point_tensor)
                / self.pl2.prob(break_point_tensor)
            )
        ) / self.new_norm

    def conditioned_prob(self, x, a, b):
        """
        Returns the conditional probability between two new boundaries [a,b]

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        a: np.array or float
            New lower boundary
        b: np.array or float
            New low_pass_max boundary
        """

        return torch.exp(self.log_conditioned_prob(x, a, b))


class BrokenPowerLawDoubleGaussian_math(object):
    """
    Class for a broken powerlaw probability with two gaussian peaks
    :math:`p(x) \\propto x^{\\alpha_1}` if :math:`min<x<m_{b}(max-min)`, :math:`p(x) \\propto x^{\\alpha_2}` if :math:`m_{b}(max-min)<x<max`. This is added to two gaussians to give
    :math: `(1-\\lambda) (1/N) BPL(\\alpha_1,\\alpha_2,b)+\\lambda \\lambda_1 \\mathcal{N}(\\mu_1,\\sigma_1)+\\lambda (1-\\lambda_1) \\mathcal{N}(\\mu_2,\\sigma_2)` where BPL is the above powerlaw function.

    Parameters
    ----------
    alpha_1: float
        Powerlaw slope for first component
    alpha_2: float
        Powerlaw slope for second component
    min_pl: float
        lower cutoff
    max_pl: float
        low_pass_max cutoff
    lambda_g: float
        fraction of prob coming in both gaussian peaks
    lambda_g_0: float
        fraction of prob in lower gaussian peak
    mu_g_0: float
        mean for the lower gaussian peak
    sigma_g_0: float
        standard deviation for the gaussian # Define the PDF as in Eq. 37 on on the tex document
    mu_g_1: float
        mean for the higher gaussian peak
    sigma_g_1: float
        standard deviation for the lower gaussian peak
    min_g: float
        minimum for the gaussian components
    max_g: float
        maximum for the gaussian components
    """

    def __init__(
        self,
        min_pl,
        max_pl,
        lambda_g,
        lambda_g_0,
        mu_g_0,
        sigma_g_0,
        mu_g_1,
        sigma_g_1,
        min_g,
        max_g,
        alpha_1,
        alpha_2,
        break_point,
        device = "cpu"
    ):

        self.minimum = _np.min([min_pl, min_g])
        self.maximum = _np.max([max_pl, max_g])

        self.lambda_g = lambda_g
        self.lambda_g_0 = lambda_g_0

        self.break_point = break_point

        self.device = device
        self.dtype = torch.get_default_dtype()

        self.pl1 = PowerLaw_math(alpha_1, min_pl, self.break_point, device = self.device)
        self.pl2 = PowerLaw_math(alpha_2, self.break_point, max_pl,  device = self.device)
        self.gg_low = Truncated_Gaussian_math(mu_g_0, sigma_g_0, min_g, max_g,  device = self.device)
        self.gg_high = Truncated_Gaussian_math(mu_g_1, sigma_g_1, min_g, max_g, device = self.device)

        self.break_point_tensor = torch.as_tensor([self.break_point], device=self.device, dtype=self.dtype)

        self.new_norm = 1 + self.pl1.prob(self.break_point_tensor) / self.pl2.prob(
            self.break_point_tensor
        )

    def set_device(self,device):
        self.pl1.set_device(device)
        self.pl2.set_device(device)
        self.gg_low.set_device(device)
        self.gg_high.set_device(device) 
        self.new_norm = self.new_norm.to(device)
        self.device=device

    def set_dtype(self, dtype):
        self.pl1.set_dtype(dtype)
        self.pl2.set_dtype(dtype)
        self.gg_low.set_dtype(dtype)
        self.gg_high.set_dtype(dtype)
        self.new_norm = self.new_norm.type(dtype)
        self.dtype=dtype

    def prob(self, x):
        """
        Returns the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        # Define the PDF as in Eq. 39-40-41 on the tex document
        return torch.exp(self.log_prob(x))

    def log_prob(self, x):
        """
        Returns the log probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """
        lambda_g = torch.tensor(self.lambda_g, device=self.device, dtype=self.dtype)
        lambda_g_0 = torch.tensor(self.lambda_g_0, device=self.device, dtype=self.dtype)

        pl1_part = self.pl1.log_prob(x)
        pl2_part = (
            self.pl2.log_prob(x)
            + self.pl1.log_prob(self.break_point_tensor)
            - self.pl2.log_prob(self.break_point_tensor)
        )
        pl_part = (
            torch.log1p(-lambda_g) + torch.logaddexp(pl1_part, pl2_part) - torch.log(self.new_norm)
        )
        g_low = self.gg_low.log_prob(x) + torch.log(lambda_g) + torch.log(lambda_g_0)
        g_high = self.gg_high.log_prob(x) + torch.log(lambda_g) + torch.log1p(-lambda_g_0)

        add_first = torch.logaddexp(pl_part,g_low)
        return torch.logaddexp(add_first, g_high)

    def log_conditioned_prob(self, x, a, b):
        """
        Returns the log conditional probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """
        lambda_g = torch.tensor(self.lambda_g, device=self.device, dtype=self.dtype)
        lambda_g_0 = torch.tensor(self.lambda_g_0, device=self.device, dtype=self.dtype)

        pl1_part = self.pl1.log_conditioned_prob(x, a, b)
        pl2_part = (
            self.pl2.log_conditioned_prob(x, a, b)
            + self.pl1.log_prob(self.break_point_tensor)
            - self.pl2.log_prob(self.break_point_tensor)
        )
        pl_part = (
            torch.log1p(-lambda_g) + torch.logaddexp(pl1_part, pl2_part) - torch.log(self.new_norm)
        )
        g_low = (
            self.gg_low.log_conditioned_prob(x, a, b)
            + torch.log(lambda_g)
            + torch.log(lambda_g_0)
        )
        g_high = (
            self.gg_high.log_conditioned_prob(x, a, b)
            + torch.log(lambda_g)
            + torch.log1p(-lambda_g_0)
        )

        add_first = torch.logaddexp(pl_part,g_low)
        return torch.logaddexp(add_first, g_high)

    def conditioned_prob(self, x, a, b):
        """
        Returns the conditional probability between two new boundaries [a,b]

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        a: np.array or float
            New lower boundary
        b: np.array or float
            New low_pass_max boundary
        """

        return torch.exp(self.log_conditioned_prob(x, a, b))

    def cdf(self, x):
        """
        Returns the cumulative density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the cumulative
        """
        lambda_g = torch.tensor(self.lambda_g, device=self.device, dtype=self.dtype)
        lambda_g_0 = torch.tensor(self.lambda_g_0, device=self.device, dtype=self.dtype)

        # Perform all calculations using torch
        pl_part = (
            (1 - lambda_g)
            * (
                self.pl1.cdf(x)
                + self.pl2.cdf(x)
                * (
                    self.pl1.prob(self.break_point_tensor)
                    / self.pl2.prob(self.break_point_tensor)
                )
            )
            / self.new_norm
        )
        g_part = self.gg_low.cdf(x) * lambda_g * lambda_g_0 + self.gg_high.cdf(x) * lambda_g * (1 - lambda_g_0)
        return pl_part + g_part

class BrokenPowerLawTripleGaussian_math(object):
    """

    Class for a probability distribution combining a broken power law with three gaussian peaks.
    The model is a mixture defined as:
    :math:`p(x) = (1 - \\lambda_g) \\text{BPL}(x) + \\lambda_g [ \\lambda_{g1} \\mathcal{N}(\\mu_1, \\sigma_1) + \\lambda_{g2} \\mathcal{N}(\\mu_2, \\sigma_2) + (1 - \\lambda_{g1} - \\lambda_{g2}) \\mathcal{N}(\\mu_3, \\sigma_3) ]`
    where :math:`\\text{BPL}(x)` is a broken power-law with :math:`p(x) \\propto x^{\\alpha_1}` for :math:`min_{pl} < x < x_{break}` and :math:`p(x) \\propto x^{\\alpha_2}` for :math:`x_{break} < x < max_{pl}`.

    Parameters
    ----------
    min_pl: float
        Lower bound for the power-law component.
    max_pl: float
        Upper bound for the power-law component.
    lambda_g: float
        The total mixing fraction for the three Gaussian components, in [0, 1].
    lambda_g_0: float
        Fraction of the Gaussian probability in the first peak.
    mu_g_0: float
        Mean of the first Gaussian peak.
    sigma_g_0: float
        Standard deviation of the first Gaussian peak.
    lambda_g_1: float
        Fraction of the Gaussian probability in the second peak.
    mu_g_1: float
        Mean of the second Gaussian peak.
    sigma_g_1: float
        Standard deviation of the second Gaussian peak.
    mu_g_2: float
        Mean of the third Gaussian peak.
    sigma_g_2: float
        Standard deviation of the third Gaussian peak.
    min_g: float
        Minimum for all truncated Gaussian components.
    max_g: float
        Maximum for all truncated Gaussian components.
    alpha_1: float
        Power-law slope for the first component.
    alpha_2: float
        Power-law slope for the second component.
    break_point: float
        Value at which the power-law breaks.
    """

    def __init__(
        self,
        min_pl,
        max_pl,
        lambda_g,
        lambda_g_0,
        mu_g_0,
        sigma_g_0,
        lambda_g_1,
        mu_g_1,
        sigma_g_1,
        mu_g_2,
        sigma_g_2,
        min_g,
        max_g,
        alpha_1,
        alpha_2,
        break_point,
        device="cpu"
    ):

        self.minimum = _np.min([min_pl, min_g])
        self.maximum = _np.max([max_pl, max_g])

        self.device = device
        self.dtype = torch.get_default_dtype()

        self.lambda_g = torch.tensor(lambda_g, device=self.device, dtype=self.dtype)
        self.lambda_g_0 = torch.tensor(lambda_g_0, device=self.device, dtype=self.dtype)
        self.lambda_g_1 = torch.tensor(lambda_g_1, device=self.device, dtype=self.dtype)

        self.log_w1 = torch.log(self.lambda_g_0)
        self.log_w2 = torch.log1p(-self.lambda_g_0) +  torch.log(self.lambda_g_1)
        self.log_w3 = torch.log1p(-self.lambda_g_0) + torch.log1p(-self.lambda_g_1)

        self.break_point = break_point
        self.break_point_tensor = torch.as_tensor([self.break_point], device=self.device, dtype=self.dtype)

        self.pl1 = PowerLaw_math(alpha_1, min_pl, self.break_point, device = self.device)
        self.pl2 = PowerLaw_math(alpha_2, self.break_point, max_pl, device = self.device)
        self.gg_1 = Truncated_Gaussian_math(mu_g_0, sigma_g_0, min_g, max_g, device = self.device)
        self.gg_2 = Truncated_Gaussian_math(mu_g_1, sigma_g_1, min_g, max_g, device = self.device)
        self.gg_3 = Truncated_Gaussian_math(mu_g_2, sigma_g_2, min_g, max_g, device = self.device)

        self.new_norm = 1 + self.pl1.prob(self.break_point_tensor) / self.pl2.prob(
            self.break_point_tensor
        )

    def set_device(self,device):
        self.pl1.set_device(device)
        self.pl2.set_device(device)
        self.gg_1.set_device(device)
        self.gg_2.set_device(device)
        self.gg_3.set_device(device)
        self.lambda_g = self.lambda_g.to(device)
        self.lambda_g_0 = self.lambda_g_0.to(device)
        self.lambda_g_1 = self.lambda_g_1.to(device)
        self.log_w1 = self.log_w1.to(device)
        self.log_w2 = self.log_w2.to(device)
        self.log_w3 = self.log_w3.to(device)
        self.break_point_tensor = self.break_point_tensor.to(device)
        self.new_norm = self.new_norm.to(device)
        self.device=device

    def set_dtype(self, dtype):
        self.pl1.set_dtype(dtype)
        self.pl2.set_dtype(dtype)
        self.gg_1.set_dtype(dtype)
        self.gg_2.set_dtype(dtype)
        self.gg_3.set_dtype(dtype)
        self.lambda_g = self.lambda_g.to(dtype)
        self.lambda_g_0 = self.lambda_g_0.to(dtype)
        self.lambda_g_1 = self.lambda_g_1.to(dtype)
        self.log_w1 = self.log_w1.to(dtype)
        self.log_w2 = self.log_w2.to(dtype)
        self.log_w3 = self.log_w3.to(dtype)
        self.break_point_tensor = self.break_point_tensor.to(dtype)
        self.new_norm = self.new_norm.to(dtype)
        self.dtype=dtype

    def prob(self, x):
        """
        Returns the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        # Define the PDF as in Eq. 39-40-41 on the tex document
        return torch.exp(self.log_prob(x))

    def log_prob(self, x):
        """
        Returns the log probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """        
        pl1_part = self.pl1.log_prob(x)
        pl2_part = (self.pl2.log_prob(x)+ 
                    self.pl1.log_prob(self.break_point_tensor) - 
                    self.pl2.log_prob(self.break_point_tensor)
        )
        pl_part = (
            torch.log1p(-self.lambda_g) + torch.logaddexp(pl1_part, pl2_part) - torch.log(self.new_norm)
        )

        g_1 = self.gg_1.log_prob(x) + torch.log(self.lambda_g) + self.log_w1
        g_2 = self.gg_2.log_prob(x) + torch.log(self.lambda_g) + self.log_w2
        g_3 = self.gg_3.log_prob(x) + torch.log(self.lambda_g) + self.log_w3

        add_first = torch.logaddexp(pl_part,g_1)
        add_second = torch.logaddexp(add_first, g_2)
        return torch.logaddexp(add_second, g_3)

    def log_conditioned_prob(self, x, a, b):
        """
        Returns the log conditional probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """
        pl1_part = self.pl1.log_conditioned_prob(x, a, b)
        pl2_part = (
            self.pl2.log_conditioned_prob(x, a, b)
            + self.pl1.log_prob(self.break_point_tensor)
            - self.pl2.log_prob(self.break_point_tensor)
        )
        pl_part = (
            torch.log1p(-self.lambda_g) + torch.logaddexp(pl1_part, pl2_part) - torch.log(self.new_norm)
        )
        g_1 = (
            self.gg_1.log_conditioned_prob(x, a, b)
            + torch.log(self.lambda_g)
            + self.log_w1
        )
        g_2 = (
            self.gg_2.log_conditioned_prob(x, a, b)
            + torch.log(self.lambda_g)
            + self.log_w2
        )
        g_3 = (
            self.gg_3.log_conditioned_prob(x, a, b)
            + torch.log(self.lambda_g)
            + self.log_w3
        )
        add_first = torch.logaddexp(pl_part,g_1)
        add_second = torch.logaddexp(add_first, g_2)
        return torch.logaddexp(add_second, g_3)

    def conditioned_prob(self, x, a, b):
        """
        Returns the conditional probability between two new boundaries [a,b]

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        a: np.array or float
            New lower boundary
        b: np.array or float
            New low_pass_max boundary
        """

        return torch.exp(self.log_conditioned_prob(x, a, b))

    def cdf(self, x):
        """
        Returns the cumulative density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the cumulative
        """
        pl_part = (
            (1 - self.lambda_g)
            * (
                self.pl1.cdf(x)
                + self.pl2.cdf(x)
                * (
                    self.pl1.prob(self.break_point_tensor)
                    / self.pl2.prob(self.break_point_tensor)
                )
            )
            / self.new_norm
        )
        g_part = self.lambda_g * ((torch.exp(self.log_w1) * self.gg_1.cdf(x))  +
        (torch.exp(self.log_w2) * self.gg_2.cdf(x)) + (torch.exp(self.log_w3) * self.gg_3.cdf(x)))
        return pl_part + g_part

class PairingFunc(object):
    """
    Class to add pairing function to mass distributions.

    Parameters
    ----------
    origin_prob : object
        original probability distribution
    pairing_function : callable
        pairing function for m1 and m2 distributions
    """

    def __init__(self, origin_dist, pairing_function, m1_samps, m2_samps, device="cpu"):
        self.device = torch.device(device)
        self.origin_dist = origin_dist
        self.pairing_function = pairing_function
        self.dtype = torch.get_default_dtype()

        if not isinstance(m1_samps, torch.Tensor):
            m1_samps = torch.tensor(m1_samps, dtype=self.dtype, device=self.device)
        if not isinstance(m2_samps, torch.Tensor):
            m2_samps = torch.tensor(m2_samps, dtype=self.dtype, device=self.device)
        self.m1_samps = m1_samps.to(self.device)
        self.m2_samps = m2_samps.to(self.device) 

        self.norm = self.calc_norm()

    def calc_norm(self):
        """
        Uses monte carlo sum to calculate new normalisation after addition of pairing function (Torch).
        """
        with torch.no_grad(): #Ensure no gradients are calculated.
            self.new_norm = torch.mean(self.pairing_function(self.m1_samps, self.m2_samps))

    def log_prob(self, x1, x2):
        """
        Calculate the log probability of a pair of values (x1, x2) (Torch).

        Parameters
        ----------
        x1 : array-like
            First set of values for which to calculate log probability.
        x2 : array-like
            Second set of values for which to calculate log probability.

        Returns
        -------
        toret : torch.Tensor
            Normalised log probabilities.
        """
        x1 = torch.as_tensor(x1, device=self.device, dtype = self.dtype)
        x2 = torch.as_tensor(x2, device=self.device, dtype = self.dtype)

        toret = (
            self.origin_dist["mass_1"].log_prob(x1)
            + self.origin_dist["mass_2"].log_prob(x2)
            + torch.log(self.pairing_function(x1, x2))
            - torch.log(self.new_norm)
        )
        toret[torch.isnan(toret)] = -torch.inf
        return toret

    def prob(self, x1, x2):
        """
        Calculate the probability of a pair of values (x1, x2) (Torch).

        Parameters
        ----------
        x1 : array-like
            First set of values for which to calculate probability.
        x2 : array-like
            Second set of values for which to calculate probability.

        Returns
        -------
        torch.Tensor
            Normalised probabilities.
        """
        return torch.exp(self.log_prob(x1, x2))

    def sample(self, Nsample):
        """
        Generate samples from the probability density function (Torch).

        Parameters
        ----------
        Nsample : int
            Number of samples to generate.

        Returns
        -------
        tuple of torch.Tensor
           arrays of m1 and m2 samples.
        """
        min_1 = self.origin_dist["mass_1"].minimum
        max_1 = self.origin_dist["mass_1"].maximum
        min_2 = self.origin_dist["mass_2"].minimum
        max_2 = self.origin_dist["mass_2"].maximum

        x1 = torch.rand(10 * Nsample, device=self.device) * (max_1 - min_1) + min_1
        x2 = torch.rand(10 * Nsample, device=self.device) * (max_2 - min_2) + min_2
        probs = self.prob(x1, x2)
        idx = torch.multinomial(probs / torch.sum(probs), Nsample, replacement=True)
        return x1[idx], x2[idx]
