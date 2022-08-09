# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''

Transform angular power spectra and correlation functions (:mod:`transformcl`)
==============================================================================

This is a minimal Python package for transformations between angular power
spectra and correlation functions.  It is currently limited to the zero spin
case.

The package can be installed using pip::

    pip install transformcl

Then import the :func:`~transformcl.cltocorr` and :func:`~transformcl.corrtocl`
functions from the package::

    from transformcl import cltocorr, corrtocl

Current functionality covers the absolutely minimal use case.  Please open an
issue on GitHub if you would like to see anything added.


Reference/API
-------------

.. autosummary::
   :toctree: api
   :nosignatures:

   cltocorr
   corrtocl
   theta
   cltovar

'''

__version__ = '2022.8.9'

__all__ = [
    'cltocorr',
    'corrtocl',
    'theta',
    'cltovar',
]


import numpy as np
from flt import dlt, idlt, theta


FOUR_PI = 4*np.pi


def cltoxi(*args, **kwargs):
    return cltocorr(*args, **kwargs)


def xitocl(*args, **kwargs):
    return corrtocl(*args, **kwargs)


def cltocorr(cl, closed=False):
    r'''transform angular power spectrum to correlation function

    Takes an angular power spectrum with :math:`\mathtt{n} = \mathtt{lmax}+1`
    coefficients and returns the corresponding angular correlation function in
    :math:`\mathtt{n}` points.

    The correlation function values can be computed either over the closed
    interval :math:`[0, \pi]`, in which case :math:`\theta_0 = 0` and
    :math:`\theta_{n-1} = \pi`, or over the open interval :math:`(0, \pi)`.

    Parameters
    ----------
    cl : (n,) array_like
        Angular power spectrum from :math:`0` to :math:`\mathtt{lmax}`.
    closed : bool
        Compute correlation function over open (``closed=False``) or closed
        (``closed=True``) interval.

    Returns
    -------
    corr : (n,) array_like
        Angular correlation function.

    See Also
    --------
    transformcl.corrtocl : the inverse operation
    transformcl.theta : compute the angles at which the correlation function is
                        evaluated

    Notes
    -----
    The computation uses the inverse discrete Legendre transform
    :func:`flt.idlt`.

    '''

    # length n of the transform
    if np.ndim(cl) != 1:
        raise TypeError('cl must be 1d array')
    n = np.shape(cl)[-1]

    # DLT coefficients = (2l+1)/(4pi) * Cl
    c = np.arange(1, 2*n+1, 2, dtype=float)
    c /= FOUR_PI
    c *= cl

    # perform the inverse DLT
    corr = idlt(c, closed=closed)

    # done
    return corr


def corrtocl(corr, closed=False):
    r'''transform angular correlation function to power spectrum

    Takes an angular function in :math:`\mathtt{n}` points and returns the
    corresponding angular power spectrum from :math:`0` to :math:`\mathtt{lmax}
    = \mathtt{n}-1`.

    The correlation function must be given at the angles returned by
    :func:`transformcl.theta`.  These can be distributed either over the closed
    interval :math:`[0, \pi]`, in which case :math:`\theta_0 = 0` and
    :math:`\theta_{n-1} = \pi`, or over the open interval :math:`(0, \pi)`.

    Parameters
    ----------
    corr : (n,) array_like
        Angular correlation function.
    closed : bool
        Compute correlation function over open (``closed=False``) or closed
        (``closed=True``) interval.

    Returns
    -------
    cl : (n,) array_like
        Angular power spectrum from :math:`0` to :math:`\mathtt{lmax}`.

    See Also
    --------
    transformcl.cltocorr : the inverse operation
    transformcl.theta : compute the angles at which the correlation function is
                        evaluated

    Notes
    -----
    The computation uses the discrete Legendre transform :func:`flt.dlt`.

    '''

    # length n of the transform
    if np.ndim(corr) != 1:
        raise TypeError('corr must be 1d array')
    n = np.shape(corr)[-1]

    # compute the DLT coefficients
    cl = dlt(corr, closed=closed)

    # DLT coefficients = (2l+1)/(4pi) * Cl
    cl /= np.arange(1, 2*n+1, 2, dtype=float)
    cl *= FOUR_PI

    # done
    return cl


def cltovar(cl):
    r'''compute variance from angular power spectrum

    Given the angular power spectrum, compute the variance of the spherical
    random field in a point.

    Parameters
    ----------
    cl : array_like
        Angular power spectrum.  Can be multidimensional, with the last axis
        representing the modes.

    Returns
    -------
    var: float
        The variance of the given power spectrum.

    Notes
    -----
    The variance :math:`\sigma^2` of the field with power spectrum :math:`C_l`
    is

    .. math::

        \sigma^2 = \sum_{l} \frac{2l + 1}{4\pi} \, C_l \;.

    '''

    l = np.arange(np.shape(cl)[-1])
    return np.sum((2*l+1)/(4*np.pi)*cl, axis=-1)
