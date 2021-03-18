# transformcl: transform angular power spectra and correlation functions
#
# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
#
# cython: language_level=3, boundscheck=False, embedsignature=True
'''

Transform angular power spectra and correlation functions (:mod:`transformcl`)
==============================================================================

This is a minimal Python package for transformations between angular power
spectra and correlation functions.  It is currently limited to the zero spin
case.

The package can be installed using pip::

    pip install transformcl

Then import the :func:`~transformcl.cltoxi` and :func:`~transformcl.xitocl`
functions from the package::

    from transformcl import cltoxi, xitocl

Current functionality covers the absolutely minimal use case.  Please open an
issue on GitHub if you would like to see anything added.


Reference/API
-------------

.. autosummary::
   :toctree: api
   :nosignatures:

   cltoxi
   xitocl
   theta

'''

__all__ = [
    'cltoxi',
    'xitocl',
    'theta',
]


import numpy as np
from scipy.fft import dct, idct

cdef extern from "dctdlt.c":
    void dctdlt(unsigned int, const double*, double*)
    void dltdct(unsigned int, const double*, double*)


FOUR_PI = 4*np.pi


def cltoxi(cl, closed=False):
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
    xi : (n,) array_like
        Angular correlation function.

    See Also
    --------
    transformcl.xitocl : the inverse operation
    transformcl.theta : compute the angles at which the correlation function is
                        evaluated

    Notes
    -----
    The computation is done in three steps:

    First, the angular power spectrum is converted to the coefficients of a
    discrete Legendre transform (DLT).

    Second, the DLT coefficients are transformed to the coefficients of a
    discrete cosine transform (DCT) using the matrix relation given by [1]_.

    Finally, the angular correlation function is computed using a DCT-III for
    the open interval, or a DCT-I for the closed interval.

    References
    ----------
    .. [1] Alpert, B. K., & Rokhlin, V. (1991). A fast algorithm for the
       evaluation of Legendre expansions. SIAM Journal on Scientific and
       Statistical Computing, 12(1), 158-179.

    '''

    # length n of the transform
    if np.ndim(cl) != 1:
        raise TypeError('cl must be 1d array')
    n = np.shape(cl)[-1]

    # type of the DCT depends on open or closed interval
    if closed:
        dcttype = 1
    else:
        dcttype = 3

    # DLT coefficients = (2l+1)/(4pi) * Cl
    b = np.arange(1, 2*n+1, 2, dtype=float)
    b /= FOUR_PI
    b *= cl

    # this holds the DCT coefficients
    a = np.empty(n, dtype=float)

    # these are memviews on a and b for C interop
    cdef double[::1] a_ = a
    cdef double[::1] b_ = b

    # transform DLT coefficients to DCT coefficients using C function
    dltdct(n, &b_[0], &a_[0])

    # perform the DCT
    xi = dct(a, type=dcttype, axis=-1, norm=None)

    # done
    return xi


def xitocl(xi, closed=False):
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
    xi : (n,) array_like
        Angular correlation function.
    closed : bool
        Compute correlation function over open (``closed=False``) or closed
        (``closed=True``) interval.

    Returns
    -------
    xi : (n,) array_like
        Angular power spectrum from :math:`0` to :math:`\mathtt{lmax}`.

    See Also
    --------
    transformcl.cltoxi : the inverse operation
    transformcl.theta : compute the angles at which the correlation function is
                        evaluated

    Notes
    -----
    The computation is done in three steps:

    First, the angular correlation function is transformed to the coefficients
    of a discrete cosine transform (DCT) using an inverse DCT-III for the open
    interval, or an inverse DCT-I for the closed interval.

    Second, the DCT coefficients are transformed to the coefficients of a
    discrete Legendre transform (DLT) using the matrix relation given by [1]_.

    Lastly, the DLT coefficients are transformed to an angular power spectrum.

    References
    ----------
    .. [1] Alpert, B. K., & Rokhlin, V. (1991). A fast algorithm for the
       evaluation of Legendre expansions. SIAM Journal on Scientific and
       Statistical Computing, 12(1), 158-179.

    '''

    # length n of the Fourier series
    if np.ndim(xi) != 1:
        raise TypeError('xi must be 1d array')
    n = np.shape(xi)[-1]

    # type of the DCT depends on open or closed interval
    if closed:
        dcttype = 1
    else:
        dcttype = 3

    # compute the DCT coefficients
    a = idct(xi, type=dcttype, axis=-1, norm=None)

    # this holds the DLT coefficients
    b = np.empty(n, dtype=float)

    # these are memviews on a and b for C interop
    cdef double[::1] a_ = a
    cdef double[::1] b_ = b

    # transform DCT coefficients to DLT coefficients using C function
    dctdlt(n, &a_[0], &b_[0])

    # DLT coefficients = (2l+1)/(4pi) * Cl
    b *= FOUR_PI/np.arange(1, 2*n+1, 2, dtype=float)

    # done
    return b


def theta(n, closed=False):
    r'''compute nodes for the angular correlation function

    Returns :math:`n` angles :math:`\theta_0, \ldots, \theta_{n-1}` for the
    angular correlation function when transforming angular power spectra using
    :func:`~transformcl.xitocl` and :func:`~transformcl.cltoxi`.

    The returned angles can be distributed either over the closed interval
    :math:`[0, \pi]`, in which case :math:`\theta_0 = 0, \theta_{n-1} = \pi`,
    or over the open interval :math:`(0, \theta)`.

    Parameters
    ----------
    n : int
        Number of nodes.

    Returns
    -------
    theta : array_like (n,)
        Angles in radians.

    '''

    if closed:
        x = np.linspace(0, np.pi, n, dtype=float)
    else:
        x = np.arange(n, dtype=float)
        x += 0.5
        x *= np.pi/n

    return x
