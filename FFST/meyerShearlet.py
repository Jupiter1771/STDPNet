from __future__ import division, print_function, absolute_import

import numpy as np


def meyeraux(x):
    """meyer wavelet auxiliary function.
    v(x) = 35*x^4 - 84*x^5 + 70*x^6 - 20*x^7.
    """
    # Auxiliary def values.
    y = np.polyval([-20, 70, -84, 35, 0, 0, 0, 0], x) * (x >= 0) * (x <= 1)
    y += (x > 1)
    return y


def meyerBump(x, meyeraux_func=meyeraux):
    int1 = meyeraux_func(x) * (x >= 0) * (x <= 1)
    y = int1 + (x > 1)
    return y


def bump(x, meyeraux_func=meyeraux):
    """compute the def psi_2^ at given points x.
    """
    y = meyerBump(1+x, meyeraux_func)*(x <= 0) + \
        meyerBump(1-x, meyeraux_func)*(x > 0)
    y = np.sqrt(y)
    return y


def meyerScaling(x, meyeraux_func=meyeraux):
    """
    mother scaling def for meyer shearlet.
    """
    xa = np.abs(x)

    int1 = ((xa < 1/2))
    int2 = ((xa >= 1/2) & (xa < 1))

    phihat = int1 + int2 * np.cos(np.pi/2*meyeraux_func(2*xa-1))

    return phihat


def _meyerHelper(x, realCoefficients=True, meyeraux_func=meyeraux):
    if realCoefficients:
        xa = np.abs(x)
    else:
        xa = -x

    int1 = ((xa >= 1) & (xa < 2))
    int2 = ((xa >= 2) & (xa < 4))

    psihat = int1 * np.sin(np.pi/2*meyeraux_func(xa-1))
    psihat = psihat + int2 * np.cos(np.pi/2*meyeraux_func(1/2*xa-1))

    y = psihat
    return y


def meyerWavelet(x, realCoefficients=True, meyeraux_func=meyeraux):
    """
    compute Meyer Wavelet.
    """
    y = np.sqrt(np.abs(_meyerHelper(x, realCoefficients, meyeraux_func))**2 +
                np.abs(_meyerHelper(2*x, realCoefficients, meyeraux_func))**2)
    return y


def meyerShearletSpect(x, y, a, s, realCoefficients=True,
                       meyeraux_func=meyeraux, scaling_only=False):
    """
    Returns the spectrum of the shearlet "meyerShearlet".
    """
    if scaling_only:
        C_hor = np.abs(x) >= np.abs(y)
        C_ver = np.abs(x) < np.abs(y)
        Psi = (meyerScaling(x, meyeraux_func) * C_hor +
               meyerScaling(y, meyeraux_func) * C_ver)
        return Psi

    y = s * np.sqrt(a) * x + np.sqrt(a) * y
    x = a * x

    xx = (np.abs(x) == 0) + (np.abs(x) > 0)*x

    Psi = meyerWavelet(x, realCoefficients, meyeraux_func) * \
        bump(y/xx, meyeraux_func)
    return Psi


def meyerSmoothShearletSpect(x, y, a, s, realCoefficients=True,
                             meyeraux_func=meyeraux, scaling_only=False):
    """
    Returns the spectrum of the smooth variant of the Meyer shearlet
    """
    if scaling_only:
        Psi = meyerScaling(x, meyeraux_func) * meyerScaling(y, meyeraux_func)
        return Psi

    if not realCoefficients:
        raise ValueError('Complex shearlets not supported for smooth Meyer '
                         'shearlets!')

    asy = s * np.sqrt(a) * x + np.sqrt(a) * y
    y = a * y
    x = a * x

    W = np.sqrt((meyerScaling(2**(-2)*x, meyeraux_func) *
                 meyerScaling(2**(-2)*y, meyeraux_func))**2 -
                (meyerScaling(x, meyeraux_func) *
                 meyerScaling(y, meyeraux_func))**2)
    Psi = W * bump(asy/x, meyeraux_func)

    Psi[np.isnan(Psi)] = 0
    return Psi
