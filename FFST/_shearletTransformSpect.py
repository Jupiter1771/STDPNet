from __future__ import division, print_function, absolute_import

import numpy as np

from .meyerShearlet import (meyerShearletSpect, meyeraux)

from ._scalesShearsAndSpectra import scalesShearsAndSpectra
from ._fft import fftshift, ifftshift, fftn, ifftn


def shearletTransformSpect(A, Psi=None, numOfScales=None,
                           realCoefficients=True, maxScale='max',
                           shearletSpect=meyerShearletSpect,
                           shearletArg=meyeraux, realReal=True):
    # parse input
    A = np.asarray(A)
    if (A.ndim != 2) or np.any(np.asarray(A.shape) <= 1):
        raise ValueError("2D image required")

    # compute spectra
    if Psi is None:
        l = A.shape
        if numOfScales is None:
            numOfScales = int(np.floor(0.5 * np.log2(np.max(l))))
            if numOfScales < 1:
                raise ValueError('image to small!')
        Psi = scalesShearsAndSpectra(l, numOfScales=numOfScales,
                                     realCoefficients=realCoefficients,
                                     shearletSpect=meyerShearletSpect,
                                     shearletArg=meyeraux)

    # shearlet transform
    if False:
        # INCORRECT TO HAVE FFTSHIFT SINCE Psi ISNT SHIFTED!
        uST = Psi * fftshift(fftn(A))[..., np.newaxis]
        ST = ifftn(ifftshift(uST, axes=(0, 1)), axes=(0, 1))
    else:
        ST = Psi * fftn(A)[..., np.newaxis]  # (256, 256, 61)

    # 由于舍入误差，虚部不是零但非常小
    # -> 舍去
    if realCoefficients and realReal and np.isrealobj(A):
        ST = ST.real

    return (ST, Psi)
