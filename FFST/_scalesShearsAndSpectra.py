from __future__ import division, print_function, absolute_import

import numpy as np
import warnings
from .meyerShearlet import meyerShearletSpect, meyeraux


def _defaultNumberOfScales(l):
    numOfScales = int(np.floor(0.5 * np.log2(np.max(l))))
    if numOfScales < 1:
        raise ValueError('image too small!')
    return numOfScales


def scalesShearsAndSpectra(shape, numOfScales=None,
                           realCoefficients=True, maxScale='max',
                           shearletSpect=meyerShearletSpect,
                           shearletArg=meyeraux, realReal=True,
                           fftshift_spectra=True):
    """
    计算给定形状和尺度数的剪切波谱。
    """
    if len(shape) != 2:
        raise ValueError("2D image dimensions required")

    if numOfScales is None:
        numOfScales = _defaultNumberOfScales(shape)

    # rectangular images
    if shape[1] != shape[0]:
        rectangular = True
    else:
        rectangular = False


    shape = np.asarray(shape)
    shape_orig = shape.copy()
    shapem = np.mod(shape, 2) == 0
    both_even = np.all(np.equal(shapem, False))
    both_odd = np.all(np.equal(shapem, True))
    shape[shapem] += 1

    if not realCoefficients:
        warnings.warn("Complex shearlet case may be buggy.  Doesn't "
                      "currently give perfect reconstruction.")

    if not (both_even or both_odd):
        raise ValueError("Mixture of odd and even axis sizes is currently "
                         "unsupported.")

    maxScale = maxScale.lower()
    if maxScale == 'max':
        X = 2 ** (2 * (numOfScales - 1) + 1)
    elif maxScale == 'min':
        X = 2 ** (2 * (numOfScales - 1))
    else:
        raise ValueError('Wrong option for maxScale, must be "max" or "min"')

    xi_x_init = np.linspace(0, X, (shape[1] + 1) // 2)
    xi_x_init = np.concatenate((-xi_x_init[-1:0:-1], xi_x_init), axis=0)
    if rectangular:
        xi_y_init = np.linspace(0, X, (shape[0] + 1) // 2)
        xi_y_init = np.concatenate((-xi_y_init[-1:0:-1], xi_y_init), axis=0)
    else:
        xi_y_init = xi_x_init

    [xi_x, xi_y] = np.meshgrid(xi_x_init, xi_y_init[::-1], indexing='xy')

    C_hor = np.abs(xi_x) >= np.abs(xi_y)
    C_ver = np.abs(xi_x) < np.abs(xi_y)

    shearsPerScale = 2 ** (np.arange(numOfScales) + 2)
    numOfAllShears = 1 + shearsPerScale.sum()

    # init
    Psi = np.zeros(tuple(shape) + (numOfAllShears,))
    # lowpass
    Psi[:, :, 0] = shearletSpect(xi_x, xi_y, np.NaN, np.NaN, realCoefficients,
                                 shearletArg, scaling_only=True)

    for j in range(numOfScales):
        idx = 2 ** j
        start_index = 1 + shearsPerScale[:j].sum()
        shift = 1
        for k in range(-2 ** j, 2 ** j + 1):
            P_hor = shearletSpect(xi_x, xi_y, 2 ** (-2 * j), k * 2 ** (-j),
                                  realCoefficients, shearletArg)
            if rectangular:
                P_ver = shearletSpect(xi_y, xi_x, 2 ** (-2 * j), k * 2 ** (-j),
                                      realCoefficients, shearletArg)
            else:
                P_ver = np.rot90(P_hor, 2).T
            if not realCoefficients:
                P_ver = np.rot90(P_ver, 2)

            if k == -2 ** j:
                Psi[:, :, start_index + idx] = P_hor * C_hor + P_ver * C_ver
            elif k == 2 ** j:
                Psi_idx = start_index + idx + shift
                Psi[:, :, Psi_idx] = P_hor * C_hor + P_ver * C_ver
            else:
                new_pos = np.mod(idx + 1 - shift, shearsPerScale[j]) - 1
                if (new_pos == -1):
                    new_pos = shearsPerScale[j] - 1
                Psi[:, :, start_index + new_pos] = P_hor
                Psi[:, :, start_index + idx + shift] = P_ver

                shift += 1

    Psi = Psi[:shape_orig[0], :shape_orig[1], :]

    if realCoefficients and realReal and (shapem[0] or shapem[1]):
        idx_finest_scale = (1 + np.sum(shearsPerScale[:-1]))
        scale_idx = idx_finest_scale + np.concatenate(
            (np.arange(1, (idx_finest_scale + 1) / 2 + 1),
             np.arange((idx_finest_scale + 1) / 2 + 2, shearsPerScale[-1])),
            axis=0)
        scale_idx = scale_idx.astype(np.int)
        if shapem[0]:  # even number of rows -> modify first row:
            idx = slice(1, shape_orig[1])
            Psi[0, idx, scale_idx] = 1 / np.sqrt(2) * (
                    Psi[0, idx, scale_idx] +
                    Psi[0, shape_orig[1] - 1:0:-1, scale_idx])
        if shapem[1]:  # even number of columns -> modify first column:
            idx = slice(1, shape_orig[0])
            Psi[idx, 0, scale_idx] = 1 / np.sqrt(2) * (
                    Psi[idx, 0, scale_idx] +
                    Psi[shape_orig[0] - 1:0:-1, 0, scale_idx])

    if fftshift_spectra:
        Psi = np.fft.ifftshift(Psi, axes=(0, 1))
    return Psi
