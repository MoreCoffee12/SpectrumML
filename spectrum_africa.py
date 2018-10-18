from . import adiabatic as ad
from . import nonintshift as nonint
from . import flowthermo as ft
from . import voleff as ve
from . import clearance as cl
from . import plotAngle as rplt
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    # matplotlib is only needed if using the debugging option.
    pass


# noinspection PyPep8Naming
def dAutoTDC_Rough(dPress_abs, dDispVol, dK, bCrankEnd, dPress_Suct_abs, dPress_Dish_abs,
                   bDebug=False):
    """

    :param dPress_abs:          One revolution of absolute cylinder pressure data in engineering units
    :type dPress_abs:           numpy.ndarray
    :param dDispVol:            One revolution of normalized absolute volume
    :type dDispVol:             numpy.ndarray
    :param dK:                  Ratio of specific heats
    :type dK:                   numpy.float64`
    :param bCrankEnd:           Set to true if it is the crank end chamber
    :type bCrankEnd:            bool
    :param dPress_Suct_abs:     Suction pressure, absolute
    :type dPress_Suct_abs:      numpy.float64
    :param dPress_Dish_abs:     Discharge pressure, absolute
    :type dPress_Dish_abs:      numpy.float64
    :param bDebug:              (Optional) set to true for extended output
    :type bDebug:               bool
    :return:
        :param dAngleCorrected:         Estimated error, degrees
        :type dAngleCorrected:          numpy.float64`
        :param dPress_abs_corr_rough:   Corrected absolute pressure curve
        :type dPress_abs_corr_rough:    numpy.ndarray


    This function makes the first estimate of the timing error
    """
    # Perform the rough estimate of TDC. This has a couple of steps. The first is to
    # calculate a theoretical curve using the maximum and minimum pressures (if
    # the index angle is really wrong the suction and discharge pressures will not
    # correct)
    iSampleCount = len(dPress_abs)
    dPressAdiabatic_abs = ad.adiabatic(dDispVol, dK,
                                       bCrankEnd,
                                       dPress_Suct_abs,
                                       dPress_Dish_abs)

    # This section performs the correlation calculation
    dEndValue = 360.0 - (360.0 / float(iSampleCount))
    iValues = np.linspace(0, dEndValue, iSampleCount)
    dCorrArray = np.real(np.fft.ifft((np.fft.fft(dPress_abs)) * np.fft.fft(np.flipud(dPressAdiabatic_abs))))
    lag = np.mod(np.argmax(dCorrArray)+1, iSampleCount)
    dAngleCorrected = iValues[lag]
    dPress_abs_corr_rough = nonint.nonintshift(dPress_abs, dAngleCorrected)[0]

    if bDebug:

        dCorrArrayPadded = np.concatenate((dCorrArray, dCorrArray, dCorrArray), axis=0)
        xvals = np.arange(-360, 720)
        plt.figure(figsize=(14, 7))
        plt.plot(xvals, dCorrArrayPadded, '.')
        plt.plot(xvals[lag+360], dCorrArrayPadded[lag+360], 'x')
        plt.xticks(np.arange(-360, 721, step=30))
        plt.xlim([dAngleCorrected-30, dAngleCorrected+30])
        plt.xlabel('Offset, Degrees')
        plt.ylabel('Normalized correlation, -')
        plt.title('Normalized correlation Vs. Offset')
        plt.legend(['X-Corr', 'Max. Value'])

    return dAngleCorrected, dPress_abs_corr_rough


# noinspection PyPep8Naming
def calcThermoState(dPress_abs, dDisp, bCrankEnd, dBaro, dTemp_Suct_abs, dK, dZ_std, dZ_Suct, dZ_Dish):

    """
    :param dPress_abs:          One revolution of absolute cylinder pressure data in engineering units
    :type dPress_abs:           numpy.ndarray
    :param dDisp:               One revolution of normalized displacement
    :type dDisp:                numpy.ndarray
    :param bCrankEnd:           Set to true if it is the crank end chamber
    :type bCrankEnd:            bool
    :param dBaro:               Site barometric pressure
    :type dBaro:                numpy.float64
    :param dTemp_Suct_abs:      Suction temperature, absolute
    :type dTemp_Suct_abs:       numpy.float64
    :param dK:                  Ratio of specific heats
    :type dK:                   numpy.float64`
    :param dZ_std:              Standard compressibility
    :type dZ_std:               numpy.float64`
    :param dZ_Suct:             Suction compressibility
    :type dZ_Suct:              numpy.float64`
    :param dZ_Dish:             Discharge compressibility
    :type dZ_Dish:              numpy.float64`

    :return:
        :param dTemp_Suct_abs:              Suction temperature, absolute
        :type dTemp_Suct_abs:               numpy.float64
        :param dPress_Dish_abs:             Discharge temperature, absolute
        :type dPress_Dish_abs:              numpy.float64
        :param dR:                          Compression ratio
        :type dR:                           numpy.float64
        :param dClearancePercent_Suct:      Clearance volume at suction conditions
        :type dClearancePercent_Suct:       numpy.float64
        :param dClearancePercent_Dish:      Clearance volume at discharge conditions
        :type dClearancePercent_Dish:       numpy.float64
        :param dFlow_Suct:                  Capacity at suction conditions (per cycle)
        :type dFlow_Suct:                   numpy.float64
        :param dFlow_Dish:                  Capacity at discharge conditions (per cycle)
        :type dFlow_Dish:                   numpy.float64
        :param dFlowBalance:                Capacity at suction/capacity at discharge conditions
        :type dFlowBalance:                 numpy.float64

    """

    (idxValveOpenSuct, idxValveOpenDish, idxValveCloseSuct, idxValveCloseDish,
     dVESuct, dVEDish) = ve.voleff(dPress_abs, dDisp, bCrankEnd)

    dTemp_STP_abs, dPress_STP_abs = ft.get_stp_USCS()

    dPress_Suct_abs = dPress_abs[idxValveCloseSuct]
    dPress_Dish_abs = dPress_abs[idxValveCloseDish]
    dTemp_Dish_abs = ft.adiabatic_temp(dPress_Dish_abs, dPress_Suct_abs, dTemp_Suct_abs, dK)
    dR = dPress_Dish_abs / dPress_Suct_abs

    # Indicated clearance values
    bUseSuct = True
    dClearancePercent_Suct = cl.clearance(dK, dVESuct, dVEDish, dR, bUseSuct)
    bUseSuct = False
    dClearancePercent_Dish = cl.clearance(dK, dVESuct, dVEDish, dR, bUseSuct)

    # Flows and flow balance
    dFlow_Suct = ft.flow_per_cycle(dPress_Suct_abs, dPress_STP_abs, dTemp_Suct_abs, dTemp_STP_abs,
                                   dZ_Suct, dZ_std, dVESuct, dDispVol=1.0)
    dFlow_Dish = ft.flow_per_cycle(dPress_Dish_abs, dPress_STP_abs, dTemp_Dish_abs, dTemp_STP_abs,
                                   dZ_Dish, dZ_std, dVEDish, dDispVol=1.0)
    dFlowBalance = dFlow_Suct / dFlow_Dish

    return (dPress_Suct_abs, dPress_Dish_abs, dTemp_Dish_abs, dR, dClearancePercent_Suct, dClearancePercent_Dish,
            dFlow_Suct, dFlow_Dish, dFlowBalance)


# noinspection PyPep8Naming
def dAutoTDC(dPress_gage, strStaticHeader, dDisp, dDispVol,
             dZ_std, dBaro, dTemp_Suct_abs, dK, bCrankEnd,
             bDebug=False):

    """

    :param dPress_gage:         One revolution of cylinder pressure data in engineering units
    :type dPress_gage:          numpy.ndarray
    :param strStaticHeader:     String with static information about the plot
    :type strStaticHeader:      str
    :param dDisp:               One revolution of normalized displacement
    :type dDisp:                numpy.ndarray
    :param dDispVol:            One revolution of normalized absolute volume
    :type dDispVol:             numpy.ndarray
    :param dZ_std:              Compressibility at standard conditions
    :type dZ_std:               numpy.float64
    :param dBaro:               Site barometric pressure
    :type dBaro:                numpy.float64
    :param dTemp_Suct_abs:      Suction temperature, absolute
    :type dTemp_Suct_abs:       numpy.float64
    :param dK:                  Ratio of specific heats
    :type dK:                   numpy.float64`
    :param bCrankEnd:           Set to true if it is the crank end chamber
    :type bCrankEnd:            bool
    :param bDebug:              (Optional) set to true for extended output
    :type bDebug:               bool
    :return:
        dAngleCorrected         The corrected offset angle. This ranges from [0,360). A negative output indicates the
                                algorithm failed to find a solution.

    This is the main routine for estimating the correct index angle
    """

    # The are parameters that define the algorithm behavior.
    dMinPressRatio = 1.20
    iSamplesTune = 5

    # Calculate the absolute pressure curve
    dPress_abs = dPress_gage + dBaro

    # Is there enough pressure change in the cylinder? If not bail.
    dPressMax_abs = np.max(dPress_abs)
    dPressMin_abs = np.min(dPress_abs)
    dPressRatio = (dPressMax_abs - dPressMin_abs) / dPressMin_abs
    if dPressRatio < dMinPressRatio:
        return -1.0

    # Since the timing could have been far off (the suction toe pressure and discharge toe pressure
    # could have even been equal) the gas properties need to be re-calculated for the
    # approximated suction and discharge pressures.
    dPress_Dish_abs = dPressMax_abs*0.95
    dPress_Suct_abs = dPressMin_abs * 1.05
    dTemp_Dish_abs = ft.adiabatic_temp(dPress_Dish_abs, dPress_Suct_abs, dTemp_Suct_abs, dK)
    dZ_Suct = ft.getZ_HP2_A(dPress_Suct_abs, dTemp_Suct_abs)
    dZ_Dish = ft.getZ_HP2_A(dPress_Dish_abs, dTemp_Dish_abs)

    # ----------------------------------------------------------------------
    # Calculate the rough estimate of the error
    # ----------------------------------------------------------------------
    dAngleCorrected, dPress_abs_corr_rough = dAutoTDC_Rough(dPress_abs, dDispVol, dK, bCrankEnd, dPressMin_abs * 1.05,
                                                            dPressMax_abs * 0.95, bDebug)

    # Get the thermo properties of this rough estimate, create header for plot
    (dPress_Suct_abs, dPress_Dish_abs, dTemp_Dish_abs, dR, dClearancePercent_Suct, dClearancePercent_Dish,
     dFlow_Suct, dFlow_Dish, dFlowBalance) = calcThermoState(dPress_abs_corr_rough, dDisp, bCrankEnd,
                                                             dBaro, dTemp_Suct_abs, dK, dZ_std, dZ_Suct,
                                                             dZ_Dish)

    # Plot the results
    if bDebug:
        print("Rough TDC Offset: {0:.2f} deg".format(dAngleCorrected))
        dPressAdiabatic_abs, fig = rplt.plotRecipAngleFunc(
            dPress_gage, (dPress_abs_corr_rough - dBaro), dDispVol, dK, bCrankEnd, dPress_Suct_abs,
            dPress_Dish_abs, dBaro,
            strStaticHeader, strMain='As-found', strAux='Corrected',
            strTitle='Pressure Vs. Crank Angle | Rough Correction: {0:.2f} deg | '.format(360-dAngleCorrected),
            bAdiab=True)
        # fig.savefig('Figure05_RoughResults.svg', format='svg')
        # fig.savefig('Figure05_RoughResults.png', format='png')

    # ----------------------------------------------------------------------
    # Calculate the flow balance and indicated clearance curves
    # ----------------------------------------------------------------------
    iValues = np.array(range(-iSamplesTune, iSamplesTune + 1))
    dClearanceDelta = np.zeros(len(iValues), dtype=np.float32)
    dFlowBalance = np.zeros(len(iValues), dtype=np.float32)

    for idx in range(0, len(iValues)):
        # Circular shift the waveform
        dPressRoll = nonint.nonintshift(dPress_abs_corr_rough, iValues[idx])[0]

        # Calculate the valve index locations and volumetric efficiencies for this pressure waveform.
        (dPress_Suct_abs, dPress_Dish_abs, dTemp_Dish_abs, dR, dClearancePercent_Suct, dClearancePercent_Dish,
         dFlow_Suct, dFlow_Dish, dFlowBalance[idx]) = calcThermoState(dPressRoll, dDisp, bCrankEnd,
                                                                      dBaro, dTemp_Suct_abs, dK, dZ_std,
                                                                      dZ_Suct, dZ_Dish)

        dClearanceDelta[idx] = dClearancePercent_Suct - dClearancePercent_Dish

    # Check for convergence and normalize the flow balance delta
    dFlowBalanceRaw = (1 - dFlowBalance)
    if dFlowBalanceRaw[0] * dFlowBalanceRaw[-1] >= 0:
        return -1
    dFlowBalanceNorm = abs(1 - dFlowBalance)
    if dClearanceDelta[0] * dClearanceDelta[-1] >= 0:
        return -1
    dClearanceDelta = abs(dClearanceDelta)

    # Adjust the offset
    dAngleFine_FB = np.argmin(dFlowBalanceNorm)
    dAngleFine_Cl = np.argmin(dClearanceDelta)
    dAngleCorrected = dAngleCorrected - (iValues[dAngleFine_Cl] + iValues[dAngleFine_FB]) / 2.0

    if bDebug:
        print("Fine TDC Offset: {0:.2f} deg".format(dAngleCorrected))
        plt.figure(figsize=(7, 3.5))
        plt.plot(iValues, dFlowBalance, '.')
        plt.plot(iValues[dAngleFine_FB], dFlowBalance[dAngleFine_FB], 'x')
        plt.xlim(-iSamplesTune, iSamplesTune)
        plt.xticks(np.arange(-iSamplesTune, iSamplesTune, step=1))
        plt.xlabel('Sample Offset, Degrees')
        plt.ylabel('Flow Balance, -')
        plt.legend(['Values', 'Best Point'])
        plt.title('Flow Balance Vs. Sample | Min. FB: {0:0.4f}'.format(dFlowBalance[dAngleFine_FB]))
        fig = plt.gcf()
        # fig.savefig('Figure06_FlowBalance.svg', format='svg')
        # fig.savefig('Figure06_FlowBalance.png', format='png')

        plt.figure(figsize=(14, 7))
        plt.plot(iValues, dFlowBalanceNorm)
        plt.xlim(-iSamplesTune, iSamplesTune)
        plt.xticks(np.arange(-iSamplesTune, iSamplesTune, step=1))
        plt.xlabel('Sample Offset, Degrees')
        plt.ylabel('Normalized Flow Balance, -')
        plt.title('Normalized Flow Balance Vs. Sample')

        plt.figure(figsize=(7, 3.5))
        plt.plot(iValues, dClearanceDelta, '.')
        plt.plot(iValues[dAngleFine_Cl], dClearanceDelta[dAngleFine_Cl], 'x')
        plt.xlim(-iSamplesTune, iSamplesTune)
        plt.xticks(np.arange(-iSamplesTune, iSamplesTune, step=1))
        plt.xlabel('Sample Offset, Degrees')
        plt.ylabel('Clearance Volume Difference, percent')
        plt.legend(['Values','Best Point'])
        plt.title('Clearance Volume Difference Vs. Sample')
        fig = plt.gcf()
        # fig.savefig('Figure07_ClearanceBalance.svg', format='svg')
        # fig.savefig('Figure07_ClearanceBalance.png', format='png')

    return dAngleCorrected
