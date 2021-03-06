from __future__ import print_function, division
import sys, os
from datetime import date
import logging
import pickle

logging.basicConfig(level=logging.WARNING)

import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.io import fits
import astropy.io.ascii as at
from scipy.signal import argrelextrema

import k2spin
from k2spin import prot
from k2spin import detrend

def k2sff_io(filename, ext):
    """ Read in a K2SFF light curve file, and return the time and flux
    arrays from the selected extension.

    Removes timepoints where the MOVING flag is set, indicating data taken
    during a thurster fire.

    Inputs
    ------
    filename: string
        a valid K2SFF file, including full or relative path

    ext: int or string
        The light curve extension to return. If 1, 0, or "best", will return
        the BESTAPER extension, which corresponds to the best light curve as
        determined by Vanderberg et al.

    Returns:
    --------
    time, flux: arrays

    """
    hdu = fits.open(filename)
    #print(hdu.info())
    #print(hdu[ext].data.dtype)

    if ext=="best":
        ext=1
    elif ext==0:
        logging.warning("There is no light curve in extension 0; \n"
                        "returning best aperture")
        ext=1

    table = hdu[ext].data
    time = table["T"][table["MOVING"]==0]
    flux = table["FCOR"][table["MOVING"]==0]
    hdu.close()
    return time,flux

def k2sc_io(filename):
    """ Read in a K2SC light curve file, and return the time and flux.

    Flux is calculated as the completely detrended flux plus the
    time-dependent trend (i.e., only the position-dependent trend is removed),
    and then normalized.

    Inputs
    ------
    filename: string
        a valid K2SC file, including full or relative path

    Returns:
    --------
    time, flux: arrays

    """
    with fits.open(filename) as hdu:
        #print(hdu.info())

        table = hdu[1].data
        good = np.isfinite(table["flux"]) & (np.isfinite(table["trend_t"]))
        med_trend = np.median(table["trend_t"][good])
        time = table["time"][good]
        flux = table["flux"][good] + table["trend_t"][good] - med_trend

    return time,flux

def choose_initial_k2sff(filename):
    """ Select which aperture from the K2SFF file to use by picking the
    light curve with the highest periodogram peak <35 days.
    (BESTAPER frequently includes a long-term trend that shouldn't dominate.)"""

    peak_power = np.zeros(21)
    peak_periods = np.zeros(21)
    with fits.open(filename) as hdu:
        for ext in np.arange(2,21,1):
            table = hdu[ext].data
            t = table["T"][table["MOVING"]==0]
            f = table["FCOR"][table["MOVING"]==0]

            ls_out = prot.run_ls(t,f,np.ones_like(f),0.1,prot_lims=[0.1,35],
                                 run_bootstrap=False)
            fund_period, fund_power, periods_to_test, periodogram, _, _ = ls_out

            peak_periods[ext] = periods_to_test[np.argmax(periodogram)]
            peak_power[ext] = max(periodogram)

    best_ext = np.arange(2,21,1)[np.argmax(peak_power)]
    return best_ext

def run_one(t,f,epic=None,secondary_file=None):
    """Run a lomb-scargle analysis on one light curve.

    Inputs:
    -------
    t: array of epochs/time points

    f: array of fluxes corresponding to time points in t

    epic: object identifier

    Returns:
    --------
    fund_period, fund_power: floats
        period and power corresponding to the highest periodogram peak

    sig_period, sig_power: floats
        period and power corresponding to the highest periodogram peak
        that is a) higher than the bootstrap significance theshold, and
        b) higher than N(=100) nearby points as selected with argrelextrema

    sigma: float
        bootstrap significance threshold
    """
    logging.info(epic)

    ylims = np.percentile(f,[0.5,99.5])

    fig = plt.figure(figsize=(8,10))
    base_grid = gridspec.GridSpec(2,1,height_ratios=[2,3])
    if epic is not None:
        plt.suptitle("EPIC {0}".format(epic),fontsize="x-large")

    top_grid = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=base_grid[0])
    # Just plot the light curve
    ax = plt.subplot(top_grid[0])
    ax.plot(t,f,'k.')
    ax.set_ylim(ylims)

    # Run the lomb-scargle periodogram on the light curve
    ls_out = prot.run_ls(t,f,np.ones_like(f),0.1,prot_lims=[0.1,70],
                         run_bootstrap=True)
    # unpack lomb-scargle results
    fund_period, fund_power, periods_to_test, periodogram, aliases, sigmas = ls_out
    logging.info("Prot={0:.3f} Power={1:.3f}".format(fund_period,fund_power))


    # Find all peaks in the periodogram
    peak_locs = argrelextrema(periodogram,np.greater,order=100)
    print(len(peak_locs[0]),periods_to_test[np.argmax(peak_locs[0])])

    # Only keep significant peaks (use bootstrap significance levels)
    sig_locs = peak_locs[0][periodogram[peak_locs[0]]>sigmas[0]]
    sig_periods = periods_to_test[sig_locs]
    sig_powers = periodogram[sig_locs]

    # Plot the periodogram
    ax = plt.subplot(top_grid[1])
    ax.plot(periods_to_test,periodogram,'k-')
    ax.axvline(fund_period,color="r",linestyle=":",linewidth=2)
    ax.axhline(sigmas[0],color="grey",linestyle="-.",linewidth=2)
    ax.set_xscale("log")
    ax.set_xlim(0.1,70)
    # Mark significant peaks (if any) on the periodogram
    num_sig = len(sig_locs)

    if num_sig>0:
        plt.plot(sig_periods,sig_powers*1.1,'kv')
        # What's the most powerful of the significant peaks?
        most_significant = np.argmax(sig_powers)
        most_sig_period = sig_periods[most_significant]
        most_sig_power = sig_powers[most_significant]

        # If there are two or more, also record the second one
        if num_sig>1:
            trim_periods = np.delete(sig_periods, most_significant)
            trim_powers = np.delete(sig_powers, most_significant)
            second_significant = np.argmax(trim_powers)
            sec_period = trim_periods[second_significant]
            sec_power = trim_powers[second_significant]
        else:
            sec_period, sec_power = -9999, -9999

        # Record all secondary peaks, just in case
        if secondary_file is not None:
            for i in range(num_sig):
                secondary_file.write("{0},{1:.4f},{2:.4f},{3:.4f}\n".format(epic,
                                     sig_periods[i],sig_powers[i],sigmas[0]))

    else:
        most_sig_period, most_sig_power = -9999,-9999
        sec_period, sec_power = -9999, -9999

    # Count the number of significant periods besides the first
    # A likely harmonic doesn't count against this
    extra_sig = 0

    # Record type of potential harmonic, if applicable
    harm_type = "-"

    if num_sig>1:
        # Compare the two most significant periods
        period_ratio = sec_period / most_sig_period
        power_ratio = sec_power / most_sig_power

        # Is the secondary peak a 1/2 or 2x harmonic?
        if abs(period_ratio-0.5)<=0.05:
            harm_type = "half"
            extra_sig = num_sig - 2
        elif abs(period_ratio-2.0)<=0.05:
            harm_type = "dbl"
            extra_sig = num_sig - 2
        else:
            extra_sig = num_sig-1

        if (harm_type!="-") and (power_ratio>0.5):
            harm_type = harm_type+"-maybe"

    # plot phase-folded periods
    num_cols = np.int(np.ceil((len(sig_periods)+1) / 2))
    bottom_grid = gridspec.GridSpecFromSubplotSpec(2,num_cols,
                                                   subplot_spec=base_grid[1])

    # Plot the phase-folded light curve corresponding to the max
    # peak in the periodogram
    ax = plt.subplot(bottom_grid[0,0])
    phased_t = t % fund_period / fund_period
    ax.plot(phased_t,f,'r.')
    ax.set_ylim(ylims)
    ax.set_xlim(0,1)
    ax.set_title(r"P$_0$={0:.2f}".format(fund_period))

    # Now plot the phase-folded light curves for all other significant peaks
    row = 0
    for i,per in enumerate(sig_periods[np.argsort(sig_powers)]):
        if (i+1)==num_cols:
            row = 1
        ax = plt.subplot(bottom_grid[row,i+1-num_cols])
        phased_t = t % per / per
        ax.plot(phased_t,f,'k.')

        ax.set_ylim(ylims)
        ax.set_xlim(0,1)
        ax.set_title("P={0:.2f}".format(per))
        ax.tick_params(labelleft=False)

    plt.subplots_adjust(hspace=0.25)

    return (fund_period, fund_power, most_sig_period, most_sig_power,
            sec_period, sec_power, sigmas[0], extra_sig, harm_type)

def run_list(list_filenames,output_filename,data_dir,plot_dir):
    """ Run a list of K2SFF or K2SC files through run_one(), and save results.

    Inputs:
    -------
    list_filenames: list or array of filename strings

    output_filenames: string, giving the output filename for a table of results

    data_dir: directory that contains the data files

    plot_dir: directory to save plots in

    """

    n_files = len(list_filenames)
    fund_periods = np.zeros(n_files)
    fund_powers = np.zeros(n_files)
    sig_periods = np.zeros(n_files)
    sig_powers = np.zeros(n_files)
    sec_periods = np.zeros(n_files)
    sec_powers = np.zeros(n_files)
    thresholds = np.zeros(n_files)
    epics = np.zeros(n_files,np.int64)
    num_sig_peaks = np.zeros(n_files,int)
    harm_types = np.empty(n_files,"S10")
    harm_types[:] = "-"

    sec_filename = output_filename.replace(".csv","_allpeaks.csv")
    sec_file = open(sec_filename,"w")
    sec_file.write("EPIC,period,power,threshold\n")

    for i,filename in enumerate(list_filenames):
        epic = filename.split("/")[0].split("-")[0].split("_")[-1]

        if "k2sff" in filename:
            best_ext = choose_initial_k2sff(data_dir+filename)
            time,flux = k2sff_io(data_dir+filename,best_ext)
            lc_type = "k2sff"
        elif "k2sc" in filename:
            time,flux = k2sc_io(data_dir+filename)
            lc_type = "k2sc"
        one_out = run_one(time,flux,epic,sec_file)

        # Unpack analysis results
        fund_periods[i],fund_powers[i],sig_periods[i] = one_out[:3]
        sig_powers[i],sec_periods[i],sec_powers[i],thresholds[i] = one_out[3:7]
        num_sig_peaks[i],harm_types[i] = one_out[7:]
        epics[i] = epic

        # Save and close the plot files
        plt.savefig("{0}EPIC{1}_{2}.png".format(plot_dir,epic,lc_type),
                    bbox_inches="tight")
        plt.close()

#        if i>=3:
#            break

    sec_file.close()

    data = {"EPIC": epics,
            "fund_period": fund_periods,
            "fund_power": fund_powers,
            "sig_period": sig_periods,
            "sig_power": sig_powers,
            "sec_period": sec_periods,
            "sec_power": sec_powers,
            "threshold": thresholds,
            "num_sig": num_sig_peaks,
            "harm_type": harm_types}
    formats = {
            "fund_period": "%0.4f",
            "fund_power": "%0.4f",
            "sig_period": "%0.4f",
            "sig_power": "%0.4f",
            "sec_period": "%0.4f",
            "sec_power": "%0.4f",
            "threshold": "%0.6f"}

    names = ["EPIC","fund_period","fund_power",
            "sig_period","sig_power","sec_period","sec_power",
            "num_sig","harm_type","threshold"]

    pickle_file = open(output_filename.replace(".csv",".pkl"),"wb")
    pickle.dump(data,pickle_file)
    pickle_file.close()

    at.write(data,output_filename,names=names,
             formats=formats,delimiter=",")

if __name__=="__main__":

    today = date.isoformat(date.today())
#    logging.basicConfig(level=logging.INFO)

    # Retrieve the input list
    if len(sys.argv)==1:
        print("Please provide a list of light curve files")
    else:
        listfile = at.read(sys.argv[1])
        file_list = listfile["filename"]

    if len(sys.argv)>2:
        outfile0 = sys.argv[2]
        outfile = "{0}_{1}.csv".format(outfile0[:-4],today)
    else:
        outfile = "c5_tables/c5_output_{0}.csv".format(today)

    # Break down the data set into subsets for parallel processing
    arrayid = int(os.getenv("PBS_ARRAYID",0))

    mini = (arrayid - 1) * 50
    maxi = min(mini + 50, len(file_list))
    if arrayid==0:
        mini = 0
        maxi = len(file_list)
    else:
        outfile = outfile.replace(".csv","_{0}.csv".format(arrayid))

    print("Array, min(i), max(i)")
    print(arrayid, mini, maxi)

    sub_list = file_list[mini:maxi]

    # base_path = "/home/stephanie/projects/praesepe/"
    # data_path = "/home/stephanie/data/c5_k2sc/"
    # plot_path = base_path+"k2_plots/"
    base_path = "/vega/astro/users/sd2706/k2/"
    data_path = base_path+"data/"
    plot_path = base_path+"c5_plots/"

#    epic = 211748286
#    test_file = "hlsp_k2sff_k2_lightcurve_{0}-c05_kepler_v1_llc.fits".format(epic)

#    t,f = k2sff_io(data_path+test_file,1)
#    run_one(t,f,epic)
#    plt.savefig("{0}EPIC{1}_lstest.png".format(plot_path,epic),
#                bbox_inches="tight")
#    plt.close()

#    list_file = base_path+"data/test_k2sff_files.lst"
#    list_file = base_path+"data/all_k2sff_files.lst"
#    test_list = at.read(list_file)
    print(sub_list)

    run_list(sub_list,base_path+outfile,data_path,plot_path)
