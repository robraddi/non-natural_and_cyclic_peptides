
# Python libraries:{{{
import biceps
import FwdModelOpt_routines as fmo
import scipy, gc, os, copy, time, string, io
from PIL import Image
import pandas as pd
import numpy as np
import mdtraj as md
from mdtraj.geometry import compute_phi
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('xtick', labelsize=16)
matplotlib.rc('ytick', labelsize=16)
matplotlib.rc('axes', labelsize=16)
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.stats, copy, matplotlib
#from matplotlib import gridspec
from scipy.optimize import fmin
import matplotlib.gridspec as gridspec
from matplotlib import ticker



#:}}}

# Append to Database:{{{
def append_to_database(dbName="database_Nd.pkl", verbose=False, **kwargs):
    data = pd.DataFrame()
    for arg in kwargs:
        #print(arg)
        data[arg] = [kwargs.get(arg)]

    # NOTE: Saving results to database
    if os.path.isfile(dbName):
       db = pd.read_pickle(dbName)
    else:
        if verbose: print("Database not found...\nCreating database...")
        db = pd.DataFrame()
        db.to_pickle(dbName)
    db = db.append(data, ignore_index=True)
    db.to_pickle(dbName)
    #gc.collect()

# }}}

# plot_landscape:{{{

def get_kernel(kernel_idx):

    from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, RationalQuadratic, ExpSineSquared, DotProduct

    if kernel_idx == -2:
        kernel = Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=1.5)#+ WhiteKernel(noise_level=1e-8)

    elif kernel_idx == -1:
        kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0)) + WhiteKernel(noise_level=1e-5)

    elif kernel_idx == 0:
        kernel = RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0)) + Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(0.1, 10.0))
        #kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.05)
#        kernel = RBF(length_scale=0.5, length_scale_bounds=(0.1, 10.0)) + Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(0.1, 10.0)) + WhiteKernel(noise_level=0.1)

    elif kernel_idx == 1:
        kernel = Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=1.5)

    elif kernel_idx == 2:
        kernel = RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0)) #+ Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(0.1, 10.0))

    elif kernel_idx == 3:
        #kernel = RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0)) + WhiteKernel(noise_level=1e-2)
        #kernel = Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=1.5) + WhiteKernel(noise_level=1e-2)
        kernel = RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0)) + Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=1.5) + WhiteKernel(noise_level=1e-2)

    elif kernel_idx == 4:
        #kernel = RationalQuadratic(length_scale=1.0, length_scale_bounds=(0.1, 10.0), alpha=0.1) + Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=2.5) + WhiteKernel(noise_level=1e-2)
        #kernel = RationalQuadratic(length_scale=1.0, length_scale_bounds=(0.1, 10.0), alpha=0.1) + Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=1.5) + WhiteKernel(noise_level=1e-2)
        kernel = RationalQuadratic(length_scale=1.0, length_scale_bounds=(0.1, 10.0), alpha=0.1)*(Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=1.5) + WhiteKernel(noise_level=1e-2))

    elif kernel_idx == 5:
        kernel = (Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=2.5) + WhiteKernel(noise_level=1e-2))*RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0))

    else:
        raise ValueError("Invalid kernel index")

    return kernel



def plot_landscape(results, figname=None, gridpoints=100, lvls=50, upper_xy_lim=None, kernel_idx=0):

    from scipy.interpolate import interp2d
    from scipy.interpolate import griddata
    import matplotlib.patheffects as pe
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.ticker import NullFormatter
    nullfmt = NullFormatter()         # no labels

    markers = matplotlib.markers.MarkerStyle.filled_markers
    colors = matplotlib.colors
    colors = np.array(list(colors.__dict__['CSS4_COLORS'].values()))[10::3]
    colors = ["b", "g", "r", "m", "c", "orange", "purple"] + list(colors)

    #facecolors = ["white"] + colors[:len(results["score"].unique()) - 1]
    facecolors = ["white"] + colors[:len(results["A"]) - 1]

    figsize = (8, 6)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2)  # Create a grid with 2 rows and 2 columns
    marker_size = 50
    main_marker_size = 100


    # heatmap_function:{{{
    def generate_heatmap(ax, x, y, score, gridpoints=100, lvls=50, upper_xy_lim=None, show_colorbar=1, kernel_idx=0):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel
        from matplotlib import ticker
        from sklearn.gaussian_process.kernels import Matern

        if isinstance(upper_xy_lim, (list, tuple, np.ndarray)):
            max_x = upper_xy_lim[0]
            max_y = upper_xy_lim[1]
        else:
            max_x =max(x)
            max_y = max(y)

        #for gridpoints in range(2, 25):
        x_grid = np.linspace(min(x), max_x, gridpoints)
        y_grid = np.linspace(min(y), max_y, gridpoints)
        X, Y = np.meshgrid(x_grid, y_grid)

        kernel = get_kernel(kernel_idx)

        gp = GaussianProcessRegressor(kernel=kernel)

        X_train = np.vstack([x, y]).T
        gp.fit(X_train, score)

        X_test = np.vstack([X.ravel(), Y.ravel()]).T
        Z, std = gp.predict(X_test, return_std=True)
        Z = Z.reshape(X.shape)
        print(gridpoints, Z.min(), Z.max())
        #exit()



        cmap = plt.cm.coolwarm
        #cmap = plt.cm.RdBu_r

        min_score = min(score)
        max_score = max(score)
        levels = np.linspace(min_score, max_score, lvls)
        norm = matplotlib.colors.Normalize(vmin=min_score, vmax=max_score)

        cont = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm)
        ax.contourf(X, Y, Z, levels=levels, cmap=cmap, norm=norm)

        # Add contour lines with dark color and increased width
        ax.contour(X, Y, Z, levels=levels, colors='black', linewidths=1.5, alpha=0.6)

        if show_colorbar:
            cbar = plt.colorbar(cont, ax=ax, extend='both')

            # Specify the tick locations
            tick_locator = ticker.MaxNLocator(nbins=10)
            cbar.locator = tick_locator

            # Format the tick labels
            tick_formatter = ticker.FormatStrFormatter("%.1f")
            cbar.formatter = tick_formatter
            cbar.ax.tick_params(labelsize=14)

            #cbar.ax.set_ylabel(r'$f$', fontsize=22, rotation=0, labelpad=10)
            cbar.ax.set_ylabel(r'$u$', fontsize=22, rotation=0, labelpad=10)
        return cont
    # }}}

    ax1 = plt.subplot(gs[0, 0])              # Subplot for A vs B
    ax2 = plt.subplot(gs[1, 0], sharex=ax1)  # Subplot for A vs C
    ax3 = plt.subplot(gs[1, 1], sharey=ax2)  # Subplot for B vs C
    #ax4 = plt.subplot(gs[1, 2])              # subplot for karplus curves

    axs = [ax1, ax2, ax3]#, ax4]
    for ax in axs:
        ax.tick_params(which="major", axis="y", direction="inout", left=1, bottom=1, right=1, top=1)
        ax.tick_params(which="major", axis="x", direction="inout", left=1, bottom=1, right=1, top=1)
        ax.tick_params(which="minor", axis="y", direction="inout", left=1, bottom=1, right=1, top=1)
        ax.tick_params(which="minor", axis="x", direction="inout", left=1, bottom=1, right=1, top=1)
        ax.grid(alpha=0.5, linewidth=0.5)


    # Heatmap + Quiver for A vs B
    cont = generate_heatmap(ax1, results["A"], results["B"], results["score"], gridpoints=gridpoints, lvls=lvls, upper_xy_lim=upper_xy_lim, show_colorbar=0, kernel_idx=kernel_idx)

    # Heatmap + Quiver for A vs C
    generate_heatmap(ax2, results["A"], results["C"], results["score"], gridpoints=gridpoints, lvls=lvls, upper_xy_lim=upper_xy_lim, show_colorbar=0, kernel_idx=kernel_idx)

    # Heatmap + Quiver for B vs C
    generate_heatmap(ax3, results["B"], results["C"], results["score"], gridpoints=gridpoints, lvls=lvls, upper_xy_lim=upper_xy_lim, show_colorbar=0, kernel_idx=kernel_idx)


    ax1_pos = ax1.get_position()
    cbar_ax = fig.add_axes([ax1_pos.width+0.10, ax1_pos.y0+0.025, 0.02, ax1_pos.height-0.025])
    #cbar_ax = fig.add_axes([0.5, 0.60, 0.02, 0.9])
#    cbar = fig.colorbar(im1, cax=cbar_ax, orientation='vertical')
    cbar = plt.colorbar(cont, cax=cbar_ax, orientation='vertical')

    # Specify the tick locations
    tick_locator = ticker.MaxNLocator(nbins=10)
    cbar.locator = tick_locator

    # Format the tick labels
    tick_formatter = ticker.FormatStrFormatter("%.1f")
    cbar.formatter = tick_formatter
    cbar.ax.tick_params(labelsize=14)

    cbar.ax.set_ylabel(r'$u$', fontsize=22, rotation=0, labelpad=10)

    res = results.iloc[np.where(results["score"].to_numpy() == results["score"].to_numpy().min())[0]]

    print(f"Score Min: {results['score'].min()}")
    print(f"Score Max: {results['score'].max()}")
    print(f"Lowest BICePs score at: {res}")

    final_parameters = [float("%0.3g"%res["A"].to_numpy()[-1]), float("%0.3g"%res["B"].to_numpy()[-1]), float("%0.3g"%res["C"].to_numpy()[-1])]

    # Get the y-axis limits
    y_min, y_max = ax1.get_ylim()
    # Calculate the range of the y-axis
    y_range = y_max - y_min
    # Define the offset as a fraction of the y-axis range
    offset_fraction = 0.0025  # Adjust this value as needed
    offset = offset_fraction * y_range

    ax1.set_xlabel(r"$A$", fontsize=16)
    ax1.set_ylabel(r"$B$", fontsize=16)
    ax2.set_xlabel(r"$A$", fontsize=16)
    ax2.set_ylabel(r"$C$", fontsize=16)
    ax3.set_xlabel(r"$B$", fontsize=16)
    ax3.set_ylabel(r"$C$", fontsize=16)

    fig.subplots_adjust(hspace=0.35, wspace=0.5)
    if figname != None: fig.savefig(figname, dpi=600)
    return fig
# }}}

# plot_landscape_with_curve:{{{
def plot_landscape_with_curve(results, figname=None, gridpoints=100, lvls=50, upper_xy_lim=None, kernel_idx=0):

    from scipy.interpolate import interp2d
    from scipy.interpolate import griddata
    import matplotlib.patheffects as pe
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.ticker import NullFormatter
    nullfmt = NullFormatter()         # no labels

    markers = matplotlib.markers.MarkerStyle.filled_markers
    colors = matplotlib.colors
    colors = np.array(list(colors.__dict__['CSS4_COLORS'].values()))[10::3]
    colors = ["b", "g", "r", "m", "c", "orange", "purple"] + list(colors)

    #facecolors = ["white"] + colors[:len(results["score"].unique()) - 1]
    facecolors = ["white"] + colors[:len(results["A"]) - 1]

    figsize = (8, 6)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2)  # Create a grid with 2 rows and 2 columns
    marker_size = 50
    main_marker_size = 100


    # heatmap_function:{{{
    def generate_heatmap(ax, x, y, score, gridpoints=100, lvls=50, upper_xy_lim=None, show_colorbar=1, kernel_idx=0):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel
        from matplotlib import ticker
        from sklearn.gaussian_process.kernels import Matern

        if isinstance(upper_xy_lim, (list, tuple, np.ndarray)):
            max_x = upper_xy_lim[0]
            max_y = upper_xy_lim[1]
        else:
            max_x = max(x)
            #max_x = max_x - max_x*0.025
            max_y = max(y)
            #max_y = max_y - max_y*0.025

        #for gridpoints in range(2, 25):
        x_grid = np.linspace(min(x), max_x, gridpoints)
        y_grid = np.linspace(min(y), max_y, gridpoints)
        X, Y = np.meshgrid(x_grid, y_grid)

        kernel = get_kernel(kernel_idx)

        gp = GaussianProcessRegressor(kernel=kernel)

        X_train = np.vstack([x, y]).T
        gp.fit(X_train, score)

        X_test = np.vstack([X.ravel(), Y.ravel()]).T
        Z, std = gp.predict(X_test, return_std=True)
        Z = Z.reshape(X.shape)
        print(gridpoints, Z.min(), Z.max())
        #exit()



        cmap = plt.cm.coolwarm
        #cmap = plt.cm.RdBu_r

        min_score = min(score)
        max_score = max(score)
        levels = np.linspace(min_score, max_score, lvls)
        norm = matplotlib.colors.Normalize(vmin=min_score, vmax=max_score)

        cont = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm)
        ax.contourf(X, Y, Z, levels=levels, cmap=cmap, norm=norm)
        # Add contour lines with dark color and increased width
        ax.contour(X, Y, Z, levels=levels, colors='black', linewidths=1.0, alpha=0.6)


        if show_colorbar:
            cbar = plt.colorbar(cont, ax=ax, extend='both')

            # Specify the tick locations
            tick_locator = ticker.MaxNLocator(nbins=10)
            cbar.locator = tick_locator

            # Format the tick labels
            tick_formatter = ticker.FormatStrFormatter("%.1f")
            cbar.formatter = tick_formatter
            cbar.ax.tick_params(labelsize=14)

            #cbar.ax.set_ylabel(r'$f$', fontsize=22, rotation=0, labelpad=10)
            cbar.ax.set_ylabel(r'$u$', fontsize=22, rotation=0, labelpad=10)
        return cont
    # }}}

    ax1 = plt.subplot(gs[0, 0])              # Subplot for A vs B
    ax2 = plt.subplot(gs[1, 0], sharex=ax1)  # Subplot for A vs C
    ax3 = plt.subplot(gs[1, 1], sharey=ax2)  # Subplot for B vs C
    ax4 = plt.subplot(gs[0, 1])              # subplot for karplus curves

    axs = [ax1, ax2, ax3, ax4]
    for ax in axs:
        ax.tick_params(which="major", axis="y", direction="inout", left=1, bottom=1, right=1, top=1)
        ax.tick_params(which="major", axis="x", direction="inout", left=1, bottom=1, right=1, top=1)
        ax.tick_params(which="minor", axis="y", direction="inout", left=1, bottom=1, right=1, top=1)
        ax.tick_params(which="minor", axis="x", direction="inout", left=1, bottom=1, right=1, top=1)
        #ax.grid(alpha=0.5, linewidth=0.5)

    # Heatmap + Quiver for A vs B
    cont = generate_heatmap(ax1, results["A"], results["B"], results["score"], gridpoints=gridpoints, lvls=lvls, upper_xy_lim=upper_xy_lim, show_colorbar=0, kernel_idx=kernel_idx)

    # Heatmap + Quiver for A vs C
    generate_heatmap(ax2, results["A"], results["C"], results["score"], gridpoints=gridpoints, lvls=lvls, upper_xy_lim=upper_xy_lim, show_colorbar=0, kernel_idx=kernel_idx)

    # Heatmap + Quiver for B vs C
    generate_heatmap(ax3, results["B"], results["C"], results["score"], gridpoints=gridpoints, lvls=lvls, upper_xy_lim=upper_xy_lim, show_colorbar=0, kernel_idx=kernel_idx)


    ax1_pos = ax1.get_position()
    #cbar_ax = fig.add_axes([ax1_pos.width+0.10, ax1_pos.y0+0.025, 0.02, ax1_pos.height-0.025])
    #cbar = plt.colorbar(cont, cax=cbar_ax, orientation='vertical')

    cbar_ax = fig.add_axes([ax1_pos.x0+0.075, ax1_pos.y0 + ax1_pos.height + 0.01, ax1_pos.width-0.05, 0.02])
    cbar = plt.colorbar(cont, cax=cbar_ax, orientation='horizontal')

    # Specify the tick locations
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator

    # Format the tick labels
    tick_formatter = ticker.FormatStrFormatter("%.1f")
    cbar.formatter = tick_formatter
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=25)

    cbar.ax.set_ylabel(r'$f$', fontsize=22, rotation=0, labelpad=10)

    cbar_ax.xaxis.set_ticks_position('top')
    cbar_ax.xaxis.set_label_position('top')


    res = results.iloc[np.where(results["score"].to_numpy() == results["score"].to_numpy().min())[0]]

    print(f"Score Min: {results['score'].min()}")
    print(f"Score Max: {results['score'].max()}")
    print(f"Lowest BICePs score at: {res}")

    final_parameters = [float("%0.3g"%res["A"].to_numpy()[-1]), float("%0.3g"%res["B"].to_numpy()[-1]), float("%0.3g"%res["C"].to_numpy()[-1])]

    # Get the y-axis limits
    y_min, y_max = ax1.get_ylim()
    # Calculate the range of the y-axis
    y_range = y_max - y_min
    # Define the offset as a fraction of the y-axis range
    offset_fraction = 0.0025  # Adjust this value as needed
    offset = offset_fraction * y_range

    ax1.set_xlabel(r"$A$", fontsize=16)
    ax1.set_ylabel(r"$B$", fontsize=16)
    ax2.set_xlabel(r"$A$", fontsize=16)
    ax2.set_ylabel(r"$C$", fontsize=16)
    ax3.set_xlabel(r"$B$", fontsize=16)
    ax3.set_ylabel(r"$C$", fontsize=16)

    fig.subplots_adjust(hspace=0.35, wspace=0.5)
    if figname != None: fig.savefig(figname, dpi=600)
    return fig
# }}}

# biceps specs:{{{
################################################################################
n_lambdas,nreplicas,nsteps,change_Nr_every,swap_every=2,8,50000,0,0
n_lambdas,nreplicas,nsteps,change_Nr_every,swap_every=7,128,50000,0,0

#n_lambdas,nreplicas,nsteps,change_Nr_every,swap_every=7,64,50000,0,0




plot_landscapes = 0
landscape_stride = 5

progress_bar = 1

verbose=True
n_converged_iters=2

max_iterations = 10 # maximum number of BICePs iterations

stat_model, data_uncertainty="GB", "single"

data_likelihood = "gaussian" #"log normal"
#data_likelihood = "log normal" #"gaussian"

attempt_move_state_every = 1
attempt_move_sigma_every = 1
attempt_move_fmp_every = 1

# Good parameters
multiprocess=1
write_every = 1000
write_every = 10
plottype="step"
burn = 50000
#burn = 100000
#burn = 0
#burn = 40000
#burn = 20000
#burn = 0

stride = 1

spec_J_str = f".J_"
#spec_J_str = f".J_C_CB"
#spec_J_str = f".J_HN_HA" # f".J_"
#spec_J_str = f".J_HN"
#f".J_C"
#spec_J_str = f".J_HN_HA"
#f".J_HN_CB"
#f".J_HN_CB"


################################################################################

if stat_model == "Students":
    phi,phi_index,stat_model,data_uncertainty=(1.0, 2.0, 1),0,"Students","single"
    beta,beta_index,stat_model,data_uncertainty=(1.0, 100.0, 10000),0,"Students","single"

#    phi,phi_index,stat_model,data_uncertainty=(1.0, 2.0, 1),0,"Students","single"
#    beta,beta_index,stat_model,data_uncertainty=(1.0, 2.0, 1),0,"Students","single"

elif stat_model == "GB":
    phi,phi_index,stat_model,data_uncertainty=(1.0, 10.0, 1000),0,"GB","single"
    #phi,phi_index,stat_model,data_uncertainty=(1.0, 2.0, 1),0,"GB","single"
    beta,beta_index,stat_model,data_uncertainty=(1.0, 2.0, 1),0,"GB","single"

else:
    phi,phi_index=(1.0, 2.0, 1),0
    beta,beta_index=(1.0, 2.0, 1),0


#print(f"nSteps of sampling: {nsteps}\nnReplicas: {nreplicas}")
lambda_values = [0.0]*4
#NOTE: [(λ,ξ), (λ,ξ)]
dsig = 1.02
sigMin = 0.001
sigMax = 200
sigMax = 100

arr = np.exp(np.arange(np.log(sigMin), np.log(sigMax), np.log(dsig)))
l = len(arr)
sigma_index = round(l*0.73)#1689 # 0.80


#:}}}

# fwd_model_function: {{{

class J_data_continer_class:
    def __init__(self, grouped_files, skip_idx):
        self.skip_idx = skip_idx

    def compute_ensemble_avg_J_with_derivatives(self, phi_angles, A, B, C, phi0):

        indices = None
        avg_J, avg_dJ, avg_d2J = [], [], []

        for i, state in enumerate(grouped_files):
            indices_phi_list = phi_angles[i*len(state):(i+1)*len(state)]
            # Use list comprehension for faster operations
            state_data = [fmo.get_scalar_couplings_with_derivatives(phi, A, B, C, phi0) for indices,phi in indices_phi_list]
            state_J, state_dJ, state_d2J = zip(*state_data)

            avg_J.append(np.delete(np.concatenate(np.mean(np.array(state_J), axis=0)), self.skip_idx, axis=0))
            avg_dJ.append(np.delete(np.concatenate(np.mean(np.array(state_dJ), axis=0)), self.skip_idx, axis=1))
            avg_d2J.append(np.delete(np.concatenate(np.mean(np.array(state_d2J), axis=0)), self.skip_idx, axis=1))

        avg_J,avg_dJ,avg_d2J = np.array(avg_J),np.array(avg_dJ),np.array(avg_d2J)
        dJ_shape = avg_dJ.shape
        d2J_shape = avg_d2J.shape
        avg_dJ = avg_dJ.reshape((dJ_shape[1], dJ_shape[0], dJ_shape[2]))
        avg_d2J = avg_d2J.reshape((d2J_shape[1], d2J_shape[0], d2J_shape[2]))
        self.indices = np.delete(np.array(phi_angles[0][0], dtype=int), self.skip_idx, axis=0)
        return avg_J, avg_dJ, avg_d2J




# }}}

# plot_kde_of_phi_angles: {{{


def get_kde_with_best_bandwidth(X, pops, verbose=False):
    # https://users.cs.duke.edu/~brd/Teaching/Bio/asmb/Papers/NMR/nilges-jmr05.pdf
    from scipy.stats import gaussian_kde
    from sklearn.model_selection import KFold

    def kde_cv_score(X, weights, bandwidth, k=10):
        kf = KFold(n_splits=k)
        scores = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            weights_train = weights[train_index]
            weights_train /= weights_train.sum()

            kde = gaussian_kde(X_train, weights=weights_train, bw_method=bandwidth)
            score = np.sum(kde.logpdf(X_test))
            scores.append(score)

        return np.mean(scores)

    bandwidths = np.linspace(0.1, 5.0, 10)  # Adjust this range based on your data

    # Repeat for kde_post
    _best_bandwidth = None
    best_score_post = -np.inf

    for width in bandwidths:
        score = kde_cv_score(X, pops, width)
        if score > best_score_post:
            best_score_post = score
            _best_bandwidth = width

    if verbose: print("Best bandwidth for kde_post:", _best_bandwidth)
    kde = gaussian_kde(X, weights=pops)
    kde.set_bandwidth(bw_method=_best_bandwidth)
    return kde




def plot_kde_of_phi_angles(total_phi_angles, prior_populations, reweighted_populations, filename="out.png"):

    # https://users.cs.duke.edu/~brd/Teaching/Bio/asmb/Papers/NMR/nilges-jmr05.pdf
    from scipy.stats import gaussian_kde
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from sklearn.model_selection import KFold
    import string

    xmin, xmax = -180, 180
    fig = plt.figure(figsize=(8,4))

    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0,0])

    total_phi_angles = np.rad2deg(total_phi_angles)


    for p in range(len(total_phi_angles[0])):
        X = total_phi_angles[:, p]


        kde_prior = get_kde_with_best_bandwidth(X, prior_populations)
        kde_post = get_kde_with_best_bandwidth(X, reweighted_populations)

        # Generate a range of values for plotting the KDE
        x = np.linspace(xmin, xmax, 1000)

        #labels = ["A", "B", "C"]
        colors = ["red", "blue"]

        _kde_prior = kde_prior(x)
        _kde_post = kde_post(x)

        avg_prior = np.sum([X[i]*w for i,w in enumerate(prior_populations)], axis=0)
        avg_post = np.sum([X[i]*w for i,w in enumerate(reweighted_populations)], axis=0)

        ax.plot(x, _kde_prior, color=colors[0], label="__no_legend__")
        ax.plot(x, _kde_post, color=colors[1], label="__no_legend__")


        # Add vertical lines and labels
        ax.axvline(x=avg_prior, ymin=0, ymax=100, color=colors[0], ls='--', lw=3)
        ax.step(x, _kde_prior, color=colors[0], where='mid', label="__no_legend__")
        #ax.step(x, _kde_prior, color="k", where='mid', label="__no_legend__")
        ax.fill_between(x, _kde_prior, step="mid", color=colors[0], alpha=0.0)

        # Add vertical lines and labels
        ax.axvline(x=avg_post, ymin=0, ymax=100, color=colors[1], ls='--', lw=3)
        #ax.step(x, _kde_post, color=colors[1], where='mid', label="__no_legend__")
        ax.step(x, _kde_post, color="k", where='mid', label="__no_legend__")
        ax.fill_between(x, _kde_post, step="mid", color=colors[1], alpha=0.4)


        #span_start = exp_parameters[p] - exp_parameters_sigma[p]  # Define the start of the shaded region
        #span_end = exp_parameters[p] + exp_parameters_sigma[p]  # Define the end of the shaded region
        #if p==len(exp_parameters)-1: ax.axvspan(span_start, span_end, color="purple", alpha=0.4, label="Bax1995 NMR/X-ray")
        #else: ax.axvspan(span_start, span_end, color="purple", alpha=0.4, label="__no_legend__")

        #span_start = exp[p] - exp_sigma[p]  # Define the start of the shaded region
        #span_end = exp[p] + exp_sigma[p]  # Define the end of the shaded region
        #if p==len(exp_parameters)-1: ax.axvspan(span_start, span_end, color="black", alpha=0.4, label="Exp. X-ray")
        #else: ax.axvspan(span_start, span_end, color="black", alpha=0.4, label="_no_legend_")


    ax.set_xlabel(r"$\phi$ [degrees]", size=16)
    ax.set_ylabel("Density", size=16)
    # Generate a range of values for plotting the KDE
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(bottom=0)


    #for i in range(len(axs)):
    #ax = axs[p]
    # Set tick labels size
    ticks = [ax.xaxis.get_minor_ticks(), ax.xaxis.get_major_ticks(), ax.yaxis.get_minor_ticks(), ax.yaxis.get_major_ticks()]
    for tick in ticks:
        for subtick in tick:
            subtick.label.set_fontsize(14)

    # Add subplot labels
    #ax.text(-0.1, 1.0, string.ascii_lowercase[0], transform=ax.transAxes, size=20, weight='bold')
    label_fontsize = 14
    ax.legend(loc='center left', bbox_to_anchor=(1.025, 0.5), fontsize=label_fontsize)
    plt.gcf().subplots_adjust(left=0.105, bottom=0.125, top=0.92, right=0.785, wspace=0.20, hspace=0.5)


    fig.tight_layout()
    fig.savefig(filename, dpi=400)

# }}}

# plot_kde_of_karplus_predictions: {{{

def plot_kde_of_karplus_predictions(parameters_for_states, pops, exp, exp_sigma, exp_parameters, exp_parameters_sigma, filename="out.png"):

    # https://users.cs.duke.edu/~brd/Teaching/Bio/asmb/Papers/NMR/nilges-jmr05.pdf
    from scipy.stats import gaussian_kde
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from sklearn.model_selection import KFold
    import string

    xmin, xmax = -3, np.max(exp_parameters) + 2 #X.min(), X.max()
    fig = plt.figure(figsize=(8,4))

    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0,0])
    for p in range(len(parameters_for_states[0])):
        #parameters_for_states[:, p]
        # Create KDEs for alpha and unfolded states using the weights
        X = parameters_for_states[:, p]

        kde_post = get_kde_with_best_bandwidth(X, pops)

        # Generate a range of values for plotting the KDE
        x = np.linspace(xmin, xmax, 1000)
        #x = np.linspace(-2, 8.5, 2000)
        #ax.set_xlim(xmin, xmax)

        labels = ["A", "B", "C"]
        colors = ["green", "red", "blue"]

#        ax.plot(x, kde_prior(x), color='Orange', label='Prior')
        ax.plot(x, kde_post(x), color=colors[p], label=labels[p] + ' Reweighted')

        # Add vertical lines and labels as before
#        ax.axvline(x=avg_prior[p], ymin=0, ymax=100, color="orange", ls='--', lw=4)
        #ax.axvline(x=avg1[p], ymin=0, ymax=100, color="green", lw=4)
        ax.axvline(x=new_parameters[p], ymin=0, ymax=100, color=colors[p], ls='--', lw=3)
        ax.step(x, kde_post(x), color=colors[p], where='mid', label="__no_legend__")
        ax.fill_between(x, kde_post(x), step="mid", color=colors[p], alpha=0.4)


        # Add a vertical shaded region instead of axvline
        #exp_parameters = [6.98, -1.38, 1.72] # [±0.04, ±0.04, ±0.03]
        #delta = [0.04, 0.04, 0.03]

        span_start = exp_parameters[p] - exp_parameters_sigma[p]  # Define the start of the shaded region
        span_end = exp_parameters[p] + exp_parameters_sigma[p]  # Define the end of the shaded region
        if p==len(exp_parameters)-1: ax.axvspan(span_start, span_end, color="purple", alpha=0.4, label="Bax1995 NMR/X-ray")
        else: ax.axvspan(span_start, span_end, color="purple", alpha=0.4, label="__no_legend__")


        #exp_parameters = [6.64, -1.43, 1.86]
        #delta = [0.11, 0.04, 0.09]
        span_start = exp[p] - exp_sigma[p]  # Define the start of the shaded region
        span_end = exp[p] + exp_sigma[p]  # Define the end of the shaded region
        if p==len(exp_parameters)-1: ax.axvspan(span_start, span_end, color="black", alpha=0.4, label="Exp. X-ray")
        else: ax.axvspan(span_start, span_end, color="black", alpha=0.4, label="_no_legend_")


#        exp_parameters = [6.51, -1.37, 2.23]
#        delta = [0.00, 0.00, 0.00]
#        span_start = exp_parameters[p] - delta[p]  # Define the start of the shaded region
#        span_end = exp_parameters[p] + delta[p]  # Define the end of the shaded region
#        if p==len(exp_parameters)-1: ax.axvspan(span_start, span_end, color="cyan", alpha=0.4, label="Opt")
#        else: ax.axvspan(span_start, span_end, color="cyan", alpha=0.4, label="_no_legend_")



    ax.set_xlabel(f"Karplus coefficents [Hz]", size=16)
    ax.set_ylabel("Density")

    ax.set_ylabel("Density", size=16)
    # Generate a range of values for plotting the KDE
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(bottom=0)


    #for i in range(len(axs)):
    #ax = axs[p]
    # Set tick labels size
    ticks = [ax.xaxis.get_minor_ticks(), ax.xaxis.get_major_ticks(), ax.yaxis.get_minor_ticks(), ax.yaxis.get_major_ticks()]
    for tick in ticks:
        for subtick in tick:
            subtick.label.set_fontsize(14)

    # Add subplot labels
    #ax.text(-0.1, 1.0, string.ascii_lowercase[0], transform=ax.transAxes, size=20, weight='bold')
    label_fontsize = 14
    ax.legend(loc='center left', bbox_to_anchor=(1.025, 0.5), fontsize=label_fontsize)
    plt.gcf().subplots_adjust(left=0.105, bottom=0.125, top=0.92, right=0.785, wspace=0.20, hspace=0.5)


    fig.tight_layout()
    fig.savefig(filename, dpi=400)

# }}}

# get_indices_to_skip_for_phi_angles:{{{
def get_indices_to_skip_for_phi_angles(structure_file, skip_idx, idx_shift=0, verbose=True):
    df = pd.DataFrame()
    frame = md.load(structure_file)
    J = biceps.J_coupling.compute_J3_HN_HA(frame, model="Bax2007")
    table = frame.topology.to_dataframe()[0]
    #for k in range(len(J[0])):
    #    idxs = [frame.topology.atom(idx) for idx in J[0][k]]
    #    print(idxs)
    #exit()
    resName = table["resName"].to_numpy()
    resNames = np.array([val+str(table["resSeq"].to_numpy()[r]) for r,val in enumerate(resName)])
    #print(resName)
    _, idx = np.unique(resNames, return_index=True)
    #print(len(idx))
    #print(len(resNames))
    df["Residue"] = resNames[np.sort(idx)]
    #print(df)
    residues = df["Residue"].to_numpy()
    #print(residues)

    listing = []
    top_res = []
    for k in range(len(J[0])):
        idxs = [frame.topology.atom(idx) for idx in J[0][k]]
        res = str(idxs[1]).split("-")[0]
        if verbose: print(res)
        top_res.append(res)
        try:
            if res in residues[skip_idx]:
                listing.append(int(np.where(res == np.array(residues[skip_idx]))[0]))
                continue
        except(Exception) as e: pass
        #print(k, idxs)
    #exit()
    if verbose: print("----")
    if verbose: print(residues)
    if verbose: print(top_res)
    if verbose: print(listing)
    # NOTE: IMPORTANT: FIXME: Make sure that this - 1 is correct on all systems
    #skip_idx = skip_idx[listing] - 1
    skip_idx = skip_idx[listing] + idx_shift
    return skip_idx
# }}}

# update_angles_and_exp:{{{
def update_angles_and_exp(angles, coeff_idx, exp=[], remove_flexible_res=True, axis=0):
    #- look at SI and find the indices of the angles you need to skip.
    add_skip = []
    if remove_flexible_res:
        if coeff_idx == 3: add_skip = np.array([0]) # L73 and R74 don't have data for J_HN_C
        else: add_skip = np.array([0,-2,-1])

        if exp != []: exp = np.delete(exp, add_skip, axis=axis)
        angles = np.delete(angles, add_skip, axis=axis)

    #else:
    #    #if coeff_idx != 3:
    #    if coeff_idx < 3:
    #        add_skip = np.array([-2,-1])
    #        if exp != []: exp = np.delete(exp, add_skip, axis=axis)
    #        angles = np.delete(angles, add_skip, axis=axis)

    if (coeff_idx <= 3):
        angles = np.delete(angles, [-1,-2], axis=axis)
        #exit()
    return angles, exp, add_skip

# }}}

# get_parameters_from_key:{{{
def get_parameters_from_key(parameter_key, _options, coefficents):
    opts = pd.DataFrame(_options)
    restraint_indices = [i for i,val in enumerate(opts["extension"].to_numpy()) if f"J" in val]
    phi0,k_labels,coeff_indices,J_type = [],[],[],[]
    all_phi_angles = []
    parameters = []
    parameters_std = []
    idx = 0
    if (parameter_key == "Raddi2024") or (parameter_key == "Bax1997") or (parameter_key == "Habeck"):
        for idx,_ in enumerate(_options):
            if idx not in restraint_indices: continue

            for i in range(len(coefficents)):
                if coefficents[i] == _["extension"].replace("J_",""):
                    coeff_indices.append(i)
                    J_type.append(coefficents[i])
                    break

            k_labels.append(karplus_labels[i])
            parameter_sets = getattr(biceps.J_coupling, "J3_%s_coefficients"%coefficents[i])
            parameter_uncert_sets = getattr(biceps.J_coupling, "J3_%s_uncertainties"%coefficents[i])
            m = list(parameter_sets.keys())[0]
            phi0.append(parameter_sets[m]["phi0"])
            if parameter_key in parameter_sets.keys():
                parameters.append([parameter_sets[parameter_key][q] for q in ["A", "B", "C"]])
    else:
        # FIXME:
        if parameter_key == "BICePs(1d3z)":
            sampler = biceps.toolbox.load_object("Ubq_results/1d3z/GB_single_sigma/50000_steps_32_replicas_2_lam_opt/sampler_obj.pkl")

        chains = sampler.fmp_traj
        for ct in range(chains.shape[2]):
            gr_input = chains[:,:,ct,:]
            chain_stats = fmo.chain_statistics(gr_input)
            mean = chain_stats["mean_over_all_chains"]
            parameters.append(mean)
            std = chain_stats["avg_uncert"]
            parameters_std.append(std)
    return parameters, parameters_std, phi0
# }}}

if __name__ == "__main__":

    # Main:{{{
    coefficents = ["HN_HA"]
    karplus_labels = [r"H^{N}H^{\alpha}"]

    # NOTE: TODO: Sequence 1 has the most experimental J-coupling data (9 data points)
    plot_landscapes = 1
    run_optimization = 1

    #nstates = 100
    nstates = 500
    if nstates == 500:
        ext = "500_states/"
    else:
        ext = ""

    J_only   = 1
    combine_cs = 1

    skip_dirs = ["biceps", "experimental_data", 'old_systems']
    main_dir = "Systems"
    system_dirs = next(os.walk(main_dir))[1]
    system_dirs = [dir for dir in system_dirs if dir not in skip_dirs]
    system_dirs = biceps.toolbox.natsorted(system_dirs)
    print(system_dirs)
    #exit()
    # 1a, 1b
    #sys_name = os.path.join(main_dir,system_dirs[0]) # FIXME::::::::::::::::::::::::::::::::::::::::
    #sys_name = os.path.join(main_dir,system_dirs[1]) # FIXME::::::::::::::::::::::::::::::::::::::::
    ## 2a, 2b
    #sys_name = os.path.join(main_dir,system_dirs[-3]) # FIXME::::::::::::::::::::::::::::::::::::::::
#?#    sys_name = os.path.join(main_dir,system_dirs[2]) # FIXME::::::::::::::::::::::::::::::::::::::::
    ## 3a, 3b
    #sys_name = os.path.join(main_dir,system_dirs[5]) # FIXME::::::::::::::::::::::::::::::::::::::::
    #sys_name = os.path.join(main_dir,system_dirs[3]) # FIXME::::::::::::::::::::::::::::::::::::::::
    ## 4a, 4b
    #sys_name = os.path.join(main_dir,system_dirs[4]) # FIXME::::::::::::::::::::::::::::::::::::::::
    #sys_name = os.path.join(main_dir,system_dirs[6]) # FIXME::::::::::::::::::::::::::::::::::::::::
    ## 5, 8
#    sys_name = os.path.join(main_dir,system_dirs[7]) # FIXME::::::::::::::::::::::::::::::::::::::::
    #sys_name = os.path.join(main_dir,system_dirs[8]) # FIXME::::::::::::::::::::::::::::::::::::::::
    ## 1, 2
    sys_name = os.path.join(main_dir,system_dirs[-1]) # FIXME::::::::::::::::::::::::::::::::::::::::
#    sys_name = os.path.join(main_dir,system_dirs[-2]) # FIXME::::::::::::::::::::::::::::::::::::::::

    print(sys_name)


    #data_dir = f"{sys_name}/{data_dir}"
    data_dir = sys_name
    print(data_dir)
    #exit()
    dir = "results"
    biceps.toolbox.mkdir(dir)
    outdir = f'{dir}/{sys_name}/{stat_model}_{data_uncertainty}_sigma'
    biceps.toolbox.mkdir(outdir)

    # NOTE: IMPORTANT: Create multiple trials to average over and check the deviation of results
    check_dirs = next(os.walk(f"{dir}/{sys_name}/{stat_model}_{data_uncertainty}_sigma/"))[1]
    trial = len(check_dirs)

    outdir = f'{dir}/{sys_name}/{ext}{stat_model}_fmo'
    print(f"data_dir: {data_dir}")
    print(f"outdir: {outdir}")
    biceps.toolbox.mkdir(outdir)



    file = f'{data_dir}/*clust*.csv'
    print(file)
    file = biceps.toolbox.get_files(file)[0]
    print(file)
    msm_pops = pd.read_csv(file, index_col=0, comment="#")
    if nstates == 100:
        msm_pops = msm_pops["populations"].to_numpy()
    elif nstates == 500:
        msm_pops = np.repeat(msm_pops["populations"].to_numpy(), 5)/5.
    msm_pops = pd.DataFrame(msm_pops, columns=["populations"])

    #################################################################################
    # NOTE: maxing uniform prior as a test
    #msm_pops["populations"] = np.ones(len(msm_pops.to_numpy()))/len(msm_pops.to_numpy())
    #################################################################################

    #print(msm_pops.columns.to_list())
    energies = -np.log(msm_pops.to_numpy())
    energies = np.concatenate(energies)
    #print(energies)
    energies = energies - energies.min()  # set ground state to zero, just in case
    print(energies)
    #exit()


    if combine_cs: sub_data_loc = "all_data"
    else: sub_data_loc = "CS_J_NOE"

    if J_only: input_data = [[file] for file in biceps.toolbox.get_files(f'{data_dir}/{ext}{sub_data_loc}/*.J')]

    dfs  = [pd.read_pickle(file) for file in input_data[0]]
    Nd  = sum([len(df) for df in dfs])
    #print(Nd)
    print(f"Input data: {biceps.toolbox.list_extensions(input_data)}")
    print(f"nSteps of sampling: {nsteps}\nnReplica: {nreplicas}")


    sigMin, sigMax, dsig = 0.001, 200, 1.02
    sigMin, sigMax, dsig = 0.001, 100, 1.02
    arr = np.exp(np.arange(np.log(sigMin), np.log(sigMax), np.log(dsig)))
    l = len(arr)
    sigma_index = round(l*0.73)



    skip_dirs = ['indices']
    # recursively loop through all the directories FF and cluster variations

    structure_file = biceps.toolbox.get_files(f"{sys_name}/*cluster0_*.pdb")[0]
    frame = md.load(structure_file)
    topology = frame.topology


    J = biceps.J_coupling.compute_J3_HN_HA(md.load(structure_file, top=frame), model="Bax2007")
    table = topology.to_dataframe()[0]
    resName = table["resName"].to_numpy()
    resNames = np.array([val+str(table["resSeq"].to_numpy()[r]) for r,val in enumerate(resName)])
    _, idx = np.unique(resNames, return_index=True)
    df = pd.DataFrame()
    df["Residue"] = resNames[np.sort(idx)]
    residues = df["Residue"].to_numpy()
    listing = []
    top_res = []
    for k in range(len(J[0])):
        idxs = [topology.atom(idx) for idx in J[0][k]]
        res = str(idxs[1]).split("-")[0]
        top_res.append(res)

    _input_data = input_data[::stride]
    snapshots = biceps.toolbox.get_files(f"{sys_name}/*cluster*_*.pdb")[::stride]
    grouped_files = [[state] for state in snapshots]
    frames = [frame for state in grouped_files for frame in state]
    total_phi_angles = np.array([compute_phi(md.load(frame))[-1][0] for frame in frames])
    data = [pd.read_pickle(file[0]) for file in _input_data]
    exp_J = data[0]["exp"].to_numpy()
    resnames = data[0]["res2"].to_numpy()

    indices = []
    for i,res in enumerate(top_res):
        if res not in resnames:
            indices.append(i)
    indices = np.array(indices)

    figsize = (10,6)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows=1, ncols=1)
    ax = fig.add_subplot(gs[0,0])

    print(len(exp_J))

    all_phi_angles = np.delete(total_phi_angles, indices, axis=1)
    #print("phi_angles_deg = ",phi_angles_deg)
    for i in range(len(all_phi_angles)):
        phi_angles_deg = np.rad2deg(all_phi_angles[i])
        ax.scatter(phi_angles_deg, exp_J, color="k", label="__no_legend__")

    ax.scatter(phi_angles_deg, exp_J, color="k", label="__no_legend__")
    ax.set_ylabel(r"$^{3}\!J(\phi)$ [Hz]", fontsize=16)
    ax.set_xlabel(r"$\phi$ [degrees]", fontsize=16)
    ax.set_xlim(-180, 180)
    ax.set_ylim(0, 12)

    tick_stride = 1
    ax.tick_params(axis='x', direction='inout')
    ax.tick_params(axis='y', direction='inout')
    ax.set_xticklabels(ax.get_xticklabels(), ha='center')
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.tight_layout()
    figname=f"{outdir}/coupling_curve_against_phi.png"
    fig.savefig(figname, dpi=400)

#    exit()


    _options = biceps.get_restraint_options(_input_data)
    for i in range(len(_options)):
        _options[i].update(dict(ref="uniform", sigma=(sigMin, sigMax, dsig), sigma_index=sigma_index,
             phi=phi, phi_index=phi_index, beta=beta, beta_index=beta_index,
             stat_model=stat_model, data_uncertainty=data_uncertainty,
             data_likelihood=data_likelihood))

    opts = pd.DataFrame(_options)
    restraint_indices = [i for i,val in enumerate(opts["extension"].to_numpy()) if f"J" in val]


    fwd_model_paras,phi0,k_labels = [],[],[]
    coeff_indices,J_type = [],[]
    idx = 0
    for idx,_ in enumerate(_options):
        if idx not in restraint_indices: continue

        for i in range(len(coefficents)):
            if coefficents[i] == _["extension"].replace("J_",""):
                coeff_indices.append(i)
                J_type.append(coefficents[i])
                break

        print(karplus_labels[i])
        #print(i)
        k_labels.append(karplus_labels[i])
        parameter_sets = getattr(biceps.J_coupling, "J3_%s_coefficients"%coefficents[i])
        parameter_uncert_sets = getattr(biceps.J_coupling, "J3_%s_uncertainties"%coefficents[i])
        m = list(parameter_sets.keys())[0]
        phi0.append(parameter_sets[m]["phi0"])
        fwd_model_paras.append([8.0, -1., 2.0])

###############################################################################
###############################################################################

    parameter_sets = getattr(biceps.J_coupling, "J3_%s_coefficients"%coefficents[0])
    #initial_parameters = [val for key,val in parameter_sets[ref_model].items() if key != "phi0"]

    fwd_model_paras1 = []
    fwd_model_paras2 = []
    fwd_model_paras3 = []
    for idx,_ in enumerate(_options):
        if idx not in restraint_indices: continue
        fwd_model_paras1.append([1.0, -1., 2.0])
        fwd_model_paras2.append([4.0, -2., 1.0])
        fwd_model_paras3.append([6.0, -1., 0.0])

    fwd_model_paras = [fwd_model_paras,fwd_model_paras1,fwd_model_paras2,fwd_model_paras3]

    # NOTE: this was used for preprint
    fwd_model_paras = [[[7, -1.0, 1.]],
                       [[7, -1.0, 1.]],
                       [[7, -1.0, 1.]],
                       [[7, -1.0, 1.]],
                       [[7, -1.0, 1.]]]


    fwd_model_paras = np.array(fwd_model_paras)


    print("\n\n\n")
    print(restraint_indices)
    print(np.rad2deg(phi0))
    print(fwd_model_paras)
    print("\n\n\n")
#    exit()

    p = np.ones(len(_input_data))/len(_input_data)
    energies = -np.log(p)
    print("nStates = ", len(energies))

    lambda_values = [0.0]*len(fwd_model_paras)
    print(lambda_values)

    if run_optimization:
        # Construct the initial ensemble
        _ensemble = biceps.ExpandedEnsemble(energies, lambda_values=lambda_values)
        _ensemble.initialize_restraints(_input_data, _options)

        Nd = 0
        for r in range(len(_ensemble.ensembles[0].ensemble[0])): # data restraint types
            Nd += len(_ensemble.ensembles[0].ensemble[0][r].restraints)
        print("Nd = ", Nd)



    #    _ensemble.fmo_restraint_indices = restraint_indices
    #    _ensemble.phi_angles = [all_phi_angles]
    #    _ensemble.phase_shifts = phi0
    #    _ensemble.fwd_model_parameters = fwd_model_paras

        #parameter_priors = np.array([["Gaussian" for i in range(len(fwd_model_paras[0][0]))]
        #parameter_priors = np.array([["GaussianSP" for i in range(len(fwd_model_paras[0][0]))]
        parameter_priors = np.array([["uniform" for i in range(len(fwd_model_paras[0][0]))]
                            for k in range(len(fwd_model_paras[0]))])
        print(parameter_priors)
        kwargs = {"phi0": phi0}
        _ensemble.initialize_fwd_model(
                #init_paras=fwd_model_paras, x=[all_phi_angles], indices=restraint_indices,
                #min_max_paras=None, parameter_priors=parameter_priors, **kwargs)
                init_paras=fwd_model_paras, x=[all_phi_angles], indices=restraint_indices,
                min_max_paras=None, parameter_priors=None, **kwargs)


    #    print(restraint_indices)
    #    print(all_phi_angles)
    #    print(phi0)
    #    print(fwd_model_paras)
    #    exit()




        local_vars = locals()
        _PSkwargs = biceps.toolbox.get_PSkwargs(local_vars, exclude=["ensemble","_ensemble"])
        _sample_kwargs = biceps.toolbox.get_sample_kwargs(local_vars)
        _sample_kwargs["progress"] = 1#progress_bar
        _sample_kwargs["verbose"] = 0
        _sample_kwargs["attempt_lambda_swap_every"] = swap_every
        _sample_kwargs["nsteps"] = nsteps
        _sample_kwargs["attempt_move_fm_prior_sigma_every"] = 1
        _PSkwargs["xi_integration"] = 0
        _PSkwargs["nreplicas"] = nreplicas
        _PSkwargs["fmo"] = 1
        _PSkwargs["fmo_method"] = "SGD"
        #_PSkwargs["fmo_method"] = "adam"


        sampler = biceps.PosteriorSampler(_ensemble, **_PSkwargs)
        sampler.sample(**_sample_kwargs)
        print(sampler.acceptance_info)
        populations = sampler.populations

        for i,pops in enumerate(populations):
            print("##############")
            print("Chain %s"%i)
            # Sort 'pops' in descending order and get the top 5
            s = np.sort(pops)[::-1][:5]
            print(f"Top 5 populations: {s}")
            indices = np.array([np.where(pops == val)[0][0] for val in s])
            indices = indices.astype(int)
            print(f"Indices of top 5 populated states: {indices}")
            for idx in indices:
                print(f"Index: {idx}, Population: {pops[idx]}, Corresponds to: {grouped_files[idx]}")
            print("##############")

        print(sampler.exchange_info)
        sampler.plot_exchange_info(xlim=(-100, nsteps), figname=f"{outdir}/lambda_swaps.png")

        biceps.toolbox.save_object(sampler, f"{outdir}/sampler_obj.pkl")

        print("Done.")
        a = biceps.Analysis(sampler, outdir=outdir, MBAR=False)
        a.plot_acceptance_trace()
        a.plot_energy_trace()
        print("Nd = ",sampler.Nd)
        print("\n\n\n\n")
        print("You started with:")
        print(fwd_model_paras)
        print("\n\n\n\n")


    if not run_optimization:
        sampler = biceps.toolbox.load_object(f"{outdir}/sampler_obj.pkl")


    chains = sampler.fmp_traj
    new_parameters = []
    new_parameters_sigma = []
    for ct in range(chains.shape[2]):
        #gr_input = chains[:,:,ct,:].reshape((chains.shape[0], chains.shape[1], chains.shape[3]))
        gr_input = chains[:,:,ct,:]
        print(gr_input.shape)
        chain_stats = fmo.chain_statistics(gr_input)
        R_hat = chain_stats["R_hat"]
        print("R_hat: ",R_hat)
        print(chain_stats["mean_of_each_chain"])
        mean = chain_stats["mean_over_all_chains"]
        new_parameters.append(mean)
        std_dev = chain_stats["std_dev_over_all_chains"]
        new_parameters_sigma.append(std_dev)
        print(["%0.2g"%mean[i] + "±%0.2g"%std_dev[i] for i in range(len(mean))])
        print("################################")
    #exit()

    data = sampler.fmp_traj#[0]


    traj = sampler.traj[0].__dict__["trajectory"]
    energies = [traj[i][1] for i in range(len(traj))]
    np.save(f"{outdir}/fmp_trajectory.npy", data)
    _chain = 1
    data = data[_chain]

    try:
        fmo.plot_fmp_posterior_histograms(sampler, k_labels=k_labels, chain=_chain, outdir=outdir)
        fmo.plot_fmp_traces(sampler, k_labels=k_labels, chain=_chain, outdir=outdir)
    except(Exception) as e:
        pass




    figsize = (10,6)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows=1, ncols=1)
    ax = fig.add_subplot(gs[0,0])

    ax.set_ylabel(r"$J(\phi)$ [Hz]", fontsize=16)
    ax.set_xlabel(r"$\phi$ [degrees]", fontsize=16)
    ax.set_xlim(-180, 180)
    ax.set_ylim(0, 12)

    angles = np.deg2rad(np.linspace(-180, 180, 100))
    y1 = fmo.get_scalar_couplings_with_derivatives(angles,
            A=new_parameters[0][0]+np.nan_to_num(new_parameters_sigma[0][0]),
            B=new_parameters[0][1]+np.nan_to_num(new_parameters_sigma[0][1]),
            C=new_parameters[0][2]+np.nan_to_num(new_parameters_sigma[0][2]), phi0=phi0[0])[0]
    y2 = fmo.get_scalar_couplings_with_derivatives(angles,
            A=new_parameters[0][0]-np.nan_to_num(new_parameters_sigma[0][0]),
            B=new_parameters[0][1]-np.nan_to_num(new_parameters_sigma[0][1]),
            C=new_parameters[0][2]-np.nan_to_num(new_parameters_sigma[0][2]), phi0=phi0[0])[0]
    ax.fill_between(np.rad2deg(angles), y1, y2, color="blue", alpha=0.5, label="BICePs")

    y = fmo.get_scalar_couplings_with_derivatives(angles, A=new_parameters[0][0], B=new_parameters[0][1], C=new_parameters[0][2], phi0=phi0[0])[0]
    ax.plot(np.rad2deg(angles), y, color="k", label="__no_legend__")

    ax.scatter(phi_angles_deg, exp_J, c="red", edgecolor="black", label="Folded state (61)")

    tick_stride = 1
    ax.tick_params(axis='x', direction='inout')
    ax.tick_params(axis='y', direction='inout')
    ax.set_xticklabels(ax.get_xticklabels(), ha='center')
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.tight_layout()
    figname=f"{outdir}/coupling_curve_against_phi.png"
    fig.savefig(figname, dpi=400)




    figsize=(14,8)
    columns = sampler.rest_type
    df = pd.DataFrame(np.array(sampler.traj[0].traces).transpose(), columns)
    df0 = df.transpose()
    nrows = np.sum([1 for col in df0.columns.to_list() if "sigma" in col])//3
    if nrows == 0: nrows = 1
    ncols=3

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)
    fig_list = []

    for r in range(len(data[0])):
        row_idx = r // ncols
        col_idx = r % ncols
        ax = fig.add_subplot(gs[row_idx,col_idx])

        results = pd.DataFrame(data[:,r,:], columns=["A", "B", "C"])
        results["score"] = np.array(energies)/nreplicas


        figname = f"{outdir}/contour.png"
        #if plot_landscapes: fig = plot_landscape(results[::landscape_stride], figname=figname, gridpoints=100, lvls=50, upper_xy_lim=None, kernel_idx=-1)

        fig_obj = plot_landscape_with_curve(results[::landscape_stride], figname=None, gridpoints=100, lvls=50, upper_xy_lim=None, kernel_idx=-1)

#        fig_obj = plot_landscape_with_curve(results[::landscape_stride], figname=None, gridpoints=100, lvls=50, upper_xy_lim=None, kernel_idx=0)
        axes = fig_obj.get_axes()
        print(axes)
        ax = axes[3]

        ############################################################################
        # Plot the initial karplus curves{{{
        figname=f"{outdir}/opt_on_karplus_curves.png"

        angles = np.deg2rad(np.linspace(-180, 180, 100))

        y1 = fmo.get_scalar_couplings_with_derivatives(angles,
                A=new_parameters[r][0]+np.nan_to_num(new_parameters_sigma[r][0]),
                B=new_parameters[r][1]+np.nan_to_num(new_parameters_sigma[r][1]),
                C=new_parameters[r][2]+np.nan_to_num(new_parameters_sigma[r][2]), phi0=phi0[r])[0]
        y2 = fmo.get_scalar_couplings_with_derivatives(angles,
                A=new_parameters[r][0]-np.nan_to_num(new_parameters_sigma[r][0]),
                B=new_parameters[r][1]-np.nan_to_num(new_parameters_sigma[r][1]),
                C=new_parameters[r][2]-np.nan_to_num(new_parameters_sigma[r][2]), phi0=phi0[r])[0]
        ax.fill_between(np.rad2deg(angles), y1, y2, color="blue", alpha=0.5, label="BICePs")

        yticks = list(range(14)[::2])
        ax.set_yticks(yticks)
        ax.set_ylim(yticks[0]-1, np.max(y1)+2.5)
        if r >= 3: ax.set_xlabel(r'$\phi$ (degrees)', fontsize=16)
        ax.set_ylabel(r'${^{3}\!J}_{%s}$ (Hz)'%k_labels[r], fontsize=16)
        ax.set_xlabel(r'$\phi$ (Degrees)', fontsize=16)

        #ax.scatter(w_phi_angles, exp, c="red", edgecolor="black")

        handles, labels = plt.gca().get_legend_handles_labels()

        fig_list.append(fig_obj)


#############################################################################
        # Save the figure object to a buffer instead of a file
        buf = io.BytesIO()
        fig_obj.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)

        # Read the image back as a PIL image
        img = Image.open(buf)

        # Only create the axes once
        img_ax = fig.add_subplot(gs[row_idx, col_idx])
        img_ax.axis('off')  # Turn off the axes

        # Display the image in the subplot, ensure the aspect ratio is maintained
        img_ax.imshow(img, aspect='auto')


        axes = fig_obj.get_axes()
        #order = [0, 2, 3, 1]
        #for i in range(4):
        #    ax = axes[i]
        #    ax.text(-0.16, 1.02, string.ascii_lowercase[order[i]],
        #        transform=ax.transAxes, size=18, weight='bold')

        figname=f"{outdir}/landscape.png"
#        fig_obj.subplots_adjust(left=0.15) #hspace=0.35, wspace=0.5)
        fig_obj.subplots_adjust(left=0.20, right=0.95) #hspace=0.35, wspace=0.5)
        fig_obj.savefig(figname, dpi=400)




        #img_ax.text(-0.14, 1.02, string.ascii_lowercase[r],
        #    transform=ax.transAxes, size=18, weight='bold')

        buf.close()


    #figname=f"{outdir}/all_landscapes.png"
    #fig.tight_layout()
    #fig.savefig(figname, dpi=400)
    #:}}}




##############################################################################

    exit()


    #:}}}











