
# Python libraries:{{{
import biceps
import FwdModelOpt_routines as fmo
import scipy, gc, os, copy, time, string, re
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
from sklearn import metrics
import scipy.stats, copy, matplotlib
#from matplotlib import gridspec
from scipy.optimize import fmin
import matplotlib.gridspec as gridspec
from matplotlib import ticker
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.transforms as transforms


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

# plot_correlation:{{{
def plot_correlation(sampler, k_labels=[], chain=0, outdir="./", figsize=(14,8), plot_type="step"):

    data = sampler.fmp_traj[chain]
    npz = sampler.traj[chain].traces[-1]
    columns = sampler.rest_type
    df = pd.DataFrame(np.array(sampler.traj[0].traces).transpose(), columns)
    df0 = df.transpose()
    nrows = np.sum([1 for col in df0.columns.to_list() if "sigma" in col])//3
    if nrows == 0: nrows = 1

    ncols=3

    if k_labels == []: k_labels = ["" for i in range(nrows*ncols)]

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)
    print(gs)

    for r in range(len(data[0])):
        row_idx = r // ncols
        col_idx = r % ncols

        results = pd.DataFrame(data[:,r,:], columns=["A", "B", "C"])
        #ax = fig.add_subplot(gs[row_idx, col_idx])
        ax = fig.add_subplot(gs[row_idx,col_idx])
        for i in range(data.shape[2]):
            labels = ["A", "B", "C"]
            colors = ["r", "b", "g"]

            if plot_type == "hist":
                ax.hist(data[:,r,i], alpha=0.5, label=labels[i], color=colors[i], edgecolor="k", bins="auto")

            elif plot_type == "step":
                counts, bin_edges = np.histogram(data[:,r,i], bins="auto")
                ax.step(bin_edges[:-1], counts, '%s-'%colors[i])
                ax.fill_between(bin_edges[:-1], counts, color=colors[i], label=labels[i], step="pre", alpha=0.4)
            else:
                print("`plot_type` needs to be either 'hist' or 'step'")
                exit()



            lower_lim, upper_lim = -5, 15
            limit = np.array([lower_lim, upper_lim])
            R2 = np.corrcoef(df["Actual"], df["posterior"])[0][1]**2
            MAE = metrics.mean_absolute_error(df["Actual"],  df["posterior"])
            RMSE = np.sqrt(metrics.mean_squared_error(df["Actual"],  df["posterior"]))
            fig.axes[0].plot(limit, limit, color="k")
            fig.axes[0].fill_between(limit, limit+1.0, limit-1.0, facecolor='wheat', alpha=0.5)
            posx,posy = np.min(df["posterior"])+0.1, np.max(df["Actual"]) - 3

            lower_lim, upper_lim = -4, 15
            limit = np.array([lower_lim, upper_lim])
            ax2.fill_between(limit, limit+1.0, limit-1.0, facecolor='brown', alpha=0.35)
            ax2.fill_between(limit, limit+2.0, limit-2.0, facecolor='wheat', alpha=0.50)
            ax2.plot(limit, limit, color="k")
            ax2.scatter(y_test, posterior, s=15)
            ax2.set_xlim(limit)
            ax2.set_ylim(limit)
            posx,posy = np.min(y_test)+.5, np.max(posterior) - 4
            string = r'''$N$ = %i
         $R^{2}$ = %0.3f
         MAE = %0.3f
         RMSE = %0.3f'''%(Nd, R2, MAE, RMSE)
            #scipy.stats.sem(posterior))
            ax2.annotate(string, xy=(posx,posy), xytext=(posx,posy))


        ax.set_title(r"${^{3}\!J}_{%s}$"%k_labels[r], fontsize=18)
        if r >= 3: ax.set_xlabel("Karplus coefficients [Hz]", fontsize=16)
        if (r == 0) or (r==3): ax.set_ylabel('Density', fontsize=16)
        # Increase the number of x ticks and improve their readability
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))  # Adjust number of x ticks
        if (r == 0): ax.legend()
        #ax.set_xlim(data.min(), data.max())
        ax.set_ylim(bottom=0)

        ax.text(-0.12, 1.02, string.ascii_lowercase[r],
            transform=ax.transAxes, size=18, weight='bold')


    fig.savefig(f"{outdir}/karplus_coefficients_histograms.png")



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
    if (parameter_key == "Bax1997") or (parameter_key == "Habeck"):
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
        if parameter_key == "BICePs(1d3z)":
            sampler = biceps.toolbox.load_object("Ubq_results/1d3z/GB_single_sigma/50000_steps_32_replicas_3_lam_opt/sampler_obj.pkl")
        if parameter_key == "BICePs(2nr2)":
            sampler = biceps.toolbox.load_object("Ubq_results/2nr2/GB_single_sigma/50000_steps_32_replicas_3_lam_opt/sampler_obj.pkl")
        if parameter_key == "BICePs(RF2)":
            sampler = biceps.toolbox.load_object("Ubq_results/rosettafold2/GB_single_sigma/50000_steps_32_replicas_3_lam_opt/sampler_obj.pkl")

        chains = sampler.fmp_traj
        for ct in range(chains.shape[2]):
            gr_input = chains[:,:,ct,:]
            chain_stats = fmo.chain_statistics(gr_input)
            mean = chain_stats["mean_over_all_chains"]
            parameters.append(mean)
            std = chain_stats["avg_uncert"]
            parameters_std.append(std)
    return parameters,parameters_std
# }}}



# Define a custom aggregation function for numpy arrays
def mean_of_array(series):
    means = []
    arr = series.to_numpy()
    means = arr.mean(axis=0)
    return means #pd.Series(means).to_numpy()[0]

def std_of_array(series):
    stds = []
    for arr in series:
        if isinstance(arr, np.ndarray) and arr.ndim > 1:
            stds.append(np.std(arr, axis=1))
        elif isinstance(arr, np.ndarray):
            stds.append(arr)  # For 1D arrays, just use the array itself
        else:
            stds.append(np.nan)  # Handle non-array values
    return pd.Series(stds)

def nanmean(series):
    return np.nanmean(series)
def nanstd(series):
    return np.nanstd(series)

def count_non_nan(arrays):
    count = 0
    for array in arrays:
        if not np.isnan(array).all():  # Check if the array does not contain only NaN
            count += 1
    return count

from uncertainties import unumpy as unp
from sklearn.metrics import mean_absolute_error
import uncertainties as u

def mae_with_uncertainty(experimental_J, reweighted_model_J):
    """
    Compute the MAE between experimental J-couplings (array of floats)
    and reweighted model J-couplings (array of ufloat).

    Parameters:
        experimental_J : array-like of floats
        reweighted_model_J : array-like of ufloats

    Returns:
        MAE : ufloat (mean absolute error with propagated uncertainty)
    """

    # Nominal values and standard deviations from ufloat array
    predicted_nominal = unp.nominal_values(reweighted_model_J)
    predicted_stddev  = unp.std_devs(reweighted_model_J)

    # Residuals: model - experiment
    residuals = predicted_nominal - np.array(experimental_J)

    # MAE nominal
    mae_nominal = np.mean(np.abs(residuals))

    # Derivatives of MAE w.r.t. each predicted point (for delta method)
    derivs = np.sign(residuals) / len(residuals)  # ∂MAE/∂J_i

    # Propagate uncertainty using: Var(MAE) = ∑ (∂MAE/∂J_i)^2 * σ_i^2
    mae_variance = np.sum((derivs * predicted_stddev) ** 2)
    mae_stddev = np.sqrt(mae_variance)

    from uncertainties import ufloat
    return ufloat(mae_nominal, mae_stddev)




if __name__ == "__main__":


    # Main:{{{
    coefficents = ["HN_HA"]
    karplus_labels = [r"H^{N}H^{\alpha}"]

    keys = ["NRV2024", "Raddi2024", "Kessler1998", "Bax2007", ]

    stride = 1
    save_obj = 1
    testing = 1
    convert_distances_to_intensities = 0
    nstates = 100
    nstates = 500
    if nstates == 500:
        ext = "500_states/"
    else:
        ext = ""

#    n_lambdas, nreplicas, nsteps, swap_every, change_Nr_every = 2, 128, 100000, 0, 0
    n_lambdas, nreplicas, nsteps, swap_every, change_Nr_every = 2, 128, 1000, 0, 0

    burn = 0

    stat_model, data_uncertainty = "GB", "single"

    all_data = 0
    J_only   = 1
    noe_only = 0
    cs_only  = 0
    rdc_only  = 0
    J_and_noe_only = 0

    combine_cs = 1
    get_combined_input_data = 0



    scale_energies = 0
    find_optimal_nreplicas = 0
    write_every = 100
    lambda_values = [0.0]
#    lambda_values = [1.0]

    #lambda_values = [0.0, 0.0]
    data_likelihood = "gaussian" #"log normal" # "gaussian"

    continuous_space=0

    attempt_move_state_every = 1
    attempt_move_sigma_every = 1

    verbose = 0#False
    if verbose:
        pd.options.display.max_columns = None
        pd.options.display.max_rows = None

    walk_in_all_dim=continuous_space
    multiprocess=1
    plottype= "step"
    ####### Data and Output Directories #######

    skip_dirs = ["biceps", "experimental_data", 'old_systems']
    main_dir = "Systems"
    system_dirs = next(os.walk(main_dir))[1]
    system_dirs = [dir for dir in system_dirs if dir not in skip_dirs]
    system_dirs = biceps.toolbox.natsorted(system_dirs)
    print(system_dirs)
    #exit()
    # 1a, 1b
    sys_name = os.path.join(main_dir,system_dirs[0]) # FIXME::::::::::::::::::::::::::::::::::::::::
#    sys_name = os.path.join(main_dir,system_dirs[1]) # FIXME::::::::::::::::::::::::::::::::::::::::
#    ### 2a, 2b
#    sys_name = os.path.join(main_dir,system_dirs[-3]) # FIXME::::::::::::::::::::::::::::::::::::::::
#    sys_name = os.path.join(main_dir,system_dirs[2]) # FIXME::::::::::::::::::::::::::::::::::::::::
#    #### 3a, 3b
#    sys_name = os.path.join(main_dir,system_dirs[5]) # FIXME::::::::::::::::::::::::::::::::::::::::
#    sys_name = os.path.join(main_dir,system_dirs[3]) # FIXME::::::::::::::::::::::::::::::::::::::::
#    #### 4a, 4b
#    sys_name = os.path.join(main_dir,system_dirs[4]) # FIXME::::::::::::::::::::::::::::::::::::::::
#    sys_name = os.path.join(main_dir,system_dirs[6]) # FIXME::::::::::::::::::::::::::::::::::::::::
#    #### 5, 8
#    sys_name = os.path.join(main_dir,system_dirs[7]) # FIXME::::::::::::::::::::::::::::::::::::::::
    #sys_name = os.path.join(main_dir,system_dirs[8]) # FIXME::::::::::::::::::::::::::::::::::::::::
#    #### 1, 2
#    sys_name = os.path.join(main_dir,system_dirs[-1]) # FIXME::::::::::::::::::::::::::::::::::::::::
#    sys_name = os.path.join(main_dir,system_dirs[-2]) # FIXME::::::::::::::::::::::::::::::::::::::::




    peptide_name = f"Peptide {sys_name.split('_')[-1]}"
    #data_dir = f"{sys_name}/{data_dir}"
    data_dir = sys_name
    print(data_dir)
    #exit()
    dir = "results"
    biceps.toolbox.mkdir(dir)

    # NOTE: IMPORTANT: Create multiple trials to average over and check the deviation of results
    check_dirs = next(os.walk(f"{dir}/{sys_name}/{ext}{stat_model}_{data_uncertainty}_sigma/"))[1]
    #trial = len(check_dirs)
    trial = 0

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
    prior_pops = np.concatenate(msm_pops.to_numpy())

    #print(f"Possible input data extensions: {biceps.toolbox.list_possible_extensions()}")

    if combine_cs: sub_data_loc = "all_data"
    else: sub_data_loc = "CS_J_NOE"

    if all_data: input_data = biceps.toolbox.sort_data(f'{data_dir}/{ext}{sub_data_loc}')
    if J_only: input_data = [[file] for file in biceps.toolbox.get_files(f'{data_dir}/{ext}{sub_data_loc}/*.J')]
    if noe_only: input_data = [[file] for file in biceps.toolbox.get_files(f'{data_dir}/{ext}{sub_data_loc}/*.noe')]
    if cs_only: input_data = biceps.toolbox.sort_data(f'{data_dir}/{ext}{sub_data_loc}/*.cs*')

    if all_data:
        _input_data = []
        for s in input_data:
            files = [r for r in s if (".rdc" not in r)]
            _input_data.append(files)
        input_data = _input_data

    if J_and_noe_only:
        _input_data = []
        input_data = biceps.toolbox.sort_data(f'{data_dir}/{ext}{sub_data_loc}')
        for s in input_data:
            files = [r for r in s if ".cs" not in r]
            _input_data.append(files)
        input_data = _input_data


    #print(input_data)
    #print(pd.read_pickle(input_data[135][-1]))
    #exit()

    dfs  = [pd.read_pickle(file) for file in input_data[0]]
    #print(dfs[0])
    restraint_indices = [df["restraint_index"].to_numpy() for df in dfs]
    Nd  = sum([len(np.unique(arr)) for arr in restraint_indices])
    print(f"Input data: {biceps.toolbox.list_extensions(input_data)}")
    print(f"nSteps of sampling: {nsteps}\nnReplica: {nreplicas}")

    print(f"Number of unique restraints: {Nd}")



    sigMin, sigMax, dsig = 0.001, 200, 1.02
    sigMin, sigMax, dsig = 0.001, 100, 1.02
    arr = np.exp(np.arange(np.log(sigMin), np.log(sigMax), np.log(dsig)))
    l = len(arr)
    sigma_index = round(l*0.73)
    #sigma_index = round(l*0.63)
    #sigma=(sigMin, sigMax, dsig)
    #sigma_index=sigma_index
    #print(arr[sigma_index])


    if stat_model == "Students":
        phi,phi_index,stat_model,data_uncertainty=(1.0, 2.0, 1),0,"Students","single"
        beta,beta_index,stat_model,data_uncertainty=(1.0, 100.0, 10000),0,"Students","single"

    elif stat_model == "GB":
        phi,phi_index,stat_model,data_uncertainty=(1.0, 100.0, 1000),0,"GB","single"
        beta,beta_index,stat_model,data_uncertainty=(1.0, 2.0, 1),0,"GB","single"

    else:
        phi,phi_index,data_uncertainty=(1.0, 2.0, 1),0,data_uncertainty
        beta,beta_index,data_uncertainty=(1.0, 2.0, 1),0,data_uncertainty


    input_data = input_data[::stride]
    snapshots = biceps.toolbox.get_files(f"{data_dir}/*cluster*_*.pdb")[::stride]
    grouped_files = [[state] for state in snapshots]
    frames = [frame for state in grouped_files for frame in state]
    total_phi_angles = np.array([compute_phi(md.load(frame))[-1][0] for frame in frames])
    data = [pd.read_pickle(file[0]) for file in input_data]
    experimental_J = data[0]["exp"].to_numpy()



    # Find out if the sequences match the experimental files
    #############################################################################
    structure_file = biceps.toolbox.get_files(f"{sys_name}/*cluster0_*.pdb")[0]
    frame = md.load(structure_file)
    topology = frame.topology
    table = topology.to_dataframe()[0]
    resNames = list(table["resName"].to_numpy())
    _ = ""
    _resNames = []
    k = 0
    for i,res in enumerate(resNames):
        if res != _:
            try:
                _resNames.append(biceps.toolbox.three2one(res+f"{k+1}"))
            except(Exception) as e:
                _resNames.append(res+f"{k+1}")
            _ = res
            k += 1
        else:
            continue
    resNames = _resNames
    exp_cs_csv_file = f"Systems/experimental_data/cs/{sys_name.split('_')[-1]}.csv"
    df = pd.read_csv(exp_cs_csv_file, comment="#")
    exp_order = [re.split(r'\d+',res)[0] for res in df["Residue"].to_numpy()]
    struct_order = [re.split(r'\d+',res)[0] for res in resNames]
    # NOTE: check to make sure exp_order and struct_order have the same length
    if len(exp_order) < len(struct_order):
        add_fake_capping_groups = 1
    else:
        add_fake_capping_groups = 0

    if add_fake_capping_groups:
        # Add a row of NaNs at the top
        df.loc[-1] = "NaN"
        df.index = df.index + 1
        df.sort_index(inplace=True)
        # Add a row of NaNs at the bottom
        df.loc[len(df)] = "NaN"

        exp_order = [re.split(r'\d+',str(res))[0] for res in df["Residue"].to_numpy()]
        struct_order = [re.split(r'\d+',str(res))[0] for res in resNames]

    exp_order = np.array(exp_order, dtype=str)
    struct_order = np.array(struct_order, dtype=str)
    # Compare arrays using list comprehensions
    a = sum([1 for x, y in zip(exp_order.tolist(), struct_order.tolist()) if x == y])
    b = sum([1 for x, y in zip(exp_order.tolist(), struct_order[::-1].tolist()) if x == y])
    if a >= b:
        reverse = False
    else:
        reverse = True
    print("Reverse order of sequence: ",reverse)

    exp_J_csv_file = f"Systems/experimental_data/J/{sys_name.split('_')[-1]}.csv"
    df = pd.read_csv(exp_J_csv_file, comment="#")
    # IMPORTANT: NOTE: IMPORTANT: order is reversed !!!
    if add_fake_capping_groups:
        # Add a row of NaNs at the top
        df.loc[-1] = np.nan
        df.index = df.index + 1
        df.sort_index(inplace=True)
        # Add a row of NaNs at the bottom
        df.loc[len(df)] = np.nan
    if reverse: df = df.iloc[::-1]
    df["Residue position"] = [i+1 for i in range(len(df))]
    df = df.reset_index(drop=True)
    exp_J = df['J (Hz)'].to_numpy()
    skip_idx_J = np.where(np.isnan(exp_J) == True)[0]
    frame = md.load(frames[0])
    J = biceps.J_coupling.compute_J3_HN_HA(frame, model="Bax2007")
    topology = frame.topology
    table = topology.to_dataframe()[0]
    resName = table["resName"].to_numpy()
    resNames = np.array([val+str(table["resSeq"].to_numpy()[r]) for r,val in enumerate(resName)])
    _, idx = np.unique(resNames, return_index=True)
    df["Residue"] = resNames[np.sort(idx)]
    residues = df["Residue"].to_numpy()
    listing = []
    top_res = []
    for k in range(len(J[0])):
        idxs = [topology.atom(idx) for idx in J[0][k]]
        res = str(idxs[1]).split("-")[0]
        top_res.append(res)
        if res in residues[skip_idx_J]:
            listing.append(int(np.where(res == np.array(residues[skip_idx_J]))[0]))
            continue
    skip_idx_J = skip_idx_J[listing] - 1
    total_phi_angles = np.delete(total_phi_angles, skip_idx_J, axis=1)





    options = biceps.get_restraint_options(input_data)
    opt = pd.DataFrame(options)
    J_rest_index = [i for i,ext in enumerate(opt["extension"]) if "J" in ext][0]
    #print(options)
    #exit()

    for i in range(len(options)):
        options[i].update(dict(ref="uniform", sigma=(sigMin, sigMax, dsig), sigma_index=sigma_index,
             phi=phi, phi_index=phi_index, beta=beta, beta_index=beta_index,
             stat_model=stat_model, data_uncertainty=data_uncertainty,
             data_likelihood=data_likelihood))


    print(pd.DataFrame(options))




    parameters = []
    parameters_sigma = []
    phi0 = []
    parameter_sets = getattr(biceps.J_coupling, "J3_%s_coefficients"%coefficents[0])
    parameter_uncert_sets = getattr(biceps.J_coupling, "J3_%s_uncertainties"%coefficents[0])
    for key in keys:
        m = list(parameter_sets.keys())[0]
        phi0.append(parameter_sets[m]["phi0"])
        if key in parameter_sets.keys():
            parameters.append([parameter_sets[key][q] for q in ["A", "B", "C"]])


    ensemble = biceps.ExpandedEnsemble(energies=energies, lambda_values=lambda_values)
    ensemble.initialize_restraints(input_data, options)



    nrows = 1
    ncols=1
    k_labels = karplus_labels[0]

    figsize=(5,4)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows, ncols + ncols, width_ratios=[1, 0.0025]*ncols)

    # Define a function to add a boxed annotation to the subplot
    def add_stats_box(ax, stats, label, unicode_marker, color='black', bgcolor='white', alpha=0.5):
        # Map your marker shapes to Unicode characters as closely as possible
        marker_to_unicode = {
            "s": "■",  # Square
            "o": "●",  # Circle
            "v": "▼",  # Triangle down
            "*": "★",  # Star
            "^": "▲",  # Triangle up
            "<": "◀",  # Triangle left
            ">": "▶",  # Triangle right
            "+": "✚",  # Plus
            # Add any additional marker shapes you have.
        }
        unicode_marker_str = marker_to_unicode.get(unicode_marker, "")

        # Create the text string with the statistics and prepend with the Unicode marker
        textstr = '\n'.join([f"{k} = {v}" for k, v in stats.items()])
        textstr = f"{unicode_marker_str} {label}\n{textstr}"
        if color == "white": color = "black"

        # Add text box with the statistics
        props = dict(boxstyle='round', ec=bgcolor, facecolor=bgcolor, alpha=alpha)
#        posy = 1.0 - i * 0.25  # Adjust 0.05 as needed for spacing between annotations
        posy = 1.0 - i * 0.28  # Adjust 0.05 as needed for spacing between annotations
        ax.text(0, posy, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='right',
                color=color, bbox=props)


    all_parameters = {}
    for i,key in enumerate(keys):
        all_parameters[key] = parameters[i]


    from matplotlib.ticker import MaxNLocator
    ax = fig.add_subplot(gs[0,0])
    ax_stats = fig.add_subplot(gs[0,0+1])
    ax_stats.axis("off")
    #colors = ["white", "b", "g", "r"]
    colors = ["g", "b", "white", "r"]
    colors = ["g", "r", "blue", "darkgoldenrod"]
    colors = ["g", "r", "black", "blue", "sienna"]
    shapes = ["o", "*", "+", "^", "s"]
    sizes = [65, 225, 65, 185, 165]
    order = [99, 98, 97, 96, 95]

    lower_lim, upper_lim = -1, 12
    limit = np.array([lower_lim, upper_lim])
    #ax.fill_between(limit, limit+2.0, limit-2.0, facecolor='wheat', alpha=0.50, edgecolor="k", ls='-.')
    #ax.fill_between(limit, limit+1.0, limit-1.0, facecolor='wheat', alpha=0.35, edgecolor="k", ls='--')
    ax.fill_between(limit, limit+2.0, limit-2.0, facecolor='gray', alpha=0.1, edgecolor="k", ls='-.')
    ax.fill_between(limit, limit+1.0, limit-1.0, facecolor='gray', alpha=0.2, edgecolor="k", ls='--')

    ax.plot(limit, limit, color="k")
    ax.set_xlim(limit)
    ax.set_ylim(limit)


    r = J_rest_index

    reweighted_model_J = []
    for model_idx in range(len(parameters)):

        i = model_idx
        A, B, C = parameters[model_idx]

        result = fmo.get_scalar_couplings_with_derivatives(total_phi_angles, A=A, B=B, C=C, phi0=phi0[model_idx])
        model_J, diff_model_J, diff2_model_J = result
        print(model_J.shape)

        for l in range(len(ensemble.expanded_values)): # thermodynamic ensembles
            for s in range(len(ensemble.ensembles[l].ensemble)): # conformational states
                for j in range(len(ensemble.ensembles[l].ensemble[s][r].restraints)): # data points (observables)
                    ensemble.ensembles[l].ensemble[s][r].restraints[j]["model"] = model_J[s][j]



        sampler = biceps.PosteriorSampler(ensemble, nreplicas,
                write_every=write_every, change_Nr_every=change_Nr_every)

        sampler.sample(nsteps, attempt_lambda_swap_every=swap_every, swap_sigmas=1,
                find_optimal_nreplicas=find_optimal_nreplicas,
                attempt_move_state_every=attempt_move_state_every,
                attempt_move_sigma_every=attempt_move_sigma_every,
                burn=burn, print_freq=100, walk_in_all_dim=walk_in_all_dim,
                verbose=0, progress=1, multiprocess=multiprocess, capture_stdout=0)


###############################################################################

            # FIXME: IMPORTANT: I think I've been using the prior MSM populatons

#        reweighted_populations = sampler.populations[0]

        reweighted_populations = prior_pops

###############################################################################

        if keys[model_idx] == "NRV2024":

            # Step 1: Karplus parameters and covariance matrix
            A = 7.3
            B = -2.3
            C = 1.6

            # Standard deviations
            sigma_A = 0.39
            sigma_B = 0.18
            sigma_C = 0.25

            cov_matrix = np.diag([sigma_A**2, sigma_B**2, sigma_C**2])

            cov_matrix = np.array([
                [sigma_A**2,  -0.00453618, -0.02054205],
                [-0.00453618,  sigma_B**2,  0.00261605],
                [-0.02054205,  0.00261605,  sigma_C**2]
                ])


            eigvals = np.linalg.eigvals(cov_matrix)
            print("Covariance matrix eigenvalues:", eigvals)
            if np.all(eigvals >= 0):
                print("✅ Covariance matrix is positive semidefinite.")
            else:
                print("❌ Covariance matrix is NOT positive semidefinite.")

            # Dihedral angle (in radians)
            _phi = total_phi_angles  # [ n_states, n_phi_angles]

            # Gradient of J with respect to [A, B, C]
            dJ_dtheta = np.array([
                np.cos(_phi + phi0[model_idx])**2,
                np.cos(_phi + phi0[model_idx]),
                1.0
            ])

            # Predict mean J-coupling
            J_mean = A * dJ_dtheta[0] + B * dJ_dtheta[1] + C

            # Propagate uncertainty
            J_var = dJ_dtheta.T @ cov_matrix @ dJ_dtheta
            J_std = np.sqrt(J_var)

            print(J_mean.shape)
            print(J_std.shape)

            u_model_J = np.array([[u.ufloat(J_mean[i, j], J_std[i, j])
                                   for j in range(J_mean.shape[1])]
                                  for i in range(J_mean.shape[0])])

            print(u_model_J)
            u_reweighted_model_J = np.sum([u_model_J[i]*w for i,w in enumerate(reweighted_populations)], axis=0)

            reweighted_model_J = np.sum([J_mean[i]*w for i,w in enumerate(reweighted_populations)], axis=0)


            R2 = np.corrcoef(experimental_J, reweighted_model_J)[0][1]**2
            MAE = mae_with_uncertainty(experimental_J, u_reweighted_model_J)
            print(MAE)
            MAE, sigma_MAE = MAE.nominal_value, MAE.std_dev
            print(MAE)
            print(sigma_MAE)


###########################################################################
        else:
            reweighted_model_J = np.sum([model_J[i]*w for i,w in enumerate(reweighted_populations)], axis=0)
            print(reweighted_model_J)
            print(experimental_J)

            R2 = np.corrcoef(experimental_J, reweighted_model_J)[0][1]**2
            MAE = metrics.mean_absolute_error(experimental_J,  reweighted_model_J)
            print(MAE)
            sigma_MAE = np.nan


#        MAE = mae_with_uncertainty(experimental_J, u_reweighted_model_J)
#        print(MAE)
#        MAE, sigma_MAE = MAE.nominal_value, MAE.std_dev
#        print(MAE)
#        print(sigma_MAE)
##        exit()




#        reweighted_model_J = np.sum([model_J[i]*w for i,w in enumerate(reweighted_populations)], axis=0)
#        print(reweighted_model_J)
#        print(experimental_J)

#        R2 = np.corrcoef(experimental_J, reweighted_model_J)[0][1]**2
#        MAE = metrics.mean_absolute_error(experimental_J,  reweighted_model_J)
        RMSE = np.sqrt(metrics.mean_squared_error(experimental_J,  reweighted_model_J))
        #ax.scatter(reweighted_model_J, experimental_J, s=sizes[i], marker=shapes[i], ec="k", fc=colors[i], alpha=1.0, zorder=order[i]) #fc="white")
        ax.scatter(experimental_J,reweighted_model_J, s=sizes[i], marker=shapes[i], ec="k", fc=colors[i], alpha=1.0, zorder=order[i]) #fc="white")

        #stats = {r"$R^2$": f"{R2:.2f}", "MAE": f"{MAE:.2f}", "RMSE": f"{RMSE:.2f}"}


#        stats = {r"$R^2$": f"{R2:.2f}", "MAE": f"{MAE:.2f}"}

        if keys[model_idx] == "NRV2024":
            stats = {r"$R^2$": f"{R2:.2f}", "MAE": f"{MAE:.2f}", r"$\sigma_{MAE}$": f"{sigma_MAE:.2f}"}
        else:
            stats = {r"$R^2$": f"{R2:.2f}", "MAE": f"{MAE:.2f}", r"$\sigma_{MAE}$": f"N/A"}

        print(keys[model_idx])
        print(stats)

        if keys[model_idx] == "Kessler1998":
            keys[model_idx] = "Kessler1988"
        add_stats_box(ax_stats, stats, label=keys[model_idx], unicode_marker=shapes[i], color=colors[i], bgcolor='white', alpha=0.0)


        ###################################################################
        _min = np.nanmin([reweighted_model_J, experimental_J])
        _max = np.nanmax([reweighted_model_J, experimental_J])

        #_min, _max = np.round(_min), np.round(_max)  # assuming _min and _max are already defined
        range_percent = (_max - _min) * 0.10
        lower_lim = _min - range_percent - 0.2
        upper_lim = _max + range_percent + 0.5
        ax.set_xlim(lower_lim, upper_lim)
        ax.set_ylim(lower_lim, upper_lim)
        ax.grid(color='k', linestyle='--', linewidth=0.5, alpha=0.25)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))

        ################################################################
    posx,posy = np.min(lower_lim), np.max(upper_lim)
    textbox = r'''$N_{d}$ = %i        %s'''%(Nd, peptide_name)
    ax.annotate(textbox, xy=(posx,posy), xytext=(posx,posy + range_percent*0.5), color="k", fontsize=12)

    ax.set_xlabel(r"${^{3}\!J}_{%s}^{Exp}$"%k_labels, fontsize=18)
    ax.set_ylabel(r"${^{3}\!J}_{%s}^{Pred}$"%k_labels, fontsize=18)
    fig.tight_layout(w_pad=0.25)
    outdir = "./"
    fig.savefig(f"{outdir}/{peptide_name.replace(' ', '')}_correlations_FMO_validation.png", dpi=400)

    exit()









#:}}}












