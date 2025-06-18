# Python libraries:{{{
import numpy as np
import sys, time, os, gc, string, re
np.set_printoptions(threshold=sys.maxsize)
import pandas as pd
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
import scipy
from sklearn import metrics
import matplotlib.pyplot as plt
#from scipy import stats
import biceps
#from biceps.decorators import multiprocess
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import uncertainties as u ################ Error Prop. Library
import uncertainties.unumpy as unumpy #### Error Prop.
from uncertainties import umath
from biceps.toolbox import three2one
import mdtraj as md
#:}}}

# Special Pairs:{{{
def special_pairs(filename, system, verbose=0):
    traj = md.load(filename)
    topology = traj.topology
    table, bonds = topology.to_dataframe()
    #print(table)
    #['RUN2_3b', 'RUN3_2b', 'RUN4_3a', 'RUN5_4a', 'RUN8_8']
    if system == "RUN0_1a":
        Hal1 = [topology.select('residue 3 and resname SER and name HG')[0],
                topology.select('residue 8 and resname SER and name OG')[0]]
        H1 = [topology.select('residue 9 and resname VAL and name O')[0],
              topology.select('residue 2 and resname GLN and name H')[0]]
        H2 = [topology.select('residue 9 and resname VAL and name H')[0],
              topology.select('residue 2 and resname GLN and name O')[0]]
        H3 = [topology.select('residue 7 and resname ALA and name O')[0],
              topology.select('residue 4 and resname VAL and name H')[0]]
        H4 = [topology.select('residue 7 and resname ALA and name H')[0],
              topology.select('residue 4 and resname VAL and name O')[0]]

    elif system == "RUN1_1b":
        Hal1 = [topology.select('residue 3 and resname SER and name HG')[0],
                topology.select('residue 8 and resname ABU and name CG')[0]]
        H1 = [topology.select('residue 9 and resname VAL and name O')[0],
              topology.select('residue 2 and resname GLN and name H')[0]]
        H2 = [topology.select('residue 9 and resname VAL and name H')[0],
              topology.select('residue 2 and resname GLN and name O')[0]]
        H3 = [topology.select('residue 7 and resname ALA and name O')[0],
              topology.select('residue 4 and resname VAL and name H')[0]]
        H4 = [topology.select('residue 7 and resname ALA and name H')[0],
              topology.select('residue 4 and resname VAL and name O')[0]]

    elif system == "RUN2_3b":
        Hal1 = [topology.select('residue 4 and resname HSM and name O2')[0],
                topology.select('residue 9 resname ABU and name CG')[0]]
        H1 = [topology.select('residue 10 and resname VAL and name O')[0],
              topology.select('residue 3 and resname GLN and name H')[0]]
        H2 = [topology.select('residue 10 and resname VAL and name H')[0],
              topology.select('residue 3 and resname GLN and name O')[0]]
        H3 = [topology.select('residue 8 and resname ALA and name O')[0],
              topology.select('residue 5 and resname VAL and name H')[0]]
        H4 = [topology.select('residue 8 and resname ALA and name H')[0],
              topology.select('residue 5 and resname VAL and name O')[0]]

    elif system == "RUN7_2a":
        Hal1 = [topology.select('residue 3 and  resname SME and name O2')[0],
                topology.select('residue 8 and resname SER and name HG')[0]]
        H1 = [topology.select('residue 9 and resname VAL and name O')[0],
              topology.select('residue 2 and resname GLN and name H')[0]]
        H2 = [topology.select('residue 9 and resname VAL and name H')[0],
              topology.select('residue 2 and resname GLN and name O')[0]]
        H3 = [topology.select('residue 7 and resname ALA and name O')[0],
              topology.select('residue 4 and resname VAL and name H')[0]]
        H4 = [topology.select('residue 7 and resname ALA and name H')[0],
              topology.select('residue 4 and resname VAL and name O')[0]]


    elif system == "RUN2_2b":
        Hal1 = [topology.select('residue 3 and  resname SME and name O2')[0],
                topology.select('residue 8 and resname ABU and name CG')[0]]
        H1 = [topology.select('residue 9 and resname VAL and name O')[0],
              topology.select('residue 2 and resname GLN and name H')[0]]
        H2 = [topology.select('residue 9 and resname VAL and name H')[0],
              topology.select('residue 2 and resname GLN and name O')[0]]
        H3 = [topology.select('residue 7 and resname ALA and name O')[0],
              topology.select('residue 4 and resname VAL and name H')[0]]
        H4 = [topology.select('residue 7 and resname ALA and name H')[0],
              topology.select('residue 4 and resname VAL and name O')[0]]

    elif system == "RUN4_3a":
        Hal1 = [topology.select('residue 4 and resname HSM and name O2')[0],
                topology.select('residue 9 and resname SER and name HG')[0]]
        H1 = [topology.select('residue 10 and resname VAL and name O')[0],
              topology.select('residue 3 and resname GLN and name H')[0]]
        H2 = [topology.select('residue 10 and resname VAL and name H')[0],
              topology.select('residue 3 and resname GLN and name O')[0]]
        H3 = [topology.select('residue 8 and resname ALA and name O')[0],
              topology.select('residue 5 and resname VAL and name H')[0]]
        H4 = [topology.select('residue 8 and resname ALA and name H')[0],
              topology.select('residue 5 and resname VAL and name O')[0]]

    elif system == "RUN3_4a":
        Hal1 = [topology.select('residue 3 and resname HSM and name O2')[0],
                topology.select('residue 8 and resname SER and name HG')[0]]
        H1 = [topology.select('residue 9 and resname GLN and name O')[0],
              topology.select('residue 2 and resname ALA and name H')[0]]
        H2 = [topology.select('residue 9 and resname GLN and name H')[0],
              topology.select('residue 2 and resname ALA and name O')[0]]
        H3 = [topology.select('residue 7 and resname ALA and name O')[0],
              topology.select('residue 4 and resname VAL and name H')[0]]
        H4 = [topology.select('residue 7 and resname ALA and name H')[0],
              topology.select('residue 4 and resname VAL and name O')[0]]

    elif system == "RUN4_4b":
        Hal1 = [topology.select('residue 3 and resname HSM and name O2')[0],
                topology.select('residue 8 and resname ABU and name CG')[0]]
        H1 = [topology.select('residue 9 and resname GLN and name O')[0],
              topology.select('residue 2 and resname ALA and name H')[0]]
        H2 = [topology.select('residue 9 and resname GLN and name H')[0],
              topology.select('residue 2 and resname ALA and name O')[0]]
        H3 = [topology.select('residue 7 and resname ALA and name O')[0],
              topology.select('residue 4 and resname VAL and name H')[0]]
        H4 = [topology.select('residue 7 and resname ALA and name H')[0],
              topology.select('residue 4 and resname VAL and name O')[0]]

    elif system == "RUN5_5":
        Hal1 = [topology.select('residue 3 and resname SME and name O2')[0],
                topology.select('residue 8 and resname ALC and name CL1')[0]]
        H1 = [topology.select('residue 9 and resname VAL and name O')[0],
              topology.select('residue 2 and resname GLN and name H')[0]]
        H2 = [topology.select('residue 9 and resname VAL and name H')[0],
              topology.select('residue 2 and resname GLN and name O')[0]]
        H3 = [topology.select('residue 7 and resname ALA and name O')[0],
              topology.select('residue 4 and resname VAL and name H')[0]]
        H4 = [topology.select('residue 7 and resname ALA and name H')[0],
              topology.select('residue 4 and resname VAL and name O')[0]]

    elif system == "RUN6_8":
        Hal1 = [topology.select('residue 3 and resname SME and name O2')[0],
                topology.select('residue 8 and resname ALB and name BR1')[0]]
        H1 = [topology.select('residue 9 and resname VAL and name O')[0],
              topology.select('residue 2 and resname GLN and name H')[0]]
        H2 = [topology.select('residue 9 and resname VAL and name H')[0],
              topology.select('residue 2 and resname GLN and name O')[0]]
        H3 = [topology.select('residue 7 and resname ALA and name O')[0],
              topology.select('residue 4 and resname VAL and name H')[0]]
        H4 = [topology.select('residue 7 and resname ALA and name H')[0],
              topology.select('residue 4 and resname VAL and name O')[0]]

    elif system == "RUN12_2":
        Hal1 = [topology.select('residue 4 and resname AAA and name I1')[0],
                topology.select('residue 9 and resname HSI and name C2')[0]]
        H1 = [topology.select('residue 1 and resname ACE and name O')[0],
              topology.select('residue 12 and resname NHE and name N')[0]]
        H2 = [topology.select('residue 3 and resname THR and name H')[0],
              topology.select('residue 10 and resname SER and name O')[0]]
        H3 = [topology.select('residue 3 and resname THR and name O')[0],
              topology.select('residue 10 and resname SER and name H')[0]]
        H4 = [topology.select('residue 5 and resname GLU and name H')[0],
              topology.select('residue 8 and resname LYS and name O')[0]]
        H5 = [topology.select('residue 5 and resname GLU and name O')[0],
              topology.select('residue 8 and resname LYS and name H')[0]]

    elif system == "RUN13_1":
        Hal1 = [topology.select('residue 4 and resname AAA and name I1')[0],
                topology.select('residue 9 and resname HSM and name O2')[0]]
        H1 = [topology.select('residue 1 and resname ACE and name O')[0],
              topology.select('residue 12 and resname NHE and name HN1')[0]]
        H2 = [topology.select('residue 3 and resname THR and name H')[0],
              topology.select('residue 10 and resname SER and name O')[0]]
        H3 = [topology.select('residue 3 and resname THR and name O')[0],
              topology.select('residue 10 and resname SER and name H')[0]]
        H4 = [topology.select('residue 5 and resname GLU and name H')[0],
              topology.select('residue 8 and resname LYS and name O')[0]]
        H5 = [topology.select('residue 5 and resname GLU and name O')[0],
              topology.select('residue 8 and resname LYS and name H')[0]]

    else: raise ValueError("'system' is not one of ['RUN0_1a', 'RUN1_1b',...]")

    pairs = np.array([H1, H2, Hal1, H3, H4])
    if verbose:
        for pair in pairs: print(table.iloc[pair])

    if (system == "RUN12_2") or (system == "RUN13_1"):
        pairs = np.array([H1, H2, Hal1, H3, H4, H5])
    return pairs
# }}}

# Methods:{{{

def distances_to_intensities(d, sigma=1.0):
    return sigma / d**6

def get_sys_pair(sys_name, system_dirs):
    if sys_name.endswith("a"):
        pair = [dir for dir in system_dirs if sys_name.split("_")[-1].replace("a", "b") in dir][0]
    elif sys_name.endswith("b"):
        pair = [dir for dir in system_dirs if sys_name.split("_")[-1].replace("b", "a") in dir][0]
    elif sys_name.endswith("5"):
        pair = [dir for dir in system_dirs if sys_name.split("_")[-1].replace("5", "8") in dir][0]
    elif sys_name.endswith("8"):
        pair = [dir for dir in system_dirs if sys_name.split("_")[-1].replace("8", "5") in dir][0]
    elif sys_name.endswith("1"):
        pair = [dir for dir in system_dirs if (sys_name.split("_")[-1].replace("1", "2") in dir) and (dir.endswith("2"))][0]
    elif sys_name.endswith("2"):
        pair = [dir for dir in system_dirs if (sys_name.split("_")[-1].replace("2", "1") in dir) and (dir.endswith("1"))][0]
    else:
        print("No pairing with this system!")
        exit()
    return pair


from mdtraj.geometry import _geometry, compute_distances
from mdtraj.utils import ensure_type

def _get_bond_triplets(topology, exclude_water=True, sidechain_only=False):
    def can_participate(atom):
        # Filter waters
        if exclude_water and atom.residue.is_water:
            return False
        # Filter non-sidechain atoms
        if sidechain_only and not atom.is_sidechain:
            return False
        # Otherwise, accept it
        return True

    def get_donors(e0, e1):
        # Find all matching bonds
        elems = {e0, e1}
        atoms = [(one, two) for one, two in topology.bonds if {one.element.symbol, two.element.symbol} == elems]

        # Filter non-participating atoms
        atoms = [atom for atom in atoms if can_participate(atom[0]) and can_participate(atom[1])]

        # Get indices for the remaining atoms
        indices = []
        for a0, a1 in atoms:
            pair = (a0.index, a1.index)
            # make sure to get the pair in the right order, so that the index
            # for e0 comes before e1
            if a0.element.symbol == e1:
                pair = pair[::-1]
            indices.append(pair)

        return indices

    # Check that there are bonds in topology
    nbonds = 0
    for _bond in topology.bonds:
        nbonds += 1
        break  # Only need to find one hit for this check (not robust)
    if nbonds == 0:
        raise ValueError(
            "No bonds found in topology. Try using "
            "traj._topology.create_standard_bonds() to create bonds "
            "using our PDB standard bond definitions.",
        )

    nh_donors = get_donors("N", "H")
    oh_donors = get_donors("O", "H")
    xh_donors = np.array(nh_donors + oh_donors)

    if len(xh_donors) == 0:
        # if there are no hydrogens or protein in the trajectory, we get
        # no possible pairs and return nothing
        return np.zeros((0, 3), dtype=int)

    acceptor_elements = frozenset(("O", "N"))
    acceptors = [a.index for a in topology.atoms if a.element.symbol in acceptor_elements and can_participate(a)]

    # Make acceptors a 2-D numpy array
    acceptors = np.array(acceptors)[:, np.newaxis]

    # Generate the cartesian product of the donors and acceptors
    xh_donors_repeated = np.repeat(xh_donors, acceptors.shape[0], axis=0)
    acceptors_tiled = np.tile(acceptors, (xh_donors.shape[0], 1))
    bond_triplets = np.hstack((xh_donors_repeated, acceptors_tiled))

    # Filter out self-bonds
    self_bond_mask = bond_triplets[:, 0] == bond_triplets[:, 2]
    return bond_triplets[np.logical_not(self_bond_mask), :]


def _compute_bounded_geometry(
    traj,
    triplets,
    distance_cutoff,
    distance_indices,
    angle_indices,
    freq=0.0,
    periodic=True,
):
    """
    Returns a tuple include (1) the mask for triplets that fulfill the distance
    criteria frequently enough, (2) the actual distances calculated, and (3) the
    angles between the triplets specified by angle_indices.
    """
    # First we calculate the requested distances
    distances = compute_distances(
        traj,
        triplets[:, distance_indices],
        periodic=periodic,
    )

    # Now we discover which triplets meet the distance cutoff often enough
    prevalence = np.mean(distances < distance_cutoff, axis=0)
    mask = prevalence > freq

    # Update data structures to ignore anything that isn't possible anymore
    triplets = triplets.compress(mask, axis=0)
    distances = distances.compress(mask, axis=1)

    # Calculate angles using the law of cosines
    abc_pairs = zip(angle_indices, angle_indices[1:] + angle_indices[:1])
    abc_distances = []

    # Calculate distances (if necessary)
    for abc_pair in abc_pairs:
        if set(abc_pair) == set(distance_indices):
            abc_distances.append(distances)
        else:
            abc_distances.append(
                compute_distances(
                    traj,
                    triplets[:, abc_pair],
                    periodic=periodic,
                ),
            )

    # Law of cosines calculation
    a, b, c = abc_distances
    cosines = (a**2 + b**2 - c**2) / (2 * a * b)
    np.clip(cosines, -1, 1, out=cosines)  # avoid NaN error
    angles = np.arccos(cosines)

    return triplets, mask, distances, angles

def get_theta_angles(traj, distance_cutoff=None):
    freq=0.1
    exclude_water=True
    periodic=True
    sidechain_only=False
    if distance_cutoff == None:
        distance_cutoff=1000.25

    angle_cutoff=120
    angle_cutoff = np.radians(angle_cutoff)

    # Get the possible donor-hydrogen...acceptor triplets
    bond_triplets = _get_bond_triplets(
        traj.topology,
        exclude_water=exclude_water,
        sidechain_only=sidechain_only,
    )
    #print(bond_triplets.shape)

    indices, mask, distances, angles = _compute_bounded_geometry(
        traj,
        bond_triplets,
        distance_cutoff,
        [1, 2],
        [0, 1, 2],
        freq=freq,
        periodic=periodic,
    )
    ## Find triplets that meet the criteria
    #presence = np.logical_and(distances < distance_cutoff, angles > angle_cutoff)
    #mask[mask] = np.mean(presence, axis=0) > freq
    #return bond_triplets.compress(mask, axis=0)

    return indices, angles


def smooth_step_function(x, x_off=3, k=4):
    """
    A logistic function that transitions from 1 to 0 as x grows. The parameter k
    controls the steepness of the curve and x_off is the point at which the
    activation function is halfway off (x = x_off, y = 0.5).
    y = \frac{1}{1 + e^{-k (x + x_{\text{off}})}} \left(1 - \frac{1}{1 + e^{-k (x - x_{\text{off}})}} \right)

    Parameters:
    x (array): Input array of distances.
    x_off (float): The point at which the activation function is halfway off (x = x_off, y = 0.5)
    k (float): The steepness of the transition.

    Returns:
    array: Output array with smooth transitions.
    """
    x_on = -x_off
    return 1 / (1 + np.exp(-k * (x - x_on))) * (1 - 1 / (1 + np.exp(-k * (x - x_off))))

def plot_logistic_function():

    x = np.linspace(0, 10, 1000)
    #result = smooth_step_function(x, k=3.5, x_off=3.0)
    result = smooth_step_function(x, k=2.75, x_off=3.0)

    fig = plt.figure(figsize=(6,6))
    ax = plt.subplot(111)
    ax.plot(x, result, color="k")
    ax.set_xlabel("r (Å)", fontsize=18)
    ax.set_ylabel("y", fontsize=18)

    index = np.argmin(np.abs(result - 0.75))
    x_line, y_line = x[index], result[index]
    ax.plot([x_line, x_line], [0, y_line], color="green", ls="--", lw=1, label=f"Strong: \nr = {x_line:.3g} Å")
    ax.plot([0, x_line], [y_line, y_line], color="green", ls="--", lw=1, label="__no_legend__")

    index = np.argmin(np.abs(result - 0.5))
    x_line, y_line = x[index], result[index]
    ax.plot([x_line, x_line], [0, y_line], color="k", ls="--", lw=1, label=f"Moderate: \nr = {x_line:.3g} Å")
    ax.plot([0, x_line], [y_line, y_line], color="k", ls="--", lw=1, label="__no_legend__")

    index = np.argmin(np.abs(result - 0.25))
    x_line, y_line = x[index], result[index]
    ax.plot([x_line, x_line], [0, y_line], color="r", ls="--", lw=1, label=f"Weak: \nr = {x_line:.3g} Å")
    ax.plot([0, x_line], [y_line, y_line], color="r", ls="--", lw=1, label="__no_legend__")

    # Adding more y-tick labels
    yticks = np.linspace(0, 1, 9)  # Generates 20 evenly spaced y-ticks between 0 and 1
    ax.set_yticks(yticks)

    ax.legend(fontsize=14)
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 1.01)

    ticks = [ax.xaxis.get_minor_ticks(),
             ax.xaxis.get_major_ticks(),]
    xmarks = [ax.get_xticklabels()]
    ymarks = [ax.get_yticklabels()]
    for k in range(0,len(ticks)):
        for tick in ticks[k]:
            tick.label.set_fontsize(12)
    for k in range(0,len(xmarks)):
        for mark in xmarks[k]:
            mark.set_size(fontsize=12)
    for k in range(0,len(ymarks)):
        for mark in ymarks[k]:
            mark.set_size(fontsize=14)
            mark.set_rotation(s=0)

    #ax.title("Logistic function", fontsize=18)
    #ax.set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(f"hbond_logistic_function.png")




#:}}}

# Append to Database:{{{
def append_to_database(A, dbName="database_Nd.pkl", verbose=False, **kwargs):
    n_lambdas = A.K
    pops = A.P_dP[:,n_lambdas-1]
    pops = np.nan_to_num(pops, nan=0.0)
    BS = A.f_df
    prior_micro_populations = kwargs.get("prior microstate populations")
    prior_macro_populations = kwargs.get("prior macrostate populations")
    macro_populations = kwargs.get("macrostate populations")

    data = pd.DataFrame()
    data["FF"] = [kwargs.get("FF")]
    data["System"] = [kwargs.get("System")]
    data["nsteps"] = [kwargs.get("nsteps")]
    data["nstates"] = [kwargs.get("nStates")]
    data["nlambda"] = [kwargs.get("n_lambdas")]
    data["nreplica"] = [kwargs.get("nreplicas")]
    data["lambda_swap_every"] = [kwargs.get("lambda_swap_every")]
    data["Nd"] = [kwargs.get("Nd")]
    data["uncertainties"] = [kwargs.get("data_uncertainty")]
    data["stat_model"] = [kwargs.get("stat_model")]

    #model_scores = A.get_model_scores(verbose=False)# True)
    #for i,lam in enumerate(kwargs.get("lambda_values")):
    #    lam = "%0.2g"%lam
    #    data["BIC Score lam=%s"%lam] = [np.float(model_scores[i]["BIC score"])]

    for i,lam in enumerate(kwargs.get("lambda_values")):
        lam = "%0.2g"%lam
        data["BICePs Score lam=%s"%lam] = [BS[i,0]]
        data["BICePs Score Std lam=%s "%lam] = [2*BS[i,1]] # at 95% C


    try:                    data["k"] = [A.get_restraint_intensity()]
    except(Exception) as e: data["k"] = [np.nan]

    data["micro pops"] = [pops]
    data["macro pops"] = [macro_populations]
    data["prior micro pops"] = [prior_micro_populations]
    data["prior macro pops"] = [prior_macro_populations]
    data["D_KL"] = [np.nansum([pops[i]*np.log(pops[i]/prior_micro_populations[i]) for i in range(len(pops))])]
    data["RMSE (prior micro pops)"] = [np.sqrt(metrics.mean_squared_error(pops, prior_micro_populations))]
    try:
        data["RMSE (prior macro pops)"] = [np.sqrt(metrics.mean_squared_error(macro_populations, prior_macro_populations))]
    except(Exception) as e:
        pass
    data["prior"] = [kwargs.get("prior")]

    acceptance_info = kwargs.get("acceptance_info")
    columns = acceptance_info.columns.to_list()
    for i,lam in enumerate(acceptance_info["lambda"].to_list()):
        row = acceptance_info.iloc[[i]]
        for k,col in enumerate(row.columns[1:]):
            data[f"{col}_lam={lam}"] = row[col].to_numpy()

    data.to_pickle(dbName)
    gc.collect()

# }}}

# MAIN:{{{
save_obj = 1
testing = 1

convert_distances_to_intensities = 0

nstates = 100
nstates = 500
if nstates == 500:
    ext = "500_states/"
else:
    ext = ""



#n_lambdas, nreplicas, nsteps, swap_every, change_Nr_every = 2, 1, 10000000, 0, 0 # NOTE: Bayesian only
n_lambdas, nreplicas, nsteps, swap_every, change_Nr_every = 2, 8, 10000000, 0, 0
n_lambdas, nreplicas, nsteps, swap_every, change_Nr_every = 2, 16, 10000000, 0, 0
n_lambdas, nreplicas, nsteps, swap_every, change_Nr_every = 2, 128, 1000000, 0, 0
#n_lambdas, nreplicas, nsteps, swap_every, change_Nr_every = 2, 1, 1000000, 0, 0


burn = 0
#burn = 10000
#burn = 100000



stat_model, data_uncertainty = "GB", "single"
#stat_model, data_uncertainty = "Students", "single"
#stat_model, data_uncertainty = "Bayesian", "single"
#stat_model, data_uncertainty = "Gaussian", "multiple"

#stat_model, data_uncertainty = "GaussianSP", "single"

all_data = 1
J_only   = 0
noe_only = 0
cs_only  = 0
rdc_only  = 0
J_and_noe_only = 0

combine_cs = 1
get_combined_input_data = 0


use_all_noes = 1


scale_energies = 0
find_optimal_nreplicas = 0
write_every = 100
lambda_values = np.linspace(0.0, 1.0, n_lambdas)

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
#sys_name = os.path.join(main_dir,system_dirs[1]) # FIXME::::::::::::::::::::::::::::::::::::::::
### 2a, 2b
#sys_name = os.path.join(main_dir,system_dirs[-3]) # FIXME::::::::::::::::::::::::::::::::::::::::
#sys_name = os.path.join(main_dir,system_dirs[2]) # FIXME::::::::::::::::::::::::::::::::::::::::
#### 3a, 3b
#sys_name = os.path.join(main_dir,system_dirs[5]) # FIXME::::::::::::::::::::::::::::::::::::::::
#sys_name = os.path.join(main_dir,system_dirs[3]) # FIXME::::::::::::::::::::::::::::::::::::::::
#### 4a, 4b
#sys_name = os.path.join(main_dir,system_dirs[4]) # FIXME::::::::::::::::::::::::::::::::::::::::
#sys_name = os.path.join(main_dir,system_dirs[6]) # FIXME::::::::::::::::::::::::::::::::::::::::
#### 5, 8
sys_name = os.path.join(main_dir,system_dirs[7]) # FIXME::::::::::::::::::::::::::::::::::::::::
#sys_name = os.path.join(main_dir,system_dirs[8]) # FIXME::::::::::::::::::::::::::::::::::::::::
#### 1, 2
#sys_name = os.path.join(main_dir,system_dirs[-1]) # FIXME::::::::::::::::::::::::::::::::::::::::
#sys_name = os.path.join(main_dir,system_dirs[-2]) # FIXME::::::::::::::::::::::::::::::::::::::::


#print(sys_name)
#exit()
#data_dir = f"{sys_name}/{data_dir}"
data_dir = sys_name
print(data_dir)
#exit()
dir = "results"
biceps.toolbox.mkdir(dir)
if use_all_noes:
    outdir = f'{dir}/{sys_name}_with_avg_sidechains/{ext}{stat_model}_{data_uncertainty}_sigma'
else:
    outdir = f'{dir}/{sys_name}/{ext}{stat_model}_{data_uncertainty}_sigma'
biceps.toolbox.mkdir(outdir)

# NOTE: IMPORTANT: Create multiple trials to average over and check the deviation of results
if use_all_noes:
    check_dirs = next(os.walk(f"{dir}/{sys_name}_with_avg_sidechains/{ext}{stat_model}_{data_uncertainty}_sigma/"))[1]
else:
    check_dirs = next(os.walk(f"{dir}/{sys_name}/{ext}{stat_model}_{data_uncertainty}_sigma/"))[1]
trial = len(check_dirs)

if use_all_noes:
    outdir = f'{dir}/{sys_name}_with_avg_sidechains/{ext}{stat_model}_{data_uncertainty}_sigma/{nsteps}_steps_{nreplicas}_replicas_{n_lambdas}_lam__swap_every_{swap_every}_change_Nr_every_{change_Nr_every}_trial_{trial}'
else:
    outdir = f'{dir}/{sys_name}/{ext}{stat_model}_{data_uncertainty}_sigma/{nsteps}_steps_{nreplicas}_replicas_{n_lambdas}_lam__swap_every_{swap_every}_change_Nr_every_{change_Nr_every}_trial_{trial}'
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


#print(f"Possible input data extensions: {biceps.toolbox.list_possible_extensions()}")

if combine_cs: sub_data_loc = "all_data"
else: sub_data_loc = "CS_J_NOE"

if use_all_noes:
    sub_data_loc = sub_data_loc + "_with_avg_sidechains"

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


def combine_input_data(input_data, pattern, readAs="pickle", write=False, verbose=False):
    import fnmatch

    if type(input_data[0][0]) != str:
        print("input data needs to be list of input data files")

    new_extension = pattern.split(".")[-1].split("_")[0].replace("*","")
    options = biceps.get_restraint_options(input_data)
    options = pd.DataFrame(options)
    extensions = options['extension'].to_numpy()
    data_filenames = []
    data = []
    for state in range(len(input_data)):
        _data = []
        file_names = input_data[state]
        # Filter the list to only include files with extensions that match the pattern
        files = np.array([(i,file) for i,file in enumerate(file_names) if fnmatch.fnmatch(file, pattern)])
        #print(files)
        remaining_files = np.array([(i,file) for i,file in enumerate(file_names) if file not in files[:,1]])
        #print(remaining_files)
        #exit()
        for i,file in remaining_files:
            df = getattr(pd, "read_%s"%readAs)
            df = df(file)
        #    print(df[["exp", "model", "restraint_index", "atom_index1", "res1"]])
            _data.append((int(i),df,file))

        _ = pd.DataFrame()
        max_restraint_index = 0
        for q,file in files:
            #if verbose: print('Loading %s as %s...'%(file, readAs))
            df = getattr(pd, "read_%s"%readAs)
            df = df(file)
            new = df.copy()
            new["restraint_index"] = df["restraint_index"].to_numpy() + max_restraint_index + 1
            _ = pd.concat([_, new.copy()], axis=0)
            max_restraint_index = new["restraint_index"].max()
        #    print(df[["exp", "model", "restraint_index", "atom_index1", "res1"]])
        #print("\n\n")
        #print(_[["exp", "model", "restraint_index", "atom_index1", "res1"]])
        #exit()
        path = os.path.dirname(files[0][1])
        outname = str(state)+"."+new_extension
        out = os.path.join(path, outname)
        _ = _.reset_index(drop=True)
        if write:
            obj = getattr(_, "to_%s"%readAs)
            obj(out)

        _data.append((int(q),_,out))
        _data = np.array(_data, dtype=object)
        #print(_data[:,0])
        arr = np.sort(_data[:,0], axis=0)
        indices = [int(np.where(val == _data[:,0])[0]) for val in arr]
        #print(arr)
        #print(indices)
        #print(len(_data))
        data.append([_data[idx][1] for idx in indices])
        #exit()
    arr = np.array(arr, dtype=int)
    new_options = options.iloc[arr]
    extensions[int(q)] = new_extension
    new_options['extension'] = extensions[arr]
    new_options = new_options.reset_index()
    new_options = new_options.to_dict(orient='records')
    return new_options, data#, data_filenames

if get_combined_input_data:
    options, input_data = combine_input_data(input_data, pattern="*.cs_*", write=1)
    exit()




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
restraint_indices = [df["restraint_index"].to_numpy() for df in dfs]
Nd  = sum([len(np.unique(arr)) for arr in restraint_indices])
#Nd  = sum([len(df) for df in dfs])
print(f"Input data: {biceps.toolbox.list_extensions(input_data)}")
print(f"nSteps of sampling: {nsteps}\nnReplica: {nreplicas}")
print("Nd = ", Nd)
#exit()




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
    #phi,phi_index,stat_model,data_uncertainty=(1.0, 2.0, 1),0,"GB","single"
    beta,beta_index,stat_model,data_uncertainty=(1.0, 2.0, 1),0,"GB","single"

elif stat_model == "GaussianSP":
    phi,phi_index,stat_model,data_uncertainty=(1.0, 2.0, 1),0,"GB",data_uncertainty
    beta,beta_index,stat_model,data_uncertainty=(1.0, 2.0, 1),0,"GB",data_uncertainty

#elif stat_model == "Bayesian":
#    phi,phi_index,stat_model,data_uncertainty=(1.0, 100.0, 1000),0,"GB","single"
#    beta,beta_index,stat_model,data_uncertainty=(1.0, 2.0, 1),0,"GB","single"

else:
    phi,phi_index,data_uncertainty=(1.0, 2.0, 1),0,data_uncertainty
    beta,beta_index,data_uncertainty=(1.0, 2.0, 1),0,data_uncertainty


options = biceps.get_restraint_options(input_data)
opt = pd.DataFrame(options)
try:
    noe_rest_index = [i for i,ext in enumerate(opt["extension"]) if "noe" in ext][0]
except(Exception) as e:
    noe_rest_index = 999
#print(options)
#exit()

for i in range(len(options)):
    options[i].update(dict(ref="uniform", sigma=(sigMin, sigMax, dsig), sigma_index=sigma_index,
         phi=phi, phi_index=phi_index, beta=beta, beta_index=beta_index,
         stat_model=stat_model, data_uncertainty=data_uncertainty,
         data_likelihood=data_likelihood))
    if i == noe_rest_index:
        options[i].update(dict(ref="uniform", sigma=(sigMin, sigMax, dsig), sigma_index=sigma_index,
             gamma=(0.1, 10.0, 1.01),
             phi=phi, phi_index=phi_index, beta=beta, beta_index=beta_index,
             stat_model=stat_model, data_uncertainty=data_uncertainty,
             data_likelihood=data_likelihood))



print(pd.DataFrame(options))
#print(input_data)
#exit()


ensemble = biceps.ExpandedEnsemble(energies=energies, lambda_values=lambda_values)
ensemble.initialize_restraints(input_data, options)
sampler = biceps.PosteriorSampler(ensemble, nreplicas,
        write_every=write_every, change_Nr_every=change_Nr_every)



sampler.sample(nsteps, attempt_lambda_swap_every=swap_every, swap_sigmas=1,
        find_optimal_nreplicas=find_optimal_nreplicas,
        attempt_move_state_every=attempt_move_state_every,
        attempt_move_sigma_every=attempt_move_sigma_every,
        burn=burn, print_freq=100, walk_in_all_dim=walk_in_all_dim,
        verbose=0, progress=1, multiprocess=multiprocess, capture_stdout=0)
sampler.save_trajectories(outdir)

if save_obj: biceps.toolbox.save_object(sampler, f"{outdir}/sampler_obj.pkl")


print(sampler.acceptance_info)
#print(sampler.exchange_info)
#sampler.plot_exchange_info(xlim=(-100, nsteps), figsize=(12,6), figname=f"{outdir}/lambda_swaps.png")

expanded_values = sampler.expanded_values

####### Posterior Analysis #######
A = biceps.Analysis(sampler, outdir=outdir, nstates=len(energies), MBAR=True,
                    scale_energies=scale_energies)

A.plot_acceptance_trace()
try:
    A.plot_energy_trace()
except(Exception) as e:
    print(e)



BS, pops = A.f_df, A.P_dP[:,n_lambdas-1]
pops = np.nan_to_num(pops, nan=0.0)
print(pops)
print(sampler.populations)
K = n_lambdas-1
dpops = A.P_dP[:,2*K]
dpops = np.nan_to_num(dpops, nan=0.0)

# Find the top populated state from BICePs (should be folded)
ntop = 10
topN_pops = pops[np.argsort(pops)[-ntop:]][::-1]
topN_labels = [np.where(topN_pops[i] == pops)[0][0] for i in range(len(topN_pops))]
top_pop = topN_labels[0]

files = biceps.toolbox.get_files(f"{sys_name}/*cluster*.pdb")
snaps_per_state = len(biceps.toolbox.get_files(f"{sys_name}/*cluster0_*.pdb"))
snapshots = biceps.toolbox.get_files(f"{sys_name}/*.pdb")
if nstates == 100:
    _files_by_state = [biceps.toolbox.get_files(f"{sys_name}/*cluster{state}_*.pdb") for state in range(nstates)]
elif nstates == 500:
    _files_by_state = [[file] for file in biceps.toolbox.get_files(f"{sys_name}/*cluster*_*.pdb")]
else:
    print("nstates is not right")
    exit()
top_pop_files = _files_by_state[top_pop]





if scale_energies == False:
    BS /= sampler.nreplicas

print(f"BICePs Score = {BS[:,0]}")
if data_uncertainty == "single":
    #try:
    #    A.plot(plottype=plottype, figsize=(12,6), figname=f"BICePs.pdf", hspace=0.85, plot_all_distributions=1)
    #except(Exception) as e:
    #    print(e)
    fig = A.plot(plottype=plottype, figsize=(12,14), figname=f"BICePs.pdf", hspace=0.85, plot_all_distributions=1)

else:
    fig = A.plot(plottype=plottype, figsize=(12,14), figname=f"BICePs.pdf",hspace=0.85, plot_all_distributions=0)
if convert_distances_to_intensities:
    for ax in fig.axes:
        x_label = ax.get_xlabel()
        y_label = ax.get_ylabel()
        if "-1/6" in x_label:
            ax.set_xlabel(x_label.replace("-1/6", "-6"))
            ax.set_ylabel(y_label.replace("-1/6", "-6"))
    fig.savefig(f"{outdir}/BICePs.pdf")
#    exit()



#NOTE: Save analysis object?
#biceps.toolbox.save_object(A, f"{outdir}/Analysis_obj_{stat_model}_{data_uncertainty}_sigma.pkl")
mlp = pd.concat([A.get_max_likelihood_parameters(model=i) for i in range(len(lambda_values))])
mlp.reset_index(inplace=True, drop=True)
mlp.to_pickle(outdir+"/mlp.pkl")
#biceps.toolbox.save_object(A, filename=outdir+"/analysis.pkl")

#traj = sampler.traj[-1].__dict__
##init, frac = biceps.find_all_state_sampled_time(traj['state_trace'], len(energies))
#C = biceps.Convergence(traj, outdir=outdir)
##biceps.toolbox.save_object(C, filename=outdir+"/convergence.pkl")
#C.plot_traces(figname="traces.pdf", xlim=(0, nsteps))
#C.get_autocorrelation_curves(method="block-avg", maxtau=5000)
#C.process()

#A.plot_population_evolution()
#A.plot_convergence()


#exit()

dbName = f"{outdir}/results.pkl"
if os.path.exists(dbName): os.remove(dbName)

try:
    #ntop = int(len(energies)/10.)
    ntop = len(energies)
    topN_pops = pops[np.argsort(pops)[-ntop:]]
    print(topN_pops)
    topN_labels = [np.where(topN_pops[i]==pops)[0][0] for i in range(len(topN_pops))]
    print(topN_labels)
    #print(pops)
    #exit()
except(Exception) as e:
    print(e)


assignments = 0
if assignments:
    # NOTE: Get reweighted populations
    df = pd.read_csv(f"{data_dir}/dihedrals_k{nstates}_msm_assignments.csv", index_col=0, skiprows=0, comment='#')
    df_pops = df.copy()
    df_pops.columns = ["macrostate"]
    df_pops["microstate"] = df_pops.index.to_numpy()
    df_pops["population"] = pops
    df_pops.to_csv(f"{outdir}/{stat_model}_{data_uncertainty}_sigma__reweighted_populations.csv")
    grouped = df_pops.groupby(["macrostate"])
    reweighted_macro_pops = grouped.sum().drop("microstate", axis=1)
    #reweighted_macro_pops["FF"] = [forcefield for i in range(len(reweighted_macro_pops.index.to_list()))]
    reweighted_macro_pops["nclusters"] = [nstates for i in range(len(reweighted_macro_pops.index.to_list()))]
    reweighted_macro_pops["System"] = [sys_name.split("/")[-1] for i in range(len(reweighted_macro_pops.index.to_list()))]
    print(reweighted_macro_pops)
    #print(df)

    # NOTE: Get Prior MSM populations
    df_pops = df.copy()
    df_pops.columns = ["macrostate"]
    df_pops["microstate"] = df_pops.index.to_numpy()
    df_pops["population"] = msm_pops
    #df_pops.to_csv(f"{outdir}/{stat_model}_{data_uncertainty}_sigma__reweighted_populations.csv")
    grouped = df_pops.groupby(["macrostate"])
    macrostate_populations = grouped.sum().drop("microstate", axis=1)
    #macrostate_populations["FF"] = [forcefield for i in range(len(macrostate_populations.index.to_list()))]
    macrostate_populations["nclusters"] = [nstates for i in range(len(macrostate_populations.index.to_list()))]
    macrostate_populations["System"] = [sys_name.split("/")[-1] for i in range(len(macrostate_populations.index.to_list()))]
    print(macrostate_populations)

###############################################################################
###############################################################################
###############################################################################

#input_data = biceps.toolbox.sort_data(f'{data_dir}/{sub_data_loc}')
if ("RUN6_8" not in sys_name) and (all_data or noe_only or J_and_noe_only):
    noe = [pd.read_pickle(i) for i in biceps.toolbox.get_files(f"{data_dir}/{ext}{sub_data_loc}/*.noe")]
    #  Get the ensemble average observable
    noe_Exp = noe[0]["exp"].to_numpy()
    noe_model = [i["model"].to_numpy() for i in noe]

    # FIXME: add reweighted uncertainty that comes from predicted populations
    noe_prior = np.array([w*noe_model[i] for i,w in enumerate(msm_pops.to_numpy())]).sum(axis=0)
    noe_reweighted = np.nansum(np.array([u.ufloat(w,dpops[i])*noe_model[i] for i,w in enumerate(pops)]), axis=0) #.sum(axis=0)

    #noe_labels = [f"{three2one(row[1]['res1'])}.{row[1]['atom_name1']}-{three2one(row[1]['res2'])}.{row[1]['atom_name2']}" for row in noe[0].iterrows()]
    noe_labels = []
    for row in noe[0].iterrows():
        try:
            noe_labels.append(f"{three2one(row[1]['res1'])}.{row[1]['atom_name1']}-{three2one(row[1]['res2'])}.{row[1]['atom_name2']}")
        except(Exception) as e:
            noe_labels.append(f"{row[1]['res1']}.{row[1]['atom_name1']}-{row[1]['res2']}.{row[1]['atom_name2']}")

    noe_label_indices = np.array([[row[1]['atom_index1'], row[1]['atom_index2']] for row in noe[0].iterrows()])


if ("RUN6_8" not in sys_name) and (all_data or J_only or J_and_noe_only):
    J = [pd.read_pickle(file) for file in biceps.toolbox.get_files(f'{data_dir}/{ext}{sub_data_loc}/*.J')]
    #  Get the ensemble average observable
    J_Exp = J[0]["exp"].to_numpy()
    J_model = [i["model"].to_numpy() for i in J]

    J_prior = np.array([w*J_model[i] for i,w in enumerate(msm_pops.to_numpy())]).sum(axis=0)
    #J_reweighted = np.array([w*J_model[i] for i,w in enumerate(pops)]).sum(axis=0)
    J_reweighted = np.nansum(np.array([u.ufloat(w,dpops[i])*J_model[i] for i,w in enumerate(pops)]), axis=0) #.sum(axis=0)

    #J_labels = [f"{three2one(row[1]['res1'])}.{row[1]['atom_name1']}\n{three2one(row[1]['res2'])}.{row[1]['atom_name2']}\n{three2one(row[1]['res3'])}.{row[1]['atom_name3']}\n{three2one(row[1]['res4'])}.{row[1]['atom_name4']}" for row in J[0].iterrows()]
    J_labels = []
    for row in J[0].iterrows():
        try:
            J_labels.append(f"{three2one(row[1]['res1'])}.{row[1]['atom_name1']}\n{three2one(row[1]['res2'])}.{row[1]['atom_name2']}\n{three2one(row[1]['res3'])}.{row[1]['atom_name3']}\n{three2one(row[1]['res4'])}.{row[1]['atom_name4']}")
        except(Exception) as e:
            J_labels.append(f"{row[1]['res1']}.{row[1]['atom_name1']}\n{row[1]['res2']}.{row[1]['atom_name2']}\n{row[1]['res3']}.{row[1]['atom_name3']}\n{row[1]['res4']}.{row[1]['atom_name4']}")

    J_label_indices = np.array([[row[1]['atom_index1'], row[1]['atom_index2'], row[1]['atom_index3'], row[1]['atom_index4']] for row in J[0].iterrows()])





try:
    if all_data or cs_only:
#        cs = [pd.read_pickle(file) for file in biceps.toolbox.get_files(f'{data_dir}/{sub_data_loc}/*.cs*')]
#        #print(cs)
#        #  Get the ensemble average observable
#        cs_Exp = cs[0]["exp"].to_numpy()
#        cs_model = [i["model"].to_numpy() for i in cs]
#
#        cs_prior = np.array([w*cs_model[i] for i,w in enumerate(msm_pops.to_numpy())]).sum(axis=0)
#        #cs_reweighted = np.array([w*cs_model[i] for i,w in enumerate(pops)]).sum(axis=0)
#        cs_reweighted = np.nansum(np.array([u.ufloat(w,dpops[i])*cs_model[i] for i,w in enumerate(pops)]), axis=0) #.sum(axis=0)
#
#        #cs_labels = [f"{three2one(row[1]['res1'])}.{row[1]['atom_name1']}" for row in cs[0].iterrows()]


         cs, cs_model = [], []
         for i in biceps.toolbox.sort_data(f'{data_dir}/{ext}{sub_data_loc}/*.cs*'):
             cs.append([pd.read_pickle(i[j]) for j in range(len(i))])
             cs_model.append([cs[-1][j]["model"].to_numpy() for j in range(len(cs[-1]))])
         #  Get the ensemble average observable
         cs_Exp = [cs[0][j]["exp"].to_numpy() for j in range(len(cs[0]))]
         cs_prior = []
         cs_reweighted = []
         cs_label_indices, cs_labels = [],[]
         for j in range(len(cs_model[0])):
             # FIXME: add reweighted uncertainty that comes from predicted populations
             cs_prior.append(np.array([w*cs_model[i][j] for i,w in enumerate(msm_pops.to_numpy())]).sum(axis=0))
             cs_reweighted.append(np.nansum(np.array([u.ufloat(w,dpops[i])*cs_model[i][j] for i,w in enumerate(pops)]), axis=0))
             #cs_labels.append([f"{three2one(row[1]['res1'])}.{row[1]['atom_name1']}}" for row in cs[0][j].iterrows()])
             #cs_label_indices.append(np.array([[row[1]['atom_index1']] for row in cs[0][j].iterrows()]))

             _cs_labels = []
             _cs_label_indices = []
             try:
                 for row in cs[0][j].iterrows():
                     try:
                         _cs_labels.append(f"{three2one(row[1]['res1'])}.{row[1]['atom_name1']}")
                     except(Exception) as e:
                         _cs_labels.append(f"{row[1]['res1']}.{row[1]['atom_name1']}")
             except(Exception) as e: pass
             #_cs_label_indices = np.array([[row[1]['atom_index1']] for row in cs[0].iterrows()])

             cs_labels.append(_cs_labels)
             cs_label_indices.append(_cs_label_indices)

except(Exception) as e:
    print(e)


if all_data:
    input_data = biceps.toolbox.sort_data(f'{data_dir}/{ext}{sub_data_loc}')
    options = biceps.get_restraint_options(input_data)

opt = pd.DataFrame(options)

if [obs for obs in opt["extension"].to_numpy() if "rdc" in obs] != []:
    if all_data or rdc_only:
        rdc, rdc_model = [], []
        for i in biceps.toolbox.sort_data(f'{data_dir}/{ext}{sub_data_loc}/*.rdc*'):
            rdc.append([pd.read_pickle(i[j]) for j in range(len(i))])
            rdc_model.append([rdc[-1][j]["model"].to_numpy() for j in range(len(rdc[-1]))])
        #  Get the ensemble average observable
        rdc_Exp = [rdc[0][j]["exp"].to_numpy() for j in range(len(rdc[0]))]
        rdc_prior = []
        rdc_reweighted = []
        rdc_label_indices, rdc_labels = [],[]
        for j in range(len(rdc_model[0])):
            # FIXME: add reweighted uncertainty that comes from predicted populations
            rdc_prior.append(np.array([w*rdc_model[i][j] for i,w in enumerate(msm_pops.to_numpy())]).sum(axis=0))
            rdc_reweighted.append(np.nansum(np.array([u.ufloat(w,dpops[i])*rdc_model[i][j] for i,w in enumerate(pops)]), axis=0))
            rdc_labels.append([f"{row[1]['res1']}.{row[1]['atom_name1']}-{row[1]['res2']}.{row[1]['atom_name2']}" for row in rdc[0][j].iterrows()])
            rdc_label_indices.append(np.array([[row[1]['atom_index1'], row[1]['atom_index2']] for row in rdc[0][j].iterrows()]))




# NOTE: make plot of reweighted data points overlaid with experimental and prior
#fig = plt.figure(figsize=(12,16))
fig = plt.figure(figsize=(12,26))
fig = plt.figure(figsize=(12,16))
gs = gridspec.GridSpec(len(options), 1)


ax1 = plt.subplot(gs[0,0])
try:ax2 = plt.subplot(gs[1,0])
except(Exception) as e: pass

#ax3 = plt.subplot(gs[2,0])
#data1 = pd.concat([pd.read_pickle(i) for i in biceps.toolbox.get_files('cineromycin_B/J_NOE/*.noe')])
data = []


if ("RUN6_8" not in sys_name) and (all_data or noe_only or J_and_noe_only):
    for i in range(len(noe_reweighted)):
        try:
            data.append({"index":i,
                "reweighted noe":noe_reweighted[i].nominal_value, "reweighted noe err":noe_reweighted[i].std_dev,
                         "prior noe":noe_prior[i], "exp noe":noe_Exp[i]*mlp['gamma_noe:0'].to_numpy()[-1], "label":noe_labels[i]
                })
        except(Exception) as e:
            data.append({"index":i,
                "reweighted noe":noe_reweighted[i], "reweighted noe err":0.0,
                         "prior noe":noe_prior[i], "exp noe":noe_Exp[i]*mlp['gamma_noe:0'].to_numpy()[-1], "label":noe_labels[i]
                })


    data1 = pd.DataFrame(data)
    #print(data1)
    #exit()

    if convert_distances_to_intensities:
        _data1 = data1.sort_values(["exp noe"], ascending=False)
    else:
        _data1 = data1.sort_values(["exp noe"])
    _data1 = _data1.reset_index()
    #exit()

#    ax1.scatter(x=_data1.index.to_numpy(), y=_data1["prior noe"].to_numpy(), s=35, color="orange", label="Prior", edgecolor='black',)
#    ax1.scatter(x=_data1.index.to_numpy(), y=_data1["exp noe"].to_numpy(), s=25, color="r", label="Exp", edgecolor='black',)
#    ax1.errorbar(x=_data1.index.to_numpy(), y=_data1["reweighted noe"].to_numpy(),

    ax1.scatter(x=_data1['label'].to_numpy(), y=_data1["prior noe"].to_numpy(), s=35, color="orange", label="Prior", edgecolor='black',)
    ax1.scatter(x=_data1['label'].to_numpy(), y=_data1["exp noe"].to_numpy(), s=25, color="r", label="Exp", edgecolor='black',)
    ax1.errorbar(x=_data1['label'].to_numpy(), y=_data1["reweighted noe"].to_numpy(),

                 yerr=_data1["reweighted noe err"].to_numpy(),
                 #capsize=25,
                 ecolor="b", label="BICePs", fmt="o",
                 mec='k', mfc='b')
    ax1.set_xlim(-1, 19)
    #ax1.scatter(x=_data1.index.to_numpy(), y=_data1["reweighted noe"].to_numpy(), s=25, color="b", label="BICePs", edgecolor='black')
    ax1.legend(fontsize=14, loc="best")
    #ax1.set_xlabel(r"distances", size=16)
    ax1.set_xlabel(r"", size=16)
    if convert_distances_to_intensities:
        ax1.set_ylabel(r"NOE Intensities", size=16)
    else:
        ax1.set_ylabel(r"NOE distance ($\AA$)", size=16)

try:
    if ("RUN6_8" not in sys_name) and (all_data or J_only or J_and_noe_only):
        data = []
        for i in range(len(J_reweighted)):
            try:
                data.append({"index":i,
                    "reweighted J":J_reweighted[i].nominal_value, "reweighted J err":J_reweighted[i].std_dev,
                    "prior J":J_prior[i], "exp J":J_Exp[i], "label":J_labels[i]
                    })
            except(Exception) as e:
                data.append({"index":i,
                    "reweighted J":J_reweighted[i], "reweighted J err":0.0,
                    "prior J":J_prior[i], "exp J":J_Exp[i], "label":J_labels[i]
                    })

        data1 = pd.DataFrame(data)

        ax2.scatter(x=data1['label'].to_numpy(), y=data1["prior J"].to_numpy(), s=35, color="orange", label="Prior", edgecolor='black',)
        ax2.scatter(x=data1['label'].to_numpy(), y=data1["exp J"].to_numpy(), s=25, color="r", label="Exp", edgecolor='black',)
        #ax2.scatter(x=data1['label'].to_numpy(), y=data1["reweighted J"].to_numpy(), s=25, color="b", label="BICePs", edgecolor='black')
        ax2.errorbar(x=data1['label'].to_numpy(), y=data1["reweighted J"].to_numpy(),
                     yerr=data1["reweighted J err"].to_numpy(),
                     #capsize=25,
                     ecolor="b", label="BICePs", fmt="o",
                     mec='k', mfc='b')

        xticks = ax2.get_xticks()
        xtick_labels = ax2.get_xticklabels()

        stride = 1
        # Select every other tick label
        new_ticks = xticks[::stride]
        new_labels = xtick_labels[::stride]

        # Set new ticks and labels
        ax2.set_xticks(new_ticks)
        ax2.set_xticklabels(new_labels)

        #ax2.legend(fontsize=14)
        #ax2.set_xlabel(r"Index", size=16)
        ax2.set_xlabel(r"", size=16)
        ax2.set_ylabel(r"J-coupling (Hz)", size=16)
except(Exception) as e:
    print(e)

try:
    if all_data or cs_only:


#        data = []
#        for i in range(len(cs_reweighted)):
#            try:
#                data.append({"index":i,
#                    "reweighted cs":cs_reweighted[i].nominal_value, "reweighted cs err":cs_reweighted[i].std_dev,
#                    "prior cs":cs_prior[i], "exp cs":cs_Exp[i], "label":cs_labels[i]
#                    })
#            except(Exception) as e:
#                data.append({"index":i,
#                    "reweighted cs":cs_reweighted[i], "reweighted cs err":0.0,
#                    "prior cs":cs_prior[i], "exp cs":cs_Exp[i], "label":cs_labels[i]
#                    })
#
#        data1 = pd.DataFrame(data)
#
#        ax3.scatter(x=data1['label'].to_numpy(), y=data1["prior cs"].to_numpy(), s=35, color="orange", label="Prior", edgecolor='black',)
#        ax3.scatter(x=data1['label'].to_numpy(), y=data1["exp cs"].to_numpy(), s=25, color="r", label="Exp", edgecolor='black',)
#        #ax3.scatter(x=data1['label'].to_numpy(), y=data1["reweighted cs"].to_numpy(), s=25, color="b", label="BICePs", edgecolor='black')
#        ax3.errorbar(x=data1["label"].to_numpy(), y=data1["reweighted cs"].to_numpy(),
#                     yerr=data1["reweighted cs err"].to_numpy(),
#                     #capsize=25,
#                     ecolor="b", label="BICePs", fmt="o",
#                     mec='k', mfc='b')
#
#        xticks = ax3.get_xticks()
#        xtick_labels = ax3.get_xticklabels()
#
#        stride = 1
#        # Select every other tick label
#        new_ticks = xticks[::stride]
#        new_labels = xtick_labels[::stride]
#
#        # Set new ticks and labels
#        ax3.set_xticks(new_ticks)
#        ax3.set_xticklabels(new_labels, rotation=45)
#
#        #ax3.legend(fontsize=14)
#        #ax3.set_xlabel(r"Index", size=16)
#        ax3.set_ylabel(r"Chemical Shift (ppm)", size=16)


        bvs = [opt["extension"].replace("cs_","").replace("_","-")
               for opt in options if "cs" in opt["extension"]]

        #ax_ = [ax4,ax6]
        if ("RUN6_8" in sys_name):
            ax_cs = [plt.subplot(gs[i,0]) for i in range(len(bvs))]
        else:
            ax_cs = [plt.subplot(gs[2+i,0]) for i in range(len(bvs))]
        for j in range(len(cs_reweighted)):
            data = []
            for i in range(len(cs_reweighted[j])):
                try:
                    data.append({"index":i,
                        "reweighted cs":cs_reweighted[j][i].nominal_value, "reweighted cs err":cs_reweighted[j][i].std_dev,
                        "prior cs":cs_prior[j][i], "exp cs":cs_Exp[j][i], "label":cs_labels[j][i]
                        })
                except(Exception) as e:
                    data.append({"index":i,
                        "reweighted cs":cs_reweighted[j][i], "reweighted cs err":0.0,
                        "prior cs":cs_prior[j][i], "exp cs":cs_Exp[j][i], "label":cs_labels[j][i]
                        })
            data1 = pd.DataFrame(data)
            # NOTE: if you want to sort RDCs
            _data1 = data1.sort_values(["exp cs"])
            _data1 = _data1.reset_index()
            #_data1 = data1.copy()
            #print(_data1)
            ax_cs[j].scatter(x=_data1['label'].to_numpy(), y=_data1["prior cs"].to_numpy(), s=35, color="orange", label="Prior", edgecolor='black',)
            ax_cs[j].scatter(x=_data1['label'].to_numpy(), y=_data1["exp cs"].to_numpy(), s=25, color="r", label="Exp", edgecolor='black',)
            ax_cs[j].errorbar(x=_data1['label'].to_numpy(), y=_data1["reweighted cs"].to_numpy(),
                         yerr=_data1["reweighted cs err"].to_numpy(),
                         #capsize=25,
                         ecolor="b", label="BICePs", fmt="o",
                         mec='k', mfc='b')

            xticks = ax_cs[j].get_xticks()
            xtick_labels = ax_cs[j].get_xticklabels()
            stride = 1
            # Select every other tick label
            new_ticks = xticks[::stride]
            new_labels = xtick_labels[::stride]
            # Set new ticks and labels
            ax_cs[j].set_xticks(new_ticks)
            ax_cs[j].set_xticklabels(new_labels, rotation=45)

            #ax_cs[j].set_xlabel(r"Index", size=16)
            ax_cs[j].set_xlabel(r"", size=16)
            ax_cs[j].set_ylabel(r"Chemical Shift (ppm)", size=16)


except(Exception) as e:
    print(e)


try:
    val = [2+i for i in range(len(bvs))][-1]
    if all_data or rdc_only:
        ax_ = [plt.subplot(gs[1+val,0])]
        for j in range(len(rdc_reweighted)):
            data = []
            for i in range(len(rdc_reweighted[j])):
                try:
                    data.append({"index":i,
                        "reweighted rdc":rdc_reweighted[j][i].nominal_value, "reweighted rdc err":rdc_reweighted[j][i].std_dev,
                        "prior rdc":rdc_prior[j][i], "exp rdc":rdc_Exp[j][i], "label":rdc_labels[j][i]
                        })
                except(Exception) as e:
                    data.append({"index":i,
                        "reweighted rdc":rdc_reweighted[j][i], "reweighted rdc err":0.0,
                        "prior rdc":rdc_prior[j][i], "exp rdc":rdc_Exp[j][i], "label":rdc_labels[j][i]
                        })
            data1 = pd.DataFrame(data)
            # NOTE: if you want to sort RDCs
#            _data1 = data1.sort_values(["exp rdc"])
#            _data1 = _data1.reset_index()
            _data1 = data1.copy()
            #print(_data1)
            ax_[j].scatter(x=_data1['label'].to_numpy(), y=_data1["prior rdc"].to_numpy(), s=35, color="orange", label="Prior", edgecolor='black',)
            ax_[j].scatter(x=_data1['label'].to_numpy(), y=_data1["exp rdc"].to_numpy(), s=25, color="r", label="Exp", edgecolor='black',)
            ax_[j].errorbar(x=_data1['label'].to_numpy(), y=_data1["reweighted rdc"].to_numpy(),
                         yerr=_data1["reweighted rdc err"].to_numpy(),
                         #capsize=25,
                         ecolor="b", label="BICePs", fmt="o",
                         mec='k', mfc='b')

            xticks = ax_[j].get_xticks()
            xtick_labels = ax_[j].get_xticklabels()
            stride = 1
            # Select every other tick label
            new_ticks = xticks[::stride]
            new_labels = xtick_labels[::stride]
            # Set new ticks and labels
            ax_[j].set_xticks(new_ticks)
            ax_[j].set_xticklabels(new_labels, rotation=45)

            #ax_[j].set_xlim(-1, 140)
            #ax_[j].scatter(x=_data1.index.to_numpy(), y=_data1["reweighted rdc"].to_numpy(), s=25, color="b", label="BICePs", edgecolor='black')
            #ax_[j].legend(fontsize=14)
            #ax_[j].set_xlabel(r"Index", size=16)
            ax_[j].set_xlabel(r"", size=16)
            ax_[j].set_ylabel(r"RDC (Hz)", size=16)

except(Exception) as e:
    print(e)






try:
    axs = [ax1,ax2]+ax_cs+ax_
    rotations = [60,0,60]+[0]*len(axs)
except(Exception) as e:
    try:
        axs = [ax1,ax2]+ax_cs
    except(Exception) as e:
        if 'RUN6_8' in sys_name:
            axs = ax_cs
        else:
            axs = [ax1]

    rotations = [60,0,60]


if combine_cs:
    for k in range(2,len(rotations[2:])):
        rotations[k] = 60

for n, ax in enumerate(axs):

    ticks = [ax.xaxis.get_minor_ticks(),
             ax.xaxis.get_major_ticks(),]
    xmarks = [ax.get_xticklabels()]
    ymarks = [ax.get_yticklabels()]
    for k in range(0,len(ticks)):
        for tick in ticks[k]:
            tick.label.set_fontsize(14)
    for k in range(0,len(xmarks)):
        for mark in xmarks[k]:
            mark.set_size(fontsize=14)
            mark.set_rotation(s=rotations[n])
    for k in range(0,len(ymarks)):
        for mark in ymarks[k]:
            mark.set_size(fontsize=16)
            mark.set_rotation(s=0)

    ax.text(-0.1, 1.0, string.ascii_lowercase[n], transform=ax.transAxes,
            size=20, weight='bold')

fig.tight_layout()
fig.savefig(f"{outdir}/reweighted_observables.png")

print(mlp)

try:

    if all_data or noe_only or J_and_noe_only:
        #print("# BICePs Posterior average x=%5.3f" % (biceps_avg_post[0]))
        #print("BICePs chix %5.3f" % (np.abs((biceps_avg_post[0]-avgx_exp))/exp_sigma[0]))
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html
        chi2_exp = scipy.stats.chi2_contingency(np.stack([unumpy.nominal_values(noe_reweighted), noe_Exp], axis=1))
        chi2_prior = scipy.stats.chi2_contingency(np.stack([unumpy.nominal_values(noe_reweighted), noe_prior], axis=1))
        print("BICePs Vs Exp.  chi^2  %5.3f" % (chi2_exp[0]))
        print("BICePs Vs Prior chi^2  %5.3f" % (chi2_prior[0]))
        #sse = np.array([(noe_reweighted[i] - noe_Exp[i])**2/np.std([noe_reweighted[i],noe_Exp[i]]) for i in range(len(noe_Exp))]).sum()
        #sse = np.array([(noe_reweighted[i] - noe_Exp[i])**2/(mlp.iloc[0]["sigma_noe"]**2 + mlp.iloc[-1]["sigma_noe"]**2) for i in range(len(noe_Exp))]).sum()
        #print(sse)
        #exit()
        columns = [col for col in mlp.columns.to_list() if "sigma_noe" in col]
        sse = []
        for k,col in enumerate(columns):
            sse.append(np.array([(noe_reweighted[i] - noe_Exp[i]*mlp['gamma_noe:0'].to_numpy()[-1])**2/(mlp.iloc[0][col]**2 + mlp.iloc[-1][col]**2) for i in range(len(noe_Exp))]).sum())
        print(sse)


    if all_data or J_only or J_and_noe_only:
        chi2_exp = scipy.stats.chi2_contingency(np.stack([unumpy.nominal_values(J_reweighted), J_Exp], axis=1))
        chi2_prior = scipy.stats.chi2_contingency(np.stack([unumpy.nominal_values(J_reweighted), J_prior], axis=1))
        print("BICePs Vs Exp.  chi^2  %5.3f" % (chi2_exp[0]))
        print("BICePs Vs Prior chi^2  %5.3f" % (chi2_prior[0]))
        #sse = np.array([(J_reweighted[i] - J_Exp[i])**2/np.std([J_reweighted[i],J_Exp[i]]) for i in range(len(J_Exp))]).sum()
        #sse = np.array([(J_reweighted[i] - J_Exp[i])**2/(mlp.iloc[0]["sigma_J"]**2 + mlp.iloc[-1]["sigma_J"]**2) for i in range(len(J_Exp))]).sum()
        #print(sse)
        columns = [col for col in mlp.columns.to_list() if "sigma_J" in col]
        sse = []
        for k,col in enumerate(columns):
            sse.append(np.array([(J_reweighted[i] - J_Exp[i])**2/(mlp.iloc[0][col]**2 + mlp.iloc[-1][col]**2) for i in range(len(J_Exp))]).sum())
        print(sse)


    if all_data or cs_only:
        chi2_exp = scipy.stats.chi2_contingency(np.stack([unumpy.nominal_values(cs_reweighted), cs_Exp], axis=1))
        chi2_prior = scipy.stats.chi2_contingency(np.stack([unumpy.nominal_values(cs_reweighted), cs_prior], axis=1))
        print("BICePs Vs Exp.  chi^2  %5.3f" % (chi2_exp[0]))
        print("BICePs Vs Prior chi^2  %5.3f" % (chi2_prior[0]))
        #sse = np.array([(cs_reweighted[i] - cs_Exp[i])**2/np.std([cs_reweighted[i],cs_Exp[i]]) for i in range(len(cs_Exp))]).sum()
        #sse = np.array([(cs_reweighted[i] - cs_Exp[i])**2/(mlp.iloc[0]["sigma_H"]**2 + mlp.iloc[-1]["sigma_H"]**2) for i in range(len(cs_Exp))]).sum()
        #print(sse)
        columns = [col for col in mlp.columns.to_list() if "sigma_H" in col]
        sse = []
        for k,col in enumerate(columns):
            sse.append(np.array([(cs_reweighted[i] - cs_Exp[i])**2/(mlp.iloc[0][col]**2 + mlp.iloc[-1][col]**2) for i in range(len(cs_Exp))]).sum())
        print(sse)

except(Exception) as e:
    print(e)

#########################################
# FIXME:: need to include macrostates
#########################################

kwargs = {"nsteps": nsteps, "nStates": nstates, "n_lambdas": n_lambdas,
    "nreplicas": nreplicas, "Nd": Nd, "data_uncertainty": data_uncertainty,
    "stat_model": stat_model,"lambda_values": lambda_values, "dir": dir,
    "prior microstate populations": msm_pops.to_numpy(),
#    "prior macrostate populations": macrostate_populations[macrostate_populations.columns[0]].to_numpy(),
    "microstate populations": pops,
#    "macrostate populations": reweighted_macro_pops[reweighted_macro_pops.columns[0]].to_numpy(),
    "lambda_swap_every": swap_every,
    "prior": energies,
    "acceptance_info": sampler.acceptance_info,
    "mlp": mlp
    }

append_to_database(A, dbName=dbName, verbose=False, **kwargs)

#:}}}



# NOTE: IMPORTANT: QUESTION:
""" How should we define the folded macrostate?
    Should it be defined by the experimental data telling us what the folded state looks like?
    Should it be defined by clustering the simulation data? What if the simulation data provides unusual conformations of the "folded state"?
    How can be reconcile the microstates with the true macrostates?
"""




#
## backbone hbond distances and angles: {{{
#
#
#sys_pair = get_sys_pair(sys_name, system_dirs)
#sys_pair = os.path.join(main_dir, sys_pair)
#
#features1,features2 = [],[]
#sys_msm_pops = []
#for p,name in enumerate([sys_name, sys_pair]):
#    data_dir = name
#
#    # NOTE: Get microstate populations for system
#    file = f'{data_dir}/*clust*.csv'
#    file = biceps.toolbox.get_files(file)[0]
#    msm_pops = pd.read_csv(file, index_col=0, comment="#")
#    if nstates == 100:
#        msm_pops = msm_pops["populations"].to_numpy()
#    elif nstates == 500:
#        msm_pops = np.repeat(msm_pops["populations"].to_numpy(), 5)/5.
#    msm_pops = pd.DataFrame(msm_pops, columns=["populations"])
#    micro_pops = np.concatenate(msm_pops.to_numpy())
#    sys_msm_pops.append(micro_pops)
#
#
#    # NOTE: get indices for features
#    structure_file = biceps.toolbox.get_files(f"{name}/*cluster0_*.pdb")[0]
#    indices = special_pairs(structure_file, name.split("/")[-1], verbose=1)
#    print(indices)
#
#    t = md.load(structure_file)
#    topology = t.topology
#    table, bonds = topology.to_dataframe()
#
#    pairs = [table.iloc[idxs] for idxs in indices]
#    labels = []
#    for pair in pairs:
#        resName = pair["resName"].to_numpy()
#        seq = pair["resSeq"].to_numpy()
#        atom = pair["name"].to_numpy()
#        label = []
#        for i in range(len(resName)):
#            label.append(str(resName[i]) + str(seq[i]) + "." + str(atom[i]))
#        label = " - ".join(label)
#        labels.append(label)
#
#
#    files = biceps.toolbox.get_files(f"{name}/*cluster*.pdb")
#    snaps_per_state = len(biceps.toolbox.get_files(f"{name}/*cluster0_*.pdb"))
#    snapshots = biceps.toolbox.get_files(f"{name}/*.pdb")
#    #print(f"nstates = {nstates}; nsnapshots = {len(snapshots)}")
#    #print(f"snaps per state = {snaps_per_state};")
#
#    if nstates == 100:
#        files_by_state = [biceps.toolbox.get_files(f"{name}/*cluster{state}_*.pdb") for state in range(nstates)]
#    elif nstates == 500:
#        files_by_state = [[file] for file in biceps.toolbox.get_files(f"{name}/*cluster*_*.pdb")]
#    else:
#        print("nstates is not right")
#        exit()
#
#    compute = 1
#    hbond_indices = [0,1,3,4]
#    print("name = ",name.split("/")[-1])
#    if (name.split("/")[-1] == "RUN12_2") or (name.split("/")[-1] == "RUN13_1"):
#        hbond_indices = [0,1,3,4,5]
#    if compute:
#        all_distances,averaged_over_snaps,all_angles,all_hbs = [],[],[],[]
#        for i,state in enumerate(files_by_state):
#            distances = []
#            _all_angles = []
#            for j,frame in enumerate(state):
#                traj = md.load(frame)
#                d = md.compute_distances(traj, indices[hbond_indices])*10. # convert nm to Å
#                distances.append(d)
#
#                # NOTE: the baker_hubbard method uses distance and angle information
#                angle_indices, angles = get_theta_angles(traj, distance_cutoff=None)
#                angles = np.concatenate(angles)
#
#                # NOTE: locating the backbone hydrogen bond pairs
#                valid_indices,valid_angles = [],[]
#                for idx_pair in indices[hbond_indices]:
#                    temp = [[k,triplet] for k,triplet in enumerate(angle_indices) if all(idx in triplet for idx in idx_pair)]
#                    idx_mask = np.array([idx for idx, trip in temp])
#                    mask = np.array([trip for idx, trip in temp])
#                    if mask.size > 0:
#                        valid_indices.append(mask[0])
#                        valid_angles.append(angles[idx_mask[0]])
#
#                angle_indices = np.array(valid_indices)
#                angles = np.array(valid_angles)
#                _all_angles.append(angles)
#
#            distances = np.array(distances)
#            all_distances.append(distances)
#            data = np.mean(distances, axis=0)[0]
#            averaged_over_snaps.append(data)
#            all_angles.append(_all_angles)
##            print(_all_angles)
#            #exit()
#        #exit()
#        #averaged_over_snaps = np.array(averaged_over_snaps)
#        all_distances = np.array(all_distances)
#        all_angles = np.array(all_angles)
#        all_distances = all_distances.reshape((all_distances.shape[0], all_distances.shape[-1]))
#        all_distances = distances_to_intensities(all_distances)
#        all_angles = all_angles.reshape((all_angles.shape[0], all_angles.shape[-1]))
#        print(all_distances.shape)
#        print(all_angles.shape)
#        if p == 0:
#            features1.append(all_distances)
#            features1.append(all_angles)
#        if p == 1:
#            features2.append(all_distances)
#            features2.append(all_angles)
#
#features = np.concatenate([np.hstack(features1), np.hstack(features2)])
#sys_msm_pops = np.array(sys_msm_pops)
##print(features.shape)
#
#
#
#
#max_K = 10
#max_K = 20
#
#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA
#from sklearn.cluster import SpectralClustering
#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#
#
## Normalize the features
#scaler_2 = StandardScaler()
#features_2_normalized = scaler_2.fit_transform(features)
#
#
#
#
## find_optimal using silhouette:{{{
#
#from sklearn.metrics import silhouette_score
## Function to determine the optimal number of macrostates using Silhouette Score
#def find_optimal_spectral(features, max_components=10):
#    silhouette_scores = []
#    models = []
#    for n in range(2, max_components + 1):
#        spectral = SpectralClustering(n_clusters=n, affinity='nearest_neighbors', random_state=42)
#        labels = spectral.fit_predict(features)
#        score = silhouette_score(features, labels)
#        silhouette_scores.append(score)
#        models.append((spectral, labels))
#    optimal_index = np.argmax(silhouette_scores)
#    return models[optimal_index][0], models[optimal_index][1], optimal_index + 2, silhouette_scores
#
## Find the optimal Spectral Clustering model and number of macrostates
#spectral_model_2, labels_2, optimal_components_2, silhouette_scores_2 = find_optimal_spectral(features_2_normalized, max_K)
#print(labels_2)
#print(labels_2.shape)
#
#
## Plot Silhouette Score against the number of macrostates
#def plot_silhouette_scores(scores, title):
#    plt.figure(figsize=(10, 6))
#    plt.plot(range(2, len(scores) + 2), scores, marker='o')
#    plt.xlabel('Number of Macrostates')
#    plt.ylabel('Silhouette Score')
#    plt.title(title)
#    plt.savefig(f"{outdir}/K_macrostates_using_Silhouette_Score.png")
#
#plot_silhouette_scores(silhouette_scores_2, 'Silhouette Score for Distances to Intensities')
## }}}
#
#
#
#
## Reduce the dimensionality of the feature matrix to 2D for visualization
#pca_2 = PCA(n_components=2)
#reduced_features_2 = pca_2.fit_transform(features_2_normalized)
#
## Plotting function
#def plot_clusters(reduced_features, labels, n_components, title):
#    plt.figure(figsize=(10, 8))
#    for i in range(n_components):
#        cluster_points = reduced_features[labels == i]
#        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Macrostate {i}')
#    plt.xlabel('PCA Component 1')
#    plt.ylabel('PCA Component 2')
#    plt.title(title)
#    plt.legend()
#    #plt.scatter(reduced_features[:, 0][top_pop], reduced_features[:, 1][top_pop], marker="+", s=100, color="k", label=f'Macrostate {i}')
#    plt.savefig(f"{outdir}/clustered_into_macrostates.png")
#
## Plot the clusters for both transformations
#plot_clusters(reduced_features_2, labels_2, optimal_components_2, 'Clustering with Distances to Intensities')
#
#
#
## Calculate and print macrostate populations
#def calculate_macropops(labels, both_sys_pops):
#    macropops = np.zeros((2, np.concatenate(labels).max() + 1))
#    for sys in range(len(both_sys_pops)):
#        for i in range(len(both_sys_pops[sys])):
#            macropops[sys, labels[sys][i]] += both_sys_pops[sys,i]
#    return macropops
#
#def get_top_n_microstates(labels, both_sys_pops, N):
#    top_microstates = []
#
#    for sys in range(len(both_sys_pops)):
#        system_top_microstates = []
#        num_macrostates = np.concatenate(labels).max() + 1
#
#        for macrostate in range(num_macrostates):
#            # Get microstates and their populations for the current macrostate
#            microstate_indices = np.where(labels[sys] == macrostate)[0]
#            microstate_pops = both_sys_pops[sys][microstate_indices]
#
#            # Get the indices of the top N populated microstates
#            if len(microstate_pops) <= N:
#                top_indices = np.argsort(microstate_pops)[::-1]
#            else:
#                top_indices = np.argsort(microstate_pops)[-N:][::-1]
#
#            top_microstates_for_macrostate = microstate_indices[top_indices]
#            top_microstate_pops = microstate_pops[top_indices]
#            system_top_microstates.append((macrostate, top_microstates_for_macrostate, top_microstate_pops))
#
#        top_microstates.append(system_top_microstates)
#
#    return top_microstates
#
#print(labels_2)
#print()
#labels_new = np.array([labels_2[:len(sys_msm_pops[0])], labels_2[len(sys_msm_pops[0]):]])
#print(labels_new)
#print(labels_new.shape)
#print(sys_msm_pops.shape)
#
#
#
## NOTE: Find which microstates belong to which macrostates
#macrostates = np.sort(list(set(np.concatenate(labels_new))))
#macro_bins = [[[] for i in range(len(macrostates))]]*2
#
#for s in range(len(labels_new)):
#    for i in range(len(labels_new[s])):
#        macro_bins[s][labels_new[s][i]].append(i)
#
#for sys in range(len(macro_bins)):
#    print(f"System {sys}:")
#    for m, macro in enumerate(macro_bins[sys]):
#        print(f"{m}: {macro}")
#    print("\n\n")
#
##print(sys_msm_pops)
#
#macropops_2 = calculate_macropops(labels_new, sys_msm_pops)
#
#print(f"Optimal number of macrostates (distances_to_intensities): {optimal_components_2}")
#for sys in range(len(macropops_2)):
#    print(f"System {sys}:")
#    for k in range(len(macropops_2[sys])):
#        print(f"Macrostate {k} (distances_to_intensities): {macropops_2[sys][k]}")
#    print("\n\n")
##print("top pop: ",labels_new[0][top_pop])
#
#
#
#macropops_2 = calculate_macropops(np.array([labels_new[0]]), np.array([pops]))
#print(f"Optimal number of macrostates (distances_to_intensities): {optimal_components_2}")
#for k in range(len(macropops_2)):
#    print(f"Macrostate {k} (distances_to_intensities): {macropops_2[k]}")
#print("top pop: ",top_pop_files)
#print("top pop in macrostate: ",labels_new[0][top_pop])
#print([sys_name, sys_pair])
#
#
#results = get_top_n_microstates(labels_new, sys_msm_pops, N=5)
#print(results)
##_files_by_state
#
#for sys in range(len(results)):
#    system = results[sys]
#    if sys > 0: break
#    for m in range(len(system)):
#        print(f"System: {[sys_name, sys_pair][sys]}; Macrostate: {m}; top_pop_files: {_files_by_state[system[m][1][0]]}")
#
#
## }}}
#



# backbone hbond distances and angles: {{{


sys_pair = get_sys_pair(sys_name, system_dirs)
sys_pair = os.path.join(main_dir, sys_pair)

features1,features2 = [],[]
sys_msm_pops = []
for p,name in enumerate([sys_name, sys_pair]):
    data_dir = name

    # NOTE: Get microstate populations for system
    file = f'{data_dir}/*clust*.csv'
    file = biceps.toolbox.get_files(file)[0]
    msm_pops = pd.read_csv(file, index_col=0, comment="#")
    if nstates == 100:
        msm_pops = msm_pops["populations"].to_numpy()
    elif nstates == 500:
        msm_pops = np.repeat(msm_pops["populations"].to_numpy(), 5)/5.
    msm_pops = pd.DataFrame(msm_pops, columns=["populations"])
    micro_pops = np.concatenate(msm_pops.to_numpy())
    sys_msm_pops.append(micro_pops)


    # NOTE: get indices for features
    structure_file = biceps.toolbox.get_files(f"{name}/*cluster0_*.pdb")[0]
    indices = special_pairs(structure_file, name.split("/")[-1], verbose=1)
    print(indices)

    t = md.load(structure_file)
    topology = t.topology
    table, bonds = topology.to_dataframe()

    pairs = [table.iloc[idxs] for idxs in indices]
    labels = []
    for pair in pairs:
        resName = pair["resName"].to_numpy()
        seq = pair["resSeq"].to_numpy()
        atom = pair["name"].to_numpy()
        label = []
        for i in range(len(resName)):
            label.append(str(resName[i]) + str(seq[i]) + "." + str(atom[i]))
        label = " - ".join(label)
        labels.append(label)


    files = biceps.toolbox.get_files(f"{name}/*cluster*.pdb")
    snaps_per_state = len(biceps.toolbox.get_files(f"{name}/*cluster0_*.pdb"))
    snapshots = biceps.toolbox.get_files(f"{name}/*.pdb")
    #print(f"nstates = {nstates}; nsnapshots = {len(snapshots)}")
    #print(f"snaps per state = {snaps_per_state};")

    if nstates == 100:
        files_by_state = [biceps.toolbox.get_files(f"{name}/*cluster{state}_*.pdb") for state in range(nstates)]
    elif nstates == 500:
        files_by_state = [[file] for file in biceps.toolbox.get_files(f"{name}/*cluster*_*.pdb")]
    else:
        print("nstates is not right")
        exit()

    compute = 1
    hbond_indices = [0,1,3,4]
    print("name = ",name.split("/")[-1])
    if (name.split("/")[-1] == "RUN12_2") or (name.split("/")[-1] == "RUN13_1"):
        hbond_indices = [0,1,3,4,5]
    if compute:
        all_distances,averaged_over_snaps,all_angles,all_hbs = [],[],[],[]
        for i,state in enumerate(files_by_state):
            distances = []
            _all_angles = []
            for j,frame in enumerate(state):
                traj = md.load(frame)
                d = md.compute_distances(traj, indices[hbond_indices])*10. # convert nm to Å
                distances.append(d)

                # NOTE: the baker_hubbard method uses distance and angle information
                angle_indices, angles = get_theta_angles(traj, distance_cutoff=None)
                angles = np.concatenate(angles)

                # NOTE: locating the backbone hydrogen bond pairs
                valid_indices,valid_angles = [],[]
                for idx_pair in indices[hbond_indices]:
                    temp = [[k,triplet] for k,triplet in enumerate(angle_indices) if all(idx in triplet for idx in idx_pair)]
                    idx_mask = np.array([idx for idx, trip in temp])
                    mask = np.array([trip for idx, trip in temp])
                    if mask.size > 0:
                        valid_indices.append(mask[0])
                        valid_angles.append(angles[idx_mask[0]])

                angle_indices = np.array(valid_indices)
                angles = np.array(valid_angles)
                _all_angles.append(angles)

            distances = np.array(distances)
            all_distances.append(distances)
            data = np.mean(distances, axis=0)[0]
            averaged_over_snaps.append(data)
            all_angles.append(_all_angles)
            #exit()
        #exit()
        #averaged_over_snaps = np.array(averaged_over_snaps)
        all_distances = np.array(all_distances)
        all_angles = np.array(all_angles)
        all_distances = all_distances.reshape((all_distances.shape[0], all_distances.shape[-1]))

        #all_distances = distances_to_intensities(all_distances)
        all_distances = smooth_step_function(all_distances, k=2.75, x_off=3.0)

        all_angles = all_angles.reshape((all_angles.shape[0], all_angles.shape[-1]))
        print(all_distances.shape)
        print(all_angles.shape)
        if p == 0:
            features1.append(all_distances)
#            features1.append(all_angles)
#            features1.append(np.cos(all_angles))
        if p == 1:
            features2.append(all_distances)
#            features2.append(all_angles)
#            features2.append(np.cos(all_angles))

features = np.concatenate([np.hstack(features1), np.hstack(features2)])
sys_msm_pops = np.array(sys_msm_pops)
#print(features.shape)




max_K = 10
max_K = 20

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Normalize the features
scaler_2 = StandardScaler()
#features_2_normalized = scaler_2.fit_transform(features)

features_2_normalized = features



# find_optimal using silhouette:{{{

from sklearn.metrics import silhouette_score
# Function to determine the optimal number of macrostates using Silhouette Score
def find_optimal_spectral(features, max_components=10):
    silhouette_scores = []
    models = []
    for n in range(2, max_components + 1):
        spectral = SpectralClustering(n_clusters=n, affinity='nearest_neighbors', random_state=42)
        labels = spectral.fit_predict(features)
        score = silhouette_score(features, labels)
        silhouette_scores.append(score)
        models.append((spectral, labels))
    optimal_index = np.argmax(silhouette_scores)
    return models[optimal_index][0], models[optimal_index][1], optimal_index + 2, silhouette_scores

# Find the optimal Spectral Clustering model and number of macrostates
spectral_model_2, labels_2, optimal_components_2, silhouette_scores_2 = find_optimal_spectral(features_2_normalized, max_K)
print(labels_2)
print(labels_2.shape)


# Plot Silhouette Score against the number of macrostates
def plot_silhouette_scores(scores, title):
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, len(scores) + 2), scores, marker='o')
    plt.xlabel('Number of Macrostates')
    plt.ylabel('Silhouette Score')
    plt.title(title)
    plt.savefig(f"{outdir}/K_macrostates_using_Silhouette_Score.png")

plot_silhouette_scores(silhouette_scores_2, 'Silhouette Score for Distances to Intensities')
# }}}




# Reduce the dimensionality of the feature matrix to 2D for visualization
pca_2 = PCA(n_components=2)
reduced_features_2 = pca_2.fit_transform(features_2_normalized)

# Plotting function
def plot_clusters(reduced_features, labels, n_components, title):
    plt.figure(figsize=(10, 8))
    for i in range(n_components):
        cluster_points = reduced_features[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Macrostate {i}')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(title)
    plt.legend()
    #plt.scatter(reduced_features[:, 0][top_pop], reduced_features[:, 1][top_pop], marker="+", s=100, color="k", label=f'Macrostate {i}')
    plt.savefig(f"{outdir}/clustered_into_macrostates.png")

# Plot the clusters for both transformations
plot_clusters(reduced_features_2, labels_2, optimal_components_2, 'Clustering with Distances to Intensities')



# Calculate and print macrostate populations
def calculate_macropops(labels, both_sys_pops):
    macropops = np.zeros((2, np.concatenate(labels).max() + 1))
    for sys in range(len(both_sys_pops)):
        for i in range(len(both_sys_pops[sys])):
            macropops[sys, labels[sys][i]] += both_sys_pops[sys,i]
    return macropops

def get_top_n_microstates(labels, both_sys_pops, N):
    top_microstates = []

    for sys in range(len(both_sys_pops)):
        system_top_microstates = []
        num_macrostates = np.concatenate(labels).max() + 1

        for macrostate in range(num_macrostates):
            # Get microstates and their populations for the current macrostate
            microstate_indices = np.where(labels[sys] == macrostate)[0]
            microstate_pops = both_sys_pops[sys][microstate_indices]

            # Get the indices of the top N populated microstates
            if len(microstate_pops) <= N:
                top_indices = np.argsort(microstate_pops)[::-1]
            else:
                top_indices = np.argsort(microstate_pops)[-N:][::-1]

            top_microstates_for_macrostate = microstate_indices[top_indices]
            top_microstate_pops = microstate_pops[top_indices]
            system_top_microstates.append((macrostate, top_microstates_for_macrostate, top_microstate_pops))

        top_microstates.append(system_top_microstates)

    return top_microstates

print(labels_2)
print()
labels_new = np.array([labels_2[:len(sys_msm_pops[0])], labels_2[len(sys_msm_pops[0]):]])
print(labels_new)
print(labels_new.shape)
print(sys_msm_pops.shape)



# NOTE: Find which microstates belong to which macrostates
macrostates = np.sort(list(set(np.concatenate(labels_new))))
macro_bins = [[[] for i in range(len(macrostates))]]*2

for s in range(len(labels_new)):
    for i in range(len(labels_new[s])):
        macro_bins[s][labels_new[s][i]].append(i)

for sys in range(len(macro_bins)):
    print(f"System {sys}:")
    for m, macro in enumerate(macro_bins[sys]):
        print(f"{m}: {macro}")
    print("\n\n")

#print(sys_msm_pops)

macropops_2 = calculate_macropops(labels_new, sys_msm_pops)

print(f"Optimal number of macrostates (distances_to_intensities): {optimal_components_2}")
for sys in range(len(macropops_2)):
    print(f"System {sys}:")
    for k in range(len(macropops_2[sys])):
        print(f"Macrostate {k} (distances_to_intensities): {macropops_2[sys][k]}")
    print("\n\n")
#print("top pop: ",labels_new[0][top_pop])



macropops_2 = calculate_macropops(np.array([labels_new[0]]), np.array([pops]))
print(f"Optimal number of macrostates (distances_to_intensities): {optimal_components_2}")
for k in range(len(macropops_2)):
    print(f"Macrostate {k} (distances_to_intensities): {macropops_2[k]}")
print("top pop: ",top_pop_files)
print("top pop in macrostate: ",labels_new[0][top_pop])
print([sys_name, sys_pair])


results = get_top_n_microstates(labels_new, sys_msm_pops, N=5)
print(results)
#_files_by_state

for sys in range(len(results)):
    system = results[sys]
    if sys > 0: break
    for m in range(len(system)):
        print(f"System: {[sys_name, sys_pair][sys]}; Macrostate: {m}; top_pop_files: {_files_by_state[system[m][1][0]]}")


# }}}



exit()




#
## GMM: {{{
#
#from sklearn.mixture import GaussianMixture
#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA
#
## Load the model and extract J, CS, NOE data
#model = sampler.model
#options = pd.DataFrame(options)
#extensions = options["extension"].to_numpy()
#noe_index = [i for i in range(len(extensions)) if "noe" in extensions[i]][0]
#
## find the top populated state from BICePs (should be folded)
## NOTE: TODO: Check with the number of backbone hydrogen bonds to confirm that it is in fact the folded state
#ntop = 10
#topN_pops = pops[np.argsort(pops)[-ntop:]][::-1]
#topN_labels = [np.where(topN_pops[i]==pops)[0][0] for i in range(len(topN_pops))]
#top_pop = topN_labels[0]
#print(top_pop)
#
#
#def distances_to_intensities(d, sigma=1.0):
#    return sigma / d**6
#
#
#_data_ = []
#for k in range(len(model[0])):
#    _ = np.array([model[i][k] for i in range(len(model))])
#    if k == noe_index:
#        # transform the distances to intensities
#        _ = distances_to_intensities(_)
#    _data_.append(_)
#
## Combine J, CS, and NOE into a single feature matrix
#features_2 = np.hstack(_data_)
#
#
## Normalize the features
#scaler_2 = StandardScaler()
#features_2_normalized = scaler_2.fit_transform(features_2)
#
## Function to determine the optimal number of macrostates using BIC
#def find_optimal_gmm(features, max_components=10, covariance_type='full'):
#    bic = []
#    models = []
#    for n in range(1, max_components + 1):
#        gmm = GaussianMixture(n_components=n, covariance_type=covariance_type, random_state=42)
#        gmm.fit(features)
#        bic.append(gmm.bic(features))
#        models.append(gmm)
#    optimal_index = np.argmin(bic)
#    return models[optimal_index], optimal_index + 1, bic
#
## Find the optimal GMM model and number of macrostates
##gmm_1, optimal_components_1, bic_1 = find_optimal_gmm(features_1_normalized)
#gmm_2, optimal_components_2, bic_2 = find_optimal_gmm(features_2_normalized)
#
## Plot BIC against the number of macrostates
#def plot_bic(bic, title):
#    plt.figure(figsize=(10, 6))
#    plt.plot(range(1, len(bic) + 1), bic, marker='o')
#    plt.xlabel('Number of Macrostates')
#    plt.ylabel('BIC')
#    plt.title(title)
#    plt.savefig(f"{outdir}/K_macrostates_using_BIC.png")
#
#plot_bic(bic_2, 'BIC for Distances to Intensities')
#
## Predict the macrostate for each microstate
#labels_2 = gmm_2.predict(features_2_normalized)
#
### Reduce the dimensionality of the feature matrix to 2D for visualization
#pca_2 = PCA(n_components=2)
#reduced_features_2 = pca_2.fit_transform(features_2_normalized)
#
## Plotting function
#def plot_clusters(reduced_features, labels, n_components, title):
#    plt.figure(figsize=(10, 8))
#    for i in range(n_components):
#        cluster_points = reduced_features[labels == i]
#        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Macrostate {i}')
#    plt.xlabel('PCA Component 1')
#    plt.ylabel('PCA Component 2')
#    plt.title(title)
#    plt.legend()
#
#    plt.scatter(reduced_features[:, 0][top_pop], reduced_features[:, 1][top_pop], marker="+", s=100, color="k", label=f'Macrostate {i}')
#
#    plt.savefig(f"{outdir}/clustered_into_macrostates.png")
#
## Plot the clusters for both transformations
#plot_clusters(reduced_features_2, labels_2, optimal_components_2, 'Clustering with Distances to Intensities')
#
## Calculate and print macrostate populations
#def calculate_macropops(labels, pops):
#    macropops = np.zeros(labels.max() + 1)
#    for i, label in enumerate(labels):
#        macropops[label] += pops[i]
#    return macropops
#
#
## NOTE: Find which microstates belong to which macrostates;
#macrostates = np.sort(list(set(labels_2)))
#macro_bins = [[] for i in range(len(macrostates))]
#for i in range(len(labels_2)):
#    macro_bins[labels_2[i]].append(i)
#
#for m,macro in enumerate(macro_bins):
#    print(f"{m}: {macro}")
#
#
#
#
#
#msm_pops = np.concatenate(msm_pops.to_numpy())
#macropops_2 = calculate_macropops(labels_2, msm_pops)
#
#print(f"Optimal number of macrostates (distances_to_intensities): {optimal_components_2}")
#for k in range(len(macropops_2)):
#    print(f"Macrostate {k} (distances_to_intensities): {macropops_2[k]}")
#print(labels_2[top_pop])
#
#
#
#macropops_2 = calculate_macropops(labels_2, pops)
#print(f"Optimal number of macrostates (distances_to_intensities): {optimal_components_2}")
#for k in range(len(macropops_2)):
#    print(f"Macrostate {k} (distances_to_intensities): {macropops_2[k]}")
#print(labels_2[top_pop])
## }}}
#


# Spectral clustering: {{{

max_K = 10
max_K = 20
#max_K = 2

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the model and extract J, CS, NOE data
model = sampler.model
options = pd.DataFrame(options)
extensions = options["extension"].to_numpy()
noe_index = [i for i in range(len(extensions)) if "noe" in extensions[i]]
if noe_index != []:
    noe_index = noe_index[0]
else:
    noe_index = -1

# Find the top populated state from BICePs (should be folded)
ntop = 10
topN_pops = pops[np.argsort(pops)[-ntop:]][::-1]
topN_labels = [np.where(topN_pops[i] == pops)[0][0] for i in range(len(topN_pops))]
top_pop = topN_labels[0]
print(top_pop)

def distances_to_intensities(d, sigma=1.0):
    return sigma / d**6

_data_ = []
for k in range(len(model[0])):
    _ = np.array([model[i][k] for i in range(len(model))])
    if k == noe_index:
        # transform the distances to intensities
        _ = distances_to_intensities(_)
    _data_.append(_)

# Combine J, CS, and NOE into a single feature matrix
features_2 = np.hstack(_data_)

# Normalize the features
scaler_2 = StandardScaler()
features_2_normalized = scaler_2.fit_transform(features_2)

## find_optimal:{{{
#
## Function to determine the optimal number of macrostates using Spectral Clustering
#def find_optimal_spectral(features, max_components=10):
#    inertia = []
#    models = []
#    for n in range(2, max_components + 1):
#        spectral = SpectralClustering(n_clusters=n, affinity='nearest_neighbors', random_state=42)
#        labels = spectral.fit_predict(features)
#        models.append((spectral, labels))
#        inertia.append(np.sum(np.min(np.linalg.norm(features[:, np.newaxis] - features[labels], axis=2), axis=1)))
#    optimal_index = np.argmin(inertia)
#    return models[optimal_index][0], models[optimal_index][1], optimal_index + 1, inertia
#
## Find the optimal Spectral Clustering model and number of macrostates
#spectral_model_2, labels_2, optimal_components_2, inertia_2 = find_optimal_spectral(features_2_normalized, max_K)
#
## Plot Inertia against the number of macrostates
#def plot_inertia(inertia, title):
#    plt.figure(figsize=(10, 6))
#    plt.plot(range(2, len(inertia) + 2), inertia, marker='o')
#    plt.xlabel('Number of Macrostates')
#    plt.ylabel('Inertia')
#    plt.title(title)
#    plt.savefig(f"{outdir}/K_macrostates_using_Inertia.png")
#
#plot_inertia(inertia_2, 'Inertia for Distances to Intensities')
## }}}
#
# find_optimal using silhouette:{{{

from sklearn.metrics import silhouette_score
# Function to determine the optimal number of macrostates using Silhouette Score
def find_optimal_spectral(features, max_components=10):
    silhouette_scores = []
    models = []
    for n in range(2, max_components + 1):
        spectral = SpectralClustering(n_clusters=n, affinity='nearest_neighbors', random_state=42)
        labels = spectral.fit_predict(features)
        score = silhouette_score(features, labels)
        silhouette_scores.append(score)
        models.append((spectral, labels))
    optimal_index = np.argmax(silhouette_scores)
    return models[optimal_index][0], models[optimal_index][1], optimal_index + 2, silhouette_scores

# Find the optimal Spectral Clustering model and number of macrostates
spectral_model_2, labels_2, optimal_components_2, silhouette_scores_2 = find_optimal_spectral(features_2_normalized, max_K)

# Plot Silhouette Score against the number of macrostates
def plot_silhouette_scores(scores, title):
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, len(scores) + 2), scores, marker='o')
    plt.xlabel('Number of Macrostates')
    plt.ylabel('Silhouette Score')
    plt.title(title)
    plt.savefig(f"{outdir}/K_macrostates_using_Silhouette_Score.png")

plot_silhouette_scores(silhouette_scores_2, 'Silhouette Score for Distances to Intensities')
# }}}




# Reduce the dimensionality of the feature matrix to 2D for visualization
pca_2 = PCA(n_components=2)
reduced_features_2 = pca_2.fit_transform(features_2_normalized)

# Plotting function
def plot_clusters(reduced_features, labels, n_components, title):
    plt.figure(figsize=(10, 8))
    for i in range(n_components):
        cluster_points = reduced_features[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Macrostate {i}')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(title)
    plt.legend()
    plt.scatter(reduced_features[:, 0][top_pop], reduced_features[:, 1][top_pop], marker="+", s=100, color="k", label=f'Macrostate {i}')
    plt.savefig(f"{outdir}/clustered_into_macrostates.png")

# Plot the clusters for both transformations
plot_clusters(reduced_features_2, labels_2, optimal_components_2, 'Clustering with Distances to Intensities')

# Calculate and print macrostate populations
def calculate_macropops(labels, pops):
    macropops = np.zeros(labels.max() + 1)
    for i, label in enumerate(labels):
        macropops[label] += pops[i]
    return macropops

def get_top_n_microstates(labels, pops, N):
    top_microstates = []
    num_macrostates = max(labels) + 1

    for macrostate in range(num_macrostates):
        # Get microstates and their populations for the current macrostate
        microstate_indices = np.where(labels == macrostate)[0]
        microstate_pops = pops[microstate_indices]

        # Get the indices of the top N populated microstates
        if len(microstate_pops) <= N:
            top_indices = np.argsort(microstate_pops)[::-1]
        else:
            top_indices = np.argsort(microstate_pops)[-N:][::-1]

        top_microstates_for_macrostate = microstate_indices[top_indices]
        top_microstate_pops = microstate_pops[top_indices]
        top_microstates.append((macrostate, top_microstates_for_macrostate, top_microstate_pops))

    return top_microstates


# NOTE: Find which microstates belong to which macrostates
macrostates = np.sort(list(set(labels_2)))
macro_bins = [[] for i in range(len(macrostates))]
for i in range(len(labels_2)):
    macro_bins[labels_2[i]].append(i)

for m, macro in enumerate(macro_bins):
    print(f"{m}: {macro}")

msm_pops = np.concatenate(msm_pops.to_numpy())
macropops_2 = calculate_macropops(labels_2, msm_pops)

print("Prior results:")
print(f"Optimal number of macrostates (distances_to_intensities): {optimal_components_2}")
for k in range(len(macropops_2)):
    print(f"Macrostate {k} (distances_to_intensities): {macropops_2[k]}")
#print("top pop: ",labels_2[top_pop])

print("BICePs reweighted results:")
macropops_2 = calculate_macropops(labels_2, pops)
print(f"Optimal number of macrostates (distances_to_intensities): {optimal_components_2}")
for k in range(len(macropops_2)):
    print(f"Macrostate {k} (distances_to_intensities): {macropops_2[k]}")
print("top pop: ",top_pop)
print("top pop: ",top_pop_files)
print("top pop in macrostate: ",labels_2[top_pop])

results = get_top_n_microstates(labels_2, msm_pops, N=5)
print(results)

for m in range(len(results)):
    macrostate = results[m]
    print(f"Macrostate: {m}; top_pop_files: {_files_by_state[results[m][1][0]]}")




# }}}


exit()








