

# Libraries:{{{
#import sys, os
from Bio import SeqUtils
import biceps
import mdtraj as md
import numpy as np
import pandas as pd
import os, re, string
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
from tqdm import tqdm # progress bar
from itertools import combinations
import subprocess
import networkx as nx
import matplotlib.pyplot as plt
#from scipy import stats
import biceps
#from biceps.decorators import multiprocess
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


import seaborn as sns
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

    elif system == "RUN3_2b":
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
                topology.select('ersidue 9 and resname SER and name HG')[0]]
        H1 = [topology.select('residue 10 and resname VAL and name O')[0],
              topology.select('residue 3 and resname GLN and name H')[0]]
        H2 = [topology.select('residue 10 and resname VAL and name H')[0],
              topology.select('residue 3 and resname GLN and name O')[0]]
        H3 = [topology.select('residue 8 and resname ALA and name O')[0],
              topology.select('residue 5 and resname VAL and name H')[0]]
        H4 = [topology.select('residue 8 and resname ALA and name H')[0],
              topology.select('residue 5 and resname VAL and name O')[0]]

    elif system == "RUN5_4a":
        # NOTE: RUN5_4a
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

    elif system == "RUN6_4b":
        Hal1 = [topology.select('residue 3 and resname HSM and name O2'),
                topology.select('residue 8 and resname ABU and name CG')]
        H1 = [topology.select('residue 9 and resname GLN and name O')[0],
              topology.select('residue 2 and resname ALA and name H')[0]]
        H2 = [topology.select('residue 9 and resname GLN and name H')[0],
              topology.select('residue 2 and resname ALA and name O')[0]]
        H3 = [topology.select('residue 7 and resname ALA and name O')[0],
              topology.select('residue 4 and resname VAL and name H')[0]]
        H4 = [topology.select('residue 7 and resname ALA and name H')[0],
              topology.select('residue 4 and resname VAL and name O')[0]]

    elif system == "RUN7_5":
        Hal1 = [topology.select('resiname 3 and resname SME and name O2')[0],
                topology.select('resiname 8 and resname ALC and name CL1')[0]]
        H1 = [topology.select('residue 9 and resname VAL and name O')[0],
              topology.select('residue 2 and resname GLN and name H')[0]]
        H2 = [topology.select('residue 9 and resname VAL and name H')[0],
              topology.select('residue 2 and resname GLN and name O')[0]]
        H3 = [topology.select('residue 7 and resname ALA and name O')[0],
              topology.select('residue 4 and resname VAL and name H')[0]]
        H4 = [topology.select('residue 7 and resname ALA and name H')[0],
              topology.select('residue 4 and resname VAL and name O')[0]]

    elif system == "RUN8_8":
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

    else: raise ValueError("'system' is not one of ['RUN0_1a', 'RUN1_1b',...]")

    pairs = np.array([H1, H2, Hal1, H3, H4])
    if verbose:
        for pair in pairs: print(table.iloc[pair])

    return pairs
# }}}



if __name__ == "__main__":

    sys = 'RUN5_4a'
    #sys = 'RUN2_3b'
    #stat_model, data_uncertainty = "GB", "single"
    stat_model, data_uncertainty = "Students", "single"
    #nreplica = 8
    nreplica = 128
    #nreplica = 32

    file1 = biceps.toolbox.get_files(f"results/Systems/{sys}/{stat_model}_{data_uncertainty}_sigma/10*{nreplica}*/populations.dat")[0]
    file2 = biceps.toolbox.get_files(f"results/Systems/{sys}/{stat_model}_{data_uncertainty}_sigma/10*{nreplica}*/populations.dat")[0]
    file3 = biceps.toolbox.get_files(f"results/Systems/{sys}/{stat_model}_{data_uncertainty}_sigma/10*{nreplica}*/populations.dat")[0]

    #file1 = biceps.toolbox.get_files("results/Systems/RUN5_4a/Students_single_sigma/NOE_only*/populations.dat")[0]
    #file2 = biceps.toolbox.get_files("results/Systems/RUN5_4a/Students_single_sigma/NOE_and_J*/populations.dat")[0]
    #file3 = biceps.toolbox.get_files("results/Systems/RUN5_4a/Students_single_sigma/NOE_J_and_cs_Ha*/populations.dat")[0]
    pops1 = np.loadtxt(file1)[:,1]
    pops2 = np.loadtxt(file2)[:,1]
    pops3 = np.loadtxt(file3)[:,1]


    #############################################################################
    skip_dirs = ["biceps", "experimental_data"]
    main_dir = "./Systems"
    system_dirs = next(os.walk(main_dir))[1]
    system_dirs = [ff for ff in system_dirs if ff not in skip_dirs]
    system_dirs = biceps.toolbox.natsorted(system_dirs)


    # NOTE: Locate the 4 important backbone hydrogen bonds + backbone halogen bond
    for _sys_name in system_dirs:
        if _sys_name != sys: continue

        sys_name = os.path.join(main_dir,_sys_name)
        structure_file = biceps.toolbox.get_files(f"{sys_name}/*cluster0_*.pdb")[0]
        indices = special_pairs(structure_file, _sys_name, verbose=1)
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
            #labels.append(label)

            label = " - ".join(label)
            labels.append(label)

#        exit()



        files = biceps.toolbox.get_files(f"{sys_name}/*cluster*.pdb")

        nstates = int(biceps.toolbox.get_files(f"{sys_name}/*.pdb")[-1].split("cluster")[-1].split("_")[0])+1
        snaps_per_state = len(biceps.toolbox.get_files(f"{sys_name}/*cluster0_*.pdb"))
        snapshots = biceps.toolbox.get_files(f"{sys_name}/*.pdb")
        print(f"nstates = {nstates}; nsnapshots = {len(snapshots)}")
        print(f"snaps per state = {snaps_per_state};")
        files_by_state = [biceps.toolbox.get_files(
            f"{sys_name}/*cluster{state}_*.pdb"
            ) for state in range(nstates)]

        # get prior populations
        file = f'{sys_name}/*clust*_{nstates}*.csv'
        file = biceps.toolbox.get_files(file)[0]
        msm_pops = pd.read_csv(file, index_col=0, comment="#")
        #msm_pops[" population(%) "] = msm_pops[" population(%) "].to_numpy()/100
        msm_pops = msm_pops["populations"].to_numpy()


        compute = 1
        if compute:
            all_data = []
            averaged_over_snaps = []
            for i,state in enumerate(files_by_state):
                distances = []
                for j,frame in enumerate(state):
                    d = md.compute_distances(md.load(frame), indices)*10. # convert nm to Å
                    distances.append(d)
                all_data.append(np.array(distances))
                data = np.mean(np.array(distances), axis=0)[0]
                averaged_over_snaps.append(data)
            all_data = np.array(all_data)
            averaged_over_snaps = np.array(averaged_over_snaps)

            np.save("all_data.npy", all_data)
            np.save("avg_data.npy", averaged_over_snaps)

        all_data = np.load("all_data.npy")
        averaged_over_snaps = np.load("avg_data.npy")

        avg_prior = np.array([w*averaged_over_snaps[i,:] for i,w in enumerate(msm_pops)]).sum(axis=0)
        avg1 = np.array([w*averaged_over_snaps[i,:] for i,w in enumerate(pops1)]).sum(axis=0)
        avg2 = np.array([w*averaged_over_snaps[i,:] for i,w in enumerate(pops2)]).sum(axis=0)
        avg3 = np.array([w*averaged_over_snaps[i,:] for i,w in enumerate(pops3)]).sum(axis=0)

        _all_data = all_data.T
        _all_data = np.concatenate(_all_data, axis=0)
        #print(_all_data.shape)

        print(avg_prior)
        print(avg1)
        print(avg2)
        print(avg3)

        plot = 0
        if plot:
            fig = plt.figure(figsize=(20,6))
            gs = gridspec.GridSpec(1, len(avg_prior))
            for i in range(len(avg_prior)):
                ax = plt.subplot(gs[0,i])
                data = np.concatenate(_all_data[i])
                ax.hist(data, alpha=0.5, bins=25,
                    edgecolor='black', linewidth=1.2, color="b")
                ax.axvline(x=avg_prior[i], ymin=0, ymax=100, color="orange", lw=4)
                ax.axvline(x=avg1[i], ymin=0, ymax=100, color="green", lw=4)
                ax.axvline(x=avg2[i], ymin=0, ymax=100, color="gray", lw=4)
                ax.axvline(x=avg3[i], ymin=0, ymax=100, color="red", lw=4)
                ax.set_xlabel(f"{labels[i]}"+r" ($\AA$)", size=16)
                ax.set_ylabel("")

                ticks = [ax.xaxis.get_minor_ticks(),
                         ax.xaxis.get_major_ticks(),
                         ax.yaxis.get_minor_ticks(),
                         ax.yaxis.get_major_ticks()]
                xmarks = [ax.get_xticklabels()]
                ymarks = [ax.get_yticklabels()]
                for k in range(0,len(ticks)):
                    for tick in ticks[k]:
                        tick.label.set_fontsize(14)
                ax.text(-0.1, 1.0, string.ascii_lowercase[i], transform=ax.transAxes,
                    size=20, weight='bold')

            fig.tight_layout()
            fig.savefig(f"{_sys_name}_distance_distributions.png", dpi=400)


        ########################################################################
        ########################################################################
        ########################################################################


        from scipy.stats import gaussian_kde
        # IMPORTANT: using averaged data
        from sklearn.cluster import KMeans
        distances_data = averaged_over_snaps

        # Perform K-means clustering for distance data
        kmeans_distances = KMeans(n_clusters=2, random_state=0).fit(distances_data)
        labels_distances = kmeans_distances.labels_

        # NOTE: Using prior as weights. Compute populations for each state in phi/psi data and distances data
        prior_pops_distances = np.bincount(labels_distances, weights=msm_pops)
        print(prior_pops_distances)

        # NOTE: Using BICeps weights. Compute populations for each state in phi/psi data and distances data
        reweighted_pops_distances = np.bincount(labels_distances, weights=pops2)
        print(reweighted_pops_distances)

        # Assign cluster indices based on the populations (largest cluster is alpha)
        folded_cluster_index_distances = np.argmax(prior_pops_distances)
        unfolded_cluster_index_distances = 1 - folded_cluster_index_distances

        # Get the indices for the alpha and unfolded states
        folded_indices_distances = np.where(labels_distances == folded_cluster_index_distances)[0]
        unfolded_indices_distances = np.where(labels_distances == unfolded_cluster_index_distances)[0]
        print(folded_indices_distances)
        print(unfolded_indices_distances)


        plot = 1
        plot_with_histograms = 0
        if plot:
            fig = plt.figure(figsize=(20,6))
            gs = gridspec.GridSpec(1, len(avg_prior))

            for i in range(len(avg_prior)):
                ax = plt.subplot(gs[0,i])

                # Separate the data for alpha and unfolded states for each distance
                folded_data = distances_data[:,i][folded_indices_distances]
                unfolded_data = distances_data[:,i][unfolded_indices_distances]

                # Plot histograms for alpha and unfolded states
                if plot_with_histograms: ax.hist(folded_data.flatten(), alpha=0.5, bins=25, color='red', edgecolor='black', linewidth=1.2, label='Folded', density=True)
                if plot_with_histograms: ax.hist(unfolded_data.flatten(), alpha=0.5, bins=25, color='blue', edgecolor='black', linewidth=1.2, label='Unfolded', density=True)


                # Create KDEs for alpha and unfolded states using the weights
                X = distances_data[:,i]
                #width = 0.5
                #kde_prior = gaussian_kde(X, weights=msm_pops)
                #kde_prior.set_bandwidth(bw_method=width)
                #kde_post = gaussian_kde(X, weights=pops2)
                #kde_post.set_bandwidth(bw_method=width)

                from sklearn.model_selection import KFold


                def kde_cv_score(X, weights, bandwidth, k=5):
                    kf = KFold(n_splits=k)
                    scores = []

                    for train_index, test_index in kf.split(X):
                        X_train, X_test = X[train_index], X[test_index]
                        weights_train = weights[train_index]

                        kde = gaussian_kde(X_train, weights=weights_train, bw_method=bandwidth)
                        score = np.sum(kde.logpdf(X_test))
                        scores.append(score)

                    return np.mean(scores)

                bandwidths = np.linspace(0.1, 5.0, 10)  # Adjust this range based on your data
                best_bandwidth_prior = None
                best_score_prior = -np.inf

                for width in bandwidths:
                    score = kde_cv_score(X, msm_pops, width)
                    if score > best_score_prior:
                        best_score_prior = score
                        best_bandwidth_prior = width

                print("Best bandwidth for kde_prior:", best_bandwidth_prior)
                kde_prior = gaussian_kde(X, weights=msm_pops)
                kde_prior.set_bandwidth(bw_method=best_bandwidth_prior)

                # Repeat for kde_post
                best_bandwidth_post = None
                best_score_post = -np.inf

                for width in bandwidths:
                    score = kde_cv_score(X, pops2, width)
                    if score > best_score_post:
                        best_score_post = score
                        best_bandwidth_post = width

                print("Best bandwidth for kde_post:", best_bandwidth_post)
                kde_post = gaussian_kde(X, weights=pops2)
                kde_post.set_bandwidth(bw_method=best_bandwidth_post)



                # Generate a range of values for plotting the KDE
                xmin, xmax = X.min(), X.max()
                x = np.linspace(xmin, xmax, 1000)
                ax.set_xlim(xmin, xmax)

                ax.plot(x, kde_prior(x), color='Orange', label='Prior')
                ax.plot(x, kde_post(x), color='Green', label='Reweighted')



                # Add vertical lines and labels as before
                ax.axvline(x=avg_prior[i], ymin=0, ymax=100, color="orange", lw=4)
                #ax.axvline(x=avg1[i], ymin=0, ymax=100, color="green", lw=4)
                ax.axvline(x=avg2[i], ymin=0, ymax=100, color="green", lw=4)
                #ax.axvline(x=avg3[i], ymin=0, ymax=100, color="red", lw=4)
                ax.set_xlabel(f"{labels[i]}"+r" ($\AA$)", size=16)
                ax.set_ylabel("")

                # Set tick labels size
                ticks = [ax.xaxis.get_minor_ticks(), ax.xaxis.get_major_ticks(), ax.yaxis.get_minor_ticks(), ax.yaxis.get_major_ticks()]
                for tick in ticks:
                    for subtick in tick:
                        subtick.label.set_fontsize(14)

                # Add subplot labels
                ax.text(-0.1, 1.0, string.ascii_lowercase[i], transform=ax.transAxes, size=20, weight='bold')

                # Add legend
                ax.legend()

            fig.tight_layout()
            fig.savefig(f"{_sys_name}_avg_distance_distributions_kernel_{nreplica}_replica.png", dpi=400)



        exit()















## old plots and tests:{{{
#
#
#
#        ########################################################################
#
#        # IMPORTANT: using all snapshots with kernel
#        from sklearn.cluster import KMeans
#        distances_data = np.concatenate(_all_data, axis=1)
#        distances_data = distances_data.T
#
#        # Perform K-means clustering for distance data
#        kmeans_distances = KMeans(n_clusters=2, random_state=0).fit(distances_data)
#        labels_distances = kmeans_distances.labels_
#
#        # NOTE: Using prior as weights. Compute populations for each state in phi/psi data and distances data
#        prior_weights = np.repeat(msm_pops,_all_data.shape[1]) / _all_data.shape[1]
#        prior_pops_distances = np.bincount(labels_distances, weights=prior_weights)
#        print(prior_pops_distances)
#
#        # NOTE: Using BICeps weights. Compute populations for each state in phi/psi data and distances data
#        post_weights = np.repeat(pops2,_all_data.shape[1]) / _all_data.shape[1]
#        reweighted_pops_distances = np.bincount(labels_distances, weights=post_weights)
#        print(reweighted_pops_distances)
#
#        # Assign cluster indices based on the populations (largest cluster is alpha)
#        folded_cluster_index_distances = np.argmax(prior_pops_distances)
#        unfolded_cluster_index_distances = 1 - folded_cluster_index_distances
#
#        # Get the indices for the alpha and unfolded states
#        folded_indices_distances = np.where(labels_distances == folded_cluster_index_distances)[0]
#        unfolded_indices_distances = np.where(labels_distances == unfolded_cluster_index_distances)[0]
#        print(folded_indices_distances)
#        print(unfolded_indices_distances)
#
#
#        plot = 1
#        if plot:
#            fig = plt.figure(figsize=(20,6))
#            gs = gridspec.GridSpec(1, len(avg_prior))
#
#            for i in range(len(avg_prior)):
#                ax = plt.subplot(gs[0,i])
#
#                # Separate the data for alpha and unfolded states for each distance
#                folded_data = np.concatenate(_all_data[i])[folded_indices_distances]
#                unfolded_data = np.concatenate(_all_data[i])[unfolded_indices_distances]
#
#                ## Plot KDE for alpha and unfolded states instead of histograms
#                #sns.kdeplot(folded_data.flatten(), ax=ax, color='red', label='Folded')
#                #sns.kdeplot(unfolded_data.flatten(), ax=ax, color='blue', label='Unfolded')
#
#
#                # Plot histograms for alpha and unfolded states
#                ax.hist(folded_data.flatten(), alpha=0.5, bins=25, color='red', edgecolor='black', linewidth=1.2, label='Folded', density=True)
#                ax.hist(unfolded_data.flatten(), alpha=0.5, bins=25, color='blue', edgecolor='black', linewidth=1.2, label='Unfolded', density=True)
#
#
#                # Create KDEs for alpha and unfolded states using the weights
#                X = np.concatenate(_all_data[i])
#                kde_prior = gaussian_kde(X, weights=prior_weights)
#                kde_post = gaussian_kde(X, weights=post_weights)
#
#                # Generate a range of values for plotting the KDE
#                xmin, xmax = X.min(), X.max()
#                x = np.linspace(xmin, xmax, 1000)
#
#                ax.plot(x, kde_prior(x), color='Orange', label='Prior')
#                ax.plot(x, kde_post(x), color='Green', label='Reweighted')
#
#
#                # Add vertical lines and labels as before
#                ax.axvline(x=avg_prior[i], ymin=0, ymax=100, color="orange", lw=4)
#                ax.axvline(x=avg1[i], ymin=0, ymax=100, color="green", lw=4)
#                ax.axvline(x=avg2[i], ymin=0, ymax=100, color="gray", lw=4)
#                ax.axvline(x=avg3[i], ymin=0, ymax=100, color="red", lw=4)
#                ax.set_xlabel(f"{labels[i]}"+r" ($\AA$)", size=16)
#                ax.set_ylabel("")
#
#                # Set tick labels size
#                ticks = [ax.xaxis.get_minor_ticks(), ax.xaxis.get_major_ticks(), ax.yaxis.get_minor_ticks(), ax.yaxis.get_major_ticks()]
#                for tick in ticks:
#                    for subtick in tick:
#                        subtick.label.set_fontsize(14)
#
#                # Add subplot labels
#                ax.text(-0.1, 1.0, string.ascii_lowercase[i], transform=ax.transAxes, size=20, weight='bold')
#
#                # Add legend
#                ax.legend()
#
#            fig.tight_layout()
#            fig.savefig(f"{_sys_name}_distance_distributions_kernel.png", dpi=400)
#        exit()
#
#
#        ########################################################################
#
#
#        # IMPORTANT: using averaged data
#        from sklearn.cluster import KMeans
#        distances_data = averaged_over_snaps
#
#        # Perform K-means clustering for distance data
#        kmeans_distances = KMeans(n_clusters=2, random_state=0).fit(distances_data)
#        labels_distances = kmeans_distances.labels_
#
#        # NOTE: Using prior as weights. Compute populations for each state in phi/psi data and distances data
#        prior_pops_distances = np.bincount(labels_distances, weights=msm_pops)
#        print(prior_pops_distances)
#
#        # NOTE: Using BICeps weights. Compute populations for each state in phi/psi data and distances data
#        reweighted_pops_distances = np.bincount(labels_distances, weights=pops2)
#        print(reweighted_pops_distances)
#
#        # Assign cluster indices based on the populations (largest cluster is alpha)
#        folded_cluster_index_distances = np.argmax(prior_pops_distances)
#        unfolded_cluster_index_distances = 1 - folded_cluster_index_distances
#
#        # Get the indices for the alpha and unfolded states
#        folded_indices_distances = np.where(labels_distances == folded_cluster_index_distances)[0]
#        unfolded_indices_distances = np.where(labels_distances == unfolded_cluster_index_distances)[0]
#        print(folded_indices_distances)
#        print(unfolded_indices_distances)
#
#
#        plot = 1
#        if plot:
#            fig = plt.figure(figsize=(20,6))
#            gs = gridspec.GridSpec(1, len(avg_prior))
#
#            for i in range(len(avg_prior)):
#                ax = plt.subplot(gs[0,i])
#
#                # Separate the data for alpha and unfolded states for each distance
#                folded_data = distances_data[:,i][folded_indices_distances]
#                unfolded_data = distances_data[:,i][unfolded_indices_distances]
#
#                # Plot histograms for alpha and unfolded states
#                ax.hist(folded_data.flatten(), alpha=0.5, bins=25, color='red', edgecolor='black', linewidth=1.2, label='Folded')
#                ax.hist(unfolded_data.flatten(), alpha=0.5, bins=25, color='blue', edgecolor='black', linewidth=1.2, label='Unfolded')
#
#                # Add vertical lines and labels as before
#                ax.axvline(x=avg_prior[i], ymin=0, ymax=100, color="orange", lw=4)
#                ax.axvline(x=avg1[i], ymin=0, ymax=100, color="green", lw=4)
#                ax.axvline(x=avg2[i], ymin=0, ymax=100, color="gray", lw=4)
#                ax.axvline(x=avg3[i], ymin=0, ymax=100, color="red", lw=4)
#                ax.set_xlabel(f"{labels[i]}"+r" ($\AA$)", size=16)
#                ax.set_ylabel("")
#
#                # Set tick labels size
#                ticks = [ax.xaxis.get_minor_ticks(), ax.xaxis.get_major_ticks(), ax.yaxis.get_minor_ticks(), ax.yaxis.get_major_ticks()]
#                for tick in ticks:
#                    for subtick in tick:
#                        subtick.label.set_fontsize(14)
#
#                # Add subplot labels
#                ax.text(-0.1, 1.0, string.ascii_lowercase[i], transform=ax.transAxes, size=20, weight='bold')
#
#                # Add legend
#                ax.legend()
#
#            fig.tight_layout()
#            fig.savefig(f"{_sys_name}_avg_distance_distributions_assigned.png", dpi=400)
#        #exit()
#
#
#
#
#
#
################################################################################
#
#        # IMPORTANT: using all snapshots
#        from sklearn.cluster import KMeans
#        distances_data = np.concatenate(_all_data, axis=1)
#        distances_data = distances_data.T
#
#        # Perform K-means clustering for distance data
#        kmeans_distances = KMeans(n_clusters=2, random_state=0).fit(distances_data)
#        labels_distances = kmeans_distances.labels_
#
#        # NOTE: Using prior as weights. Compute populations for each state in phi/psi data and distances data
#        prior_pops_distances = np.bincount(labels_distances, weights=np.repeat(msm_pops,_all_data.shape[1]) / _all_data.shape[1])
#        print(prior_pops_distances)
#
#        # NOTE: Using BICeps weights. Compute populations for each state in phi/psi data and distances data
#        reweighted_pops_distances = np.bincount(labels_distances, weights=np.repeat(pops2,_all_data.shape[1]) / _all_data.shape[1])
#        print(reweighted_pops_distances)
#
#        # Assign cluster indices based on the populations (largest cluster is alpha)
#        folded_cluster_index_distances = np.argmax(prior_pops_distances)
#        unfolded_cluster_index_distances = 1 - folded_cluster_index_distances
#
#        # Get the indices for the alpha and unfolded states
#        folded_indices_distances = np.where(labels_distances == folded_cluster_index_distances)[0]
#        unfolded_indices_distances = np.where(labels_distances == unfolded_cluster_index_distances)[0]
#        print(folded_indices_distances)
#        print(unfolded_indices_distances)
#
#
#        plot = 1
#        if plot:
#            fig = plt.figure(figsize=(20,6))
#            gs = gridspec.GridSpec(1, len(avg_prior))
#
#            for i in range(len(avg_prior)):
#                ax = plt.subplot(gs[0,i])
#
#                # Separate the data for alpha and unfolded states for each distance
#                folded_data = np.concatenate(_all_data[i])[folded_indices_distances]
#                unfolded_data = np.concatenate(_all_data[i])[unfolded_indices_distances]
#
#                # Plot histograms for alpha and unfolded states
#                ax.hist(folded_data.flatten(), alpha=0.5, bins=25, color='red', edgecolor='black', linewidth=1.2, label='Folded')
#                ax.hist(unfolded_data.flatten(), alpha=0.5, bins=25, color='blue', edgecolor='black', linewidth=1.2, label='Unfolded')
#
#                # Add vertical lines and labels as before
#                ax.axvline(x=avg_prior[i], ymin=0, ymax=100, color="orange", lw=4)
#                ax.axvline(x=avg1[i], ymin=0, ymax=100, color="green", lw=4)
#                ax.axvline(x=avg2[i], ymin=0, ymax=100, color="gray", lw=4)
#                ax.axvline(x=avg3[i], ymin=0, ymax=100, color="red", lw=4)
#                ax.set_xlabel(f"{labels[i]}"+r" ($\AA$)", size=16)
#                ax.set_ylabel("")
#
#                # Set tick labels size
#                ticks = [ax.xaxis.get_minor_ticks(), ax.xaxis.get_major_ticks(), ax.yaxis.get_minor_ticks(), ax.yaxis.get_major_ticks()]
#                for tick in ticks:
#                    for subtick in tick:
#                        subtick.label.set_fontsize(14)
#
#                # Add subplot labels
#                ax.text(-0.1, 1.0, string.ascii_lowercase[i], transform=ax.transAxes, size=20, weight='bold')
#
#                # Add legend
#                ax.legend()
#
#            fig.tight_layout()
#            fig.savefig(f"{_sys_name}_distance_distributions_assigned.png", dpi=400)
#
#        exit()
#
## }}}
#





    exit()



