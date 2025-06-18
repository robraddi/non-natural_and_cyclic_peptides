

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
import uncertainties as u ################ Error Prop. Library
import uncertainties.unumpy as unumpy #### Error Prop.
from uncertainties import umath
import seaborn as sns
from scipy import stats
#:}}}


# Special Pairs:{{{
def special_pairs(filename, system, verbose=0):
    traj = md.load(filename)
    topology = traj.topology
    table, bonds = topology.to_dataframe()
    #print(table)
    #['RUN2_3b', 'RUN3_2b', 'RUN4_3a', 'RUN5_4a', 'RUN8_8']
    if (system == "RUN0_1a") or (system == "1a"):
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

    elif (system == "RUN1_1b") or (system == "1b") :
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

    elif (system == "RUN2_3b") or (system == "3b"):
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

    elif (system == "RUN7_2a") or (system == "2a"):
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


    elif (system == "RUN2_2b") or (system == "2b"):
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

    elif (system == "RUN4_3a") or (system == "3a"):
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

    elif (system == "RUN3_4a") or (system == "4a"):
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

    elif (system == "RUN4_4b") or (system == "4b"):
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

    elif (system == "RUN5_5") or (system == "5"):
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

    elif (system == "RUN6_8") or (system == "8"):
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

    elif (system == "RUN12_2") or (system == "2"):
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

    elif (system == "RUN13_1") or (system == "1"):
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

    if (system == "RUN12_2") or (system == "RUN13_1") or (system == "2") or (system == "1"):
        pairs = np.array([H1, H2, Hal1, H3, H4, H5])
    return pairs
# }}}

# garbage {{{
def cluster_distance_data():
    exit()

    #-----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # First subplot (phi vs psi)
    #ax1.scatter(phi_angles_all_snaps, psi_angles_all_snaps, edgecolor="k", color='blue', label=r'$\unfolded$')
    ax1.scatter(phi_angles_all_snaps[folded_indices_phi_psi], psi_angles_all_snaps[folded_indices_phi_psi], edgecolor="k", color='red', label=r'$\alpha$')
    ax1.scatter(phi_angles_all_snaps[unfolded_indices_phi_psi], psi_angles_all_snaps[unfolded_indices_phi_psi], edgecolor="k", color='blue', label=r'$\unfolded$')

    ax1.scatter(reweighted_phi, reweighted_psi, edgecolor="k", color='green', label=r'BICePs', s=100, alpha=0.5)
    ax1.scatter(weighted_phi, weighted_psi, edgecolor="k", color='orange', label=r'Prior', s=100, alpha=0.5)
    ax1.set_xlabel(r'$\psi$ D52', fontsize=16)
    ax1.set_ylabel(r'$\phi$ G53', fontsize=16)
    ax1.set_ylim(-180, 180)
    ax1.set_xlim(-180, 180)
    ax1.legend()
    # Second subplot (E24_G53 vs E24_D52)
    #ax2.scatter(E24_G53_distance_all_snaps, E24_D52_distance_all_snaps, edgecolor="k", color='blue', label=r'$\unfolded$')
    ax2.scatter(E24_G53_distance_all_snaps[folded_indices_distances], E24_D52_distance_all_snaps[folded_indices_distances], edgecolor="k", color='red', label=r'$\alpha$')
    ax2.scatter(E24_G53_distance_all_snaps[unfolded_indices_distances], E24_D52_distance_all_snaps[unfolded_indices_distances], edgecolor="k", color='blue', label=r'$\unfolded$')
    ax2.scatter(reweighted_E24_G53, reweighted_E24_D52, edgecolor="k", color='green', label=r'BICePs', s=100, alpha=0.5)
    ax2.scatter(weighted_E24_G53, weighted_E24_D52, edgecolor="k", color='orange', label=r'Prior', s=100, alpha=0.5)
    ax2.set_xlabel('d (E24 sc - G53 NH) (Å)', fontsize=16)
    ax2.set_ylabel('d (E24 NH - D52 CO) (Å)', fontsize=16)
    ax2.set_ylim(1, 10)
    ax2.set_xlim(1, 9)
    ax2.legend()
    fig.tight_layout()
    #fig.savefig("folded_unfolded_populations_100_states_all_snapshots.png")
    fig.savefig("folded_unfolded_populations_100_states.png")
    #-----------------------------------------------------------------------




# }}}



def calculate_free_energy(folded_state_pop):
    f = folded_state_pop
    u = 100. - f
    deltaG = - 8.314*298./1000. * np.log(f/u)
    print("deltaG = ",deltaG, "kJ/mol")
    deltaG = - 8.314*298./4184 * np.log(f/u)
    print("deltaG = ",deltaG, "kcal/mol")
    return deltaG



def convert_to_population(free_energy, free_energy_sigma):
    F = u.ufloat(free_energy, free_energy_sigma)
    print(F)
    #F_kT = (1/4184) * 1000 * F /0.5959
    # (kcal / J)
    # (1000 J / kJ)
    # (kJ/mol)
    # (1 kT / 0.5959 kcal / mol)
    F_kT = (1/4184) * 1000 * F *(1/0.5959)

    print(F_kT)
    print(umath.exp(-F_kT))







if __name__ == "__main__":

#    print("System 1:")
#    convert_to_population(free_energy=2.1, free_energy_sigma=0.2)
#    print()
#    print("System 2:")
#    convert_to_population(free_energy=3.0, free_energy_sigma=0.1)
#    exit()


    sys= 'RUN0_1a'
#    sys= 'RUN1_1b'
#    sys= 'RUN7_2a'
#    sys= 'RUN2_2b'
##    sys= 'RUN4_3a'
##    sys= 'RUN2_3b'
#    sys= 'RUN3_4a'
#    sys= 'RUN4_4b'
#    sys= 'RUN5_5'
#    sys= 'RUN6_8'
#    sys= 'RUN13_1'
#    sys= 'RUN12_2'


    use_all_noes = 0

    joint_models = 0

    nstates = 100
    nstates = 500

    if nstates == 500:
        ext = "500_states/"
    else:
        ext = ""


#
#    #exit()


    stat_model, data_uncertainty = "GB", "single"
#    stat_model, data_uncertainty = "Bayesian", "single"
#    stat_model, data_uncertainty = "Students", "single"
#    stat_model, data_uncertainty = "Gaussian", "multiple"
#    stat_model, data_uncertainty = "GaussianSP", "single"

    #nreplica = 8
    nreplica = 128
    #nreplica = 16
    #nreplica = 32



    if joint_models:
        sys = sys.split("_")[-1]
        pop_files = biceps.toolbox.get_files(f"joint_results/joint/{sys}/{ext}{stat_model}_{data_uncertainty}_sigma/10000*/populations.dat")
    else:
        if use_all_noes:
            #pop_files = biceps.toolbox.get_files(f"results/Systems/{sys}_with_sidechains/{ext}{stat_model}_{data_uncertainty}_sigma/10000*/populations.dat")
            pop_files = biceps.toolbox.get_files(f"results/Systems/{sys}_with_avg_sidechains/{ext}{stat_model}_{data_uncertainty}_sigma/10000*/populations.dat")
        else:
            pop_files = biceps.toolbox.get_files(f"results/Systems/{sys}/{ext}{stat_model}_{data_uncertainty}_sigma/10000*/populations.dat")
    #file2 = biceps.toolbox.get_files(f"results/Systems/{sys}/{stat_model}_{data_uncertainty}_sigma/10*{nreplica}*/populations.dat")[0]
    fp_results = []
    for file1 in pop_files:
        print(file1)

        #file1 = biceps.toolbox.get_files("results/Systems/RUN5_4a/Students_single_sigma/NOE_only*/populations.dat")[0]
        pops1 = np.loadtxt(file1)[:,1]
        pops1 = np.nan_to_num(pops1)
        #print(pops1)
        #exit()
        #pops2 = np.loadtxt(file2)[:,1]
        #pops3 = np.loadtxt(file3)[:,1]

        outdir = file1.replace("/populations.dat", "")


        #############################################################################
        skip_dirs = ["biceps", "experimental_data"]
        if joint_models: main_dir = "../joint"
        else: main_dir = "./Systems"
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
            #exit()

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

    #        nstates = int(biceps.toolbox.get_files(f"{sys_name}/*.pdb")[-1].split("cluster")[-1].split("_")[0])+1
            snaps_per_state = len(biceps.toolbox.get_files(f"{sys_name}/*cluster0_*.pdb"))
            snapshots = biceps.toolbox.get_files(f"{sys_name}/*.pdb")
            print(f"nstates = {nstates}; nsnapshots = {len(snapshots)}")
            print(f"snaps per state = {snaps_per_state};")

            if nstates == 100:
                files_by_state = [biceps.toolbox.get_files(
                    f"{sys_name}/*cluster{state}_*.pdb"
                    ) for state in range(nstates)]
            elif nstates == 500:
                files_by_state = [[file] for file in biceps.toolbox.get_files(
                    f"{sys_name}/*cluster*_*.pdb"
                    )]
            else:
                print("nstates is not right")
                exit()

            # get prior populations
            file = f'{sys_name}/*clust*_*.csv'
            file = biceps.toolbox.get_files(file)[0]
            msm_pops = pd.read_csv(file, index_col=0, comment="#")
            #msm_pops[" population(%) "] = msm_pops[" population(%) "].to_numpy()/100
            if nstates == 100:
                msm_pops = msm_pops["populations"].to_numpy()
            elif nstates == 500:
                msm_pops = np.repeat(msm_pops["populations"].to_numpy(), 5)/5.


            compute = 1
            hbond_indices = [0,1,3,4]

            if (sys == "RUN12_2") or (sys == "RUN13_1"):
                hbond_indices = [0,1,3,4,5]
            if compute:
                all_data,averaged_over_snaps,bc,all_hbs = [],[],[],[]
                for i,state in enumerate(files_by_state):
                    distances = []
                    hbs = []
                    for j,frame in enumerate(state):
                        d = md.compute_distances(md.load(frame), indices)*10. # convert nm to Å
                        distances.append(d)

                        # NOTE: the baker_hubbard method uses distance and angle information
                        hbonds = md.baker_hubbard(md.load(frame), distance_cutoff=0.25)

                        # NOTE: the wernet_nilsson method uses distance and angle information
                        # a “cone” criterion where the distance cutoff depends on the angle.
                        #hbonds = md.wernet_nilsson(md.load(frame))[0]

                        ## NOTE: the kabsch_sander method only uses distance information
                        #hbonds = md.kabsch_sander(md.load(frame))[0]

                        #print(hbonds)
                        #exit()

                        hb_count = 0
                        counts = np.zeros(len(indices))
                        for hbond in hbonds:
                            for b,pair in enumerate(indices):
                                if set(hbond[1:]) == set(pair):
                                    hb_count += 1
                                    counts[b] += 1
                        hbs.append(counts)


                    distances = np.array(distances)
                    all_data.append(distances)
                    data = np.mean(distances, axis=0)[0]
                    averaged_over_snaps.append(data)

                    hbs = np.array(hbs)
                    all_hbs.append(np.mean(hbs, axis=0))


                all_data = np.array(all_data)
                averaged_over_snaps = np.array(averaged_over_snaps)

            all_hbs = np.array(all_hbs)

            bond_character = all_hbs.sum(axis=1)
            top10 = np.argsort(bond_character)[::-1][:10]
            #exit()

            HB_criteria = 3.0
            state_indices = np.where(bond_character >= HB_criteria)[0]
            print(state_indices)

    #
    #        top_indices = [61,73,9,56,55,41,35,64,52,39] # 1 from Gaussian
    #
    #
    #        #top_indices = [0,73,44,45,3,37,87,16,34,51] # 1a from Students
    ##        top_indices = [0,73,44,3,37,59,87,34,45,16] # 1a from Gaussian
    #
    #        ##top_indices = [35,4,80,44,88,26,33,91,60,97] # 1b from Gaussian
    ##        top_indices = [35,4,80,44,88,26] # 1b from Gaussian
    #        print(top_indices)
    #        print(msm_pops[top_indices].sum())
    #        print(pops1[top_indices].sum())
    #        exit()
    #


            prior_folded = msm_pops[state_indices].sum()
            reweighted_folded = pops1[state_indices].sum()
            bc_criteria = HB_criteria/len(hbond_indices)*100
            print(f"At {bc_criteria}% HBond character:")

            print(prior_folded)
            print(reweighted_folded)
            print(bond_character)

            y1,y2 = [],[]
    #        x = np.linspace(0, np.max(bond_character), 1000)
            x = np.linspace(0, len(hbond_indices), 1000)
            for HB_crit in x:
                y1.append(100.*msm_pops[np.where(bond_character >= HB_crit)[0]].sum())
                y2.append(100.*pops1[np.where(bond_character >= HB_crit)[0]].sum())
            print("At 100% HBond character:")
            print("prior: ",y1[-1])
            print("reweighted: ",y2[-1])
            #exit()

            x = x/np.max(x) * 100.
            fig = plt.figure(figsize=(6,6))
            ax = plt.subplot(111)
            ax.plot(x, y1, label="prior", color="red")
            ax.plot(x, y2,  label="reweighted", color="blue")
            ax.legend()
            ax.set_xlabel("HB Character (%)", fontsize=16)
            ax.set_ylabel("Folded state (%)", fontsize=16)
            ax.set_xlim(0, 100)
            fig.tight_layout()
            fig.savefig(f"{outdir}/{sys}_HB_character_def_pops.png")




            #print(bond_character)
            #print(bond_character.max())
            #exit()



            avg_prior = np.array([w*averaged_over_snaps[i,:] for i,w in enumerate(msm_pops)]).sum(axis=0)
            avg1 = np.array([w*averaged_over_snaps[i,:] for i,w in enumerate(pops1)]).sum(axis=0)
            #avg2 = np.array([w*averaged_over_snaps[i,:] for i,w in enumerate(pops1)]).sum(axis=0)
            #avg3 = np.array([w*averaged_over_snaps[i,:] for i,w in enumerate(pops3)]).sum(axis=0)

            _all_data = all_data.T
            _all_data = np.concatenate(_all_data, axis=0)
            #print(_all_data.shape)

            #print(avg_prior)
            #print(avg1)
            #print(avg2)
            #print(avg3)

            plot = 0
            if plot:
                fig = plt.figure(figsize=(20,6))
                gs = gridspec.GridSpec(1, len(avg_prior))
                for i in range(len(avg_prior)):
                    ax = plt.subplot(gs[0,i])
                    data = np.concatenate(_all_data[i])
                    ax.hist(data, alpha=0.5, bins=25,
                        edgecolor='black', linewidth=1.2, color="b")
                    ax.axvline(x=avg_prior[i], ymin=0, ymax=100, color="#FF8C00", lw=4) #color="orange", lw=4)
                    ax.axvline(x=avg1[i], ymin=0, ymax=100, color="blue", lw=4)
                    #ax.axvline(x=avg2[i], ymin=0, ymax=100, color="gray", lw=4)
                    #ax.axvline(x=avg3[i], ymin=0, ymax=100, color="red", lw=4)
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
                fig.savefig(f"{outdir}/{_sys_name}_distance_distributions.png", dpi=400)

            ########################################################################
            ########################################################################
            ########################################################################


            from scipy.stats import gaussian_kde
            # IMPORTANT: using averaged cond character data
            from sklearn.cluster import KMeans
            #distances_data = bc
            distances_data = all_hbs
    #        distances_data = averaged_over_snaps

    #        # Perform K-means clustering for distance data
    #        kmeans_distances = KMeans(n_clusters=2, random_state=0).fit(distances_data)
    #        labels_distances = kmeans_distances.labels_

    #------------------------------------------------------------------------------

            folded_indices = state_indices.copy()
            #print(folded_indices)
            _labels = np.zeros(len(bond_character))
            _labels[folded_indices] = np.ones(len(folded_indices))
            labels_distances = np.array(_labels, dtype=int)
            print(labels_distances)
    #        exit()

    #------------------------------------------------------------------------------
            states = np.array(list(range(len(labels_distances))))
            states = states[np.where(1 == labels_distances)[0]]
            #print(f"top states: {states}")
            print(f"top states: {str(states).replace(' ',',')}")
            print(f"top pops:   {pops1[states]}")
            print("------------------------------------")

            sorted_pops = np.argsort(msm_pops)[::-1][:10]
            print(f"top 10 prior pops:   {msm_pops[sorted_pops]}")
            print(f"top 10 states: {str(sorted_pops).replace(' ',',')}")
            print(f"top states prior populations: {msm_pops[states].sum()}")
            print("------------------------------------")


            sorted_pops = np.argsort(pops1)[::-1][:10]
            print(f"top 10 pops:   {pops1[sorted_pops]}")
            print(f"top 10 states: {str(sorted_pops).replace(' ',',')}")
            print(f"top states BICePs populations: {pops1[states].sum()}")

            print(file1)
            print(msm_pops[states].sum())
            fp_results.append(pops1[states].sum())
            print(fp_results[-1])
            calculate_free_energy(msm_pops[states].sum())
            calculate_free_energy(pops1[states].sum())
    #        exit()

            # NOTE: Using prior as weights. Compute populations for each state in phi/psi data and distances data
            prior_pops_distances = np.bincount(labels_distances, weights=msm_pops)
            print(f"prior: {prior_pops_distances}")

            # NOTE: Using BICeps weights. Compute populations for each state in phi/psi data and distances data
            reweighted_pops_distances = np.bincount(labels_distances, weights=pops1)
            print(f"reweighted: {reweighted_pops_distances}")

            # Assign cluster indices based on the populations (largest cluster is alpha)
            folded_cluster_index_distances = 1 #np.argmax(prior_pops_distances)
            unfolded_cluster_index_distances = 1 - folded_cluster_index_distances

            # Get the indices for the alpha and unfolded states
            folded_indices_distances = np.where(labels_distances == folded_cluster_index_distances)[0]
            unfolded_indices_distances = np.where(labels_distances == unfolded_cluster_index_distances)[0]
            #print(folded_indices_distances)
            #print(unfolded_indices_distances)


            avg_hbs_prior = np.array([w*all_hbs[i,:] for i,w in enumerate(msm_pops)]).sum(axis=0)
            avg_hbs_post = np.array([w*all_hbs[i,:] for i,w in enumerate(pops1)]).sum(axis=0)

            distances_data = averaged_over_snaps
            plot = 1
            plot_with_histograms = 0
            if plot:
                fig = plt.figure(figsize=(20,6))
                gs = gridspec.GridSpec(2, len(avg_prior))
                axs = []
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
                    best_bandwidth_prior = None
                    best_score_prior = -np.inf

                    for width in bandwidths:
                        score = kde_cv_score(X, msm_pops, width)
                        if score > best_score_prior:
                            best_score_prior = score
                            best_bandwidth_prior = width

                    #print("Best bandwidth for kde_prior:", best_bandwidth_prior)
                    kde_prior = gaussian_kde(X, weights=msm_pops)
                    kde_prior.set_bandwidth(bw_method=best_bandwidth_prior)

                    # Repeat for kde_post
                    best_bandwidth_post = None
                    best_score_post = -np.inf

                    for width in bandwidths:
                        score = kde_cv_score(X, pops1, width)
                        if score > best_score_post:
                            best_score_post = score
                            best_bandwidth_post = width

                    #print("Best bandwidth for kde_post:", best_bandwidth_post)
                    kde_post = gaussian_kde(X, weights=pops1)
                    kde_post.set_bandwidth(bw_method=best_bandwidth_post)



                    # Generate a range of values for plotting the KDE
                    xmin, xmax = X.min(), X.max()
                    x = np.linspace(xmin, xmax, 1000)
                    ax.set_xlim(xmin, xmax)

                    ax.plot(x, kde_prior(x), color="#FF8C00", label='Prior')
                    ax.plot(x, kde_post(x), color='blue', label='Reweighted')


                    # Add vertical lines and labels as before
                    #ax.axvline(x=avg_prior[i], ymin=0, ymax=100, color="orange", ls='--', lw=4)
                    #ax.axvline(x=avg1[i], ymin=0, ymax=100, color="green", ls='--', lw=4)

                    ax.axvline(x=avg_prior[i], ymin=0, ymax=100, color="#FF8C00", ls='--', lw=4) #color="orange", lw=4)
                    ax.axvline(x=avg1[i], ymin=0, ymax=100, color="blue", ls='--', lw=4)

                    #ax.axvline(x=avg2[i], ymin=0, ymax=100, color="green", ls='--', lw=4)
                    #ax.axvline(x=avg3[i], ymin=0, ymax=100, color="red", lw=4)
                    ax.set_xlabel(f"{labels[i]}"+r" ($\AA$)", size=16)
                    ax.set_ylabel("")
                    # Add legend
                    if i == 0:
                        ax.legend()
                        ax.set_ylabel("Density", size=16)

                    axs.append(ax)

                for i in range(len(avg_prior)):
                    if i not in hbond_indices: continue
                    ax = plt.subplot(gs[1,i])


                    # Separate the data for alpha and unfolded states for each distance
                    folded_data = all_hbs[:,i][folded_indices_distances]
                    unfolded_data = all_hbs[:,i][unfolded_indices_distances]

                    X = all_hbs[:,i]
                    X = X*100 + np.random.normal(0, 0.001, X.shape)


                    def kde_cv_score(X, weights, bandwidth, k=10):
                        kf = KFold(n_splits=k)
                        scores = []
                        #X = X + np.random.normal(0, 0.001, X.shape)
                        #exit()

                        for train_index, test_index in kf.split(X):
                            X_train, X_test = X[train_index], X[test_index]
                            weights_train = weights[train_index]
                            weights_train /= weights_train.sum()

                            kde = gaussian_kde(X_train, weights=weights_train, bw_method=bandwidth)
                            score = np.sum(kde.logpdf(X_test))
                            scores.append(score)

                        return np.mean(scores)

                    #bandwidths = np.linspace(0.1, 5.0, 10)  # Adjust this range based on your data
                    bandwidths = np.linspace(0.1, 0.99, 10)  # Adjust this range based on your data
                    best_bandwidth_prior = None
                    best_score_prior = -np.inf

                    for width in bandwidths:
                        #print(width)
                        score = kde_cv_score(X, msm_pops, width)
                        if score > best_score_prior:
                            best_score_prior = score
                            best_bandwidth_prior = width

                    #print("Best bandwidth for kde_prior:", best_bandwidth_prior)
                    kde_prior = gaussian_kde(X, weights=msm_pops)
                    kde_prior.set_bandwidth(bw_method=best_bandwidth_prior)

                    # Repeat for kde_post
                    best_bandwidth_post = None
                    best_score_post = -np.inf

                    for width in bandwidths:
                        score = kde_cv_score(X, pops1, width)
                        if score > best_score_post:
                            best_score_post = score
                            best_bandwidth_post = width

                    #print("Best bandwidth for kde_post:", best_bandwidth_post)
                    kde_post = gaussian_kde(X, weights=pops1)
                    kde_post.set_bandwidth(bw_method=best_bandwidth_post)

                    # Generate a range of values for plotting the KDE
                    xmin, xmax = X.min(), X.max()
                    x = np.linspace(xmin, xmax, 1000)
                    ax.set_xlim(xmin, xmax)

                    ax.plot(x, kde_prior(x), color='Orange', label='Prior')
                    ax.plot(x, kde_post(x), color='Green', label='Reweighted')








                    ax.set_xlim(0, 100)
                    ## Define bin edges
                    #bin_edges = np.arange(0, 1.2, 0.1) * 100
                    ## Plot histograms for alpha and unfolded states
                    #ax.hist(folded_data.flatten()*100, alpha=0.5, bins=bin_edges, color='red', edgecolor='black', linewidth=1.2, label='Folded', density=False)
                    #ax.hist(unfolded_data.flatten()*100, alpha=0.5, bins=bin_edges, color='blue', edgecolor='black', linewidth=1.2, label='Unfolded', density=False)

                    # Add vertical lines and labels as before
                    ax.axvline(x=100*avg_hbs_prior[i], ymin=0, ymax=100, color="orange", ls='--', lw=4)
                    ax.axvline(x=100*avg_hbs_post[i], ymin=0, ymax=100, color="green", ls='--', lw=4)

                    ax.set_xlabel(f"HB content "+r"(%)"+f"\n{labels[i]}", size=16)
                    ax.set_ylabel("")


                    # Add legend
                    if i == 0:
                        ax.legend()
                        ax.set_ylabel("Density", size=16)
                    axs.append(ax)




                for i in range(len(axs)):
                    ax = axs[i]
                    # Set tick labels size
                    ticks = [ax.xaxis.get_minor_ticks(), ax.xaxis.get_major_ticks(), ax.yaxis.get_minor_ticks(), ax.yaxis.get_major_ticks()]
                    for tick in ticks:
                        for subtick in tick:
                            subtick.label.set_fontsize(14)

                    # Add subplot labels
#                    ax.text(-0.1, 1.0, string.ascii_lowercase[i], transform=ax.transAxes, size=20, weight='bold')


                fig.tight_layout()
                fig.savefig(f"{outdir}/{_sys_name}_avg_distance_dist_with_HB_cluster_kernel_{nreplica}_replica.png", dpi=400)


    folded_pop = np.mean(fp_results)
    folded_pop_sigma = np.std(fp_results)
    folded_pop_sigma_sem = stats.sem(fp_results)
    print(f"Predicted folded pop: {folded_pop} \pm {folded_pop_sigma} (Std Dev)")
    print(f"Predicted folded pop: {folded_pop} \pm {folded_pop_sigma_sem} (SEM)")

    exit()



## Hold:{{{
#            ########################################################################
#            ########################################################################
#            ########################################################################
#
#
#            from scipy.stats import gaussian_kde
#            # IMPORTANT: using averaged data
#            from sklearn.cluster import KMeans
#            distances_data = averaged_over_snaps
#
#            # Perform K-means clustering for distance data
#            kmeans_distances = KMeans(n_clusters=2, random_state=0).fit(distances_data)
#            labels_distances = kmeans_distances.labels_
#
#            # NOTE: Using prior as weights. Compute populations for each state in phi/psi data and distances data
#            prior_pops_distances = np.bincount(labels_distances, weights=msm_pops)
#            print(prior_pops_distances)
#
#            # NOTE: Using BICeps weights. Compute populations for each state in phi/psi data and distances data
#            reweighted_pops_distances = np.bincount(labels_distances, weights=pops1)
#            print(reweighted_pops_distances)
#
#            # Assign cluster indices based on the populations (largest cluster is alpha)
#            folded_cluster_index_distances = np.argmax(prior_pops_distances)
#            unfolded_cluster_index_distances = 1 - folded_cluster_index_distances
#
#            # Get the indices for the alpha and unfolded states
#            folded_indices_distances = np.where(labels_distances == folded_cluster_index_distances)[0]
#            unfolded_indices_distances = np.where(labels_distances == unfolded_cluster_index_distances)[0]
#            print(folded_indices_distances)
#            print(unfolded_indices_distances)
#
#
#            plot = 1
#            plot_with_histograms = 0
#            if plot:
#                fig = plt.figure(figsize=(20,6))
#                gs = gridspec.GridSpec(1, len(avg_prior))
#
#                for i in range(len(avg_prior)):
#                    ax = plt.subplot(gs[0,i])
#
#                    # Separate the data for alpha and unfolded states for each distance
#                    folded_data = distances_data[:,i][folded_indices_distances]
#                    unfolded_data = distances_data[:,i][unfolded_indices_distances]
#
#                    # Plot histograms for alpha and unfolded states
#                    if plot_with_histograms: ax.hist(folded_data.flatten(), alpha=0.5, bins=25, color='red', edgecolor='black', linewidth=1.2, label='Folded', density=True)
#                    if plot_with_histograms: ax.hist(unfolded_data.flatten(), alpha=0.5, bins=25, color='blue', edgecolor='black', linewidth=1.2, label='Unfolded', density=True)
#
#
#                    # Create KDEs for alpha and unfolded states using the weights
#                    X = distances_data[:,i]
#                    #width = 0.5
#                    #kde_prior = gaussian_kde(X, weights=msm_pops)
#                    #kde_prior.set_bandwidth(bw_method=width)
#                    #kde_post = gaussian_kde(X, weights=pops1)
#                    #kde_post.set_bandwidth(bw_method=width)
#
#                    from sklearn.model_selection import KFold
#
#
#                    def kde_cv_score(X, weights, bandwidth, k=5):
#                        kf = KFold(n_splits=k)
#                        scores = []
#
#                        for train_index, test_index in kf.split(X):
#                            X_train, X_test = X[train_index], X[test_index]
#                            weights_train = weights[train_index]
#
#                            kde = gaussian_kde(X_train, weights=weights_train, bw_method=bandwidth)
#                            score = np.sum(kde.logpdf(X_test))
#                            scores.append(score)
#
#                        return np.mean(scores)
#
#                    bandwidths = np.linspace(0.1, 5.0, 10)  # Adjust this range based on your data
#                    best_bandwidth_prior = None
#                    best_score_prior = -np.inf
#
#                    for width in bandwidths:
#                        score = kde_cv_score(X, msm_pops, width)
#                        if score > best_score_prior:
#                            best_score_prior = score
#                            best_bandwidth_prior = width
#
#                    print("Best bandwidth for kde_prior:", best_bandwidth_prior)
#                    kde_prior = gaussian_kde(X, weights=msm_pops)
#                    kde_prior.set_bandwidth(bw_method=best_bandwidth_prior)
#
#                    # Repeat for kde_post
#                    best_bandwidth_post = None
#                    best_score_post = -np.inf
#
#                    for width in bandwidths:
#                        score = kde_cv_score(X, pops1, width)
#                        if score > best_score_post:
#                            best_score_post = score
#                            best_bandwidth_post = width
#
#                    print("Best bandwidth for kde_post:", best_bandwidth_post)
#                    kde_post = gaussian_kde(X, weights=pops1)
#                    kde_post.set_bandwidth(bw_method=best_bandwidth_post)
#
#
#
#                    # Generate a range of values for plotting the KDE
#                    xmin, xmax = X.min(), X.max()
#                    x = np.linspace(xmin, xmax, 1000)
#                    ax.set_xlim(xmin, xmax)
#
#                    ax.plot(x, kde_prior(x), color='Orange', label='Prior')
#                    ax.plot(x, kde_post(x), color='Green', label='Reweighted')
#
#
#
#                    # Add vertical lines and labels as before
#                    ax.axvline(x=avg_prior[i], ymin=0, ymax=100, color="orange", lw=4)
#                    ax.axvline(x=avg1[i], ymin=0, ymax=100, color="green", lw=4)
#                    #ax.axvline(x=avg2[i], ymin=0, ymax=100, color="green", lw=4)
#                    #ax.axvline(x=avg3[i], ymin=0, ymax=100, color="red", lw=4)
#                    ax.set_xlabel(f"{labels[i]}"+r" ($\AA$)", size=16)
#                    ax.set_ylabel("")
#
#                    # Set tick labels size
#                    ticks = [ax.xaxis.get_minor_ticks(), ax.xaxis.get_major_ticks(), ax.yaxis.get_minor_ticks(), ax.yaxis.get_major_ticks()]
#                    for tick in ticks:
#                        for subtick in tick:
#                            subtick.label.set_fontsize(14)
#
#                    # Add subplot labels
#                    ax.text(-0.1, 1.0, string.ascii_lowercase[i], transform=ax.transAxes, size=20, weight='bold')
#
#                    # Add legend
#                    ax.legend()
#
#                fig.tight_layout()
#                fig.savefig(f"{_sys_name}_avg_distance_distributions_kernel_{nreplica}_replica.png", dpi=400)
#
## }}}
#















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



