# Libraries:{{{
#import sys, os
from Bio import SeqUtils
import biceps
import mdtraj as md
import numpy as np
import pandas as pd
import os, re
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
from tqdm import tqdm # progress bar
from itertools import combinations
import subprocess
import networkx as nx

#:}}}

# Methods:{{{

def edit_proton_name(label):
    label = label.replace('N', '')
    label = label.replace('α', 'A')
    label = label.replace('β', 'B')
    label = label.replace('γ', 'G')
    label = label.replace('δ', 'D')
    label = label.replace('ε', 'E')
    return label


def find_corresponding_value_in_reversed_list(value, min_val, max_val):
    """
    Given a value, and the minimum and maximum values of a list, this function
    returns the corresponding value in the reversed list.

    :param value: The value to find the corresponding value for in the reversed list.
    :param min_val: The minimum value in the original list.
    :param max_val: The maximum value in the original list.
    :return: The corresponding value in the reversed list.
    """
    if min_val <= value <= max_val:
        # Calculate the corresponding value in the reversed list
        return max_val - (value - min_val)
    else:
        # Value is outside the range of the list
        return None


def split(string):
    match = re.match(r"([a-z]+)([0-9]+)", string[0].upper()+string[1:].lower(), re.I)
    items = match.groups()
    return items[0]

def get_rdc_indices(frame, data):
    indices = []
    frame = md.load(frame)
    topology = frame.topology
    table = topology.to_dataframe()[0]
    #print(data)
    #print(table)
    #"resIdx1","atom1","resIdx2","atom2"
    for k in range(len(data["resIdx1"].to_numpy())):
        atom1,atom2 = data["atom1"].to_numpy()[k],data["atom2"].to_numpy()[k]
        #print(atom1,atom2)
        resIdx1,resIdx2 = data["resIdx1"].to_numpy()[k],data["resIdx2"].to_numpy()[k]
        #print(resIdx1,resIdx2)
        group1 = table.iloc[np.where(int(resIdx1) == table["resSeq"].to_numpy())[0]]
        group2 = table.iloc[np.where(int(resIdx2) == table["resSeq"].to_numpy())[0]]
        #print(k, group1, group2)
        row1 = group1.iloc[np.where(atom1 == group1["name"].to_numpy())[0]]
        row2 = group2.iloc[np.where(atom2 == group2["name"].to_numpy())[0]]
        #print(k, row1, row2)
        indices.append([row1["serial"].values[0]-1,row2["serial"].values[0]-1])
        #print(indices)
    indices = np.array(indices)
    return indices

def compute_ensemble_avg_RDC(grouped_files, exp_data_df, outdir, skip_idx=[], verbose=False):
    """
    """
    #file = "../../ubq_biceps/experimental_data/1d3z_mr.str"
    #df = get_exp_RDC_file(file, k=2) # 2, 5, 6, 10
    #atom_name_1, atom_name_2 = df[["atom1", "atom2"]].to_numpy().T
    #exp = df["exp"].to_numpy(dtype=float)

    df = exp_data_df
    exp = df["exp"].to_numpy(dtype=float)

    indices = None
    for i,state in enumerate(grouped_files):
        _state = []
        for j,frame in enumerate(state):

            rdcp = biceps.RDC_predictor(frame)
            rdcp.get_atom_indices(df)
            rdcs = rdcp.predict_RDCs_with_SVD(exp)
            stats = rdcp.statistics
            parameters = rdcp.parameters
            _state.append(rdcs["rdc"].to_numpy())
            #print(rdcs)
            #print(parameters)
            #print(stats)
            #exit()

        data = np.mean(np.array(_state), axis=0)
        data = np.delete(data, skip_idx, axis=0)
#        print(data)
#        exit()
        if verbose: print(data)
        np.savetxt(f'{outdir}/rdc_{i}.txt', data)
    return data






#def compute_ensemble_avg_J(grouped_files, outdir, model="Bax2007", skip_idx=[], return_index=False, verbose=False):
#def compute_ensemble_avg_J(grouped_files, outdir, model="Raddi2024", skip_idx=[], return_index=False, verbose=False):
#def compute_ensemble_avg_J(grouped_files, outdir, model="Kessler1998", skip_idx=[], return_index=False, verbose=False):
def compute_ensemble_avg_J(grouped_files, outdir, model="NRV2024", skip_idx=[], return_index=False, verbose=False):
    """Additional models are possible, such as 'Habeck'
    """

    indices = None
    for i,state in enumerate(grouped_files):
        _state = []
        for j,frame in enumerate(state):
            t = md.load(frame, top=frame)
            J = biceps.J_coupling.compute_J3_HN_HA(t, model = model)
            #J = biceps.toolbox.get_J3_HN_HA(frame, model=model,
            #        outname=f'{outdir}/J_state_{i}_snapshot_{j}.npy', verbose=False)
            _state.append(J[-1][0])

        frame = md.load(frame, top=state[0])
        topology = frame.topology
        table = topology.to_dataframe()[0]
        #print(table)
#        listing = []
#        for k in range(len(J[0])):
##            if k in skip_idx: continue
#            print(k, [topology.atom(idx) for idx in J[0][k]])
#            listing.append([topology.atom(idx) for idx in J[0][k]])
#        exit()

        data = np.mean(np.array(_state), axis=0)
        data = np.delete(data, skip_idx, axis=0)

        if verbose: print(data)
        np.savetxt(f'{outdir}/J_{i}.txt', data)
    indices = np.delete(np.array(J[0], dtype=int), skip_idx, axis=0)
    return indices, data


def get_cs_indices(structure_file, data, col, verbose=False):

    t = md.load(structure_file)
    topology = t.topology
    table, bonds = topology.to_dataframe()
    #print(table)

    #print(df)
    # PDB_residue_no_1 PDB_residue_name_1 PDB_atom_name_1
    # PDB_residue_no_2 PDB_residue_name_2 PDB_atom_name_2
    indices = []
    exp_data,exp_err,restraint_index = [],[],[]
    _id, id = 0, 0

    error_res = []
    #"resIdx1","atom1","resIdx2","atom2"
    for k in range(len(data)):
        row = data.iloc[[k]]

        #atom1 = data["atom1"].to_numpy()[k]
        #resIdx1 = data["resIdx1"].to_numpy()[k]
        #print(resIdx1, atom1)
        #group1 = table.iloc[np.where(int(resIdx1) == table["resSeq"].to_numpy())[0]]
        #print(atom1)
        #print(group1)
        #row1 = group1.iloc[np.where(atom1 == group1["name"].to_numpy())[0]]


        #resIndex1 = int(row["PDB_residue_no_1"].to_numpy()[0])
        resName1 = str(row["PDB_residue_name_1"].to_numpy()[0])
        symbol1 = str(row["PDB_atom_name_1"].to_numpy()[0])
        resSeq1 = int(row["Seq_ID_1"].to_numpy()[0])

        print(row)
        print(resName1,symbol1,resSeq1)

        if symbol1[-1].isdigit():
            try:
                index1 = np.concatenate([topology.select(f'residue {resSeq1} and resname {resName1} and name {symbol1}'),
                           topology.select(f'residue {resSeq1} and resname {resName1} and name {symbol1}1'),
                           topology.select(f'residue {resSeq1} and resname {resName1} and name {symbol1}2')])
            except(ValueError) as e:
                print(e)
                error_res.append(k)
                continue

        else:
            try:
                index1 = topology.select(f'residue {resSeq1} and resname {resName1} and name {symbol1}')
            except(ValueError) as e:
                print(e)
                error_res.append(k)
                continue

        index1 = index1.astype(int)
        print(index1)

        if (len(index1) == 0) or (np.isnan(row[col].to_numpy()[0])):
            if verbose: print("Warning: could not find index...")
            error_res.append(k)
        elif (len(index1) == 1):
            new_id = int(row.index.to_numpy()[0])
            if (_id != new_id):
                _id = new_id
                id += 1
            idx1 = table.iloc[index1[0]]["serial"]
            indices.append(np.array(idx1))
            exp_data.append(row[col].to_numpy()[0])
            restraint_index.append(id)
        else:
            for j in range(len(index1)):
                new_id = int(row.index.to_numpy()[0])
                if (_id != new_id):
                    _id = new_id
                    id += 1
                idx1 = table.iloc[index1[j]]["serial"]
                indices.append(np.array(idx1))
                exp_data.append(row[col].to_numpy()[0])
                restraint_index.append(id)


    indices = np.array(indices) - 1
    #return indices, error_res
    return np.array(indices),np.array(restraint_index),np.array(exp_data), np.array(error_res)



def compute_ensemble_avg_CS(grouped_files, outdir, temp=298.0, pH=5.0,
        atom_sel="H", skip_idx=[], return_index=False, verbose=False, compute=True):
    for i,state in enumerate(grouped_files):
        _state = []
        for j,frame in enumerate(state):
            if verbose: print(f"Loading {frame} ...")
            frame = md.load(frame, top=state[0])
            topology = frame.topology
            table = topology.to_dataframe()[0]
            seq = np.array([split(str(res)) for res in topology.residues])
            #skip_idx = np.where(seq=="Gly")[0]
            #print(seq)
            #exit()

            if compute:
                shifts = md.nmr.chemical_shifts_shiftx2(frame, pH, temp)
                shifts = shifts[0].unstack(-1)
                shifts.to_pickle(f'{outdir}/CS_state_{i}_snapshot_{j}.pkl')
            else:
                shifts = pd.read_pickle(f'{outdir}/CS_state_{i}_snapshot_{j}.pkl')
            #print(shifts)
            #print(len(shifts))
            #exit()
            #columns = shifts.columns.to_list()
            #print(skip_idx)
            if skip_idx == []: skip = []
            else: skip = np.concatenate([np.where(shifts.index.to_numpy() == idx)[0] for idx in skip_idx])
            #print(skip)

#            # Reindex the DataFrame to include all rows in the specified range
#            full_index = range(shifts.index[0], shifts.index[-1] + 1)
#            shifts = shifts.reindex(full_index)
#            # If you want NaNs instead of None values in the DataFrame
#            shifts = shifts.replace({None: float('nan')})

            #print(shifts)
            _shifts = shifts[atom_sel]#.to_numpy()
            #print(np.delete(shifts.to_numpy(), skip_idx, axis=0))
            _state.append(np.delete(_shifts.to_numpy(), skip, axis=0))
            #print(len(_state[-1]))
            #exit()
        data = np.mean(np.array(_state), axis=0)
        data = data[~np.isnan(data)]
        if verbose: print(data)
        np.savetxt(f'{outdir}/CS_{i}.txt', data)
#    print(shifts)
    #print(data)
    idxs = np.stack([_shifts.index.to_numpy(dtype=int),_shifts.to_numpy()], axis=1)[~np.isnan(_shifts.to_numpy())][:,0].astype(int) - 1
    return idxs



def compute_ensemble_avg_NOE(grouped_files, indices, outdir,
            convert_distances_to_intensities=False, verbose=False):

    indices = np.loadtxt(indices, dtype=int)
    for i,state in enumerate(grouped_files):
        distances = []
        for j,frame in enumerate(state):
            d = md.compute_distances(md.load(frame), indices)*10. # convert nm to Å
            distances.append(d)
        data = np.mean(np.array(distances), axis=0)
        if convert_distances_to_intensities:
            # smallest distance can't be smaller than 0.1 Å
            data[data < 0.1] = np.ones(len(data[data < 0.1]))*0.1
            data = data**(-6)
        #print(np.mean([d[0][1] for d in distances]))
        #print(data)
        #exit()
        np.savetxt(f'{outdir}/NOE_{i}.txt', data)


#:}}}

# get_noe_indices:{{{

def get_NOE_indices(structure_file, exp_df, verbose=False):

    t = md.load(structure_file)
    topology = t.topology
    table, bonds = topology.to_dataframe()
    print(table)
#    exit()
    #serial  name element  resSeq resName  chainID segmentID

    #df = pd.read_csv(exp_csv_file, index_col=0)
    df = exp_df.copy()
    df = df.replace('.', np.NAN)
    df = df.dropna(axis=1, how='all')

    #print(df)
    # PDB_residue_no_1 PDB_residue_name_1 PDB_atom_name_1
    # PDB_residue_no_2 PDB_residue_name_2 PDB_atom_name_2
    indices = []
    exp_data,exp_err,restraint_index = [],[],[]
    _id, id = 0, 0
    for i in range(len(df)):
        row = df.iloc[[i]]

        #resIndex1 = int(row["PDB_residue_no_1"].to_numpy()[0])
        resName1 = str(row["PDB_residue_name_1"].to_numpy()[0])
        symbol1 = str(row["PDB_atom_name_1"].to_numpy()[0])
        resSeq1 = int(row["Seq_ID_1"].to_numpy()[0])

        ###
        #resIndex2 = int(row["PDB_residue_no_2"].to_numpy()[0])
        resName2 = str(row["PDB_residue_name_2"].to_numpy()[0])
        symbol2 =  str(row["PDB_atom_name_2"].to_numpy()[0])
        resSeq2 = int(row["Seq_ID_2"].to_numpy()[0])

        ## if testing:
        print(row)
        print(resName1,symbol1,resSeq1)
        print(resName2,symbol2,resSeq2)


        #index1 = list(set(np.where((resName1 == table["resName"].to_numpy()))[0]).intersection(
        #    np.where((symbol1 == table["name"].to_numpy()))[0]).intersection(
        #                        np.where((resSeq1 == table["resSeq"].to_numpy()))[0]))
        #index2 = list(set(np.where((resName2 == table["resName"].to_numpy()))[0]).intersection(
        #    np.where((symbol2 == table["name"].to_numpy()))[0]).intersection(
        #                        np.where((resSeq2 == table["resSeq"].to_numpy()))[0]))
        #print(index1)
        #print(index2)


        index1 = topology.select(f'residue {resSeq1} and resname {resName1} and name {symbol1}')
        if index1.size == 0:
            index1_1 = topology.select(f'residue {resSeq1} and resname {resName1} and name {symbol1}1')
            index1_2 = topology.select(f'residue {resSeq1} and resname {resName1} and name {symbol1}2')
            index1 = np.concatenate((index1_1, index1_2))
        index1 = np.array(index1).astype(int)

        index2 = topology.select(f'residue {resSeq2} and resname {resName2} and name {symbol2}')
        if index2.size == 0:
            index2_1 = topology.select(f'residue {resSeq2} and resname {resName2} and name {symbol2}1')
            index2_2 = topology.select(f'residue {resSeq2} and resname {resName2} and name {symbol2}2')
            index2 = np.concatenate((index2_1, index2_2))
        index2 = np.array(index2).astype(int)


#        index1 = np.concatenate([topology.select(f'residue {resSeq1} and resname {resName1} and name {symbol1}'),
#                       topology.select(f'residue {resSeq1} and resname {resName1} and name {symbol1}1'),
#                       topology.select(f'residue {resSeq1} and resname {resName1} and name {symbol1}2')])
#        index1 = index1.astype(int)
#
#        index2 = np.concatenate([topology.select(f'residue {resSeq2} and resname {resName2} and name {symbol2}'),
#                       topology.select(f'residue {resSeq2} and resname {resName2} and name {symbol2}1'),
#                       topology.select(f'residue {resSeq2} and resname {resName2} and name {symbol2}2')])
#        index2 = index2.astype(int)

        print(index1)
        print(index2)


        #exit()

        if (len(index1) == 0) or (len(index2) == 0):
            if verbose: print("Warning: could not find index...")
            #exit()
        elif (len(index1) == 1) and (len(index2) == 1):
            #new_id = int(row["ID"].to_numpy()[0])
            new_id = int(row.index.to_numpy()[0])
            if (_id != new_id):
                _id = new_id
                id += 1
            idx1,idx2 = table.iloc[index1[0]]["serial"], table.iloc[index2[0]]["serial"]
            indices.append(np.array([idx1,idx2]))
            exp_data.append(row["Distance_val"].to_numpy()[0])
            restraint_index.append(id)
        else:
            if len(index1) > len(index2):

                for j in range(len(index1)):
                    new_id = int(row.index.to_numpy()[0])
                    if (_id != new_id):
                        _id = new_id
                        id += 1
                    idx1,idx2 = table.iloc[index1[j]]["serial"], table.iloc[index2[0]]["serial"]
                    indices.append(np.array([idx1,idx2]))
                    exp_data.append(row["Distance_val"].to_numpy()[0])
                    restraint_index.append(id)

            if len(index1) <= len(index2):

                for j in range(len(index2)):
                    new_id = int(row.index.to_numpy()[0])
                    if (_id != new_id):
                        _id = new_id
                        id += 1
                    idx1,idx2 = table.iloc[index1[0]]["serial"], table.iloc[index2[j]]["serial"]
                    indices.append(np.array([idx1,idx2]))
                    exp_data.append(row["Distance_val"].to_numpy()[0])
                    restraint_index.append(id)

    return np.array(indices),np.array(restraint_index),np.array(exp_data)

# }}}


if __name__ == "__main__":


    #nstates = 100
    nstates = 500

    use_all_noes = 1

    convert_distances_to_intensities = 0

    skip_dirs = ["biceps", "experimental_data"]
    system_dirs = next(os.walk("./"))[1]
    system_dirs = [ff for ff in system_dirs if ff not in skip_dirs]
    system_dirs = biceps.toolbox.natsorted(system_dirs)
    system_dirs = [dir for dir in system_dirs if dir != 'old_systems']
    #print(system_dirs)
    #exit()

    #['RUN0_1a', 'RUN1_1b', 'RUN2_2b', 'RUN2_3b', 'RUN3_4a', 'RUN4_3a', 'RUN4_4b', 'RUN5_5', 'RUN6_8', 'RUN7_2a', 'RUN12_2', 'RUN13_1']

    pbar = tqdm(total=len(system_dirs))

    #system_dirs = system_dirs[-2:]
    for sys_name in system_dirs:
        if sys_name == 'RUN0_1a': continue
        if sys_name == 'RUN1_1b': continue
        if sys_name == 'RUN2_2b': continue
        if sys_name == 'RUN2_3b': continue
        if sys_name == 'RUN3_4a': continue
        if sys_name == 'RUN4_3a': continue
        if sys_name == 'RUN4_4b': continue
#        if sys_name == 'RUN5_5': continue
        if sys_name == 'RUN6_8': continue
        if sys_name == 'RUN7_2a': continue
        if sys_name == 'RUN12_2': continue
        if sys_name == 'RUN13_1': continue

        print(sys_name)
        #exit()

        # Find out if the sequences match the experimental files
        #############################################################################
        structure_file = biceps.toolbox.get_files(f"{sys_name}/*cluster0_*.pdb")[0]
        frame = md.load(structure_file)
        topology = frame.topology
        table = topology.to_dataframe()[0]
        #print(table)
        #exit()
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
        print(f"Structure file order: {resNames}")

        exp_cs_csv_file = f"experimental_data/cs/{sys_name.split('_')[-1]}.csv"
        print(exp_cs_csv_file)
        df = pd.read_csv(exp_cs_csv_file, comment="#")
        print(df)
        print(f"Exp file order: {df['Residue'].to_numpy()}")
        #exit()

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
        #exp_order = np.char.strip(exp_order)
        #struct_order = np.char.strip(struct_order)
        print(exp_order.shape)
        print(struct_order.shape)

        # Compare arrays using list comprehensions
        a = sum([1 for x, y in zip(exp_order.tolist(), struct_order.tolist()) if x == y])
        b = sum([1 for x, y in zip(exp_order.tolist(), struct_order[::-1].tolist()) if x == y])

#        a = np.sum(np.equal(exp_order, struct_order))
#        b = np.sum(np.equal(exp_order, struct_order[::-1]))
        if a >= b:
            reverse = False
        else:
            reverse = True
        print("Reverse order of sequence: ",reverse)
        #print(exp_order)
        #print(struct_order)
        #exit()


        #############################################################################
        exp_cs_csv_file = f"experimental_data/cs/{sys_name.split('_')[-1]}.csv"

        #----------------------------------------------------------------------
        # NOTE: all of the cs experiments were performed at 25°C except system 8 (-10°C)
        # TODO: need to find the pH of solution
        #----------------------------------------------------------------------
        if sys_name == "RUN6_8": temp=263.15 # -10°C
        else: temp=298.15 # 25°C



        df = pd.read_csv(exp_cs_csv_file, comment="#")
        # IMPORTANT: NOTE: IMPORTANT: order is reversed !!!
        if add_fake_capping_groups:
            # Add a row of NaNs at the top
            df.loc[-1] = "NaN"
            df.index = df.index + 1
            df.sort_index(inplace=True)
            # Add a row of NaNs at the bottom
            df.loc[len(df)] = "NaN"

        if reverse: df = df.iloc[::-1]

        atom_sel_cs="HA" # H
        extension_cs="Ha"

        # IMPORTANT:
        # Hα   Hα1   Hα2    Hβ   Hβ1   Hβ2    Hγ   Hγ1   Hγ2   Hδ1  Hδ2
        # FIXME: only getting Ha for right now....
        # IMPORTANT:
        #print(df)
        exp_cs_Ha = df["Hα"].to_numpy(dtype=float)
        cs_df = df.copy()
        cs_df = cs_df.reset_index()

        _ = []
        for res in cs_df["Residue"].to_numpy():
            try:
                _.append(biceps.toolbox.one2three(res))
            except(Exception) as e:
                _.append(res)
                pass
        cs_df["resName"] = _
        #cs_df["resName"] = [str(re.split(r'\d+',res)[0]) for res in cs_df["Residue"].to_numpy()]

#        cs_df["resIdx1"] = [re.split(r'\D+',res)[1] for res in cs_df["Residue"].to_numpy()]

        cs_df["Seq_ID_1"] = [i+1 for i in range(len(cs_df["Residue"].to_numpy()))]
        cs_df["PDB_atom_name_1"] = [atom_sel_cs for i in range(len(cs_df["Seq_ID_1"].to_numpy()))]
        cs_df["PDB_residue_name_1"] = [re.split(r'\d+',res)[0].upper() for res in cs_df["resName"].to_numpy()]

        print(cs_df)
        #exit()

        # NOTE: the experimental file has index 0, but there's no HA on the zeroith residue
        #the output of shiftx2 suggests that it starts with index 1
        states = biceps.toolbox.get_files(f"{sys_name}/*cluster0_*.pdb")
        indices_cs_Ha = np.where(~np.isnan(np.array(exp_cs_Ha)))[0]
        skip_idx_cs = [i for i in range(len(cs_df)) if i not in indices_cs_Ha]
        print(skip_idx_cs)

        exp_cs_Ha = exp_cs_Ha[indices_cs_Ha]
        # NOTE: if resName is not a natural AA, then we have to delete that experiemntal
        # data point, b/c shiftx2 does not predict those
        residues = [str(re.split(r'\d+',res)[0]) for res in cs_df["resName"].iloc[indices_cs_Ha].to_numpy()]
        AAs = SeqUtils.IUPACData.protein_letters_3to1.keys()
        nonAA_indices = [r for r,res in enumerate(residues) if res not in AAs]

        indices_cs_Ha,restraint_index,exp_data, error_res = get_cs_indices(states[0], cs_df, col="Hα")
        print()
        print(indices_cs_Ha)
        print(restraint_index)
        print(exp_data)
        print(error_res)
        skip_idx_cs = [] #error_res
        #exit()

        #indices_cs_Ha = np.delete(indices_cs_Ha, nonAA_indices, axis=0)

        #if error_res == []: remove_Idx = nonAA_indices
        #else: remove_Idx = np.concatenate([nonAA_indices, error_res])

        #exp_cs_Ha = np.delete(exp_cs_Ha, remove_Idx, axis=0)
        exp_cs_Ha = np.stack([restraint_index, exp_data], axis=1)
        print(exp_cs_Ha)
        #exit()

        #np.savetxt(f"{sys_name}/exp_cs_Ha.txt", indices_cs_Ha)
        #print(cs_df)
        #print(indices_cs_Ha)
        #print(exp_cs_Ha)
        #exit()
        np.savetxt(f"{sys_name}/exp_cs_{extension_cs}.txt", exp_cs_Ha)
        #exit()




        #############################################################################
        if (sys_name == "RUN5_5") and (use_all_noes):
#            exp_NOE_csv_file = f"experimental_data/NOE/{sys_name.split('_')[-1]}_with_sidechains.csv"
            exp_NOE_csv_file = f"experimental_data/NOE/{sys_name.split('_')[-1]}_with_avg_sidechains.csv"
        else:
            exp_NOE_csv_file = f"experimental_data/NOE/{sys_name.split('_')[-1]}.csv"
        if sys_name != "RUN6_8":
            df = pd.read_csv(exp_NOE_csv_file, comment="#")
            #print(df)
            #exit()

            # Have to make sure the column is named Distance_val for consistency with function
            df["Distance_val"] = df['Distance rAB (Å)'].to_numpy()
            exp_noe = df['Distance_val'].to_numpy()
            #print(exp_noe)
            _A,_B = [],[]
            resA_list = df["Residue A"].to_numpy()
            resB_list = df["Residue B"].to_numpy()
            for i in range(len(resA_list)):
                resA, resB = resA_list[i], resB_list[i]
                label = f"{resA} - {resB}"
    #            print(label)

                try:
                    _A.append(biceps.toolbox.one2three(resA))
                except(Exception) as e:
                    _A.append(resA)

                try:
                    _B.append(biceps.toolbox.one2three(resB))
                except(Exception) as e:
                    _B.append(resB)

            df["Seq_ID_1"]  = [re.split(r'\D+',res)[1] for res in _A]
            df["PDB_residue_name_1"] = [re.split(r'\d+',res)[0].upper() for res in _A]
            #print(df)
            #exit()
            #print(df["PDB_residue_name_1"])
            df["Seq_ID_2"]  = [re.split(r'\D+',res)[1] for res in _B]

            _temp = []
            for res in _B:
                _temp.append(re.split(r'\D+',res)[1])

            df["Seq_ID_2"]  = _temp


#            # Assuming _B is one of the columns in your DataFrame
#            df["Seq_ID_2"] = [
#                re.split(r'\D+', str(res))[1] if isinstance(res, str) else np.nan
#                for res in df['Seq_ID_1']
#            ]

            #df["PDB_residue_name_2"] = [re.split(r'\d+',res)[0].upper() for res in _B]
            df["PDB_residue_name_2"] = [
                re.split(r'\d+',res)[0].upper() if isinstance(res, str) and len(re.split(r'\d+', str(res))) > 1 else np.nan
                for res in _B
            ]
#            print(df)
#            print(df["PDB_residue_name_2"])
#            exit()



            # IMPORTANT: NOTE: IMPORTANT: order is reversed !!!
            if reverse:
                _min, _max = table["resSeq"].to_numpy()[0], table["resSeq"].to_numpy()[-1]
                df["Seq_ID_1"] = [find_corresponding_value_in_reversed_list(int(val), _min, _max) for val in df["Seq_ID_1"].to_numpy()]
                df["Seq_ID_2"] = [find_corresponding_value_in_reversed_list(int(val), _min, _max) for val in df["Seq_ID_2"].to_numpy()]


            structure_file = biceps.toolbox.get_files(f"{sys_name}/*cluster0_*.pdb")[0]
            frame = md.load(structure_file)
            topology = frame.topology
            table = topology.to_dataframe()[0]
            resNames = list(set(table["resName"].to_numpy()))

            # NOTE: Here I will use difflib to find the nearest match for each
            # of the residue names given in the experimental files
            import difflib
            new_res_1, new_res_2 = [],[]
            print(resNames)
            print(df)
            for i in range(len(df["PDB_residue_name_1"])):
                A, B = df["PDB_residue_name_1"].to_numpy()[i], df["PDB_residue_name_2"].to_numpy()[i]
                if A == "X": A = "ABU"
                if B == "X": B = "ABU"

                if A == "A(CL)": A = "ALC"
                if B == "A(CL)": B = "ALC"

                if A == "F(I)": A = "AAA"
                if B == "F(I)": B = "AAA"
                if sys_name == "RUN12_2":
                    #if A == "NLE": A = "HIS"
                    #if B == "NLE": B = "HIS"
                    if A == "NLE": A = "HSI"
                    if B == "NLE": B = "HSI"


#['ACE1', 'L2', 'T3', 'AAA4', 'E5', 'P6', 'G7', 'K8', 'H9', 'S10', 'I11', 'NHE12']
#['GLU', 'LYS', 'SER', 'LEU', 'THR', 'PRO', 'HIS', 'ACE', 'GLY', 'NHE', 'AAA', 'ILE']

                print(A, B)
                #print(resNames)
                new_res_1.append(difflib.get_close_matches(A, resNames, n=1)[0])
                print("B, resNames = ", B, resNames, " close_mtch_idx = ",difflib.get_close_matches(B, resNames, n=1))
                new_res_2.append(difflib.get_close_matches(B, resNames, n=1)[0])
            df["PDB_residue_name_1"] = new_res_1
            df["PDB_residue_name_2"] = new_res_2

            # NOTE: TODO: Do the same thing for atom pairs
            df["PDB_atom_name_1"] = ["H"+edit_proton_name(re.split(r'\d+',res)[1]) for res in resA_list]
            df["PDB_atom_name_2"] = ["H"+edit_proton_name(re.split(r'\d+',res)[1]) for res in resB_list]
            #print(df)
            atomNames = list(set(table["name"].to_numpy()))
            #print(df["PDB_atom_name_2"])
            #exit()
            new_atom_1, new_atom_2 = [],[]
            print(atomNames)
            for i in range(len(df["PDB_atom_name_1"])):
                A, B = str(df["PDB_atom_name_1"].to_numpy()[i]).upper(), str(df["PDB_atom_name_2"].to_numpy()[i]).upper()
                if A.startswith("HH"): A[1:]
                if B.startswith("HH"): B[1:]
                print(i, A, B)
                print(atomNames)
                print(difflib.get_close_matches(A, atomNames, n=1), difflib.get_close_matches(B, atomNames, n=1))
                new_atom_1.append(difflib.get_close_matches(A, atomNames, n=1)[0])
                new_atom_2.append(difflib.get_close_matches(B, atomNames, n=1)[0])
            #exit()
            df["PDB_atom_name_1"] = new_atom_1
            df["PDB_atom_name_2"] = new_atom_2

            # IMPORTANT: Need to remap the index of each residue since the experimentalist made things difficult...
            df['Seq_ID_1'] = df['Seq_ID_1'].astype(int)
            df['Seq_ID_2'] = df['Seq_ID_2'].astype(int)
            max_value = int(np.max(df['Seq_ID_1'].to_numpy()))
    # FIXME:?
    #################
            # check if the index needs to be shifted
            x,y = 0,0
            for res, seq in zip(df["PDB_residue_name_1"].to_numpy(), df['Seq_ID_1'].to_numpy()):
                print(res, seq)
                try:
                    if table.iloc[np.where(table["resSeq"].to_numpy() == seq-1)[0]]["resName"].to_numpy()[0] == res:
                        x += 1
                except(Exception) as e: print(e)

                try:
                    if table.iloc[np.where(table["resSeq"].to_numpy() == seq)[0]]["resName"].to_numpy()[0] == res:
                        y += 1
                except(Exception) as e: print(e)

            if x >= y: shift = -1
            else: shift = 0

    #        ordering = np.array(range(max_value))[::-1] + 1
    #        _a, _b = [],[]
    #        for a,b in zip(df['Seq_ID_1'].to_numpy(), df['Seq_ID_2'].to_numpy()):
    #            _a.append(ordering[a+shift])
    #            _b.append(ordering[b+shift])
    #        df['Seq_ID_1'] = _a
    #        df['Seq_ID_2'] = _b

            _a, _b = [],[]
            for a,b in zip(df['Seq_ID_1'].to_numpy(), df['Seq_ID_2'].to_numpy()):
                _a.append(a+shift)
                _b.append(b+shift)
            df['Seq_ID_1'] = _a
            df['Seq_ID_2'] = _b
            print(df)


            indices_dir = f"{sys_name}/indices"
            biceps.toolbox.mkdir(indices_dir)
            skip_dirs = ['indices']
            # NOTE: Get indices for NOE distances
            indices_NOE = f"{indices_dir}/NOE_indices.txt"
            indices,restraint_index,exp_data = get_NOE_indices(structure_file, exp_df=df, verbose=False)
            index_correction = 1
            indices = indices - index_correction
            print(indices)
            print(restraint_index)
            #print(indices)
            #exit()
            np.savetxt(indices_NOE, indices, fmt="%i")

            if convert_distances_to_intensities:
                exp_data = exp_data**(-6)

            exp_data = np.array([restraint_index, exp_data]).T
            #print(exp_data)
            #exit()
            np.savetxt(f"{sys_name}/exp_NOE.txt", exp_data)

            #print(df)
            #exit()
            #############################################################################

            exp_J_csv_file = f"experimental_data/J/{sys_name.split('_')[-1]}.csv"
            print(exp_J_csv_file)
            df = pd.read_csv(exp_J_csv_file, comment="#")
            # IMPORTANT: NOTE: IMPORTANT: order is reversed !!!
            if add_fake_capping_groups:
                # Add a row of NaNs at the top
                df.loc[-1] = np.nan
                df.index = df.index + 1
                df.sort_index(inplace=True)
                # Add a row of NaNs at the bottom
                df.loc[len(df)] = np.nan
                #print(df)
                #exit()

            if reverse: df = df.iloc[::-1]
            df["Residue position"] = [i+1 for i in range(len(df))]
            df = df.reset_index(drop=True)
            print(df)



            #print(df.columns)
            #df.columns = [val for val in df.columns.to_list()]
            exp_J = df['J (Hz)'].to_numpy()
            print(exp_J)

            skip_idx_J = np.where(np.isnan(exp_J) == True)[0]
            print(skip_idx_J)
            #exit()

            #exp_J = np.delete(exp_J, skip_idx_J)
            exp_J = exp_J[np.where(~np.isnan(np.array(exp_J)))[0]]

            # NOTE: need to remove the last index, no Gly
            #skip_idx_J = np.delete(skip_idx_J, len(skip_idx_J)-1)
            exp_J = np.stack([list(range(len(exp_J))), exp_J], axis=1)
            #print(skip_idx_J)
            exp_J_data = f"{sys_name}/exp_J.txt"
            np.savetxt(exp_J_data, exp_J)
            #print(exp_J)
            #print(skip_idx_J)
            #exit()


            J = biceps.J_coupling.compute_J3_HN_HA(md.load(structure_file, top=frame), model="Bax2007")
            table = topology.to_dataframe()[0]
            resName = table["resName"].to_numpy()
            resNames = np.array([val+str(table["resSeq"].to_numpy()[r]) for r,val in enumerate(resName)])
            print(J)
            print(resName)
            _, idx = np.unique(resNames, return_index=True)
            print(len(idx))
            print(len(resNames))
            #print(np.sort(idx))
            df["Residue"] = resNames[np.sort(idx)]
            print(df)
            #exit()
            residues = df["Residue"].to_numpy()
            print(residues)
            print(skip_idx_J)

            listing = []
            top_res = []
            for k in range(len(J[0])):
                idxs = [topology.atom(idx) for idx in J[0][k]]
                #res = str(idxs[1])[3:].split("-")[0]
                res = str(idxs[1]).split("-")[0]
                print(res)
                top_res.append(res)
                if res in residues[skip_idx_J]:
                    listing.append(int(np.where(res == np.array(residues[skip_idx_J]))[0]))
                    #listing.append(k)
                    continue
                print(k, idxs)
            #exit()
            print("----")
            print(residues)
            print(top_res)
            print(listing)
            # NOTE: IMPORTANT: FIXME: Make sure that this - 1 is correct on all systems
            skip_idx_J = skip_idx_J[listing] - 1
            #if sys_name == "RUN12_2": skip_idx_J = np.array([skip_idx_J[0]])
            print(skip_idx_J)


            #############################################################################
            if (sys_name == "RUN13_1"):
                print(table)
                exp_RDC_csv_file = f"experimental_data/rdc/{sys_name.split('_')[-1]}.csv"
                exp_RDC_data_df = pd.read_csv(exp_RDC_csv_file, index_col=0, comment="#")
                #print(exp_RDC_data_df)
                #exit()

                atom1, atom2 = exp_RDC_data_df['atom1'].to_numpy()[0], exp_RDC_data_df['atom2'].to_numpy()[0]
                rdc_extension = f"rdc_{atom1}_{atom2}"
                exp_rdc = np.stack([list(range(len(exp_RDC_data_df["exp"].to_numpy()))), exp_RDC_data_df["exp"].to_numpy(dtype=float)], axis=1)
                np.savetxt(f"{sys_name}/exp_{rdc_extension}.txt", exp_rdc)

                structure_file = states[0]
                indices_rdc = get_rdc_indices(structure_file, exp_RDC_data_df)
                #print(indices_rdc)
                #exit()
                #np.savetxt(indices_rdc, "indices.txt", fmt="%i")



        #############################################################################
        #############################################################################

        data_dir = f"{sys_name}"

        if nstates == 500:
            ext = "500_states/"
        else:
            ext = ""

        #indices_NOE = f"{indices_dir}/NOE_indices.txt"
        #indices_cs = f"{indices_dir}/cs_indices.txt"
        #indices_J = f"{indices_dir}/J_indices.txt"
        #continue
        # Create output directories for biceps input files
        outdir_CS_J_NOE = f"{data_dir}/{ext}CS_J_NOE/"
        biceps.toolbox.mkdir(outdir_CS_J_NOE)

        outdir_J = f"{data_dir}/{ext}J/"
        biceps.toolbox.mkdir(outdir_J)

        outdir_CS = f"{data_dir}/{ext}CS/"
        biceps.toolbox.mkdir(outdir_CS)

        outdir_NOE = f"{data_dir}/{ext}NOE/"
        biceps.toolbox.mkdir(outdir_NOE)

        if (sys_name == "RUN13_1"):
            outdir_RDC = f"{data_dir}/{ext}{rdc_extension}/"
            biceps.toolbox.mkdir(outdir_RDC)

#        nstates = int(biceps.toolbox.get_files(f"{data_dir}/*.pdb")[-1].split("cluster")[-1].split("_")[0])+1
        snaps_per_state = len(biceps.toolbox.get_files(f"{data_dir}/*cluster0_*.pdb"))
        snapshots = biceps.toolbox.get_files(f"{data_dir}/*.pdb")
        print(f"nstates = {nstates}; nsnapshots = {len(snapshots)}")
        print(f"snaps per state = {snaps_per_state};")
        if nstates == 100:
            files_by_state = [biceps.toolbox.get_files(
                f"{data_dir}/*cluster{state}_*.pdb"
                ) for state in range(nstates)]
        elif nstates == 500:
            files_by_state = [[file] for file in biceps.toolbox.get_files(f"{data_dir}/*cluster*_*.pdb")]
        else:
            print("nstates is not right")
            exit()




        #print(files_by_state)
        #exit()

        ## NOTE: if you want to do testing with fewer files:
        #files_by_state = [[biceps.toolbox.get_files(
        #    f"{data_dir}/*cluster{state}_*.pdb"
        #    )[0]] for state in range(nstates)]



        if (sys_name == "RUN13_1"):
            compute_ensemble_avg_RDC(files_by_state, exp_RDC_data_df, outdir=outdir_RDC, skip_idx=[], verbose=False)
            #exit()


        if sys_name != "RUN6_8":
            compute_ensemble_avg_NOE(files_by_state, indices=indices_NOE, outdir=outdir_NOE, convert_distances_to_intensities=convert_distances_to_intensities)
#            _indices_J, model_J_data = compute_ensemble_avg_J(files_by_state, outdir_J, skip_idx=skip_idx_J, return_index=1)

#NOTE: HERE

#        #if sys_name != "RUN12_2":
#        # FIXME: the temperature should be 25°C and I'm not sure about the pH
#        compute_ensemble_avg_CS(files_by_state, outdir_CS, temp=temp, pH=7.0,
#                atom_sel=atom_sel_cs, skip_idx=skip_idx_cs, return_index=False, verbose=False, compute=1)



        # NOTE: code to read in the experimental chemical shift data
        exp_cs_data = f"{sys_name}/exp_cs_{extension_cs}.txt"
        model_cs_data = f"{outdir_CS}/*.txt"

        exp_J_data = f"{sys_name}/exp_J.txt"
        model_J_data = f"{outdir_J}/*.txt"

        exp_NOE_data = f"{sys_name}/exp_NOE.txt"
        model_NOE_data = f"{outdir_NOE}/*.txt"

        if (sys_name == "RUN13_1"):
            exp_RDC_data = f"{sys_name}/exp_{rdc_extension}.txt"
            model_RDC_data = f"{outdir_RDC}/*.txt"

        verbose=True
        # NOTE: Prepare the input files for biceps
        preparation = biceps.Restraint.Preparation(nstates=nstates, top_file=structure_file, outdir=outdir_CS_J_NOE)
        #if sys_name != "RUN12_2":
        print(exp_cs_Ha)
        print(model_cs_data)
#        preparation.prepare_cs(exp_cs_Ha, model_cs_data, indices_cs_Ha, extension=extension_cs, verbose=verbose)
        if sys_name != "RUN6_8":
#             preparation.prepare_J(exp_J_data, model_J_data, _indices_J, extension="J", verbose=verbose)
             preparation.prepare_noe(exp_NOE_data, model_NOE_data, indices_NOE, verbose=verbose)
             exit()

        if (sys_name == "RUN13_1"):
             preparation.prepare_rdc(exp_RDC_data, model_RDC_data, indices_rdc, extension=f"{rdc_extension}", verbose=verbose)
             #exit()

################################################################################

        df = pd.read_csv(exp_cs_csv_file, comment="#")
        columns = df.columns.to_list()
        columns = [edit_proton_name(col) for col in columns]
        _ = []
        for col in columns:
            s = ""
            for char in col:
                if char.isdigit():
                    s += str(int(char)+1)
                else:
                    s += char
            _.append(s)
        columns = _
        df.columns = columns

        if add_fake_capping_groups:
            # Add a row of NaNs at the top
            df.loc[-1] = "NaN"
            df.index = df.index + 1
            df.sort_index(inplace=True)
            # Add a row of NaNs at the bottom
            df.loc[len(df)] = "NaN"

        if reverse: df = df.iloc[::-1]

        for col in columns[1:]:
        #for col in columns[5:]:

            print(df)
            #exit()
            atom_sel_cs = col

            try: extension_cs= atom_sel_cs[:1] + atom_sel_cs[1].lower() + atom_sel_cs[2:]
            except(Exception) as e: extension_cs= atom_sel_cs

            print(extension_cs)
            exp_cs = df[col].to_numpy(dtype=float)
            cs_df = df.copy()
            cs_df = cs_df.reset_index()
            print(exp_cs)


            _ = []
            for res in cs_df["Residue"].to_numpy():
                try:
                    _.append(biceps.toolbox.one2three(res))
                except(Exception) as e:
                    _.append(res)
                    pass
            cs_df["resName"] = _



            cs_df["Seq_ID_1"] = [i+1 for i in range(len(cs_df["Residue"].to_numpy()))]
            cs_df["PDB_atom_name_1"] = [atom_sel_cs for i in range(len(cs_df["Seq_ID_1"].to_numpy()))]
            cs_df["PDB_residue_name_1"] = [re.split(r'\d+',res)[0].upper() for res in cs_df["resName"].to_numpy()]
            print(cs_df)

            # NOTE: the experimental file has index 0, but there's no HA on the zeroith residue
            #the output of shiftx2 suggests that it starts with index 1
            states = biceps.toolbox.get_files(f"{sys_name}/*cluster0_*.pdb")
            indices_cs = np.where(~np.isnan(np.array(exp_cs)))[0]
            skip_idx_cs = [i for i in range(len(cs_df)) if i not in indices_cs]
            print(skip_idx_cs)

            exp_cs = exp_cs[indices_cs]
            # NOTE: if resName is not a natural AA, then we have to delete that experiemntal
            # data point, b/c shiftx2 does not predict those
            residues = [str(re.split(r'\d+',res)[0]) for res in cs_df["resName"].iloc[indices_cs].to_numpy()]
            AAs = SeqUtils.IUPACData.protein_letters_3to1.keys()
            nonAA_indices = [r for r,res in enumerate(residues) if res not in AAs]

            indices_cs,restraint_index,exp_data, error_res = get_cs_indices(states[0], cs_df, col=col)
            print()
            print(indices_cs)
            print(restraint_index)
            print(exp_data)
            print(error_res)
            skip_idx_cs = [] #error_res
            exp_cs = np.stack([restraint_index, exp_data], axis=1)
            print(exp_cs)

            if indices_cs.size == 0: continue
            #if indices_cs.shape[0] != exp_cs.shape[0]:
            print("Shapes: (indices_cs.shape[0], exp_cs.shape[0]) = ",indices_cs.shape[0], exp_cs.shape[0])
            if indices_cs.shape[0] != exp_cs.shape[0]: continue

            np.savetxt(f"{sys_name}/exp_cs_{extension_cs}.txt", exp_cs)

            try:
                # FIXME: the temperature should be 25°C and I'm not sure about the pH
                idxs = compute_ensemble_avg_CS(files_by_state, outdir_CS, temp=temp, pH=7.0,
                        atom_sel=atom_sel_cs, skip_idx=skip_idx_cs, return_index=False, verbose=False, compute=0)
            except(Exception) as e: print(e);continue

################################################################################
            indices_cs,restraint_index,exp_data, error_res = get_cs_indices(states[0], cs_df, col=col)
            #print("indices_cs = ",indices_cs)
            print(exp_data)
            arr = cs_df[col].to_numpy(dtype=float)
            arr[np.where(arr == "NaN")[0]] = np.nan
            print(idxs)
            arr = np.array(arr[idxs])
            print(arr)
            matches = np.concatenate([np.where(np.array(exp_data)==val)[0] for val in arr])
            exp_cs = np.stack([restraint_index, exp_data], axis=1)[matches]
            indices_cs = np.array(indices_cs[matches])
            print("exp_cs = ",exp_cs)
            print("indices_cs = ",indices_cs)
            #exit()
            print(indices_cs.shape)

################################################################################

            # NOTE: code to read in the experimental chemical shift data
            exp_cs_data = f"{sys_name}/exp_cs_{extension_cs}.txt"
            model_cs_data = f"{outdir_CS}/*.txt"

            verbose=True
            # NOTE: Prepare the input files for biceps
            try:
                preparation.prepare_cs(exp_cs, model_cs_data, indices=indices_cs, extension=extension_cs, verbose=verbose)
            except(IndexError) as e:
                # NOTE: this IndexError is a result of degenerate protons.
                _, idx = np.unique(exp_cs, axis=0, return_index=True)
                idx = np.sort(idx)
                exp_cs = exp_cs[idx]
                indices_cs = indices_cs[idx]
                print("exp_cs = ",exp_cs)
                print("indices_cs = ",indices_cs)
                preparation.prepare_cs(exp_cs, model_cs_data, indices=indices_cs, extension=extension_cs, verbose=verbose)



        pbar.update(1)
        print(pbar)
#        exit()
    pbar.close()



#def double_check_data(structure_file, df, col, model_indices):
#    indices_cs,restraint_index,exp_data, error_res = get_cs_indices(states[0], cs_df, col=col)
#    arr = cs_df[col].to_numpy(dtype=float)
#    arr[np.where(arr == "Nan")[0]] = np.nan
#    matches = np.concatenate([np.where(np.array(exp_data)==val)[0] for val in arr])
#    exp_cs = np.stack([restraint_index, exp_data], axis=1)[matches]
#    indices_cs = indices_cs[matches]
#    return indices_cs, exp_cs




