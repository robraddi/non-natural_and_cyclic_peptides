import mdtraj as md
import biceps
import os
import tempfile



def align_and_save_pdbs(pdb_files, output_file):
    # Load the first pdb file to use as a reference
    reference = md.load(pdb_files[0])
    #reference = md.load(pdb_files[317])

    with open(output_file, 'w') as output:
        for pdb_file in pdb_files:#[::100]:
            # Load and align the pdb file
            traj = md.load(pdb_file)

            atom_indices = traj.topology.select("all")
            #atom_indices = traj.topology.select("backbone")
            #atom_indices = traj.topology.select("resname PRO")
            traj.superpose(reference, atom_indices=atom_indices)
            rmsd_values = md.rmsd(traj, reference, atom_indices=atom_indices)
            print(rmsd_values)


            # Save the aligned trajectory to a temporary file
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.pdb', delete=False) as tmp_file:
                traj.save(tmp_file.name)

                # Go back to the start of the temporary file
                tmp_file.seek(0)

                # Insert a comment line and write the contents of the temporary file
                output.write(f"REMARK Original file: {pdb_file}\n")
                for line in tmp_file:
                    if not line.startswith("CRYST1") and not line.startswith("MODEL") and not line.startswith("END"):
                        output.write(line)
                output.write("CONECT   1    3  2   122\n")
                output.write("END\n")

            # Remove the temporary file
            os.remove(tmp_file.name)


if __name__ == "__main__":

    # List of PDB files to align and save
    #files = biceps.toolbox.get_files("System_01/ncluster_100/RUN0_kmean100_*n0.pdb")

#    files = biceps.toolbox.get_files("RUN0_1a/*.pdb")
    files = biceps.toolbox.get_files("RUN5_5/*.pdb")
#    files = biceps.toolbox.get_files("RUN13_1/*.pdb")
#    files = biceps.toolbox.get_files("*_2a/*.pdb")
    #print(files)
    #exit()

    # Output file
#    output_file = '1a.pdb'
    output_file = '5.pdb'
#    output_file = '1.pdb'
#    output_file = '2a.pdb'
#    output_file = '2a.pdb'
    #output_file = 'combined_.pdb'

    # Save aligned structures to a single PDB file with comments
    align_and_save_pdbs(files, output_file)





