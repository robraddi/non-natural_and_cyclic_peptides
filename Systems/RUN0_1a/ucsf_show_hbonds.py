
import chimera
from chimera import runCommand as rc
import sys, glob, re
from chimera import replyobj
import numpy as np


def get_files(path):
    """Return a sorted list of files that will be globbed from the path given.
    First, this function can handle decimals and multiple numbers that are
    seperated by characters.
    https://pypi.org/project/natsort/

    Args:
        path(str) - path that will be globbed
    Returns:
        sorted list
    """

    globbed = glob.glob(path)
    return globbed


# special_pairs:{{{
def special_pairs(system):
    # Define your special pairs selection logic here
    # For example:
    if system == "RUN0_1a":
        # Define your selections for this system
        return [
                ":3@HG :8@CG",
                ":9@O :2@H",
                ":9@H :2@O",
                ":7@O :4@H",
                ":7@H :4@O"
             ]
    elif system == "RUN1_1b":
        return [
                #':3@HG :8@CG',
                ':9@O 2@H',
                ':9@H 2@O',
                ':7@O 4@H',
                ':7@H 4@O']

    elif system == "RUN2_3b":
        return [
                #':4@O2 :9@
                ':10@O 3@H',
                ':10@H 3@O',
                ':8@O 5@H',
                ':8@H 5@O',
             ]

    elif system == "RUN3_2b":
        return [
                #':3@O2 :8@CG',
                ':9@O 2@H',
                ':9@H 2@O',
                ':7@O 4@H',
                ':7@H 4@O',
        ]

    elif system == "RUN4_3a":
        return [
                #':4@O2 :9@HG',
                ':10@O 3@H',
                ':10@H 3@O',
                ':8@O 5@H',
                ':8@H 5@O',
             ]

    elif system == "RUN5_4a":
        return [
                # NOTE: RUN5_4a
                #':3@O2,:8@HG',
                ':9@O :2@H',
                ':9@H :2@O',
                ':7@O :4@H',
                ':7@H :4@O',
             ]

    elif system == "RUN6_4b":
        return [
                #':3@O2 :8@CG',
                ':9@O :2@H',
                ':9@H :2@O',
                ':7@O :4@H',
                ':7@H :4@O',
                ]

    elif system == "RUN7_5":
        return [
                #':3@O2 :8@CL1',
                ':9@O 2@H',
                ':9@H 2@O',
                ':7@O 4@H',
                ':7@H 4@O',
             ]

    elif system == "RUN8_8":
        return [
                #':3@O2 :8@BR1',
                ':9@O 2@H',
                ':9@H 2@O',
                ':7@O 4@H',
                ':6@H 4@O',
             ]

    elif system == "RUN12_2":
        return [
                #':4@I1 :9@C2',
                ':1@O 12@H',
                ':10@O 3@H',
                ':3@O 10@H',
                ':5@H 8@O',
                ':8@H 5@O',
             ]

    elif system == "RUN13_1":
        return [
                #':4@I1 :9@O2',
                ':1@O 12@H',
                ':10@O 3@H',
                ':3@O 10@H',
                ':5@H 8@O',
                ':8@H 5@O',
             ]

    else: raise ValueError("'system' is not one of ['RUN0_1a', 'RUN1_1b',...]")

# }}}



def main():
    '''
    if len(sys.argv) < 3:
        print("Usage: chimera --script \"ucsf_show_hbonds.py <pdb_file> <system>\"")
        sys.exit(1)
    '''


    #files = [get_files("RUN*_cluster%s_*.pdb"%i)[0] for i in list(sys.argv[1].split(","))]
#    files = [get_files("RUN*_cluster%s_*.pdb"%i)[1] for i in list(sys.argv[1].split(","))]
    files = ["cluster0_frame13607.pdb",
#             "cluster37_frame12964.pdb"]

             #"cluster87_frame10122.pdb"]
             #"cluster87_frame16856.pdb"]
             #"cluster87_frame20302.pdb"]
             #"cluster87_frame13277.pdb"]
             #"cluster87_frame17499.pdb"]

             #"cluster3_frame10627.pdb"]
#             "cluster3_frame16009.pdb"]
             #"cluster3_frame7356.pdb"]
             #"cluster3_frame12766.pdb"]
             "cluster3_frame18549.pdb"]




    system = "RUN0_1a"

    for file in files:
        rc("open " + file)

    # Perform matchmaker alignment
    rc("sel :")
    rc("~ribbon sel")
    rc("mmaker #0 sel")


    rc("mmaker #0:4-8.A #1:4-8.A")


    rc("show full")
    rc('display :pro')
    # NOTE: uncomment if you want to show the Br and Cl
#    rc('display :@Cl1')
#    rc('display :@Br1')

    # Select specific atoms/residues based on system and find hbonds
    selections = special_pairs(system)
    for f,file in enumerate(files):
        selection = "#%s"%f
        for sel in selections:
            s = re.split(r'\D+',sel)[1:3]
            sel = "#%s:"%f+s[0]+","+s[1]
            rc('sel ' + sel)
            rc("findhbond color green retain true selRestrict sel")

    rc("setattr g lineType 2")  # Set line to dashed
    rc("setattr g lineWidth 2") # Increase line width
    rc("~sel")



if __name__ == "__main__":
    main()





