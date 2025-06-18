
## **High-Resolution Tuning of Non-Natural and Cyclic Peptide Folding Landscapes against NMR Measurements Using Markov Models and Bayesian Inference of Conformational Populations**

[![DOI: 10.1021/acs.jctc.5c00489](https://img.shields.io/badge/DOI-10.1021%2Facs.jctc.5c00489-blue.svg)](https://doi.org/10.1021/acs.jctc.5c00489)

**Authors**
* Thi Dung Nguyen
    - Department of Chemistry, Temple University
* Robert M. Raddi
    - Department of Chemistry, Temple University
* Vincent A. Voelz
    - Department of Chemistry, Temple University
    

We model the folding landscapes of 12 linear and cyclic peptide $\beta$-hairpin mimics studied by the Erdelyi group, with the goal of reproducing the effects of subtle chemical modifications on peptide folding stability. The Bayesian Inference of Conformational Populations (BICePs) algorithm was first used to refine Karplus parameters to obtain an optimal forward model for scalar coupling constants; then, BICePs was used to reweight conformational ensembles against experimental NMR observables.  Compared to previous estimates of folded populations made using the NAMFIS algorithm, BICePs-reweighted folding landscapes predict that the introduction of a side chain hydrogen- or halogen-bonding group changes the folding stability by no more than 2 kJ mol$^{–1}$. The overall agreement between simulated and experimental NMR observables suggests that our approach is highly robust, offering a reliable pathway for designing foldable non-natural and cyclic peptides.



### Below is a description of what this repository contains:

- [`Systems/`](Systems/): a directory containing folders for each system with corresponding structure files
- [`Systems/experimental_data`](Systems/experimental_data): all experimental data for each observable
- [`scripts/`](scripts/): scripts that use `biceps` to perform forward model optimization, reweighting, and plotting.
  - [`scripts/runme.py`](scripts/runme.py): performs BICePs reweighting for a given peptide
  - [`scripts/fmo_peptide.py`](scripts/fmo_peptide.py): performs forward model optimization for a particular peptide
  - [`scripts/...`](scripts/...): the remaining scripts are for plotting and other analysis.
 


### Installation requirements:

To run these scripts and notebook, you will need to install [`biceps`](https://github.com/vvoelz/biceps). Specifically, you will want [`biceps v3.0a`](https://github.com/vvoelz/biceps/tree/biceps_v3.0a).








