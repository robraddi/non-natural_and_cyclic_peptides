

# Libraries:{{{
import biceps
import numpy as np
import pandas as pd
import os, re, string, copy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import uncertainties as u ################ Error Prop. Library
import uncertainties.unumpy as unumpy #### Error Prop.
from uncertainties import umath
import seaborn as sns
from scipy import stats
#:}}}


import numpy as np

def calculate_free_energy(folded_state_pop):
    if unumpy.isnan(folded_state_pop.nominal_value):
        return u.ufloat(np.nan, np.nan)

    F = folded_state_pop
    U = 100.0 - F

    # Check if F or U are outside valid ranges
    if F.nominal_value <= 0 or U.nominal_value <= 0:
        return u.ufloat(np.nan, np.nan)

    deltaG = -8.314 * 298.0 / 1000.0 * unumpy.log(F / U)
    return deltaG


results = [
{"group": 2, "sys":"1a", "Exp":u.ufloat(np.nan, np.nan), "NAMFIS":u.ufloat(88,0)          , "MSM":u.ufloat(18,0), "BICePs":u.ufloat(53,1.4)},
{"group": 2, "sys":"1b", "Exp":u.ufloat(np.nan, np.nan), "NAMFIS":u.ufloat(50,0)          , "MSM":u.ufloat(30,0), "BICePs":u.ufloat(39,1.5)},
{"group": 3, "sys":"2a", "Exp":u.ufloat(np.nan, np.nan), "NAMFIS":u.ufloat(58,0)          , "MSM":u.ufloat(45,0), "BICePs":u.ufloat(54,1.8)},
{"group": 3, "sys":"2b", "Exp":u.ufloat(np.nan, np.nan), "NAMFIS":u.ufloat(29,0)          , "MSM":u.ufloat(33,0), "BICePs":u.ufloat(49,1.8)},
{"group": 4, "sys":"3a", "Exp":u.ufloat(np.nan, np.nan), "NAMFIS":u.ufloat(11,0)          , "MSM":u.ufloat(9 ,0), "BICePs":u.ufloat(10,1.4)},
{"group": 4, "sys":"3b", "Exp":u.ufloat(np.nan, np.nan), "NAMFIS":u.ufloat(0 ,0)          , "MSM":u.ufloat(32,0), "BICePs":u.ufloat(10,1.1)},
{"group": 5, "sys":"4a", "Exp":u.ufloat(np.nan, np.nan), "NAMFIS":u.ufloat(36,0)          , "MSM":u.ufloat(9 ,0), "BICePs":u.ufloat(28,0.8)},
{"group": 5, "sys":"4b", "Exp":u.ufloat(np.nan, np.nan), "NAMFIS":u.ufloat(31,0)          , "MSM":u.ufloat(52,0), "BICePs":u.ufloat(28,0.6)},
{"group": 8, "sys":"5",  "Exp":u.ufloat(np.nan, np.nan), "NAMFIS":u.ufloat(74,0)          , "MSM":u.ufloat(44,0), "BICePs":u.ufloat(54,1.3)},
{"group": 8, "sys":"8",  "Exp":u.ufloat(np.nan, np.nan), "NAMFIS":u.ufloat(np.nan, np.nan), "MSM":u.ufloat(41,0), "BICePs":u.ufloat(45,0.0)},
{"group": 1, "sys":"1",  "Exp":u.ufloat(30, 3.2       ), "NAMFIS":u.ufloat(56,0)          , "MSM":u.ufloat(31,0), "BICePs":u.ufloat(31,0.7)},
{"group": 1, "sys":"2",  "Exp":u.ufloat(23, 1.7       ), "NAMFIS":u.ufloat(39,0)          , "MSM":u.ufloat(24,0), "BICePs":u.ufloat(26,0.6)}
]


# Create a new group for system 2a and system 5 comparison
results.append(copy.copy(results[2])) # system 2a
results[-1]["group"] = 6
results.append(copy.copy(results[8])) # system 5
results[-1]["group"] = 6

# Create a new group for system 2b and system 5 comparison
results.append(copy.copy(results[8])) # system 5
results[-1]["group"] = 7
results.append(copy.copy(results[3])) # system 2b
results[-1]["group"] = 7

# Create a new group for system 2a and system 8 comparison
results.append(copy.copy(results[2])) # system 2a
results[-1]["group"] = 9
results.append(copy.copy(results[9])) # system 8
results[-1]["group"] = 9

# Create a new group for system 2b and system 8 comparison
results.append(copy.copy(results[3])) # system 2b
results[-1]["group"] = 10
results.append(copy.copy(results[9])) # system 8
results[-1]["group"] = 10





df = pd.DataFrame(results)

models = ["Exp", "NAMFIS", "MSM", "BICePs"]
fe_results = []
for model in models:
    dGs,labels = [],[]
    for val in df[model].to_numpy():
        dGs.append(calculate_free_energy(val))
    df[model] = dGs

print(df)

df.sort_values(by=["group"], inplace=True)
# Group by "group" and calculate the difference for each column in the models list
ddG = df.groupby("group").apply(
    lambda x: pd.concat([x.assign(**{f"∆∆G {model}": x[f"{model}"].diff().shift(-1)}) for model in models], axis=1)
).dropna()

ddG["label"] = df.groupby("group")["sys"].shift(-1) + r" $\rightarrow$ " + df.groupby("group")["sys"].shift(0)

ddG.drop(columns=models, inplace=True)
ddG.drop(columns=["group", "sys"], inplace=True)
ddG = ddG.reset_index(drop=True)
print(ddG)
#exit()


models = ddG.columns[:-1]
colors = ["red", "green", "orange", "blue", "cyan", "purple", "grey"]
fig = plt.figure(figsize=(5, 4))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax = fig.add_subplot(gs[0])
for i,model in enumerate(models):

    # IMPORTANT: adding negative sign to correct the free energy difference
    #y = [val.nominal_value for val in ddG[model].to_numpy()]
    y = [-val.nominal_value for val in ddG[model].to_numpy()]
    yerr = [val.std_dev for val in ddG[model].to_numpy()]
    label = model.replace("∆∆G ", "")
    ax.axhline(y=0, ls="--", lw=0.9, color="k")
    ax.errorbar(ddG["label"].to_numpy(), y, yerr=yerr, marker="o", capsize=4,
                markeredgecolor="k", ls="", color=colors[i], label=label)#, label="__no_legend__")



ax.axvline(x=0.5, ls="-", lw=1, color="k")
ax.axvline(x=4.5, ls="-", lw=1, color="k")
ax.axvline(x=6.5, ls="-", lw=1, color="k")
#ax.axvline(x=4.5, ls="-", lw=1, color="k")

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="center")
ax.legend(loc="best")
ax.set_xlabel("IDs", fontsize=16)
ax.set_ylabel(r"$\Delta\Delta G$ [kJ mol$^{-1}$]", fontsize=16)
fig.tight_layout()
fig.savefig("∆∆G_predictions.png")


val = ddG["∆∆G Exp"].to_numpy()[-2]
print(val - val*0.3)


exit()











