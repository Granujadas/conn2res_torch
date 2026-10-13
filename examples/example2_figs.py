"""
Example 2: Anatomical inferences
=======================================================================
This script includes the analysis code for example 2

Note: this code includes a one-way ANOVA analysis. To run the ANOVA we
used the statsmodels Python package, which is not included in the
conn2res toolbox.
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from conn2res import plotting

import sys 
sys.path.insert(0, '/mnt/d/0_CUCEI/modular/original')

PROJ_DIR = '/mnt/d/0_CUCEI/modular/original'
DATA_DIR = '/mnt/d/0_CUCEI/modular/original/examples/data/human'
OUTPUT_DIR = '/mnt/d/0_CUCEI/modular/original/examples/results/example2'
FIG_DIR = '/mnt/d/0_CUCEI/modular/original/examples/figs'

task = 'PerceptualDecisionMaking'
rsn = [
    # 'VIS',  # not included because used as input
    'SM',
    'DA',
    'VA',
    'FP',
    'LIM',
    'DMN'
]

# #####################################################################
# Concatenate scores across task iterations
# #####################################################################
def concat_scores(connectome):
    """ Concatenate scores
    """
    scores = []
    for sample_id in range(5):
        try:
            df = pd.read_csv(
                f'{OUTPUT_DIR}/{connectome}_xy_{sample_id}_scores.csv'
            ).reset_index(drop=True)

            df['sample_id'] = sample_id

            scores.append(
                df[['sample_id', 'alpha', 'module', 'n_nodes', 'balanced_accuracy_score']]
            )
        except:
            pass

    scores = pd.concat(scores).reset_index(drop=True)

    scores.to_csv(
        f'{OUTPUT_DIR}/{connectome}_scores.csv',
        index=False
        )

    return scores

scores = concat_scores('subj_0')

# plot empirical scores as a function of alpha - per module
scores = scores.loc[scores['module'].isin(rsn), :]
plotting.plot_performance(
    df=scores.copy(), x='alpha', y='balanced_accuracy_score', normalize=True,
    hue='module', hue_order=rsn,
    show=True, savefig=True,
    fname=f'{FIG_DIR}/perf_curve_human_rsn_{task}',
    ax_params={'ylabel': task,
               },
    rc_params={'savefig.format': 'png'}
)

# #####################################################################
# one-way ANOVA
# #####################################################################
# import statsmodels.api as sm
# from statsmodels.formula.api import ols
# import statsmodels.stats.multicomp as mc
#
# def anova_table(aov):
#     aov['mean_sq'] = aov[:]['sum_sq']/aov[:]['df']
#     cols = ['df', 'sum_sq', 'mean_sq', 'F', 'PR(>F)']
#     aov = aov[cols]
#     return aov
#
# scores = concat_scores('subj_0')
# scores = scores.loc[scores['module'].isin(rsn), :]
# scores = scores.loc[scores['alpha'] == 1.0, :]
#
# # one-way ANOVA model
# model = ols('balanced_accuracy_score ~ C(module)', data=scores).fit()
# aov_table = sm.stats.anova_lm(model, typ=1)
#
# print('\n')
# print(anova_table(aov_table))
