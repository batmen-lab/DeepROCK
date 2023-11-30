import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.metrics import roc_auc_score

def pairwise_interaction_auc(interactions, ground_truth):
    strengths = []
    gt_binary_list = []
    for inter, strength in interactions:
        inter_set = set(inter)
        strengths.append(strength)
        if any(inter_set <= gt for gt in ground_truth):
            gt_binary_list.append(1)
        else:
            gt_binary_list.append(0)

    auc = roc_auc_score(gt_binary_list, strengths)
    return auc

def get_interaction_type_idx(inter_indices_arr, p):
    TD_idx = []
    TT_idx = []
    DD_idx = []
    for idx, indices in enumerate(inter_indices_arr):
        if all(i < p for i in indices):
            TT_idx.append(idx)
        elif all(i >= p for i in indices):
            DD_idx.append(idx)
        else:
            TD_idx.append(idx)
    return TT_idx, TD_idx, DD_idx

def get_selected_interactions(interactions, p, q, abs_diff=True):
    interactions = interactions.copy()
    interactions.sort(key=lambda x: x[1], reverse=True)
    inter_idices_arr = np.array([i for i, _ in interactions])
    inter_score_arr = np.array([s for _, s in interactions])
    TT_idx, TD_idx, DD_idx = get_interaction_type_idx(inter_idices_arr, p)
    cutoff = np.inf
    for indices, t in interactions:
        TT_count = (inter_score_arr[TT_idx] >= t).sum()
        TD_count = (inter_score_arr[TD_idx] >= t).sum()
        DD_count = (inter_score_arr[DD_idx] >= t).sum()
        if abs_diff:
            score = abs((TD_count-DD_count)) / max(1.0, TT_count)
        else:
            score = (TD_count-DD_count) / max(1.0, TT_count)
        if (t < cutoff) and (score <= q):
            cutoff = t
        if score > q:
            break
    selected_interaction = []
    for idx, score in interactions:
        i, j = idx
        if score >= cutoff and (i < p) and (j < p):
            selected_interaction.append((idx, score))
    return selected_interaction, cutoff

def get_gt_bins(interactions, ground_truth):
    """get binary label and the score"""
    scores = []
    gt_binaries = []
    interactions = interactions.copy()
    interactions.sort(key=lambda x: x[1], reverse=True)
    for inter, score in interactions:
        inter_set = set(inter)
        scores.append(score)
        if any(inter_set <= gt for gt in ground_truth):
            gt_binaries.append(1)
        else:
            gt_binaries.append(0)
    return gt_binaries[::-1], scores[::-1]

def get_interaction_fdp(sel_interactions, ground_truth):
    binaries, _ = get_gt_bins(sel_interactions, ground_truth)
    fdp = (len(binaries) - sum(binaries)) / max(1, len(binaries))
    return fdp

def get_interaction_power(sel_interactions, ground_truth):
    binaries, _ = get_gt_bins(sel_interactions, ground_truth)
    num_gt = 0
    all_gt = []
    for gt in ground_truth:
        curr_gt = list(combinations(gt, 2))
        all_gt += curr_gt
    all_gt = set(all_gt)
    num_gt = len(all_gt)
    power = sum(binaries) / max(1, num_gt)
    return power

def get_interaction_df(interactions, p, import_gt=None, inter_gt=None, feat_names=None, cutoff=None):
    feat_inter_df = {"value": [], "type": [], "i": [], "j": [], "name": [], "selected": []}
    for (i, j), score in interactions:
        feat_inter_df["value"].append(score)
        feat_inter_df["i"].append(i)
        feat_inter_df["j"].append(j)
        if feat_names is None:
            feat_inter_df["name"].append(f"{i} - {j}")
        else:
            if i < p:
                i_name = f"{feat_names[i]}"
            else:
                i_name = f"{feat_names[i-p]} (knockoff)"
            if j < p:
                j_name = f"{feat_names[j]}"
            else:
                j_name = f"{feat_names[j-p]} (knockoff)"
            feat_inter_df["name"].append(f"{i_name} - {j_name}")
        if cutoff is None:
            feat_inter_df["selected"].append(False)
        else:
            if score >= cutoff:
                feat_inter_df["selected"].append(True)
            else:
                feat_inter_df["selected"].append(False)
        if inter_gt is not None and import_gt is not None:
            if i < p and j < p:
                if any({i,j} <= gt for gt in inter_gt):
                    feat_inter_df["type"].append("Ground Truth Interactions")
                elif i in import_gt and j in import_gt:
                    feat_inter_df["type"].append("marginal-marginal")
                elif i in import_gt or j in import_gt:
                    feat_inter_df["type"].append("marginal-irrelevant")
                else:
                    feat_inter_df["type"].append("irrelevant-irrelevant")
            elif i < p or j < p:
                if i in import_gt or j in import_gt:
                    feat_inter_df["type"].append("marginal-knockoff")
                else:
                    feat_inter_df["type"].append("irrelevant-knockoff")
            else:
                feat_inter_df["type"].append("knockoff-knockoff")
        elif inter_gt is not None:
            if i < p and j < p:
                if any({i,j} <= gt for gt in inter_gt):
                    feat_inter_df["type"].append("Ground Truth Interactions")
                else:
                    feat_inter_df["type"].append("original-original")
            elif i < p or j < p:
                feat_inter_df["type"].append("original-knockoff")
            else:
                feat_inter_df["type"].append("knockoff-knockoff")
        else:
            if i < p and j < p:
                feat_inter_df["type"].append("original-original")
            elif i < p or j < p:
                feat_inter_df["type"].append("original-knockoff")
            else:
                feat_inter_df["type"].append("knockoff-knockoff")
    feat_inter_df = pd.DataFrame(feat_inter_df)
    return feat_inter_df
