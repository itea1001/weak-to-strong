import bisect
import functools
from dataclasses import dataclass
from random import Random
from typing import Any, Callable, Optional
import matplotlib.pyplot as plt
import random
import time

import sys

current_directory = sys.path.pop(0)

from datasets import Dataset as HfDataset
from datasets import load_dataset as hf_load_dataset

sys.path.insert(0, current_directory)


def generate_random_index(n_tot, n_aug, seed):
    random.seed(seed)
    return random.sample(range(n_tot), n_aug)

def strong_gt_aug(n_aug, seed, strong_ds, gt_ds):
    #substitute half of strong labels with ground truth labels
    new_dp = []
    n_tot = len(strong_ds)
    ind_list = generate_random_index(n_tot, n_aug, seed)
    print('random index: ', ind_list[:10])
    for i in range(0, n_tot):
        if i in ind_list:
            ind = i
            dp = strong_ds[ind]
            tmp_dp = dp.copy()
            if ind < 10:
                print(tmp_dp['hard_label'], gt_ds['hard_label'][ind])
            tmp_dp['hard_label'] = gt_ds['hard_label'][ind]
            tmp_dp['soft_label'] = gt_ds['soft_label'][ind]
            if ind < 10:
                print(tmp_dp['hard_label'], gt_ds['hard_label'][ind])
            new_dp.append(tmp_dp)
        else:
            dp = strong_ds[i]
            tmp_dp = dp.copy()
            new_dp.append(tmp_dp)
    new_ds = HfDataset.from_list(new_dp)
    return new_ds

def strong_weak_aug(n_aug, seed, strong_ds, weak_ds):
    #substitute half of strong labels with weak labels
    new_dp = []
    n_tot = len(strong_ds)
    ind_list = generate_random_index(n_tot, n_aug, seed)
    print('random index: ', ind_list[:10])
    for i in range(0, n_tot):
        if i in ind_list:
            ind = i
            dp = strong_ds[ind]
            tmp_dp = dp.copy()
            if ind < 10:
                print(tmp_dp['hard_label'], weak_ds['hard_label'][ind])
            tmp_dp['hard_label'] = weak_ds['hard_label'][ind]
            tmp_dp['soft_label'] = weak_ds['soft_label'][ind]
            if ind < 10:
                print(tmp_dp['hard_label'], weak_ds['hard_label'][ind])
            new_dp.append(tmp_dp)
        else:
            dp = strong_ds[i]
            tmp_dp = dp.copy()
            new_dp.append(tmp_dp)
    new_ds = HfDataset.from_list(new_dp)
    return new_ds

def analyze_softlabel(inf_ds, path):
    #analyze the soft labels (confidence) of inference dataset
    conf_dict={}
    conf_levels = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    conf_values = []
    n_tot = len(inf_ds)
    for i in range(0,len(conf_levels)):
        conf_dict[conf_levels[i]] = 0
    for i in range(0, n_tot):
        dp = inf_ds[i]
        conf_level = max(dp['soft_label'][0], dp['soft_label'][1])
        ind = 0
        for j in range(0, len(conf_levels)):
            if conf_level >= conf_levels[j]:
                ind = max(ind, j)
        conf_dict[conf_levels[ind]] += 1
    for i in range(0, len(conf_levels)):
        conf_values.append(conf_dict[conf_levels[i]])

    plt.figure()
    plt.bar(range(len(conf_values)), conf_values, tick_label=conf_levels)
    plt.savefig(path)
    print('soft label analysis saved, path = ', path)
    return

def weak_label_conf_flip(seed, weak_ds):
    #flip weak labels based (based on confidence level)
    rng = random.Random(seed)
    new_dp = []
    n_tot = len(weak_ds)
    for i in range(0,n_tot):
        dp = weak_ds[i]
        tmp_dp = dp.copy()
        conf_level = max(tmp_dp['soft_label'][0], tmp_dp['soft_label'][1])
        flip = int(rng.random() > conf_level)
        if flip:
            tmp_dp['hard_label'] = 1 - tmp_dp['hard_label']
        new_dp.append(tmp_dp)
    new_ds = HfDataset.from_list(new_dp)
    return new_ds

