import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from explainers_utility import create_array_with_largest_area

import torch
import torch.nn as nn
import numpy as np
import shap

from data_utility import get_dataset, get_level_flat
from generic_classifier import Model, predict_proba, predict

import argparse
import json
import matplotlib.pyplot as plt, pdb
from constants import CAVE_COLS, CAVE_ROWS, CAVE_CHANNELS, MARIO_COLS, MARIO_ROWS, MARIO_CHANNELS, SUPERCAT_CHANNELS, SUPERCAT_COLS, SUPERCAT_ROWS
from constants import cave_chars_unique, mario_chars_unique, supercat_chars_unique
from sturgeon import util_common

CAVE_SHAPE = (CAVE_COLS,CAVE_ROWS,CAVE_CHANNELS)
MARIO_SHAPE = (MARIO_COLS,MARIO_ROWS,MARIO_CHANNELS)
CONSTANT_HIGH = 10
CONSTANT_MID = 2
CONSTANT_LOW = 1

class Shapley():
    def __init__(self, game):
        self.game = game
        self.X, self.y = get_dataset(game)
        if game == "cave":
            self.cols = CAVE_COLS
            self.rows = CAVE_ROWS
            self.channels = CAVE_CHANNELS
        elif game == 'mario':
            self.cols = MARIO_COLS
            self.rows = MARIO_ROWS
            self.channels = MARIO_CHANNELS
        elif game == 'supercat':
            self.cols = SUPERCAT_COLS
            self.rows = SUPERCAT_ROWS
            self.channels = SUPERCAT_CHANNELS

        # get model
        path = "./models/" + game + "_" + 'generic_classifier_py' + ".pth"
        self.num_channels = len(self.X[0][0][0])
        self.model = Model(self.cols, self.rows, self.num_channels)
        self.model.load_state_dict(torch.load(path))

        desired_shape = (1, self.cols * self.rows)
        desired_value = [0]
        result_array1 = np.full(desired_shape, desired_value)

        desired_shape = (1, self.cols * self.rows)
        desired_value = [1]
        result_array2 = np.full(desired_shape, desired_value)

        result_array = np.concatenate((result_array1, result_array2), axis=0)
        self.explainer = shap.KernelExplainer(self.f, result_array)
        
    def get_cols(self):
        return self.cols
    
    def get_rows(self):
        return self.rows
        
    def f(self, x):

        if args.game == "cave":
            int2char = dict(enumerate(cave_chars_unique))
        elif args.game == "mario":
            int2char = dict(enumerate(mario_chars_unique))
        elif args.game == "supercat":
            int2char = dict(enumerate(supercat_chars_unique))
        char2int = {ch: ii for ii, ch in int2char.items()}
        num_tiles = len(char2int)

        tmp = x.copy()
        tmp = np.array(np.eye(num_tiles, dtype='uint8')[tmp]).reshape(-1,self.cols,self.rows,self.channels)
        tmp = np.transpose(tmp, (0, 3, 1, 2))
        tmp = torch.tensor(tmp, dtype=torch.float32)
        return np.array(predict_proba(self.model, tmp))

    def get_shapley_values(self, level):
        level_input = get_level_flat(self.game, level)
        if args.game == "cave":
            int2char = dict(enumerate(cave_chars_unique))
        elif args.game == "mario":
            int2char = dict(enumerate(mario_chars_unique))
        elif args.game == "supercat":
            int2char = dict(enumerate(supercat_chars_unique))
        char2int = {ch: ii for ii, ch in int2char.items()}
        num_tiles = len(char2int)

        tmp = level_input.copy()
        tmp = np.array(np.eye(num_tiles, dtype='uint8')[tmp]).reshape(-1,self.cols,self.rows,self.channels)
        level_input_transposed = np.transpose(tmp, (0, 3, 1, 2))
        level_input_transposed = torch.tensor(level_input_transposed, dtype=torch.float32)
        model_answer = predict(self.model, level_input_transposed)

        shap_values = self.explainer.shap_values(level_input.reshape(1,self.cols * self.rows)  , nsamples=3000)
        shap_values = np.array(shap_values).reshape(2,1,self.cols,self.rows)
        d = shap_values[0, :, :, :]
        return d

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Resnet classifier')

    parser.add_argument('--game', required=True, type=str, help='cave/icarus/mario')
    parser.add_argument('--level', required=True, type=str, help='string level to be repaired')
    parser.add_argument('--outfile', required=True, nargs='+', type=str, help='string level to be repaired')
    parser.add_argument('--outimage', required=False, type=str, help='string level to be repaired')

    # get dataset
    args = parser.parse_args()
    game = args.game
    level = args.level
    print('running ' + 'Command: python kernel_shap.py ' + ' '.join([f'--{k} {v}' for k, v in vars(args).items()]))


    shapley = Shapley(game)
    util_common.timer_start()
    shap_values = shapley.get_shapley_values(level)
    util_common.write_time("")

    shapley_value = shap_values[0]

    output0 = np.array(shapley_value).reshape(shapley.get_cols(),shapley.get_rows())

    path = str(args.outfile[0])
    with open(path, 'w') as json_file:
        json.dump(output0.tolist(), json_file)

    output1 = create_array_with_largest_area(shapley_value)
    path = str(args.outfile[1])
    with open(path, 'w') as json_file:
        json.dump(output1.tolist(), json_file)

    if args.outimage:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        cmap0 = plt.get_cmap('viridis')
        cmap1 = plt.get_cmap('coolwarm')
        norm0 = plt.Normalize(vmin=output0.min(), vmax=output0.max())
        norm1 = plt.Normalize(vmin=output1.min(), vmax=output1.max())
        im0 = axes[0].imshow(output0, cmap=cmap0, norm=norm0)
        axes[0].set_title('original Kernel Shapley')
        cbar0 = plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(output1, cmap=cmap1, norm=norm1)
        axes[1].set_title('rescaled Kernel Shapley')
        cbar1 = plt.colorbar(im1, ax=axes[1])

        plt.tight_layout()
        plt.savefig(args.outimage)
    
