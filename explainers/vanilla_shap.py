import os
import sys
from explainers.explainers_utility import create_array_with_largest_area
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import numpy as np
import shap
from data_utility import get_dataset, get_level
from generic_classifier import Model, predict_proba, predict
import argparse
import json
import matplotlib.pyplot as plt
from constants import CAVE_COLS, CAVE_ROWS, CAVE_CHANNELS, MARIO_COLS, MARIO_ROWS, MARIO_CHANNELS, SUPERCAT_CHANNELS, SUPERCAT_COLS, SUPERCAT_ROWS
from sturgeon import util
import pdb

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
        path = 'models/model_' + game + '.pth'
        self.num_channels = len(self.X[0][0][0])
        self.model = Model(self.cols, self.rows, self.num_channels)
        self.model.load_state_dict(torch.load(path))

        self.explainer = shap.Explainer(self.f, self.custom_masker)
        
    def get_cols(self):
        return self.cols
    
    def get_rows(self):
        return self.rows    
    def f(self, x):
        tmp = x.copy()
        tmp = np.transpose(tmp, (0, 3, 1, 2))
        tmp = torch.tensor(tmp, dtype=torch.float32)
        return predict_proba(self.model, tmp)

    # A masking function takes a binary mask vector as the first argument and
    # the model arguments for a single sample after that
    # It returns a masked version of the input x, where you can return multiple
    # rows to average over a distribution of masking types
    def custom_masker(self, mask, x):
        if self.game == 'cave':
            mask = mask.reshape(x.shape).reshape(-1, CAVE_COLS, CAVE_ROWS, 4)
            result_array = np.where(mask[:, :, None] == np.array([1, 0, 0, 0])[None, None, :], [0, 1, 0, 0], [1, 0, 0, 0])
            res = result_array.reshape(-1, CAVE_COLS, CAVE_ROWS, 4)
        elif self.game == 'mario':
            mask = mask.reshape(x.shape).reshape(-1, MARIO_COLS, MARIO_ROWS, 4)
            result_array = np.where(mask[:, :, None] == np.array([1, 0, 0, 0])[None, None, :], [0, 1, 0, 0], [1, 0, 0, 0])
            res = result_array.reshape(-1, MARIO_COLS, MARIO_ROWS, 4)

        return res

    def get_shapley_values(self, level):
        level_input = get_level(self.game, level)

        level_input_transposed = np.transpose(level_input, (0, 3, 1, 2))
        level_input_transposed = torch.tensor(level_input_transposed, dtype=torch.float32)
        model_answer = predict(self.model, level_input_transposed)

        print("true answer: ", 0)
        print("model answer: ", model_answer)

        shap_values = self.explainer(level_input, max_evals="auto", batch_size=10)
        d = shap_values[0, :, :, :, 0].values
        data = np.sum(d, axis=2)
        print(np.min(data))
        print(np.max(data))

        return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Resnet classifier')

    parser.add_argument('--game', required=True, type=str, help='cave/icarus/mario')
    parser.add_argument('--level', required=True, type=str, help='string level to be repaired')
    parser.add_argument('--outfile', required=True, nargs='+', type=str, help='string level to be repaired')
    parser.add_argument('--outimage', required=True, type=str, help='string level to be repaired')

    # get dataset
    args = parser.parse_args()
    game = args.game
    level = args.level
    outfile = args.outfile
    util.timer_start()

    print('running ' + 'Command: python vanilla_shap.py ' + ' '.join([f'--{k} {v}' for k, v in vars(args).items()]))

    shapley = Shapley(game)

    shap_values = shapley.get_shapley_values(level)

    output0 = np.array(shap_values).reshape(shapley.get_cols(),shapley.get_rows())
    util.write_time("")

    path = str(args.outfile[0])
    with open(path, 'w') as json_file:
        json.dump(output0.tolist(), json_file)

    output1 = create_array_with_largest_area(shap_values)
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
        plt.savefig(args.outimage)
