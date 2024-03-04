import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from explainers_utility import create_array_with_largest_area
from generic_classifier import Model, predict_proba, predict

import torch
import torch.nn as nn
import numpy as np
import shap

from data_utility import get_dataset, get_level_flat
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

        model = Model(self.cols, self.rows, self.channels)
        path = "./models/new/" + game + "_" + 'generic_classifier_py' + ".pth"
        model.load_state_dict(torch.load(path))
        self.model = model
        background = torch.tensor(self.X[:100], dtype=torch.float32)
        self.explainer = shap.DeepExplainer(model, background)
        
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

        shap_values = self.explainer.shap_values(torch.tensor(tmp,dtype=torch.float32), check_additivity=False)
        return shap_values[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Resnet classifier')

    parser.add_argument('--game', required=True, type=str, help='cave/icarus/mario')
    parser.add_argument('--level', required=True, type=str)
    parser.add_argument('--outfile', required=True, nargs='+', type=str)
    parser.add_argument('--outimage', required=False, type=str)

    # get dataset
    args = parser.parse_args()
    game = args.game
    level = args.level

    print('running ' + 'Command: python deep_shap.py ' + ' '.join([f'--{k} {v}' for k, v in vars(args).items()]))

    shapley = Shapley(game)
    util_common.timer_start()
    shap_values = shapley.get_shapley_values(level)
    util_common.write_time("")

    path = str(args.outfile[0])
    with open(path, 'w') as json_file:
        json.dump(shap_values.tolist(), json_file)
    output1 = create_array_with_largest_area(np.sum(shap_values.squeeze(), axis=2))
    path = str(args.outfile[1])
    with open(path, 'w') as json_file:
        json.dump(output1.tolist(), json_file)
