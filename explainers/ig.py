import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from generic_classifier import Model
from explainers_utility import create_array_with_largest_area

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import captum
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
from sturgeon import util_common
import matplotlib.pyplot as plt

import os, sys
import json

import numpy as np
from PIL import Image
import constants
import data_utility

def predict(model, input_tensor):
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)
        return predicted_class
    
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
    outfile = args.outfile

    print('running ' + 'Command: python ig.py ' + ' '.join([f'--{k} {v}' for k, v in vars(args).items()]))

    if game == "cave":
        cols = constants.CAVE_COLS
        rows = constants.CAVE_ROWS
        channels = constants.CAVE_CHANNELS
    elif game == 'mario':
        cols = constants.MARIO_COLS
        rows = constants.MARIO_ROWS
        channels = constants.MARIO_CHANNELS
    elif game == 'supercat':
        cols = constants.SUPERCAT_COLS
        rows = constants.SUPERCAT_ROWS
        channels = constants.SUPERCAT_CHANNELS


    input_img = data_utility.get_level(game, level)
    input_img_transposed = np.transpose(input_img, (0, 3, 1, 2))
    input_img_transposed = torch.tensor(input_img_transposed, dtype=torch.float32)

    model = Model(cols, rows, channels)
    path = "./models/" + game + "_" + 'generic_classifier_py' + ".pth"
    model.load_state_dict(torch.load(path))
    model = model.eval()
    pred_label_idx = predict(model, input_img_transposed)

    print("true answer: ", 0)
    print("model answer: ", pred_label_idx)

    util_common.timer_start()
    # Initialize the attribution algorithm with the model
    integrated_gradients = IntegratedGradients(model)
    # Ask the algorithm to attribute our output target to
    attributions_ig = integrated_gradients.attribute(input_img_transposed, target=pred_label_idx, n_steps=200)
    util_common.write_time("")

    path = str(args.outfile[0])
    with open(path, 'w') as json_file:
        json.dump(attributions_ig.tolist(), json_file)

    output = create_array_with_largest_area(np.sum(attributions_ig.squeeze().cpu().detach().numpy(), axis=0))
    path = str(args.outfile[1])
    with open(path, 'w') as json_file:
        json.dump(output.tolist(), json_file)

    if args.outimage:

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        cmap0 = plt.get_cmap('viridis')
        cmap1 = plt.get_cmap('coolwarm')
        norm0 = plt.Normalize(vmin=attributions_ig.min(), vmax=attributions_ig.max())
        norm1 = plt.Normalize(vmin=output.min(), vmax=output.max())
        im0 = axes[0].imshow(attributions_ig, cmap=cmap0, norm=norm0)
        axes[0].set_title('original IG')
        cbar0 = plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(output, cmap=cmap1, norm=norm1)
        axes[1].set_title('rescaled IG')
        cbar1 = plt.colorbar(im1, ax=axes[1])

        plt.tight_layout()
        plt.savefig(args.outimage)