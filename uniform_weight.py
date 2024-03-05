import argparse
from constants import CAVE_COLS, CAVE_ROWS, MARIO_COLS, MARIO_ROWS, SUPERCAT_COLS, SUPERCAT_ROWS
from sturgeon import util_common
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create start end level')

    parser.add_argument('--game', required=True, type=str)
    parser.add_argument('--outfile', required=True, type=str)
    parser.add_argument('--outimage', required=False, type=str, help='string level to be repaired')

    args = parser.parse_args()

    print('running' + ': python uniform_weight.py ' + ' '.join([f'--{k} {v}' for k, v in vars(args).items()]))
    util_common.timer_start()

    name = args.game
    if name == 'cave':
       num_cols = CAVE_COLS
       num_rows = CAVE_ROWS
    elif name == 'mario':
       num_cols = MARIO_COLS
       num_rows = MARIO_ROWS
    elif args.game == "supercat":
        num_cols = SUPERCAT_COLS
        num_rows = SUPERCAT_ROWS
    

    my_2d_list = [[1] * num_rows for _ in range(num_cols)]

    try:
        with open(args.outfile, 'w') as new_file:
            new_file.write(str(my_2d_list))
    except Exception as e:
        print(f"An error occurred: {e}")

    temp = np.array(my_2d_list)
    if args.outimage:
        cmap = plt.get_cmap('coolwarm')
        norm = plt.Normalize(vmin=temp.min(), vmax=temp.max())
        im = plt.imshow(temp, cmap=cmap, norm=norm)
        cbar = plt.colorbar(im)
        plt.tight_layout()
        plt.axis('off')
        plt.savefig(args.outimage)
    
    util_common.write_time("")
