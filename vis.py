from PIL import Image, ImageDraw
import argparse

def draw_on_image(path, orig_lvl, repaired, weights, tile_size = 22):
    input_image = Image.open(path)
    draw = ImageDraw.Draw(input_image)
    rows = len(orig_lvl)
    cols = len(orig_lvl[0])
    
    if weights is not None:
        for i in range(cols):
            for j in range(rows):
                if weights[i][j] == 1:
                    rect_start = (j * tile_size, i * tile_size)
                    rect_end = ((j + 1) * tile_size, (i + 1) * tile_size)
                    draw.rectangle([rect_start, rect_end], outline='black', width=2)
    
    if repaired is not None:
        for i in range(cols):
            for j in range(rows):
                if repaired[i][j] != orig_lvl[i][j]:
                    rect_start = (j * tile_size, i * tile_size)
                    rect_end = ((j + 1) * tile_size, (i + 1) * tile_size)
                    draw.rectangle([rect_start, rect_end], outline='green', width=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create start end level')

    parser.add_argument('--path', required=True, type=str)
    parser.add_argument('--orig', required=True, type=str)
    parser.add_argument('--repaired', required=True, type=str)
    parser.add_argument('--weights', required=True, type=str)

    args = parser.parse_args()
    image_paths = args.images
    output_path = args.outfile

    print('running' + 'python viz.py ' + ' '.join([f'--{k} {v}' for k, v in vars(args).items()]))


    draw_on_image(image_paths, output_path)