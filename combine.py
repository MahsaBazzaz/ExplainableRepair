from PIL import Image, ImageDraw, ImageFont
import argparse

def combine_images(image_paths, output_path):
    # Open the images
    images = [Image.open(path) for path in image_paths]

    # Get the width and height of the images
    widths, heights = zip(*(i.size for i in images))

    # Calculate the total width and maximum height
    total_width = sum(widths)
    max_height = max(heights)

    # Create a new image with the calculated dimensions
    new_image = Image.new('RGB', (total_width, max_height))

    # Paste the images into the new image
    x_offset = 0
    for image in images:
        new_image.paste(image, (x_offset, 0))
        x_offset += image.width

    # Save the combined image
    new_image.save(output_path)

def combine_images_with_titles(image_paths, titles, output_path):
    # Open the images
    images = [Image.open(path) for path in image_paths]

    # Get the width and height of the images
    widths, heights = zip(*(i.size for i in images))

    # Calculate the total width and maximum height
    total_width = sum(widths)
    max_height = max(heights)

    # Create a new image with the calculated dimensions
    new_image = Image.new('RGB', (total_width, max_height + 50))  # Add extra height for titles

    # Paste the images into the new image and add titles
    draw = ImageDraw.Draw(new_image)
    font = ImageFont.load_default()  # You can change the font if needed

    x_offset = 0
    for image, title in zip(images, titles):
        new_image.paste(image, (x_offset, 50))  # Add 50 pixels offset for titles
        draw.text((x_offset, 0), title, fill="white", font=font)  # Add titles at the top
        x_offset += image.width

    # Save the combined image
    new_image.save(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create start end level')

    parser.add_argument('--images', required=True, type=str, nargs='+')
    parser.add_argument('--outfile', required=True, type=str)

    args = parser.parse_args()
    image_paths = args.images
    output_path = args.outfile

    print('running' + 'python viz.py ' + ' '.join([f'--{k} {v}' for k, v in vars(args).items()]))

    titles = ["original", "IG", "DEEP SHAP", "UNI"]
    combine_images_with_titles(image_paths, titles, output_path)