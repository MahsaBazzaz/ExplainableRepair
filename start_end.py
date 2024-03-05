import argparse

def replace_and_filter(input_array):
    result_array = []
    for line in input_array:
        new_line = ''.join([',' if char not in '{}' else char for char in line])
        result_array.append(new_line)
    return result_array

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create start end level')

    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--out', required=True, type=str)

    args = parser.parse_args()
    file_path = args.input
    output_path = args.out

    print('running' + ': python start_end.py ' + ' '.join([f'--{k} {v}' for k, v in vars(args).items()]))

    level = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.rstrip('\n')
            if not line.startswith('META'):
                level.append(line)
    
    levels_new = replace_and_filter(level)
    try:
        with open(output_path, 'w') as new_file:
            for i, pl in enumerate(levels_new):
                new_file.write(pl)
                new_file.write('\n')
    except Exception as e:
        print(f"An error occurred: {e}")