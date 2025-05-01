_old_path = "/mnt/data1/ZYDdata/data/WHBUS/buvimgs"
_new_path = "datasets/tsm_buv_imgs/rawframes"

def update_paths(input_txt, output_txt):
    with open(input_txt, 'r') as infile, open(output_txt, 'w') as outfile:
        for line in infile:
            parts = line.strip().split()
            if len(parts) != 3:
                continue  # skip malformed lines
            path, count, label = parts
            # Replace the path prefix
            new_path = path.replace(
                _old_path, _new_path
            )
            # Write the updated line
            outfile.write(f"{new_path} {count} {label}\n")


update_paths('datasets/tsm_buv_imgs/tsm_train_paths.txt',
             'datasets/tsm_buv_imgs/my_tsm_train_paths.txt')
