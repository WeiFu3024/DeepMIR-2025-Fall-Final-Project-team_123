import warnings
warnings.filterwarnings("ignore")

import os
import allin1
import matplotlib
matplotlib.use('Agg') # Must be called before importing plt
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse


def get_filelist(directory, extensions):
    filelist = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                filelist.append(os.path.join(root, file))
    return filelist

def parse_args():   # partition for parallel processing
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", type=int, default=0)
    parser.add_argument("--num_partitions", type=int, default=1)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    partition = args.partition
    num_partitions = args.num_partitions

    input_dir = 'RapBank/data/bgm_cut'
    output_dir = 'all_in_one_results_bgm_cut'
    output_viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_viz_dir, exist_ok=True)
    filelist = get_filelist(input_dir, ['.wav'])
    filelist = filelist[partition::num_partitions]  # partitioning the filelist
    
    # filewise analysis
    success_files = []
    error = {}
    
    for file in tqdm(filelist):
        json_file = os.path.join(output_dir, os.path.basename(file).replace('.wav', '.json'))
        if os.path.exists(json_file):
            print(f"Skipping {file}, already processed.")
            continue  # Skip already processed files
        try:
            result = allin1.analyze(file, out_dir=output_dir)
            fig = allin1.visualize(result, out_dir=output_viz_dir)
            # if need visualization, uncomment this
            # plt.close(fig)  # Close the figure to free memory
            # del fig
            success_files.append(file)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            error[file] = str(e)

    print(f"Processed {len(success_files)}/{len(filelist)} files successfully.")
    if error:
        print("Errors encountered in the following files:")
        for file, err in error.items():
            print(f"{file}: {err}")