# no need, since all in one will run demucs internally
import demucs.separate
import os

def get_filelist(directory, extension):
    """Get a list of files with the given extension in the specified directory."""
    return [f for f in os.listdir(directory) if f.endswith(extension)]

def separate_drums(input_file, output_dir):
    if os.path.exists(os.path.join(output_dir, 'htdemucs', os.path.basename(input_file).replace('.wav', ''), 'drums.wav')):
        pass  # Skip if already processed
    else:
        try:
            demucs.separate.main(
                [
                    "--two-stems", "drums",
                    "-n", "htdemucs",
                    "-o", output_dir,
                    input_file
                ]
            )
        except Exception as e:
            print(f"Error processing {input_file}: {e}")

if __name__ == "__main__":

    target_dir = "RapBank/data/bgm_cut"
    output_dir = "bgm_cut_drums"

    target_files = get_filelist(target_dir, ".wav")
    for i, file_name in enumerate(target_files):
        print(f"Processing file {i + 1}/{len(target_files)}: {file_name}")
        input_path = os.path.join(target_dir, file_name)
        separate_drums(input_path, output_dir)