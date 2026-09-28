import os
import re
import shutil
import argparse
from tqdm import tqdm


def get_sample_index(filename):
    match = re.match(r"sample(\d+)\.hdf5$", filename)

    if match is None:
        return None

    return int(match.group(1))


def get_max_index(directory):
    max_index = 0

    if not os.path.isdir(directory):
        return max_index

    for filename in os.listdir(directory):
        index = get_sample_index(filename)

        if index is not None:
            max_index = max(max_index, index)

    return max_index


def get_hdf5_files(directory):
    files = []

    for filename in os.listdir(directory):
        index = get_sample_index(filename)

        if index is not None:
            files.append((index, filename))

    files.sort(key=lambda x: x[0])

    return [filename for _, filename in files]


def merge_datasets(moises_dir, musdb_dir, output_dir):
    if os.path.exists(output_dir):
        raise FileExistsError("Output directory already exists: {}".format(output_dir))

    shutil.copytree(moises_dir, output_dir)

    categories = ["bass", "drums", "other", "vocals"]

    for category in categories:
        source_dir = os.path.join(musdb_dir, category)
        target_dir = os.path.join(output_dir, category)

        os.makedirs(target_dir, exist_ok=True)
        start_index = get_max_index(target_dir) + 1
        files = get_hdf5_files(source_dir)

        print("Merging {}, starting from sample{}".format(category, start_index))

        for offset, filename in enumerate(tqdm(files, dynamic_ncols=True)):
            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(target_dir, "sample{}.hdf5".format(start_index + offset))
            shutil.copy2(source_path, target_path)

    mixture_source = os.path.join(musdb_dir, "mixture")
    mixture_target = os.path.join(output_dir, "mixture")

    if os.path.isdir(mixture_source):
        shutil.copytree(mixture_source, mixture_target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--moises_dir", required=True)
    parser.add_argument("--musdb_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()
    merge_datasets(args.moises_dir, args.musdb_dir, args.output_dir)