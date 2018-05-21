# contains utility functions that are related to retrieving paths, file names, creating folders, etc
import os
import glob
import errno


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def get_file_paths(path, return_file_names=False):
    """
    :param path:
    :return: :rtype: a list of filepaths that are in the folder
    """
    if os.path.isdir(path):
        paths = glob.glob(path + "/*")
    else:
        paths = [path]  # that means there is only one file

    if return_file_names:
        paths = [(p, p.split('/')[-1]) for p in paths]
    return paths



def get_subdir_number(path):
    """
    Checks the number of subdirectories, and returns it. Useful for automatic output folders generation
    """
    if not os.path.exists(path):
        return 0
    subdirectories = get_immediate_subdirectories(path)
    return len(subdirectories)


def create_folders_if_not_exist(filename):
    if os.path.dirname(filename) and not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def append_to_file(file_path, str):
    create_folders_if_not_exist(file_path)
    with open(file_path, "a") as f:
        f.write(str+" \n")


def files_len(path):
    file_paths = get_file_paths(path)
    total = 0
    for file_path in file_paths:
      total += files_len(file_path)
    return total


def file_len(file_path):
    with open(file_path) as f:
            for i, l in enumerate(f):
                pass
    return i + 1


def count_number_of_tokens(folder_path):
    """
    Counts the number of tokens in all files in the folder
    """
    nr_tokens = 0
    filenames = glob.glob(folder_path + "/*")
    for fname in filenames:
       with open(fname) as f:
            print fname
            for sentence in f:
                words = sentence.lower().split()
                nr_tokens += len(words)
    return nr_tokens


def merge_text_files(input_folder, output_file_path):
    with open(output_file_path, 'w') as output_file:
        for file_path in get_file_paths(input_folder):
            with open(file_path) as input_file:
                for line in input_file:
                    output_file.write(line)