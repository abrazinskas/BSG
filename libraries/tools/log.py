import os
from libraries.utils.paths_and_files import create_folders_if_not_exist
from time import strftime


# A general purpose class for logging
class Log():
    def __init__(self, folder):
        self.file_path = os.path.join(folder, "log_"+strftime("%b_%d_%H_%M_%S")+'.txt')
        create_folders_if_not_exist(self.file_path)

    def write(self, string, also_print=True, include_timestamp=True):
        """
        :param string: what string to write to the log
        :param also_print: if set to True, will also print to the console
        :param include_timestamp: whether include the timestamp
        """
        if include_timestamp:
            string = "%s [INFO]: %s" % (strftime("%H:%M:%S"), string)
        if also_print:
            print(string)
        with open(self.file_path, "a") as f:
            f.write(string+" \n")
