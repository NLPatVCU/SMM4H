
import os

class File:

    def read_from_file(self, file):
        """
        Reads external files and insert the content to a list. It also removes whitespace
        characters like `\n` at the end of each lines.

        :param file: name of the input file.
        :return : content of the file in list format
        """
        if not os.path.isfile(file):
            raise FileNotFoundError("Not a valid file path")

        with open(file) as f:
            content = f.readlines()
        content = [x.strip() for x in content]

        return content

    def write_from_list(self, list, file_path):
        """
        Creates a csv file from a list

        :param list: list to become a csv file
        :param file_path: path/name to the new file.
        """
        with open(file_path, mode='w') as file:
            for item in list:
                file.write(str(item)+"\n")
