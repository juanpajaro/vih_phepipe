import os
from typing import Any, Optional

import pandas as pd


# The `CodeTranslator` class reads a CSV file to create a dictionary mapping values from the first
# column to values from the second column, allowing for easy translation of codes.
class CodeTranslator:
    codes: Optional[dict]

    def __init__(self, file_path: str = None) -> Optional[Any]:
        self.codes = self.get_umls_codes_dict(file_path) if file_path else dict()

    def __call__(self, key: str) -> str:
        return self.codes[key] if self.codes and key in self.codes else key

    def get_umls_codes_dict(self, file_path: str) -> dict:
        """
        The function `get_umls_codes_dict` reads a CSV file and returns a dictionary mapping values from the
        first column to values from the second column.

        :param file_path: The `file_path` parameter in the `get_umls_codes_dict` function is a string that
        represents the file path to a CSV file containing data that will be read and processed to create a
        dictionary of UMLS codes
        :type file_path: str
        :return: A dictionary is being returned, where the keys are the values in the first column of the
        CSV file and the values are the values in the second column of the CSV file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError("file not found: '{}'".format(file_path))

        df = pd.read_csv(file_path)
        return {row[3]: row[4] for i, row in df.iterrows()}
