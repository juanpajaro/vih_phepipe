import os
import pickle
from ast import literal_eval
from typing import Any, Optional

from pydantic import BaseModel

from app.model.errors import AIXUnserializeObjectError


# The `AbstractWrapper` class provides methods for unpickling objects from a file and parsing
# parameters from a text string.
class AbstractWrapper(BaseModel):
    @classmethod
    def path_validator(cls, path: str, asset_name: str = "asset") -> Optional[Any]:
        """
        The function `path_validator` validates a given file path and raises appropriate exceptions if
        the path is empty, not a string, or does not exist.
        
        :param cls: The `cls` parameter in the `path_validator` function is typically used as a
        reference to the class itself. It is a convention in Python to use `cls` as the first parameter
        in class methods to refer to the class object. However, in the provided function, the `cls`
        parameter
        :param path: The `path` parameter is a string that represents the file path that you want to
        validate. It should point to the location of the file you are trying to access or work with
        :type path: str
        :param asset_name: The `asset_name` parameter is a string that represents the name of the asset
        being validated in the `path_validator` method. It is used in the error messages to provide
        context about the specific asset that is being checked for validity, defaults to asset
        :type asset_name: str (optional)
        :return: The function `path_validator` is returning the `path` variable if it passes all the
        validation checks.
        """
        if not path:
            raise FileNotFoundError("empty {} path".format(asset_name))

        is_good_path = path and isinstance(path, str) and bool(path.strip())
        if not is_good_path:
            raise ValueError("is not a good path: '{}'".format(path))
        
        path = path.strip()

        if not os.path.exists(path):
            raise FileNotFoundError("{} file do not exist: '{}'".format(asset_name, path))

        return path

    @classmethod
    def save(cls, obj: Any, path: str):
        """
        The function `save` saves an object to a file using pickle serialization.
        
        :param cls: In the provided code snippet, the `cls` parameter seems to be a class method or a
        static method within a class. It is a conventional naming convention to use `cls` as the first
        parameter in a class method to refer to the class itself. However, in the context of the `save
        :param obj: The `obj` parameter in the `save` method is the object that you want to save to a
        file. This object can be of any type
        :type obj: Any
        :param path: The `path` parameter in the `save` function is a string that represents the file
        path where the object will be saved or serialized. This is the location where the object will be
        written to as a binary file using the `pickle` module
        :type path: str
        """
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> Optional[Any]:
        """
        The `load` function reads and deserializes an object from a file using pickle in Python, raising
        appropriate errors if the file is not found or the object cannot be loaded.

        :param path: The `path` parameter in the `load` method is a string that represents the file path
        from which an object needs to be loaded
        :type path: str
        :return: The `load` method is returning the object loaded from the specified file path using pickle.
        If the object is successfully loaded, it will be returned. If the loaded object is `None`, an
        `AIXUnserializeObjectError` exception will be raised with a message indicating the inability to load
        the object from the specified path.
        """
        if not os.path.exists(path):
            raise FileNotFoundError("file not found: '{}'".format(path))

        with open(path, "rb") as f:
            obj = pickle.load(f)

            if obj is None:
                raise AIXUnserializeObjectError("unable to load object from: '{}'".format(path))

            return obj

    @classmethod
    def parse_parameters(cls, text: str) -> Optional[Any]:
        """
        The function `parse_parameters` attempts to parse a string input using `literal_eval` and
        returns the result or `None` if an exception occurs.

        :param cls: The `cls` parameter in the `parse_parameters` method is a reference to the class
        itself. It is used in class methods to access class variables and methods
        :param text: The `text` parameter is a string that contains the data that needs to be parsed
        into a Python object using the `literal_eval` function
        :type text: str
        :return: The `parse_parameters` method is returning the result of `literal_eval(text)` if the
        text can be successfully evaluated using `literal_eval`. If an exception occurs during the
        evaluation, it will return `None`.
        """
        try:
            return literal_eval(text)
        except Exception:
            return None
