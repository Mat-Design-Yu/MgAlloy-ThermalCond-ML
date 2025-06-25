import pickle
import os
from pathlib import Path, PureWindowsPath, PurePosixPath


class CustomPickler(pickle.Pickler):
    def persistent_id(self, obj):
        if isinstance(obj, (Path, PureWindowsPath, PurePosixPath)):
            return ("Path", str(obj))
        return None


class CustomUnpickler(pickle.Unpickler):
    def persistent_load(self, pid):
        type_tag, value = pid
        if type_tag == "Path":
            return Path(value)
        else:
            raise pickle.UnpicklingError("Unsupported persistent object")


def save_state(obj, folder_path):
    """
    Save an object to a file using custom pickling for cross-platform compatibility.

    Args:
        obj: The object to save.
        folder_path (str or Path): The path to save the object to.

    Raises:
        IOError: If there's an error saving the file.
    """
    try:
        folder_path = Path(folder_path)
        folder_path.parent.mkdir(parents=True, exist_ok=True)

        with open(folder_path, "wb") as file:
            CustomPickler(file).dump(obj)
        print(f"Object saved to {folder_path}")
    except IOError as e:
        print(f"Error saving object to {folder_path}: {e}")
        raise


def load_state(folder_path):
    """
    Load an object from a file using custom unpickling for cross-platform compatibility.

    Args:
        folder_path (str or Path): The path to load the object from.

    Returns:
        The loaded object.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        pickle.UnpicklingError: If there's an error unpickling the object.
    """
    try:
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"File not found: {folder_path}")

        with open(folder_path, "rb") as file:
            obj = CustomUnpickler(file).load()
        print(f"Object loaded from {folder_path}")
        return obj
    except (FileNotFoundError, pickle.UnpicklingError) as e:
        print(f"Error loading object from {folder_path}: {e}")
        raise
