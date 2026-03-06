import io
import logging
import os
import shutil
import zipfile
from typing import Any, Dict, List, Tuple, Union

import fsspec


def reset_fsspec() -> None:
    """
    Reset fsspec defaults to make it work with multiprocessing.
    For more details, see https://github.com/fsspec/gcsfs/issues/379
    """
    try:
        # this section ensures that the execution
        # doesn't hang when running with multiprocessing
        fsspec.asyn.iothread[0] = None
        fsspec.asyn.loop[0] = None
    except AttributeError:
        # this logic is fs dependent. Ignore errors caused by
        # fs incompatibility
        pass


def get_fs(path: str) -> Tuple[Any, str]:
    """
    Retrieves a filesystem object based on the given path.

    Args:
        path: A string representing the path to a resource. This can include a protocol
              scheme (like 'file://', 'http://', etc.) or be a local path.

    Returns:
        A tuple containing:
        - An instance of a filesystem object according to the detected protocol.
        - The updated path string, which may be altered if it initially pointed to
          a directory with a single file.

    """
    # Check if the path contains a recognized scheme
    if "://" not in path:
        protocol = "file"
        path = f"file://{path}"
    else:
        protocol = path.split("://")[0]

    # needed for multiprocessing
    reset_fsspec()

    fs = fsspec.filesystem(protocol)

    # if root dir points to a directory with a single file, reset it
    # to that file. Needed for correct support of zip archives.
    if fs.isdir(path):
        contents = fs.ls(path)
        if len(contents) == 1:
            path = contents[0]

    return fs, path


def list_files(root_dir: str, include_protocol: bool = False) -> List[str]:
    """
    Lists files recursively from a specified directory.

    Args:
        root_dir: Path to the directory to list the files from.
        include_protocol: Whether to include the protocol in the file paths.
    Returns:
        A list of file paths.
    """
    fs, root_dir = get_fs(root_dir)

    if not include_protocol:
        protocol = ""
    else:
        if "://" in root_dir:
            protocol = root_dir.split("://")[0]
        else:
            protocol = "file"
        protocol = f"{protocol}://"

    # For zip archives, handle them specifically
    if root_dir.endswith(".zip"):
        with fs.open(root_dir, "rb") as f:
            paths = []
            with zipfile.ZipFile(io.BytesIO(f.read())) as zf:
                for name in zf.namelist():
                    # Assuming you want the paths of files inside the zip
                    paths.append(f"{protocol}{name}")
            return paths

    # Find all paths in the given directory
    paths = fs.find(root_dir)

    # this code is better but slower
    # paths = [p for p in paths if not fs.isdir(p) and not p.endswith(".DS_Store")]
    paths = [
        f"{protocol}{p}"
        for p in paths
        if not p.endswith("/") and not p.endswith(".DS_Store")
    ]

    return paths


def read_files(root_dir: str) -> Dict[str, str]:
    """
    Reads files recursively from a specified directory.

    Args:
        root_dir: Path to the directory to read the data from.

    Returns:
        A dictionary with file paths/names as keys and their contents as values.
    """
    fs, root_dir = get_fs(root_dir)

    file_data = {}

    if root_dir.endswith(".zip"):
        with fs.open(root_dir, "rb") as f:
            with zipfile.ZipFile(io.BytesIO(f.read())) as zf:
                for name in zf.namelist():
                    with zf.open(name) as zfile:
                        file_data[name] = zfile.read().decode("utf-8")

        return file_data

    paths = fs.find(root_dir)

    for p in paths:
        if p.endswith(".DS_Store"):
            continue
        with fs.open(p, "r") as f:
            content = f.read()
            # Use the full path instead of only the base name for nested files
            file_data[p] = content

    return file_data


def write_file(data: Union[str, bytes], path: str) -> None:
    """
    Writes data to a specified path. The path can be a local filesystem or any
    other supported protocol by fsspec (e.g., "s3://", "gcs://").

    Args:
        data: The data to write to the file.
        path: The path (with or without a protocol prefix) where the data should be written.
    """
    # Check if the root_dir contains a recognized scheme
    if "://" not in path:
        protocol = "file"
    else:
        protocol = path.split("://")[0]

    # if root dir points to a directory with a single file, reset it
    # to that file. Needed for correct support of zip archives.
    fs = fsspec.filesystem(protocol)

    # Open the file and write data
    mode = "wb" if isinstance(data, bytes) else "w"
    with fs.open(path, mode) as f:
        f.write(data)


def read_file(path: str, mode: str = "rb") -> str:
    """
    Reads data from a specified path. The path can be a local filesystem or any
    other supported protocol by fsspec (e.g., "s3://", "gs://").

    Args:
        path: The path from which the data should be read.

    Returns:
        The data read from the file as a string.
    """
    fs, path = get_fs(path)
    with fs.open(path, mode) as f:
        result = f.read()
        return result


def exists(path: str) -> bool:
    """
    Checks if a file or directory exists at the specified path.

    Args:
        path: The path (with or without a protocol prefix) to check for existence.

    Returns:
        True if the file or directory exists, False otherwise.
    """
    fs, _ = get_fs(path)
    return fs.exists(path)


# make this generic
def upload_file(from_path: str, to_path: str) -> None:
    """
    Uploads a file from a local path to a specified Google Cloud Storage path.

    Args:
        from_path: The local path to the file that needs to be uploaded.
        to_path: The Google Cloud Storage path where the file should be uploaded.
    """
    try:
        with open(from_path, "rb") as local_file:
            write_file(local_file.read(), to_path)
        logging.debug(f"Uploaded {from_path} to {to_path}")
    except Exception as e:
        raise Exception(f"Error uploading {from_path} to {to_path}: {str(e)}")


def upload_directory(local_directory: str, gcs_directory: str) -> None:
    """
    Uploads all files from a local directory to a specified Google Cloud Storage directory.
    Args:
        local_directory: The local directory containing files to upload.
        gcs_directory: The Google Cloud Storage directory where the files should be uploaded.
    """
    uploaded_files = 0

    for root, _, files in os.walk(local_directory):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_directory)
            gcs_path = os.path.join(gcs_directory, relative_path).replace("\\", "/")
            upload_file(local_path, gcs_path)
            uploaded_files += 1
            if uploaded_files % 100 == 0:  # Log progress every 100 files
                logging.info(f"Uploaded {uploaded_files} files.")


def cleanup(path: str) -> None:
    """
    Deletes the specified directory and its contents if it exists.

    This function checks if the directory at the given path exists. If it does,
    the directory and all of its contents are deleted. If the directory does
    not exist, a message indicating this is printed.

    Args:
        path (str): The path to the directory to be deleted.
    """
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Directory {path} has been deleted.")
    else:
        print(f"Directory {path} does not exist.")
