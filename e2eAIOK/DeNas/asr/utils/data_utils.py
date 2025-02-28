import os
import re
import shutil
import urllib.request
import collections.abc
import torch
import tqdm
import pathlib

from asr.utils.utils import is_main_process, ddp_barrier


def undo_padding(batch, lengths):
    """Produces Python lists given a batch of sentences with
    their corresponding relative lengths.

    Arguments
    ---------
    batch : tensor
        Batch of sentences gathered in a batch.
    lengths : tensor
        Relative length of each sentence in the batch.
    """
    batch_max_len = batch.shape[1]
    as_list = []
    for seq, seq_length in zip(batch, lengths):
        actual_size = int(torch.round(seq_length * batch_max_len))
        seq_true = seq.narrow(0, 0, actual_size)
        as_list.append(seq_true.tolist())
    return as_list


def get_all_files(
    dirName, match_and=None, match_or=None, exclude_and=None, exclude_or=None
):
    """Returns a list of files found within a folder.

    Different options can be used to restrict the search to some specific
    patterns.

    Arguments
    ---------
    dirName : str
        The directory to search.
    match_and : list
        A list that contains patterns to match. The file is
        returned if it matches all the entries in `match_and`.
    match_or : list
        A list that contains patterns to match. The file is
        returned if it matches one or more of the entries in `match_or`.
    exclude_and : list
        A list that contains patterns to match. The file is
        returned if it matches none of the entries in `exclude_and`.
    exclude_or : list
        A list that contains pattern to match. The file is
        returned if it fails to match one of the entries in `exclude_or`.
    """

    # Match/exclude variable initialization
    match_and_entry = True
    match_or_entry = True
    exclude_or_entry = False
    exclude_and_entry = False

    # Create a list of file and sub directories
    listOfFile = os.listdir(dirName)
    allFiles = list()

    # Iterate over all the entries
    for entry in listOfFile:

        # Create full path
        fullPath = os.path.join(dirName, entry)

        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + get_all_files(
                fullPath,
                match_and=match_and,
                match_or=match_or,
                exclude_and=exclude_and,
                exclude_or=exclude_or,
            )
        else:

            # Check match_and case
            if match_and is not None:
                match_and_entry = False
                match_found = 0

                for ele in match_and:
                    if ele in fullPath:
                        match_found = match_found + 1
                if match_found == len(match_and):
                    match_and_entry = True

            # Check match_or case
            if match_or is not None:
                match_or_entry = False
                for ele in match_or:
                    if ele in fullPath:
                        match_or_entry = True
                        break

            # Check exclude_and case
            if exclude_and is not None:
                match_found = 0

                for ele in exclude_and:
                    if ele in fullPath:
                        match_found = match_found + 1
                if match_found == len(exclude_and):
                    exclude_and_entry = True

            # Check exclude_or case
            if exclude_or is not None:
                exclude_or_entry = False
                for ele in exclude_or:
                    if ele in fullPath:
                        exclude_or_entry = True
                        break

            # If needed, append the current file to the output list
            if (
                match_and_entry
                and match_or_entry
                and not (exclude_and_entry)
                and not (exclude_or_entry)
            ):
                allFiles.append(fullPath)

    return allFiles


def download_file(
    source, dest, unpack=False, dest_unpack=None, replace_existing=False
):
    """Downloads the file from the given source and saves it in the given
    destination path.

     Arguments
    ---------
    source : path or url
        Path of the source file. If the source is an URL, it downloads it from
        the web.
    dest : path
        Destination path.
    unpack : bool
        If True, it unpacks the data in the dest folder.
    replace_existing : bool
        If True, replaces the existing files.
    """
    try:
        if is_main_process():

            class DownloadProgressBar(tqdm.tqdm):
                """ DownloadProgressBar class."""

                def update_to(self, b=1, bsize=1, tsize=None):
                    """Needed to support multigpu training."""
                    if tsize is not None:
                        self.total = tsize
                    self.update(b * bsize - self.n)

            # Create the destination directory if it doesn't exist
            dest_dir = pathlib.Path(dest).resolve().parent
            dest_dir.mkdir(parents=True, exist_ok=True)
            if "http" not in source:
                shutil.copyfile(source, dest)

            elif not os.path.isfile(dest) or (
                os.path.isfile(dest) and replace_existing
            ):
                print(f"Downloading {source} to {dest}")
                with DownloadProgressBar(
                    unit="B",
                    unit_scale=True,
                    miniters=1,
                    desc=source.split("/")[-1],
                ) as t:
                    urllib.request.urlretrieve(
                        source, filename=dest, reporthook=t.update_to
                    )
            else:
                print(f"{dest} exists. Skipping download")

            # Unpack if necessary
            if unpack:
                if dest_unpack is None:
                    dest_unpack = os.path.dirname(dest)
                print(f"Extracting {dest} to {dest_unpack}")
                shutil.unpack_archive(dest, dest_unpack)
    finally:
        ddp_barrier()


def pad_right_to(
    tensor: torch.Tensor, target_shape: (list, tuple), mode="constant", value=0,
):
    """
    This function takes a torch tensor of arbitrary shape and pads it to target
    shape by appending values on the right.

    Parameters
    ----------
    tensor : input torch tensor
        Input tensor whose dimension we need to pad.
    target_shape : (list, tuple)
        Target shape we want for the target tensor its len must be equal to tensor.ndim
    mode : str
        Pad mode, please refer to torch.nn.functional.pad documentation.
    value : float
        Pad value, please refer to torch.nn.functional.pad documentation.

    Returns
    -------
    tensor : torch.Tensor
        Padded tensor.
    valid_vals : list
        List containing proportion for each dimension of original, non-padded values.
    """
    assert len(target_shape) == tensor.ndim
    pads = []  # this contains the abs length of the padding for each dimension.
    valid_vals = []  # this contains the relative lengths for each dimension.
    i = len(target_shape) - 1  # iterating over target_shape ndims
    j = 0
    while i >= 0:
        assert (
            target_shape[i] >= tensor.shape[i]
        ), "Target shape must be >= original shape for every dim"
        pads.extend([0, target_shape[i] - tensor.shape[i]])
        valid_vals.append(tensor.shape[j] / target_shape[j])
        i -= 1
        j += 1

    tensor = torch.nn.functional.pad(tensor, pads, mode=mode, value=value)

    return tensor, valid_vals


def batch_pad_right(tensors: list, mode="constant", value=0):
    """Given a list of torch tensors it batches them together by padding to the right
    on each dimension in order to get same length for all.

    Parameters
    ----------
    tensors : list
        List of tensor we wish to pad together.
    mode : str
        Padding mode see torch.nn.functional.pad documentation.
    value : float
        Padding value see torch.nn.functional.pad documentation.

    Returns
    -------
    tensor : torch.Tensor
        Padded tensor.
    valid_vals : list
        List containing proportion for each dimension of original, non-padded values.

    """

    if not len(tensors):
        raise IndexError("Tensors list must not be empty")

    if len(tensors) == 1:
        # if there is only one tensor in the batch we simply unsqueeze it.
        return tensors[0].unsqueeze(0), torch.tensor([1.0])

    if not (
        any(
            [tensors[i].ndim == tensors[0].ndim for i in range(1, len(tensors))]
        )
    ):
        raise IndexError("All tensors must have same number of dimensions")

    # FIXME we limit the support here: we allow padding of only the first dimension
    # need to remove this when feat extraction is updated to handle multichannel.
    max_shape = []
    for dim in range(tensors[0].ndim):
        if dim != 0:
            if not all(
                [x.shape[dim] == tensors[0].shape[dim] for x in tensors[1:]]
            ):
                raise EnvironmentError(
                    "Tensors should have same dimensions except for the first one"
                )
        max_shape.append(max([x.shape[dim] for x in tensors]))

    batched = []
    valid = []
    for t in tensors:
        # for each tensor we apply pad_right_to
        padded, valid_percent = pad_right_to(
            t, max_shape, mode=mode, value=value
        )
        batched.append(padded)
        valid.append(valid_percent[0])

    batched = torch.stack(batched)

    return batched, torch.tensor(valid)


def recursive_to(data, *args, **kwargs):
    """Moves data to device, or other type, and handles containers.

    Very similar to torch.utils.data._utils.pin_memory.pin_memory,
    but applies .to() instead.
    """
    if isinstance(data, torch.Tensor):
        return data.to(*args, **kwargs)
    elif isinstance(data, collections.abc.Mapping):
        return {
            k: recursive_to(sample, *args, **kwargs)
            for k, sample in data.items()
        }
    elif isinstance(data, tuple) and hasattr(data, "_fields"):  # namedtuple
        return type(data)(
            *(recursive_to(sample, *args, **kwargs) for sample in data)
        )
    elif isinstance(data, collections.abc.Sequence):
        return [recursive_to(sample, *args, **kwargs) for sample in data]
    elif hasattr(data, "to"):
        return data.to(*args, **kwargs)
    # What should be done with unknown data?
    # For now, just return as they are
    else:
        return data


np_str_obj_array_pattern = re.compile(r"[SaUO]")


def mod_default_collate(batch):
    """Makes a tensor from list of batch values.

    Note that this doesn't need to zip(*) values together
    as PaddedBatch connects them already (by key).

    Here the idea is not to error out.

    This is modified from:
    https://github.com/pytorch/pytorch/blob/c0deb231db76dbea8a9d326401417f7d1ce96ed5/torch/utils/data/_utils/collate.py#L42
    """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        try:
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        except RuntimeError:  # Unequal size:
            return batch
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        try:
            if (
                elem_type.__name__ == "ndarray"
                or elem_type.__name__ == "memmap"
            ):
                # array of string classes and object
                if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    return batch
                return mod_default_collate([torch.as_tensor(b) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        except RuntimeError:  # Unequal size
            return batch
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    else:
        return batch
