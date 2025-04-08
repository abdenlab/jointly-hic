"""Utilities to create multivec files."""

from math import ceil, log

import numpy as np
from h5py import File as h5py_File
from scipy.special import logsumexp


def nansum_agg(x):
    """Sum the values in x, ignoring NaNs."""
    return np.nansum(x.T.reshape((x.shape[1], -1, 2)), axis=2).T


def nanmean_agg(x):
    """Calculate the mean of the values in x, ignoring NaNs."""
    return np.nanmean(x.T.reshape((x.shape[1], -1, 2)), axis=2).T


def logsumexp_agg(x):
    """Calculate the log sum of the exponentials of the values in x, ignoring NaNs."""
    a = x.T.reshape((x.shape[1], -1, 2))
    return logsumexp(a, axis=2).T


def make_multivec(
    outpath,
    df,
    feature_names,
    base_res,
    chromsizes,
    tilesize=1024,
    agg=nansum_agg,
    chunksize=int(1e5),
    h5opts=None,
):
    """Create a multivec file from a pandas DataFrame.

    Parameters
    ----------
    outpath : str
        Path to the output file.
    df : pandas.DataFrame
        The dataframe containing the data to be stored.
    feature_names : list of str
        The names of the columns in the dataframe to be stored.
    base_res : int
        The resolution of the data in the dataframe.
    chromsizes : dict
        A dictionary-like object mapping chromosome names to their lengths.
    tilesize : int
        The size of the tiles in the output file.
    agg : function
        The function to use to aggregate data when creating lower-resolution
        zoom levels. The function should take a 2D array of shape (n, m) and
        return a 2D array of shape (n, m // 2).
    chunksize : int
        The size of the chunks to use when writing the data to the output file.
    h5opts : dict
        A dictionary of options to pass to h5py when creating the output file.

    """
    if h5opts is None:
        h5opts = {"compression": "gzip", "compression_opts": 6, "shuffle": True}

    chromosomes = list(chromsizes.keys())
    grouped = df.groupby("chrom")
    array_dict = {chrom: grouped.get_group(chrom).loc[:, feature_names].values for chrom in chromosomes}
    chroms, lengths = zip(*chromsizes.items())
    chrom_array = np.array(chroms, dtype="S")
    feature_names = np.array(feature_names, dtype="S")

    # this will be the file that contains our multires data
    with h5py_File(outpath, "w") as f:
        # store some metadata
        f.create_group("info")
        f["info"].attrs["tile-size"] = tilesize
        f.create_group("chroms")
        f["chroms"].create_dataset(
            "name",
            shape=(len(chroms),),
            dtype=chrom_array.dtype,
            data=chrom_array,
            **h5opts,
        )
        f["chroms"].create_dataset("length", shape=(len(chroms),), data=lengths, **h5opts)

        # the data goes here
        f.create_group("resolutions")
        # the maximum zoom level corresponds to the number of aggregations
        # that need to be performed so that the entire extent of
        # the dataset fits into one tile
        total_length = sum(lengths)
        max_zoom = ceil(log(total_length / (tilesize * base_res)) / log(2))

        # start with a resolution of 1 element per pixel
        res = base_res
        grp = f["resolutions"].create_group(str(res))
        # add information about each of the rows
        if feature_names is not None:
            grp.attrs["row_infos"] = feature_names
        # hard links
        grp.create_group("chroms")
        grp["chroms"]["name"] = f["chroms"]["name"]
        grp["chroms"]["length"] = f["chroms"]["length"]
        # add the data
        grp.create_group("values")
        for chrom, length in zip(chroms, lengths):
            if chrom not in array_dict:
                continue

            dset = grp["values"].create_dataset(str(chrom), array_dict[chrom].shape, **h5opts)
            start = 0
            step = int(min(chunksize, len(dset)))
            while start < len(dset):
                # see above section
                dset[start : start + step] = array_dict[chrom][start : start + step]
                start += int(min(chunksize, len(array_dict[chrom]) - start))

        # we're going to go through and create the data for the different
        # zoom levels by summing adjacent data points
        prev_res = res
        for i in range(max_zoom):
            # each subsequent zoom level will have half as much data
            # as the previous
            res = prev_res * 2
            prev_grp, grp = grp, f["resolutions"].create_group(str(res))
            # add information about each of the rows
            if feature_names is not None:
                grp.attrs["row_infos"] = feature_names
            # hard links
            grp.create_group("chroms")
            grp["chroms"]["name"] = f["chroms"]["name"]
            grp["chroms"]["length"] = f["chroms"]["length"]
            # add the data
            grp.create_group("values")
            for chrom, length in zip(chroms, lengths):
                if chrom not in prev_grp["values"]:
                    continue

                prev_dset = prev_grp["values"][chrom]
                start = 0
                step = int(min(chunksize, len(prev_dset)))

                shape = (ceil(prev_dset.shape[0] / 2), prev_dset.shape[1])
                dset = grp["values"].create_dataset(chrom, shape, **h5opts)
                while start < len(prev_dset):
                    prev_data = prev_dset[start : start + step]

                    # this is a sort of roundabout way of calculating the
                    # shape of the aggregated array, but all its doing is
                    # just halving the first dimension of the previous shape
                    # without taking into account the other dimensions
                    if len(prev_data) % 2 != 0:
                        # we need our array to have an even number of elements
                        # so we just add the last element again
                        prev_data = np.concatenate((prev_data, [prev_data[-1]]))
                        step += 1

                    data = agg(prev_data)
                    dset[int(start / 2) : int(start / 2 + step / 2)] = data
                    start += int(min(chunksize, len(prev_dset) - start))

            prev_res = res
