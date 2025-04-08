"""Utilities for working with ENCODE data."""

import os.path
import shutil
import tempfile
from multiprocessing.pool import ThreadPool

import bbi
import pandas as pd
import requests

# Set the cache directory, default to ~/cache if not set
CACHE_DIR = os.getenv("CACHE_DIR", tempfile.gettempdir())


def pick_bigwig(filelist, output_type="fold change over control") -> str:
    """Find the best bigWig from a list: ENCODE4, max bioreps, fold change over control, etc.

    Parameters
    ----------
    filelist : list
        List of file paths from an ENCODE experiment
    output_type : str
        The output type to filter on, default is "fold change over control"

    Returns
    -------
    str
        The accession of the best bigWig file

    """
    file_candidates = []
    with ThreadPool(50) as pool:
        filedata_results = pool.map(
            lambda url: requests.get(url, timeout=60).json(),
            [f"https://www.encodeproject.org{fpath}?format=json" for fpath in filelist],
        )

    for filedata in filedata_results:
        if filedata["file_format"] != "bigWig":
            continue
        if filedata["assembly"] != "GRCh38":
            continue
        if filedata["status"] != "released":
            continue
        if filedata["award"]["viewing_group"] != "ENCODE4":
            continue
        if filedata["output_type"] != output_type:
            continue

        # Now, we know it's an ENCODE bigWig aligned to hg38
        result = {
            "accession": filedata["title"],
            "reps": filedata["biological_replicates"],
            "replen": len(filedata["biological_replicates"]),
        }
        file_candidates.append(result)
    if len(file_candidates) == 0:
        return None
    files = pd.DataFrame(file_candidates).sort_values("replen", ascending=False).reset_index()
    return str(files["accession"][0])


def get_bigwigs_from_experiment_report(seach_term: str) -> dict:
    """Create a dict of {name: accession} for fold change ENCODE4 bigwigs from a url to an experiment report."""
    url = (
        f"https://www.encodeproject.org/search/?type=Experiment&searchTerm={seach_term}&"
        f"assay_title=Histone+ChIP-seq&status=released&assembly=GRCh38&format=json"
    )
    response = requests.get(url, timeout=60)
    payload = response.json()
    result = {}
    for experiment in payload["@graph"]:  # The list of experiments
        biosample = experiment["biosample_ontology"]["term_name"]
        target = experiment["target"]["label"]
        exp_accession = experiment["accession"]
        filelist = [f["@id"] for f in experiment["files"]]
        bigwig_accession = pick_bigwig(filelist)
        if bigwig_accession is None:
            continue
        acc_dict = {f"{biosample}.{target}.{exp_accession}": bigwig_accession}
        result.update(acc_dict)
    return result


def get_signal(path, bins):
    """Get signal from bigwig file."""
    with bbi.open(path) as bw:
        signal = bw.stackup(bins["chrom"], bins["start"], bins["end"], bins=1).ravel()
    return signal


class EncodeFile:
    """Implement context manager that downloads, caches, and moves an ENCODE file to/from temporary storage."""

    def __init__(self, accession: str, extension: str = "bw"):
        """Initialize the EncodeFile object."""
        self.accession = accession
        self.extension = extension
        self.cache_path = f"{CACHE_DIR}/{self.accession}.{self.extension}"
        self.local_file = f"{self.accession}.{self.extension}"

    def __enter__(self):
        """Download file if not exists, and move to temp."""
        if os.path.exists(self.cache_path):
            shutil.copyfile(self.cache_path, self.local_file)
        else:
            if self.extension == "bw":
                extension = "bigWig"
            else:
                extension = self.extension
            url = f"https://www.encodeproject.org/files/{self.accession}/@@download/{self.accession}.{extension}"
            with requests.get(url, stream=True, allow_redirects=True, timeout=60) as r:
                with open(self.local_file, "wb") as f:
                    shutil.copyfileobj(r.raw, f)
            shutil.copy(self.local_file, self.cache_path)
        return self.local_file

    def __exit__(self, exc_type, exc_value, traceback):
        """Remove the temporary file."""
        try:
            os.remove(self.local_file)
        except FileNotFoundError:
            pass
