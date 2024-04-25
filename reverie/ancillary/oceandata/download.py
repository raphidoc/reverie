"""
gets ancillary data from the ocean data server
written by Quinten Vanhellemont, RBINS for the PONDER project (2017-10-17)
modified by Raphael Mabit (2024-01-18)
"""

import os
import netrc

import urllib.parse
import requests


def download(
    anc_file: str,
    local_dir=None,
    override=False,
    oceandata_url="https://oceandata.sci.gsfc.nasa.gov/ob/getfile/",
):
    # TODO: persistent save GMAO data

    file_url = urllib.parse.urljoin(oceandata_url, anc_file)

    local_file = os.path.join(local_dir, anc_file)

    if os.path.exists(local_file) & (not override):
        print("File %s exists" % local_file)

    else:
        if not os.path.exists(os.path.dirname(local_file)):
            os.makedirs(os.path.dirname(local_file))
        print("Downloading file %s" % anc_file)

        nr = netrc.netrc()
        auth = nr.authenticators("earthdata")
        auth = (auth[0], auth[2])

        with requests.Session() as session:
            auth_rep = session.get(file_url, verify=True)
            if (auth_rep.status_code != 200):
                raise ValueError("Could not download file %s" % anc_file)

            data_rep = session.get(auth_rep.url, auth=auth, verify=True)

            with open(local_file, mode="wb") as file:
                for chunk in data_rep.iter_content(chunk_size=10 * 1024):
                    file.write(chunk)

        print("Finished downloading file %s" % anc_file)

    return local_file
