import os
import math
import re
import csv
import logging

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from reverie import ReveCube
from reverie.image.tile import Tile

def compute_dark_spectrum(l1c: ReveCube, method = ["ols", "min"], dev = False):

    image = l1c.in_ds
    # filter image for bad bands
    bad_band_list = image["radiance_at_sensor"].bad_band_list
    if isinstance(bad_band_list, str):
        bad_band_list = str.split(bad_band_list, ", ")
    bad_band_list = np.array(bad_band_list)
    good_band_indices = np.where(bad_band_list == "1")[0]
    good_bands_slice = slice(min(good_band_indices), max(good_band_indices) + 1)
    image = image.isel(wavelength=good_bands_slice)

    num_pixels = 1000

    dark_spectrum = []
    coordinates = []

    if (method == "ols"):
        for wavelength in image['wavelength']:
            data = image["radiance_at_sensor"].sel(wavelength=wavelength).values.flatten()
            data = data[~np.isnan(data)]

            indices = np.argsort(data)[:num_pixels]
            values = data[indices]

            # Create an array of indices
            indices = np.arange(len(values))

            # Add a constant (intercept) to the model
            X = sm.add_constant(indices)

            # Fit the OLS model
            model = sm.OLS(values, X).fit()

            # Get the intercept
            intercept = model.params[0]
            dark_spectrum.append(intercept)
            coordinates.append((None, None))

            if (dev):
                plt.figure(figsize=(10, 6))
                plt.plot(values, marker='o', linestyle='-', color='b', label='Values')
                plt.plot(indices, model.predict(X), color='r', label=f'OLS Model (Intercept: {intercept:.2f})')
                plt.xlabel('Indices')
                plt.ylabel('Radiance at sensor')
                plt.title(wavelength.values.round())
                plt.grid(True)
                plt.legend()
                # plt.show()

                # Save the plot to the dev folder
                match = re.search(r'(.*?/reverie)', __file__)
                top_folder = match.group(1)
                dev_folder = os.path.join(top_folder, 'dev')
                os.makedirs(dev_folder, exist_ok=True)
                plot_filename = os.path.join(dev_folder, f'dark_pixels_{wavelength.values.round()}.png')
                plt.savefig(plot_filename)
                plt.close()

    if (method == "min"):
        for wavelength in image['wavelength']:
            logging.info(f"processing {wavelength.values.round()}")
            # dark_pixel = image["radiance_at_sensor"].sel(wavelength=wavelength)
            #
            # min_idx = dark_pixel.argmin(dim=["x", "y"])
            # dark_pixel = dark_pixel.isel(min_idx)
            #
            # dark_spectrum.append(dark_pixel.values)
            # coordinates.append((dark_pixel["x"].values, dark_pixel["y"].values))

            dark_pixel = np.nanmin(np.nanmin(image["radiance_at_sensor"].sel(wavelength=wavelength)))
            dark_spectrum.append(dark_pixel)
            coordinates.append((None, None))

    if (dev):
        # Plot the dark spectrum
        plt.figure(figsize=(10, 6))
        plt.plot(image['wavelength'].values, dark_spectrum, marker='o', linestyle='-', color='b')
        plt.xlabel('Wavelength')
        plt.ylabel('Radiance at sensor')
        plt.title('Dark Spectrum')
        plt.grid(True)
        # plt.show()

        match = re.search(r'(.*?/reverie)', __file__)
        top_folder = match.group(1)
        dev_folder = os.path.join(top_folder, 'dev')
        os.makedirs(dev_folder, exist_ok=True)
        plot_filename = os.path.join(dev_folder, f'dark_spectrum.png')
        plt.savefig(plot_filename)
        plt.close()

    # Write dark spectrum and coordinates to CSV
    csv_filename = os.path.join(top_folder, 'dark_spectrum.csv')
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Wavelength', 'Radiance at Sensor', 'X Coordinate', 'Y Coordinate'])
        for wavelength, radiance, (x, y) in zip(image['wavelength'].values, dark_spectrum, coordinates):
            writer.writerow([wavelength, radiance, x, y])

    return dark_spectrum

def fit_dark_spectrum(dark_spectrum):

    return aod550

if __name__ == "__main__":
    image_dir = "/D/Data/WISE/"

    images = [
        # "ACI-10A/220705_ACI-10A-WI-1x1x1_v01-L1CG.nc",
        # "ACI-11A/220705_ACI-11A-WI-1x1x1_v01-L1CG.nc",
        "ACI-12A/220705_ACI-12A-WI-1x1x1_v01-L1CG.nc",
        # "ACI-13A/220705_ACI-13A-WI-1x1x1_v01-L1CG.nc",
        # "MC-50A/190818_MC-50A-WI-2x1x1_v02-L1CG.nc",
        # "MC-37A/190818_MC-37A-WI-1x1x1_v02-L1CG.nc",
        # "MC-10A/190820_MC-10A-WI-1x1x1_v02-L1G.nc",
    ]

    for image in images:
        l1c = ReveCube.from_reve_nc(os.path.join(image_dir, image))

        dark_spectrum = compute_dark_spectrum(l1c, method="min")

        aod550 = fit_dark_spectrum(dark_spectrum)
