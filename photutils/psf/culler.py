# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This module defines an interface for filtering (culling) photometric detections
"""
import abc

import numpy as np
from astropy.table import Table
from .models import FittableImageModel
from astropy.nddata.utils import overlap_slices
from astropy.modeling.fitting import LevMarLSQFitter


class Culler(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def cull(self, detections: Table) -> Table:
        pass

    def __call__(self, *args, **kwargs):
        return self.cull(*args, **kwargs)




class ChiSquareCuller(Culler):
    """
    sum((pixel_value - model_value)**2/pixel_value) / n_pixel
    """

    def __init__(self, cutoff_chisquare: float, image: np.ndarray, model: FittableImageModel, fitter=LevMarLSQFitter()):
        if len(image.shape) != 2:
            raise ValueError('Image must be 2D array')

        self.cutoff_chisquare = cutoff_chisquare
        self.model = model
        self.image = image
        self.fitter = fitter

    expected_columns = {'xcentroid', 'ycentroid'}
    colname = 'model_chisquare'

    # TODO should this make use of the star groups to account for close sources?
    #  that would make it a lot more complex and harder to interpret the cuttoff
    #  Maybe the single-source approach with iterative subtraction works better?
    #  Looks like there's an issue that close sources from PSF artifacts
    #  will just converge to the center and therefore look decent.
    #  Maybe restrict maximum fit deviation to 1px or something?
    def cull(self, detections: Table) -> Table:
        if not self.expected_columns.issubset(detections.colnames):
            raise ValueError('table must contain columns {}'.format(self.expected_columns))

        cutout_shape = (np.array(self.model.data.shape) /
                        np.array(self.model.oversampling)).astype(int)

        y_template, x_template = np.indices(cutout_shape).astype(float)
        # by default EPSF models are centered around 0, 0. Start from there to converge faster
        x_template -= (cutout_shape[1]-1)/2  # shift by half of last element of array
        y_template -= (cutout_shape[0]-1)/2

        # TODO it would be nice to save this loop and paralellize or vectorize
        detections.add_column(col=np.nan, name=self.colname)
        for row in detections:
            xcenter, ycenter = row['xcentroid'], row['ycentroid']
            slices_large, slices_small = overlap_slices(self.image.shape,
                                             cutout_shape,
                                             (ycenter, xcenter), mode='trim')

            x = x_template[slices_small]
            y = y_template[slices_small]
            fitted_model = self.fitter(self.model, x, y, self.image[slices_large])

            # TODO is purely poisson always a good idea?
            #  looks like some additional scaling is needed to make the criterion uniform
            variance = self.image[slices_large]
            chisquares = (self.image[slices_large] - fitted_model(x, y))**2 / variance
            reduced_chisquare = np.nansum(chisquares) / chisquares.size
            row[self.colname] = reduced_chisquare

        return detections[detections[self.colname] < self.cutoff_chisquare]
