# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This module defines an interface for filtering (culling) photometric detections
"""
import abc

import numpy as np
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.nddata.utils import overlap_slices
from astropy.stats import sigma_clipped_stats
from astropy.table import Table

from .models import FittableImageModel
from .utils import _extract_psf_fitting_names

__all__ = ['CullerBase', 'ChiSquareCuller']


class CullerBase(metaclass=abc.ABCMeta):
    # TODO in the architecture draft this is supposed to be CullerAndEnderBase. I'm not sure it makes sense
    #  to always couple the concerns. Alternatively one could define the concerns separately and then
    #  inherit the final CullerAndEnder from both a separate culler and Ender.

    @abc.abstractmethod
    def cull_data(self, detections: Table) -> Table:
        """Return a filtered detection table. Optionally add culling criterion info to input table"""
        pass

    def __call__(self, *args, **kwargs):
        return self.cull_data(*args, **kwargs)


class ChiSquareCuller(CullerBase):
    r"""
    This class implements culling of detection tables according to a reduced χ² metric:
    .. math:: \chi^2_r =\frac{1}{\mathrm n\_pixel}
        \sum_{\mathrm pixel} \frac{ ({\mathrm image} - {\mathrm model})^2}{pixel\_variance}

    TODO it would be nice to support some form of masking to handle bad pixels
    Parameters
    ----------
    cutoff_chisquare : float
        filter out detections with a smaller χ² < ``cutoff_chisquare``
    image : np.ndarray
        Image in which the detections are occuring
    model : FittableImageModel
        Model to compare detections against. E.g. derived EPSF or custom model.
        Needs to have x/y parameters recognized by ``util._extract_psf_fitting_names``
        TODO if cutout_shape becomes a parameter, we can handle analytical models here...
    fitter : TODO is there a type for this?
        object that handles fitting of model to image.
        Needs to offer ``fitter(model, *fitparams, expected)`` interface
    """

    def __init__(self, cutoff_chisquare: float, image: np.ndarray, model: FittableImageModel, fitter=LevMarLSQFitter()):
        if len(image.shape) != 2:
            raise ValueError('Image must be 2D array')

        self.cutoff_chisquare = cutoff_chisquare
        self.model = model
        self.image = image
        self.fitter = fitter

        _, _, self.image_std = sigma_clipped_stats(self.image)

    # TODO seems a bit brittle to rely on these exact columns being present.
    #  Would be nicer to define a set of allowed multiple names or make the user pass the columns directly
    expected_columns = {'xcentroid', 'ycentroid'}
    colname = 'model_chisquare'

    # TODO should this make use of the star groups to account for close sources?
    #  that would make it a lot more complex and harder to interpret the cuttoff
    #  or Maybe the single-source approach with iterative subtraction works better?
    def cull_data(self, detections: Table) -> Table:
        if not self.expected_columns.issubset(detections.colnames):
            raise ValueError('table must contain columns {}'.format(self.expected_columns))

        cutout_shape = (np.array(self.model.data.shape) /
                        np.array(self.model.oversampling)).astype(int)

        y_template, x_template = np.indices(cutout_shape).astype(float)
        # center model and x/ys  around 0, 0. Start from there to converge faster
        x_template -= (cutout_shape[1] - 1) / 2  # shift by half of last element of array
        y_template -= (cutout_shape[0] - 1) / 2

        xname, yname, _ = _extract_psf_fitting_names(self.model)
        getattr(self.model, xname).value = 0.
        getattr(self.model, yname).value = 0.

        # TODO it would be nice to save this loop and paralellize or vectorize
        detections.add_column(col=np.nan, name=self.colname)
        for row in detections:
            xcenter, ycenter = row['xcentroid'], row['ycentroid']
            slices_large, slices_small = overlap_slices(self.image.shape,
                                                        cutout_shape,
                                                        (ycenter, xcenter), mode='trim')

            x = x_template[slices_small]
            y = y_template[slices_small]

            # TODO This is doing extra work that is already done/will be done in Photometry
            #  could this be saved/unified?
            fitted_model = self.fitter(self.model, x, y, self.image[slices_large])

            # If the model was adjusted by more than two pixels, it is safe to assume that the fit
            # converged so something else than the desired detection.
            # TODO this seems to work, but may require some sort of de-trending around very bright
            #  AO-PSF stars as the model has a tendency to walk up the halo.
            if abs(getattr(fitted_model, xname)) > 2 or abs(getattr(fitted_model, yname)) > 2:
                row[self.colname] = np.nan
            else:
                # TODO is this the correct formula to apply here?
                #  Looks like some scaling is needed to make this uniform across all images
                poisson_variance = self.image[slices_large]
                chisquares = (self.image[slices_large] - fitted_model(x, y)) ** 2 / \
                             (poisson_variance + self.image_std ** 2)
                reduced_chisquare = np.nansum(chisquares) / chisquares.size
                row[self.colname] = reduced_chisquare

        return detections[detections[self.colname] < self.cutoff_chisquare]
