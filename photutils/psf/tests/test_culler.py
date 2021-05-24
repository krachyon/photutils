import numpy as np

from photutils.psf import culler
from photutils.psf.models import FittableImageModel

from astropy.table import Table


def test_smoke():
    image = np.ones((10, 10))
    model = FittableImageModel(np.array([[0, 0, 0], [0, 1., 0], [0, 0, 0]]), degree=1)

    my_culler = culler.ChiSquareCuller(2, image, model)
    detections = Table([[0, 1], [0, 0]], names=['xcentroid', 'ycentroid'])

    culled = my_culler.cull(detections)


def test_badpixels():
    # Do nan, negative value etc.
    pass