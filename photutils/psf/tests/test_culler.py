import numpy as np
import pytest
from astropy.modeling.functional_models import Gaussian2D, AiryDisk2D
from astropy.table import Table

from photutils.psf.culler import ChiSquareCuller
from photutils.psf.models import FittableImageModel

names = ['xcentroid', 'ycentroid']


def test_smoke():
    """a simple application of Culler to see if anything explodes outright"""
    image = np.ones((10, 10))
    model = FittableImageModel(np.array([[0, 0, 0], [0, 1., 0], [0, 0, 0]]), degree=1)

    my_culler = ChiSquareCuller(2, image, model)

    detections = Table([[0, 1], [0, 0]], names=names)
    culled = my_culler.cull_data(detections)
    # TODO without redoing the table it can't add an existing column. This is a bit annoying to use
    detections = Table([[0, 1], [0, 0]], names=names)
    culled = my_culler(detections)


@pytest.fixture
def gauss_test_image():
    size = 128
    ygrid, xgrid = np.indices((size, size))

    xy_gauss = np.array([[30., 30.], [70.1, 70.1], [81.2, 10.3], [20., 60.], [20.5, 61.]])
    gauss = Gaussian2D(x_stddev=1., y_stddev=1., x_mean=0., y_mean=0.)

    image = np.zeros((size, size))
    for x, y in xy_gauss:
        image += gauss(xgrid - x, ygrid - y)

    detections = Table(xy_gauss, names=names)
    return image, gauss, detections


def test_use_gaussian(gauss_test_image):
    image, model_template, detections = gauss_test_image
    x, y = np.mgrid[-2:2:5j, -2:2:5j]
    model = FittableImageModel(model_template(x, y))

    cutoff = 0.00001
    my_culler = ChiSquareCuller(cutoff, image, model)

    culled = my_culler(detections)

    # first entry is far away from everything and at pixel phase 0, should fit perfectly
    assert np.isclose(detections[0][ChiSquareCuller.colname], 0)
    # only first detection should 'survive'
    assert len(culled) == 1

    # others still should be pretty good even though pixel phase and crowding play a role
    for row in detections[1:]:
        assert row[ChiSquareCuller.colname] < 0.1


@pytest.fixture
def various_test_image():
    size = 128
    ygrid, xgrid = np.indices((size, size))

    # two groups of
    xy = np.array([[25., 10.], [10., 25.], [100., 80.], [80., 100.]])

    gauss = Gaussian2D(x_stddev=1., y_stddev=1., x_mean=0., y_mean=0.)
    airy = AiryDisk2D(radius=1.)

    gauss_big = Gaussian2D(x_stddev=2., y_stddev=2.)
    airy_big = AiryDisk2D(radius=2.)

    image = np.zeros((size, size))
    image += gauss(xgrid - xy[0, 0], ygrid - xy[0, 1])
    image += airy(xgrid - xy[1, 0], ygrid - xy[1, 1])
    image += gauss_big(xgrid - xy[2, 0], ygrid - xy[2, 1])
    image += airy_big(xgrid - xy[3, 0], ygrid - xy[3, 1])

    detections = Table(xy, names=names)
    return image, gauss, detections


def test_different_shapes(various_test_image):
    image, model_template, detections = various_test_image
    x, y = np.mgrid[-2:2:5j, -2:2:5j]
    model = FittableImageModel(model_template(x, y))

    my_culler = ChiSquareCuller(1., image, model)

    culled = my_culler(detections)

    # first should match exactly
    assert np.isclose(detections[0][ChiSquareCuller.colname], 0)

    # everything else should be worse
    for row in detections[1:]:
        assert row[ChiSquareCuller.colname] > detections[0][ChiSquareCuller.colname]

    # big airy should be worse than airy with similar radius to gaussian
    assert detections[1][ChiSquareCuller.colname] > detections[3][ChiSquareCuller.colname]


@pytest.mark.xfail
def test_badpixels():
    # Do nan, negative value etc.
    raise NotImplemented
