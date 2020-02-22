import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


class DensityExtractor:
    def __init__(self, image_filename, args, resolution=100):
        img = plt.imread(image_filename)
        self.ymax, self.xmax, _ = img.shape
        self.xmin, self.ymin = 0, 0
        self.xx = None
        self.positions = None
        self.point_estimate = args.point_estimate
        self.resolution = resolution

    def create_grid(self):
        # Create meshgrid
        self.xx = np.mgrid[self.xmin:self.xmax:self.resolution*1j,
                           self.ymin:self.ymax:self.resolution*1j]
        self.positions = np.vstack([i.ravel() for i in self.xx])

    def estimate(self, data):
        """Return the complete estimated distribution and the mean on both axes
        """
        if self.point_estimate:
            kernel = st.gaussian_kde(data)
            f = np.reshape(kernel(self.positions).T, self.xx.shape)
            imean = np.argmax(f)
            xtmean = int(round(self.xx[0][imean // self.resolution, imean % self.resolution]))
            self.xtmean = xtmean
            ytmean = int(round(self.xx[1][imean // self.resolution, imean % self.resolution]))
            self.ytmean = ytmean
            return f, (xtmean, ytmean)
        else:
            data1 = data[:2]
            data2 = data[2:]
            kernel1 = st.gaussian_kde(data1)
            kernel2 = st.gaussian_kde(data2)
            f1 = np.reshape(kernel1(self.positions).T, self.xx[0].shape)
            f2 = np.reshape(kernel2(self.positions).T, self.xx[0].shape)

            imean = np.argmax(f1)
            xtmean = int(round(self.xx[0][imean // self.resolution, imean % self.resolution]))
            ytmean = int(round(self.xx[1][imean // self.resolution, imean % self.resolution]))

            imean = np.argmax(f2)
            xbmean = int(round(self.xx[0][imean // self.resolution, imean % self.resolution]))
            ybmean = int(round(self.xx[1][imean // self.resolution, imean % self.resolution]))
            self.xtmean = xtmean
            self.ytmean = ytmean
            self.xbmean = xbmean
            self.ybmean = ybmean
            return f1, f2, (xtmean, ytmean), (xbmean, ybmean)
