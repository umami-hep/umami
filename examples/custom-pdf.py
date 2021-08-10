import numpy as np

from umami.preprocessing import PDFSampling

# create some dummy data
x = np.random.default_rng().normal(size=1000)
y = np.random.default_rng().normal(1, 2, size=1000)

# get 2d histogram of our dummy data
h_original, x_bins, y_bins = np.histogram2d(x, y, [4, 5])

# calculate a custom function
pt = np.cos(x ** 2) + np.sin(x + y) + np.exp(x)
eta = 20 - y ** 2

h_target, _, _ = np.histogram2d(pt, eta, bins=[x_bins, y_bins])

ps = PDFSampling()
ps.CalculatePDFRatio(h_target, h_original, x_bins, y_bins)
ps.save("custom-pdf.pkl")
