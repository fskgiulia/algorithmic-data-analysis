## Idea
Implementation of a function to compute the discrete wavelet transform of a time-series.
In particular, it computes:
* the weights of the decomposition,
* the corresponding matrix of basis vectors,
* the approximated time-series when retaining only a chosen number/fraction of weights with largest normal-
ized values, and
* the corresponding ratio of energy from the original time-series retained in the approximate time-series.

Then, the discrete wavelet transform is compared with the the decomposition obtained with the discrete Fourier transform (computed using the numpy.fft package).


## Dataset
The function is tested on local weather observations from the Finnish Meteorological Institute (https://en.ilmatieteenlaitos.fi/download-observations
!/).
