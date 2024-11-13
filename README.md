# magnetometer-calibrate

This library provides a C API for collecting raw magnetometer samples and generating hard iron offset and soft iron matrix calibrations using Qingde Li and John G.Griffiths's Ellipsoid Fitting algorithm. Requires the [GSL](https://www.gnu.org/software/gsl/) library.

## Relevant Linkks

* [Least Squares Ellipsoid Specific Fitting](https://www.researchgate.net/profile/Qingde-Li/publication/4070857_Least_squares_ellipsoid_specific_fitting/links/565c4e3e08aefe619b252553/Least-squares-ellipsoid-specific-fitting.pdf) original publication
* [MATLAB demonstration](https://www.mathworks.com/matlabcentral/fileexchange/23377-ellipsoid-fitting) files
* [MagCal C# implementation](https://github.com/hightower70/MagCal/blob/master/Program.cs)
* [Magneto C explanation](https://sites.google.com/view/sailboatinstruments1/d-implementation), [full C reference](https://sites.google.com/view/sailboatinstruments1/g-c-language-implementation), and [.exe download](https://sites.google.com/view/sailboatinstruments1/a-download-magneto-v1-2) (works well on Linux with `wine`)
