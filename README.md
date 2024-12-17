# Biosensing_analysis
Spectral grating image analysis

Python image analysis of hyperspectral 1D GMR gratings. The code analyses the brightness of gratings to assess the behaviour of the resonance wavelength as a function of voltage. 

## Operation
+ A universal ROI is selected from the first image, which is then reused for all images for consistency
+ Average brightness is next found for a given ROI by averaging all pixels across the area
+ Process is repeated for each voltage for each concentration
+ Data is plotted and a Gaussian fit is applied to extract mu (peak x val)