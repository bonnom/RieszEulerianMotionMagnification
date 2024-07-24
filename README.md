# Motion Magnification Using Riesz Pyramid

This Python script performs Eulerian video magnification of motion on video frames, adapting concepts from the Riesz Pyramid method. It is based on the work described in the article by Neal Wadhwa,Michael Rubinstein, William Freeman, Fredo Durand, "Riesz Pyramids for Fast Phase-Based Video Magnification". It must be noted however that this uses an traditional laplacian pyramid and not their Laplacian-like pyramid.
This makes the reconstruction of the original image not as good.

## Key Components

### IIR Temporal Filter
Implements an Infinite Impulse Response (IIR) filter to smooth temporal variations and isolate the frequencies of interest in the phase signal.

### Riesz Pyramid
- **Gaussian Pyramid:** Constructs a multi-resolution representation of the image by progressively downscaling it.
- **Laplacian Pyramid:** Laplacian Pyramid
- **Riesz Filters:** Applies directional derivatives (Riesz filters) to each level of the pyramid to capture directional information relevant for motion magnification.

### Phase Difference and Amplitude Computation
Uses quaternion-based calculations to compute the phase difference and amplitude between consecutive frames, quantifying temporal variations.

### Amplitude Weighted Blur
Applies Gaussian blur to the product of the temporally filtered phase and amplitude to smooth out noise and refine the motion magnification.

### Phase Shift Coefficient
Calculates the phase shift coefficient to adjust the Riesz coefficients based on the filtered phase and amplitude information.

### Reconstruction
Reconstructs the motion magnified image from the processed pyramid levels by combining the adjusted coefficients.

### Normalization
Normalizes the final image to the [0, 255] range, converting it to an 8-bit grayscale format suitable for display and saving.

### Video Processing
Reads a video file frame by frame, applies the motion magnification techniques, and writes the processed frames to a new video file. Provides real-time display of the motion magnified video.

## Important Notes

- **Limitations:** This script uses the traditional Laplacian Pyramid. Not the Laplacian-like pyramid from the article. This makes the reconstruction of the image from the laplacian pyramid different.

## Usage

To use this script, ensure you have the required dependencies (`opencv-python`, `numpy`, and `scipy`). Run the script with the following command:

```bash
python riesz_motion_magnification.py
