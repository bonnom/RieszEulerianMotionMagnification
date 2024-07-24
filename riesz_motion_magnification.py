import cv2
import numpy as np
from scipy.signal import butter

class IIRTemporalFilter:
    def __init__(self, B, A):
        """Initialize the IIR temporal filter with filter coefficients B and A."""
        self.B = B
        self.A = A
        self.register0 = None
        self.register1 = None

    def filter(self, phase):
        """Apply the IIR filter to the input phase signal."""
        if self.register0 is None:
            self.register0 = np.zeros_like(phase)
            self.register1 = np.zeros_like(phase)

        temporally_filtered_phase = self.B[0] * phase + self.register0
        self.register0 = self.B[1] * phase + self.register1 - self.A[1] * temporally_filtered_phase
        self.register1 = self.B[2] * phase - self.A[2] * temporally_filtered_phase
        return temporally_filtered_phase

class RieszPyramid:
    def __init__(self, image, max_levels=3):
        """Initialize the Riesz pyramid with the given image and number of levels."""
        self.max_levels = max_levels
        self.image = image.astype(np.float32)
        self.expanded_pyramid = []
        self.laplacian_pyramid = []
        self.pyramid = self.build_riesz_pyramid(self.image, max_levels)

    def __getitem__(self, index):
        """Return the pyramid level at the specified index."""
        return self.pyramid[index]

    def build_gaussian_pyramid(self, image, max_levels):
        """Construct a Gaussian pyramid."""
        pyramid = [image]
        for _ in range(max_levels):
            image = cv2.pyrDown(image)
            pyramid.append(image)
        return pyramid

    def build_laplacian_pyramid(self, gaussian_pyramid):
        """Construct a Laplacian pyramid from the Gaussian pyramid."""
        pyramid = []
        self.expanded_pyramid = []
        for i in range(len(gaussian_pyramid) - 1, 0, -1):
            expanded = cv2.pyrUp(gaussian_pyramid[i])
            laplacian = cv2.subtract(gaussian_pyramid[i - 1], expanded)
            pyramid.append(laplacian)
            self.expanded_pyramid.append(expanded)
        pyramid.append(gaussian_pyramid[-1])
        return pyramid

    def apply_riesz_filters(self, octave):
        """Apply Riesz filters to the image octave."""
        kernel_x = np.array([[-0.49, 0, 0.49]], dtype=np.float32)
        kernel_y = kernel_x.T
        riesz_x = cv2.filter2D(octave, -1, kernel_x)
        riesz_y = cv2.filter2D(octave, -1, kernel_y)
        return riesz_x, riesz_y

    def build_riesz_pyramid(self, image, max_levels):
        """Construct the Riesz pyramid by combining Gaussian and Laplacian pyramids with Riesz filters."""
        gaussian_pyramid = self.build_gaussian_pyramid(image, max_levels)
        self.laplacian_pyramid = self.build_laplacian_pyramid(gaussian_pyramid)
        riesz_pyramid = [self.apply_riesz_filters(level) for level in self.laplacian_pyramid]
        return riesz_pyramid

    def get_levels(self):
        """Return the Gaussian and Laplacian pyramids."""
        return self.pyramid, self.laplacian_pyramid

def compute_phase_difference_and_amplitude(current_real, current_x, current_y, previous_real, previous_x, previous_y):
    """Compute phase difference and amplitude between current and previous Riesz pyramid levels."""
    eps = np.finfo(float).eps  # Small constant to prevent division by zero

    # Compute quaternion conjugate product
    q_conj_prod_real = current_real * previous_real + current_x * previous_x + current_y * previous_y
    q_conj_prod_x = -current_real * previous_x + previous_real * current_x
    q_conj_prod_y = -current_real * previous_y + previous_real * current_y

    # Compute amplitude and phase difference
    q_conj_prod_amplitude = np.sqrt(q_conj_prod_real**2 + q_conj_prod_x**2 + q_conj_prod_y**2)
    phase_difference = np.arccos(q_conj_prod_real / (eps + q_conj_prod_amplitude))
    cos_orientation = q_conj_prod_x / (eps + np.sqrt(q_conj_prod_x**2 + q_conj_prod_y**2))
    sin_orientation = q_conj_prod_y / (eps + np.sqrt(q_conj_prod_x**2 + q_conj_prod_y**2))

    # Compute the quaternionic phase
    phase_difference_cos = phase_difference * cos_orientation
    phase_difference_sin = phase_difference * sin_orientation

    # Compute the amplitude
    amplitude = np.sqrt(q_conj_prod_amplitude)

    return phase_difference_cos, phase_difference_sin, amplitude

def amplitude_weighted_blur(temporally_filtered_phase, amplitude, blur_kernel_size, sigma=1):
    """Apply amplitude-weighted Gaussian blur to the temporally filtered phase."""
    # Apply Gaussian blur to the amplitude
    blurred_amplitude = cv2.GaussianBlur(amplitude, blur_kernel_size, sigma)
    
    # Apply Gaussian blur to the product of temporally filtered phase and amplitude
    numerator = cv2.GaussianBlur(temporally_filtered_phase * amplitude, blur_kernel_size, sigma)
    
    # Avoid division by zero by adding a small epsilon
    denominator = np.finfo(float).eps + blurred_amplitude
    
    # Compute the spatially smooth temporally filtered phase
    spatially_smooth_temporally_filtered_phase = numerator / denominator
    
    return spatially_smooth_temporally_filtered_phase

def phase_shift_coefficient_real_part(riesz_real, riesz_x, riesz_y, phase_cos, phase_sin):
    """Compute the real part of the phase shift coefficient."""
    eps = np.finfo(float).eps  # Small constant to prevent division by zero
    phase_magnitude = np.sqrt(phase_cos**2 + phase_sin**2)
    
    # Add eps to phase_magnitude to avoid division by zero
    phase_magnitude_safe = phase_magnitude + eps
    
    exp_phase_real = np.cos(phase_magnitude)
    exp_phase_x = phase_cos / phase_magnitude_safe * np.sin(phase_magnitude)
    exp_phase_y = phase_sin / phase_magnitude_safe * np.sin(phase_magnitude)
    
    result = exp_phase_real * riesz_real - exp_phase_x * riesz_x - exp_phase_y * riesz_y
    
    return result

def reconstruct_from_pyramid_expanded(pyramid, expanded_pyramid, levels):
    """Reconstruct the image from the pyramid, including expanded pyramid levels."""
    reconstructed_frame = pyramid[0]
    for i in range(1, levels):
        reconstructed_frame = cv2.pyrUp(reconstructed_frame) + pyramid[i] + 0.5 * expanded_pyramid[i]
    return reconstructed_frame

def reconstruct_from_pyramid(pyramid, levels):
    """Reconstruct the image from the pyramid itself."""
    reconstructed_frame = pyramid[0]
    for i in range(1, levels):
        reconstructed_frame = cv2.pyrUp(reconstructed_frame) + pyramid[i]
    return reconstructed_frame

def normalize_to_uint8(motion_magnified_frame):
    """Normalize the image to the uint8 range [0, 255]."""
    motion_magnified_frame = np.asarray(motion_magnified_frame)
    min_val = np.min(motion_magnified_frame)
    max_val = np.max(motion_magnified_frame)
    if max_val == min_val:
        return np.zeros_like(motion_magnified_frame, dtype=np.uint8)
    normalized_frame = (motion_magnified_frame - min_val) / (max_val - min_val)
    scaled_frame = normalized_frame * 255
    uint8_frame = scaled_frame.astype(np.uint8)
    return uint8_frame

def magnify_motion(video_name, video_output_name, 
                   low_cutoff, high_cutoff, amplification, levels, 
                   color='gray', blur='amplitude', reconstruct_expanded=True):
    """
    Magnify motion in a video and save the result.

    Parameters:
    - video_name (str): Path to the input video file.
    - video_output_name (str): Path to the output video file.
    - low_cutoff (float): Lower cutoff frequency for bandpass filtering.
    - high_cutoff (float): Upper cutoff frequency for bandpass filtering.
    - amplification (float): Magnification factor for detected motion.
    - levels (int): Number of levels in the Riesz pyramid.
    - color (str): Color mode ('gray' for grayscale or 'color' for color images).
    - blur (str): Type of blur ('amplitude' for amplitude-weighted blur or 'gauss' for Gaussian blur).
    - reconstruct_expanded (bool): If True, use expanded pyramid reconstruction; otherwise, use standard pyramid reconstruction.
    """
    
    # Open the video file
    cap = cv2.VideoCapture(video_name)
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Retrieve video properties
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second of the video

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for output video
    if color=='gray':
        video_output_color = False
    else:
        video_output_color = True
    out = cv2.VideoWriter(video_output_name, fourcc, fps, (int(cap.get(3)), int(cap.get(4))), video_output_color)

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error reading video")
        return
    
    # Convert frame to grayscale if specified
    if color == 'gray':
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        frame = frame.astype(np.float32)
    
    # Create Riesz pyramid and Laplacian pyramid for the initial frame
    previous_riesz_pyramid = RieszPyramid(frame, levels)
    previous_laplacian_pyramid = previous_riesz_pyramid.laplacian_pyramid

    # Initialize phase and amplitude arrays
    phase_cos = [np.zeros_like(p[0]) for p in previous_riesz_pyramid]
    phase_sin = [np.zeros_like(p[1]) for p in previous_riesz_pyramid]
    motion_magnified_laplacian_pyramid = [np.zeros_like(p) for p in previous_laplacian_pyramid]

    # Design bandpass filter
    nyquist = 0.5 * fps  # Nyquist frequency
    low = low_cutoff / nyquist  # Normalized low cutoff frequency
    high = high_cutoff / nyquist  # Normalized high cutoff frequency
    B, A = butter(1, [low, high], btype='band')  # Bandpass filter coefficients

    # Create IIR filters for phase components
    iir_filter_cos = [IIRTemporalFilter(B, A) for _ in range(levels)]
    iir_filter_sin = [IIRTemporalFilter(B, A) for _ in range(levels)]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale if specified
        if color == 'gray':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            frame = frame.astype(np.float32)
        
        # Compute Riesz pyramid and Laplacian pyramid for the current frame
        riesz_pyramid = RieszPyramid(frame, levels)
        current_laplacian_pyramid = riesz_pyramid.laplacian_pyramid

        # Process each pyramid level
        for i in range(levels - 1):
            # Compute phase difference and amplitude between current and previous frames
            phase_difference_cos, phase_difference_sin, amplitude = compute_phase_difference_and_amplitude(
                riesz_pyramid.laplacian_pyramid[i], riesz_pyramid[i][0], riesz_pyramid[i][1],
                previous_riesz_pyramid.laplacian_pyramid[i], previous_riesz_pyramid[i][0], previous_riesz_pyramid[i][1]
            )
            # Update phase arrays
            phase_cos[i] += phase_difference_cos
            phase_sin[i] += phase_difference_sin

            # Filter the phase components using the IIR filter
            phase_filtered_cos = iir_filter_cos[i].filter(phase_cos[i])
            phase_filtered_sin = iir_filter_sin[i].filter(phase_sin[i])

            # Apply the selected blur
            if blur == 'amplitude':
                phase_filtered_cos = amplitude_weighted_blur(phase_filtered_cos, amplitude, (5, 5))
                phase_filtered_sin = amplitude_weighted_blur(phase_filtered_sin, amplitude, (5, 5))
            elif blur == 'gauss':
                phase_filtered_cos = cv2.GaussianBlur(phase_filtered_cos, (5, 5), 0)
                phase_filtered_sin = cv2.GaussianBlur(phase_filtered_sin, (5, 5), 0)
            else:
                pass
            
            # Amplify the filtered phases
            phase_magnified_filtered_cos = amplification * phase_filtered_cos
            phase_magnified_filtered_sin = amplification * phase_filtered_sin

            # Compute the motion magnified Laplacian pyramid
            motion_magnified_laplacian_pyramid[i] = phase_shift_coefficient_real_part(
                riesz_pyramid.laplacian_pyramid[i], riesz_pyramid[i][0], riesz_pyramid[i][1],
                phase_magnified_filtered_cos, phase_magnified_filtered_sin
            )

        # Set the last level of the Laplacian pyramid without modification
        motion_magnified_laplacian_pyramid[levels - 1] = current_laplacian_pyramid[levels - 1]

        # Reconstruct the motion magnified frame
        if reconstruct_expanded:
            motion_magnified_frame = reconstruct_from_pyramid_expanded(
                motion_magnified_laplacian_pyramid, riesz_pyramid.expanded_pyramid, levels
            )
        else:
            motion_magnified_frame = reconstruct_from_pyramid(
                motion_magnified_laplacian_pyramid, levels
            )

        # Normalize the frame to 8-bit unsigned integer format
        motion_magnified_frame = normalize_to_uint8(motion_magnified_frame)
        
        previous_riesz_pyramid = riesz_pyramid
        
        # Write the frame to the output video
        out.write(motion_magnified_frame)
        # Display the frame
        cv2.imshow('Motion Magnified', motion_magnified_frame)

        # Exit if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
# Example usage
if __name__ == "__main__":
    magnify_motion("baby.mp4", "baby_amplified_levels6.mp4", 
                   low_cutoff=0.1, high_cutoff=2, amplification=12, levels=3,
                   color='gray', blur='amplitude', reconstruct_expanded=True)

