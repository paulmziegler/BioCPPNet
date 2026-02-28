import os
import glob
import numpy as np
import soundfile as sf
import cv2
from src.spatial.beamforming import Beamformer

def get_acoustic_heatmap(bf, signal_chunk, az_range, el_range, res=5):
    """
    Generate an acoustic energy heatmap over the specified FOV.
    """
    azimuths = np.arange(az_range[0], az_range[1] + res, res)
    elevations = np.arange(el_range[0], el_range[1] + res, res)
    
    heatmap = np.zeros((len(elevations), len(azimuths)))
    
    # Calculate energy for each direction
    for i, el in enumerate(elevations):
        for j, az in enumerate(azimuths):
            # Delay and sum
            bf_signal = bf.delay_and_sum(signal_chunk, azimuth_deg=az, elevation_deg=el)
            heatmap[i, j] = np.var(bf_signal)
            
    return heatmap, azimuths, elevations

def process_file(wav_file, avi_file, output_dir):
    print("Processing: " + os.path.basename(wav_file) + " & " + os.path.basename(avi_file))
    
    # Load audio
    data, fs = sf.read(wav_file)
    signal = data.T  # (channels, samples)
    
    # Setup Beamformer (loads coordinates from config)
    bf = Beamformer(sample_rate=fs)
    print(f"  Using {bf.n_channels}-channel array configuration.")
    
    # Open video
    cap = cv2.VideoCapture(avi_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    
    # Define Camera Field of View (FOV) and grid resolution
    # Assumed FOV: 60 deg horizontal (-30 to 30), 40 deg vertical (-20 to 20)
    az_range = (-30, 30)
    el_range = (-20, 20)
    grid_res = 2  # 2 degrees for higher resolution heatmap
    
    duration = int(signal.shape[1] / fs)
    
    for sec in range(duration):
        # 1. Extract 1-second audio chunk
        start_sample = sec * fs
        end_sample = start_sample + fs
        chunk = signal[:, start_sample:end_sample]
        
        # 2. Extract corresponding video frame (middle of the second)
        frame_idx = int((sec + 0.5) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"  Warning: Could not read video frame at {sec}s.")
            break
            
        height, width = frame.shape[:2]
        
        # 3. Generate acoustic heatmap
        print(f"  Calculating heatmap for second {sec}...")
        heatmap, _, _ = get_acoustic_heatmap(bf, chunk, az_range, el_range, res=grid_res)
        
        # Normalize heatmap to 0-255
        heatmap_norm = heatmap - np.min(heatmap)
        max_val = np.max(heatmap_norm)
        if max_val > 0:
            heatmap_norm = (heatmap_norm / max_val * 255).astype(np.uint8)
        else:
            heatmap_norm = heatmap_norm.astype(np.uint8)
            
        # Optional: Thresholding to only highlight dominant sources (top 40% energy)
        _, mask = cv2.threshold(heatmap_norm, 153, 255, cv2.THRESH_BINARY)
        heatmap_norm = cv2.bitwise_and(heatmap_norm, mask)
        
        # 4. Map Spherical Coordinates to 2D Image Plane
        # The heatmap was computed with elevation rows, azimuth cols.
        # Elevations range from -20 (down) to +20 (up). 
        # OpenCV image index 0 is TOP. So we must flip the heatmap vertically.
        heatmap_flipped = np.flipud(heatmap_norm)
        
        # Resize heatmap to match video frame dimensions using cubic interpolation
        heatmap_resized = cv2.resize(heatmap_flipped, (width, height), interpolation=cv2.INTER_CUBIC)
        
        # Apply colormap (JET goes from blue to red)
        colored_heatmap = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        
        # Create alpha mask based on intensity to blend smoothly
        alpha = heatmap_resized.astype(float) / 255.0
        # Square the alpha to emphasize the hottest spots and make background transparent
        alpha = alpha ** 2
        
        # 5. Blend frame and heatmap
        overlay = np.zeros_like(frame)
        for c in range(3):
            overlay[:, :, c] = (1. - alpha) * frame[:, :, c] + alpha * colored_heatmap[:, :, c]
            
        final_frame = overlay.astype(np.uint8)
        
        # Add timestamp text
        cv2.putText(final_frame, f"Acoustic Camera - T+{sec}s", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save image
        out_name = f"{os.path.basename(wav_file).replace('.wav', '')}_sec_{sec:02d}.jpg"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, final_frame)
        print(f"  Saved {out_path}")
        
    cap.release()

if __name__ == "__main__":
    output_dir = "results/acoustic_camera"
    os.makedirs(output_dir, exist_ok=True)
    
    wav_files = sorted(glob.glob("recordings/outside/audio_*.wav"))
    for wav_file in wav_files:
        # Construct corresponding .avi filename
        avi_file = wav_file.replace("audio_", "video_").replace(".wav", ".avi")
        if os.path.exists(avi_file):
            process_file(wav_file, avi_file, output_dir)
        else:
            print(f"Warning: No matching video found for {wav_file}")
