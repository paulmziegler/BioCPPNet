import sys
import os
import time
import threading
import queue

import numpy as np
import sounddevice as sd
import cv2
import pyqtgraph as pg

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QComboBox, QLabel, QSlider, QSizePolicy, QLineEdit,
    QCheckBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QImage, QPixmap


# --- Worker for Audio/Video Recording (Non-GUI Thread) ---
class RecordingWorker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, audio_device_id, video_device_id, audio_sample_rate, num_audio_channels, output_dir):
        super().__init__()
        self.audio_device_id = audio_device_id
        self.video_device_id = video_device_id
        self.audio_sample_rate = audio_sample_rate
        self.num_audio_channels = num_audio_channels
        self.output_dir = output_dir
        self._is_recording = False
        self.audio_file_path = None
        self.video_file_path = None

    def start_recording(self):
        self._is_recording = True
        self.audio_queue = queue.Queue()
        self.video_queue = queue.Queue()

        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.audio_file_path = os.path.join(self.output_dir, f"audio_{timestamp}.wav")
        self.video_file_path = os.path.join(self.output_dir, f"video_{timestamp}.avi")

        try:
            # Audio recording thread
            self.audio_thread = threading.Thread(target=self._record_audio_stream)
            self.audio_thread.start()

            # Video recording thread
            self.video_thread = threading.Thread(target=self._record_video_stream)
            self.video_thread.start()

        except Exception as e:
            self.error.emit(f"Failed to start recording: {e}")
            self._is_recording = False
            self.finished.emit()

    def stop_recording(self):
        self._is_recording = False
        if hasattr(self, 'audio_thread') and self.audio_thread.is_alive():
            self.audio_thread.join()
        if hasattr(self, 'video_thread') and self.video_thread.is_alive():
            self.video_thread.join()
        self.finished.emit()

    def _record_audio_stream(self):
        try:
            with sf.SoundFile(self.audio_file_path, mode='w', 
                             samplerate=self.audio_sample_rate, 
                             channels=self.num_audio_channels, 
                             subtype='FLOAT') as file:
                with sd.InputStream(device=self.audio_device_id,
                                    samplerate=self.audio_sample_rate,
                                    channels=self.num_audio_channels,
                                    callback=self._audio_callback) as stream:
                    while self._is_recording:
                        sd.sleep(100) # Keep stream open and active

                # Write any remaining data in queue
                while not self.audio_queue.empty():
                    file.write(self.audio_queue.get())
        except Exception as e:
            self.error.emit(f"Audio recording error: {e}")

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        if self._is_recording:
            self.audio_queue.put(indata.copy())

    def _record_video_stream(self):
        cap = cv2.VideoCapture(self.video_device_id, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.error.emit(f"Could not open video device {self.video_device_id}")
            return

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        fps = cap.get(cv2.CAP_PROP_FPS) or 30 # Default to 30 FPS if not detected
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if width == 0 or height == 0:
            self.error.emit(f"Could not get valid frame size from video device {self.video_device_id}. Check camera settings.")
            cap.release()
            return

        out = cv2.VideoWriter(self.video_file_path, fourcc, fps, (width, height))
        if not out.isOpened():
            self.error.emit(f"Could not open video writer for {self.video_file_path}")
            cap.release()
            return

        try:
            while self._is_recording:
                ret, frame = cap.read()
                if not ret:
                    self.error.emit("Failed to grab frame from video device.")
                    break
                out.write(frame)
        except Exception as e:
            self.error.emit(f"Video recording error: {e}")
        finally:
            cap.release()
            out.release()

# --- GUI Application ---        
class BioacousticRecorderApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bioacoustic Recorder")
        self.setGeometry(100, 100, 1000, 700)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.audio_input_stream = None
        self.is_recording = False
        self.recording_worker = None
        self.recording_thread = None
        
        # Output directory
        self.output_dir = "recordings"
        os.makedirs(self.output_dir, exist_ok=True)

        self._init_ui()
        self._list_devices()
        self._start_preview_timer()

    def _init_ui(self):
        # Device Selection
        device_selection_layout = QHBoxLayout()
        
        self.audio_label = QLabel("Audio Input:")
        self.audio_combo = QComboBox()
        self.audio_combo.currentIndexChanged.connect(self._on_audio_device_selected)
        device_selection_layout.addWidget(self.audio_label)
        device_selection_layout.addWidget(self.audio_combo)

        self.video_label = QLabel("Video Input:")
        self.video_combo = QComboBox()
        self.video_combo.currentIndexChanged.connect(self._on_video_device_selected)
        device_selection_layout.addWidget(self.video_label)
        device_selection_layout.addWidget(self.video_combo)
        
        self.layout.addLayout(device_selection_layout)

        # Output Folder Selection
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Folder:"))
        self.output_dir_line_edit = QLineEdit(self.output_dir)
        output_layout.addWidget(self.output_dir_line_edit)
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self._browse_for_folder)
        output_layout.addWidget(self.browse_button)
        self.layout.addLayout(output_layout)
        
        # Recording Controls
        control_layout = QHBoxLayout()
        self.record_button = QPushButton("Start Recording")
        self.record_button.clicked.connect(self._toggle_recording)
        control_layout.addWidget(self.record_button)

        self.status_label = QLabel("Ready")
        control_layout.addWidget(self.status_label)
        self.layout.addLayout(control_layout)

        # Live Previews
        previews_and_plots_layout = QHBoxLayout()
        
        # Video Preview
        video_stream_layout = QVBoxLayout()
        self.video_label = QLabel("Video Preview")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_display = QLabel()
        self.video_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        video_stream_layout.addWidget(self.video_label)
        video_stream_layout.addWidget(self.video_display)
        previews_and_plots_layout.addLayout(video_stream_layout)
        
        # --- Audio Plots Container ---
        audio_plots_container = QVBoxLayout()
        
        # Audio Waveform Plot
        waveform_header_layout = QHBoxLayout()
        self.waveform_label = QLabel("Audio Waveform (0 Channels)")
        self.waveform_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        waveform_header_layout.addWidget(self.waveform_label)
        
        self.auto_scale_checkbox = QCheckBox("Auto-Scale Y-Axis")
        self.auto_scale_checkbox.stateChanged.connect(self._toggle_auto_scale)
        waveform_header_layout.addWidget(self.auto_scale_checkbox)
        waveform_header_layout.addStretch() # Push checkbox to left but keep label centered-ish (or just let them flow)
        
        audio_plots_container.addLayout(waveform_header_layout)

        self.audio_plot_widget = pg.PlotWidget()
        self.audio_plot_item = self.audio_plot_widget.getPlotItem()
        self.audio_plot_item.setYRange(-1.0, 1.0)
        self.audio_plot_data = [] # Will be populated dynamically
        audio_plots_container.addWidget(self.audio_plot_widget)
        
        # Audio Spectrum Plot
        self.spectrum_label = QLabel("Frequency Spectrum (Channel 0)")
        self.spectrum_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spectrum_plot_widget = pg.PlotWidget()
        self.spectrum_plot_item = self.spectrum_plot_widget.getPlotItem()
        self.spectrum_plot_item.setLogMode(x=True, y=True)
        self.spectrum_plot_item.setLabel('bottom', 'Frequency', units='Hz')
        self.spectrum_plot_data = self.spectrum_plot_item.plot(pen='c')
        audio_plots_container.addWidget(self.spectrum_label)
        audio_plots_container.addWidget(self.spectrum_plot_widget)
        
        previews_and_plots_layout.addLayout(audio_plots_container)
        self.layout.addLayout(previews_and_plots_layout)

        # Timer for updating previews
        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self._update_previews)
        self.preview_interval_ms = 100 # Update ~10 times per second
        
        # Audio buffer for plot - will be re-initialized when device is selected
        self.plot_audio_buffer = None 
        self.audio_plot_queue = queue.Queue()
        
        self.cap = None # OpenCV VideoCapture object
        self.audio_stream = None # Sounddevice InputStream object

    def _toggle_auto_scale(self, state):
        if state == 2: # Checked
            self.audio_plot_item.enableAutoRange('y', True)
        else: # Unchecked
            self.audio_plot_item.enableAutoRange('y', False)
            self.audio_plot_item.setYRange(-1.0, 1.0)

    def _browse_for_folder(self):
        from PyQt6.QtWidgets import QFileDialog
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_dir = folder
            self.output_dir_line_edit.setText(folder)
            
    def _list_devices(self):
        # List Audio Devices
        self.audio_devices = sd.query_devices()
        for i, device in enumerate(self.audio_devices):
            if device['max_input_channels'] > 0:
                self.audio_combo.addItem(f"{device['name']} (ID: {i})", userData=i)
        
        # List Video Devices (using OpenCV)
        self.video_devices = []
        for i in range(10): # Try IDs from 0 to 9
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                backend_name = cap.getBackendName()
                # Only add if it's a real DirectShow device
                if backend_name == 'DSHOW':
                    self.video_devices.append({'name': f"DirectShow Camera (ID: {i})", 'id': i})
                    self.video_combo.addItem(f"DirectShow Camera (ID: {i})", userData=i)
                cap.release()
            
        if self.audio_combo.count() == 0:
            self.status_label.setText("No usable audio input devices found!")
            self.record_button.setEnabled(False)
        if not self.video_devices:
            self.status_label.setText("No video devices found!")
            self.record_button.setEnabled(False)

    def _start_preview_timer(self):
        self.preview_timer.start(self.preview_interval_ms)
        self._on_audio_device_selected(self.audio_combo.currentIndex()) # Start audio preview
        self._on_video_device_selected(self.video_combo.currentIndex()) # Start video preview
        
    def _on_audio_device_selected(self, index):
        if self.audio_input_stream:
            self.audio_input_stream.stop()
            self.audio_input_stream.close()
        
        device_id = self.audio_combo.itemData(index)
        if device_id is not None:
            device_info = sd.query_devices(device_id)
            self.selected_audio_device_id = device_id
            self.selected_audio_channels = device_info['max_input_channels']
            
            # Update label to show detected channels
            self.waveform_label.setText(f"Audio Waveform ({self.selected_audio_channels} Channels Detected)")

            # Dynamically create the plot buffer
            self.plot_audio_buffer = np.zeros((2048, self.selected_audio_channels), dtype=np.float32)

            # Clear old plot lines and create new ones
            self.audio_plot_item.clear()
            # Extended color palette for up to 16 channels
            colors = [
                '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF', '#FFFFFF',
                '#FFA500', '#800080', '#008000', '#800000', '#008080', '#000080', '#808000',
                '#FFC0CB', '#A52A2A'
            ]
            self.audio_plot_data = [self.audio_plot_item.plot(pen=pg.mkPen(colors[i % len(colors)], width=1)) 
                                   for i in range(self.selected_audio_channels)]
            self.audio_plot_item.setYRange(-1.0, 1.0) # Reset range
            
            # Start a new preview stream
            try:
                self.audio_input_stream = sd.InputStream(device=self.selected_audio_device_id,
                                                       samplerate=int(self.audio_devices[device_id]['default_samplerate']),
                                                       channels=self.selected_audio_channels,
                                                       callback=self._audio_preview_callback)
                self.audio_input_stream.start()
                self.status_label.setText(f"Audio device selected: {device_info['name']}")
            except Exception as e:
                self.status_label.setText(f"Error opening audio stream: {e}")
                print(f"Error opening audio stream: {e}")
                self.audio_input_stream = None

    def _audio_preview_callback(self, indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        # Put data into queue for plotting in GUI thread
        self.audio_plot_queue.put(indata.copy())

    def _on_video_device_selected(self, index):
        if self.cap:
            self.cap.release()
            self.video_display.clear()
            
        device_id = self.video_combo.itemData(index)
        if device_id is not None:
            self.selected_video_device_id = device_id
            self.cap = cv2.VideoCapture(self.selected_video_device_id, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.status_label.setText(f"Error opening video device {device_id}")
                print(f"Error opening video device {device_id}")
                self.cap = None
            else:
                self.status_label.setText(f"Video device selected: DirectShow Camera (ID: {device_id})")

    def _update_previews(self):
        # Update Video Preview
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Convert to RGB (OpenCV uses BGR)
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(convert_to_qt_format)
                # Scale pixmap to fit label while maintaining aspect ratio
                self.video_display.setPixmap(pixmap.scaled(self.video_display.size(), 
                                                         Qt.AspectRatioMode.KeepAspectRatio, 
                                                         Qt.TransformationMode.SmoothTransformation))
            else:
                self.video_display.clear()
        
        # Update Audio Plots
        if self.plot_audio_buffer is None:
            return

        new_data = False
        latest_indata = None
        while not self.audio_plot_queue.empty():
            indata = self.audio_plot_queue.get()
            latest_indata = indata
            num_samples = indata.shape[0]
            new_data = True
            
            # If the new data is larger than the buffer, only use the most recent part of it
            if num_samples > self.plot_audio_buffer.shape[0]:
                indata = indata[-self.plot_audio_buffer.shape[0]:, :]
                num_samples = indata.shape[0]

            # --- Update Waveform Plot ---
            self.plot_audio_buffer = np.roll(self.plot_audio_buffer, -num_samples, axis=0)
            self.plot_audio_buffer[-num_samples:, :] = indata
            
        if new_data:
            for i in range(len(self.audio_plot_data)):
                self.audio_plot_data[i].setData(self.plot_audio_buffer[:, i])
                
            # --- Update Spectrum Plot ---
            if latest_indata is not None and latest_indata.size > 0:
                fft_data = latest_indata[:, 0] # Use channel 0 of the latest chunk
                
                # Apply Hanning window to reduce spectral leakage
                fft_result = np.fft.rfft(fft_data * np.hanning(len(fft_data)))
                fft_magnitude = np.abs(fft_result)

                # Get sample rate for frequency axis calculation
                device_id = self.selected_audio_device_id
                sample_rate = int(self.audio_devices[device_id]['default_samplerate'])
                fft_freqs = np.fft.rfftfreq(len(fft_data), 1.0 / sample_rate)

                # Update plot data (avoiding the DC component at index 0 for log plot)
                if self.spectrum_plot_data is not None:
                    self.spectrum_plot_data.setData(fft_freqs[1:], fft_magnitude[1:])
                    self.spectrum_plot_item.enableAutoRange('y')

    def _toggle_recording(self):
        if not self.is_recording:
            if self.selected_audio_device_id is None or self.selected_video_device_id is None:
                self.status_label.setText("Please select both audio and video devices.")
                return
            
            self.status_label.setText("Recording...")
            self.record_button.setText("Stop Recording")
            self.is_recording = True

            # Stop preview streams/captures before starting recording worker
            self.preview_timer.stop()
            if self.audio_input_stream:
                self.audio_input_stream.stop()
                self.audio_input_stream.close()
                self.audio_input_stream = None # Ensure it's reset
            if self.cap:
                self.cap.release()
                self.cap = None # Ensure it's reset

            self.output_dir = self.output_dir_line_edit.text()
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Start recording worker in a new thread
            self.recording_worker = RecordingWorker(self.selected_audio_device_id, 
                                                  self.selected_video_device_id, 
                                                  int(self.audio_devices[self.selected_audio_device_id]['default_samplerate']),
                                                  self.selected_audio_channels,
                                                  self.output_dir)
            self.recording_thread = threading.Thread(target=self.recording_worker.start_recording)
            self.recording_worker.finished.connect(self._on_recording_finished)
            self.recording_worker.error.connect(self._on_recording_error)
            self.recording_thread.start()
            
        else:
            self.status_label.setText("Stopping recording...")
            if self.recording_worker:
                self.recording_worker.stop_recording()
            
    def _on_recording_finished(self):
        self.is_recording = False
        self.record_button.setText("Start Recording")
        self.status_label.setText(f"Recording saved to {self.recording_worker.audio_file_path} and {self.recording_worker.video_file_path}")
        self._start_preview_timer() # Restart previews

    def _on_recording_error(self, message):
        self.is_recording = False
        self.record_button.setText("Start Recording")
        self.status_label.setText(f"Error: {message}")
        self._start_preview_timer() # Restart previews
        
    def closeEvent(self, event):
        if self.audio_input_stream:
            self.audio_input_stream.stop()
            self.audio_input_stream.close()
        if self.cap:
            self.cap.release()
        if self.recording_worker and self.is_recording:
            self.recording_worker.stop_recording()
        super().closeEvent(event)

if __name__ == "__main__":
    # Ensure soundfile is imported for the worker to use it directly
    import soundfile as sf
    app = QApplication(sys.argv)
    window = BioacousticRecorderApp()
    window.show()
    sys.exit(app.exec())
