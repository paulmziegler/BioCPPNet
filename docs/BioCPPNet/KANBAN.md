---

kanban-plugin: basic

---

## Backlog

- [ ] Identify and download public mono datasets (e.g., Earth Species Project) for spatialization
- [ ] Develop interactive demo (Gradio) for multichannel file upload
- [ ] Investigate super-resolution techniques for TDOA estimation


## Signal Pipeline

- [x] Update project_config.yaml with coordinate-based array geometry (x, y, z)
- [x] Implement DataMixer with physics-based delay simulation (Virtual Array)
- [x] Create geometry-based delay calculation logic (Steering Vectors)
- [x] Verify simulation physics (sine wave delay test)
- [x] Handle high sampling rates (up to 250kHz)


## Modeling

- [ ] Implement Denoising Autoencoder (DAE) for initial noise reduction
- [ ] Reimplement BioCPPNet U-Net architecture
- [ ] Implement L1 Waveform Loss
- [ ] Implement STFT L1 Loss
- [ ] Implement Spectral Convergence Loss
- [ ] Setup training loop with YAML config


## Array Processing

- [x] Implement Delay-and-Sum Beamforming
- [ ] Implement Sub-sample Delay Estimation (GCC-PHAT + Interpolation)
- [ ] Implement MUSIC algorithm for DoA estimation
- [ ] Integrate beamformed signal as input to BioCPPNet


## Results

- [ ] SI-SDR Benchmarking
- [ ] Downstream classification accuracy evaluation
- [ ] Documentation and Obsidian export


***

## Archive

- [ ] Initial Project Setup
- [x] Add unit tests for STFT inversions and beamforming delays

%% kanban:settings
```
{"kanban-plugin":"basic"}
```
%%
