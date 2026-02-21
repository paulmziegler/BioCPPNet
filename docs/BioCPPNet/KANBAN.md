---

kanban-plugin: basic

---

## Visual Board

```mermaid
graph TD
  subgraph Backlog
    B1[Identify mono datasets]
    B2[Gradio Demo]
    B3[Atmos Absorption]
    B4[Trajectories]
  end
  subgraph Signal_Pipeline
    S1[250kHz Support]
    S2[Noise Models]
    S3[Multi-Source]
    S4[Reverb]
    S5[Sensor Perturb]
    S6[Source Movement]
  end
  subgraph Modeling
    M1[OnlineDataset]
    M2[DAE]
    M3[U-Net]
    M4[Loss Functions]
    M5[Training Loop]
    M6[E2E Pipeline]
  end
  subgraph Array_Processing
    A1[Beamforming]
    A2[GCC-PHAT]
    A3[MUSIC]
    A4[E2E Integration]
  end
  subgraph Results
    R1[SI-SDR]
  end
```

## Backlog

- [ ] Identify and download public mono datasets (e.g., Earth Species Project) for spatialization
- [ ] Develop interactive demo (Gradio) for multichannel file upload
- [ ] Investigate super-resolution techniques for TDOA estimation
- [ ] Implement Atmospheric Absorption (Frequency-dependent attenuation)


## Signal Pipeline

- [x] Update project_config.yaml with coordinate-based array geometry (x, y, z)
- [x] Implement DataMixer with physics-based delay simulation (Virtual Array)
- [x] Create geometry-based delay calculation logic (Steering Vectors)
- [x] Verify simulation physics (sine wave delay test)
- [x] Handle high sampling rates (up to 250kHz)
- [x] Implement Noise Models (White, Pink, Rain, Cocktail Party)
- [x] **Multi-Source Mixing:** Support arbitrary N sources.
- [x] **Reverberation:** Convolution with synthetic/real RIRs.
- [x] **Sensor Perturbation:** Random gain/phase mismatch per channel.
- [x] **Source Movement:** Trajectory-based delays.


## Modeling

- [x] **Create PyTorch `OnlineDataset` for on-the-fly augmentation**
- [x] **Implement Denoising Autoencoder (DAE) for initial noise reduction**
- [x] **Setup training loop with YAML config**
- [x] **Reimplement BioCPPNet U-Net architecture**
- [x] **Implement Loss Functions (L1, STFT, SC)**
- [x] Integrate End-to-End Pipeline (Beamformer + DAE + U-Net)


## Array Processing

- [x] Implement Delay-and-Sum Beamforming
- [x] **Implement Sub-sample Delay Estimation (GCC-PHAT + Interpolation)**
- [x] Implement MUSIC algorithm for DoA estimation
- [x] Integrate beamformed signal as input to BioCPPNet


## Results

- [ ] SI-SDR Benchmarking
- [ ] Downstream classification accuracy evaluation
- [ ] Documentation and Obsidian export


***

## Archive

- [x] Initial Project Setup
- [x] Add unit tests for STFT inversions and beamforming delays

%% kanban:settings
```
{"kanban-plugin":"basic"}
```
%%
