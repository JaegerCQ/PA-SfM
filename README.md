# PA-SfM: Tracker-free differentiable acoustic radiation for freehand 3D photoacoustic imaging    

[***Preprint paper***](https://doi.org/10.13140/RG.2.2.32941.04328)

We introduce PA-SfM, a tracker-free differentiable acoustic structure-from-motion (SfM) framework that recovers relative imaging poses directly from PA measurements. By integrating a differentiable acoustic radiation model with hierarchical optimization and rigid array constraints, PA-SfM jointly estimates inter-view transformations and reconstructs 3D PA volumes without external pose measurements. We demonstrate genuine freehand 3D PAI of human hand vasculature, in which arbitrary hand motion over approximately 1 s provides multi-view measurements from which PA-SfM recovers the relative poses and jointly reconstructs a large FOV vascular network without motion tracking or predefined trajectories.  

![image](pictures/sequential_display.png)           
_Repeatability validation of PA-SfM freehand 3D reconstruction of hand vessels._        

![image](pictures/freehand.png)        
_PA-SfM freehand 3D reconstructions of hand vessels._

![image](pictures/pipeline_final.png)        
_The overview of PA-SfM pipeline._    

## Create Conda Environment   

This edition was tested on a single NVIDIA A100-SXM4-40GB with Python 3.10.20, PyTorch 2.5.1+cu121, Triton 3.1.0, NumPy 1.26.4, and SciPy 1.15.3. Create the pinned environment with the following commands; pip downloads the locked packages from their online sources.

```bash
conda create -n PA_SfM --file locks/conda-explicit.txt
conda activate PA_SfM

python -m pip install \
  --require-hashes \
  -r locks/requirements.pip-hash-lock.txt
```

## Data Layout   

(Note: The data was acquired with 3D-PanoPACT system from Professor Junhui Shi and Dr. Xuanhao Wang. If you need to use this data in any context, please make sure to contact us.)
Place input files under `data/` with names expected by `run_group3_pose_range.sh`, for example:

```text
data/
  sensor_location_group03_pose000.txt
  processed_signal_group03_pose000.txt
  processed_signal_group03_pose001.txt
  ...
```

## Settings

The default configuration in `run_group3_pose_range.sh` uses one A100 and processes poses 000 through 009:

```shellscript
GPU_IDS=(0)
START_POSE=0
END_POSE=9
```

The script enables batched Triton localization, vectorized time-gradient scattering, and sensor-tiled backward projection by default. On the tested A100, volume-training forward projection groups sources into 16 x 16 x 16 cubes. Each 64-thread GPU block uses two independent 256-bin shared-memory histograms, one per warp, to reduce contention. Contributions are quantized individually to int64 before accumulation; the two histograms are merged into global memory after synchronization. Contributions outside the local caches go directly to global memory. The same projector is used for full signal generation.

For the default 1024-sensor volume fit, each backward program owns a source block. It accumulates each sensor lane's already-quantized integers throughout the loop, reduces the four lanes once after the loop, and writes once without atomics. This removes repeated intermediate reductions while preserving the per-pair floating calculation and quantization.

`TRAIN_FORWARD_PROJECTOR=auto` selects the shared histogram on SM80 GPUs when the grid size is divisible by 16; other configurations retain the original Triton projector. Runtime CUDA compilation uses NVRTC 12.1.105 from the already pinned `nvidia-cuda-nvrtc-cu12` package and the NVIDIA driver.

Localization uses batches of 16 sensors, a cache covering the full sigma schedule, and CUDA Graph replay of all 600 original Adam steps. Rigid refinement uses a fused continuous Gaussian operator with analytic position gradients, skipping blocks whose Gaussian exponentials are already zero in float32.

On one A100-SXM4-40GB, a **fresh two-pose run measured 287.8 seconds (4 minutes 48 seconds)** using the normal shell pipeline with `START_POSE=0 END_POSE=1 TARGET_POSE=0`. Both volumes were freshly trained for 100 epochs, and their volume-training stages took **102 and 102 seconds**.

| Stage | Measured seconds |
| --- | ---: |
| Initial pose000 DAS reconstruction | 5 |
| Pose000 volume training, 100 epochs | 102 |
| Pose000 full signal generation | 6 |
| Pose001 volume training, 100 epochs | 102 |
| Pose001 full signal generation | 6 |
| Pose001 localization, 256 sensors | 42 |
| Pose001 RANSAC and rigid refinement, 100 epochs | 19 |
| Two-pose joint DAS reconstruction | 6 |

Within this run, adding pose001 after pose000 took **175 seconds (2 minutes 55 seconds)** according to the shell's adaptive-pipeline timer. The runtime for all 10 poses is only approximately 28 minutes.

To run the two-pose case from scratch, set `END_POSE=1` in the script or launch it with `END_POSE=1 bash ./run_group3_pose_range.sh`. For diagnostic comparisons, `TRAIN_FORWARD_PROJECTOR=triton` selects the original Triton forward projector, `LOCALIZATION_FINE_EXECUTION=eager` retains the original localization loop, and `REFINE_BACKEND=torch` retains the original direct PyTorch refinement operator.

Setting `START_POSE` above zero resumes an existing run and requires the preceding pose's checkpoint and recovered coordinates in this directory.

## Run Pipeline  

Run these commands from the `PA-SfM` directory.

```bash
conda activate PA_SfM
chmod +x run_group3_pose_range.sh
nohup ./run_group3_pose_range.sh > main_group3_pose_range.log 2>&1 &
```

Monitor progress:

```bash
tail -f main_group3_pose_range.log
```

For an optional input, dependency, and GPU check without starting reconstruction, activate the environment and run:

```bash
bash ./run_group3_pose_range.sh --check
```

## Citation 

```   
@article{li2026pa,   
  title={PA-SfM: Tracker-free differentiable acoustic radiation for freehand 3D photoacoustic imaging},        
  author={Li, Shuang and Gao, Jian and Kim, Chulhong and Choi, Seongwook and Huang, Hao and Wang, Xuanhao and Shi, Junhui and Chen, Qian and Wang, Yibing and Wu, Shuang and Zhang, Yu and Huang, Tingting and Zhou, Yucheng and Yao, Boxin and Yao, Yao and Li, Changhui},      
  journal={bioRxiv},       
  pages={2026--04},       
  year={2026},      
  publisher={Cold Spring Harbor Laboratory}  
}    
```

```
@article{wang2025cross,  
  title={Cross-regional real-time visualization of systemic physiology and dynamics with 3D panoramic photoacoustic computed tomography (3D-PanoPACT)},  
  author={Wang, Xuanhao and Meng, Yuqian and Sun, Mingli and Gao, Xiali and Wang, Yuqi and Wang, Shaobo and Wang, Kaiyue and Wang, Ruofan and Ren, Danyang and Yin, Yonggang and others},  
  journal={Nature Communications},  
  volume={16},  
  number={1},  
  pages={10077},  
  year={2025},  
  publisher={Nature Publishing Group UK London}  
}  
```

```
@article{choi2023deep,
  title={Deep learning enhances multiparametric dynamic volumetric photoacoustic computed tomography in vivo (DL-PACT)},
  author={Choi, Seongwook and Yang, Jinge and Lee, Soo Young and Kim, Jiwoong and Lee, Jihye and Kim, Won Jong and Lee, Seungchul and Kim, Chulhong},
  journal={Advanced Science},
  volume={10},
  number={1},
  pages={2202089},
  year={2023},
  publisher={Wiley Online Library}
}
```

## Ackonwledgement

We are deeply grateful to Professor Chulhong Kim, Professor Junhui Shi, Dr. Seongwook Choi, Dr. Xuanhao Wang, Dr. Hao Huang and Dr. Zhibo Xiao for providing the invaluable in vivo experimental data.
