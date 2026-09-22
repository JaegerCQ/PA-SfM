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

On the current machine, the tested environment already exists as `cryoet`. Run `conda activate cryoet` and proceed directly to the pipeline; use this activation command in place of `conda activate PA_SfM` below.

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

The script enables batched Triton localization, vectorized time-gradient scattering, sensor-fast forward projection, and sensor-tiled backward projection by default. No additional environment exports are needed.

The complete two-pose pipeline (pose000 + pose001) was measured at **13 minutes 7 seconds** on one A100-SXM4-40GB, including approximately **280 seconds for each pose's volume training**. And for one new pose, the runtime is about only **450 seconds**. The runtime for all 10 poses is approximately **1 hour 13 minutes**.

To run only the measured two-pose case, set `END_POSE=1` in the script or launch it with `END_POSE=1 bash ./run_group3_pose_range.sh`.

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
