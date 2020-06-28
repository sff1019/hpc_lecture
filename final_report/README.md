# Final Report

Rewrote the 2-D Navier-Stokes code in C++ and CUDA.

## Environment
Both tested on Tsubame. CUDA code ran on NVIDIA TESLA P100.

Module list is shown below:

```
$ module list
Currently Loaded Modulefiles:
  1) cuda/10.2.89   2) cudnn/7.4      3) nccl/2.4.2     4) tmux/2.7       5) gcc/8.3.0
```

# Results (plot)
## Original
<p align="center">
  <img src="https://github.com/sff1019/hpc_lecture/blob/master/final_report/original_nt700.png?raw=true" alt="Original result"/>
</p>
src: original_nt.700.png

## C++
<p align="center">
  <img src="https://github.com/sff1019/hpc_lecture/blob/master/final_report/cpp_nt700.png?raw=true" alt="C++ result"/>
</p>
src: cpp_nt.700.png

## CUDA
<p align="center">
  <img src="https://github.com/sff1019/hpc_lecture/blob/master/final_report/cu_nt700.png?raw=true" alt="CUDA result"/>
</p>

src: cu_nt.700.png
