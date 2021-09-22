# Cactus: Benchmarking GPGPU with Real Applications

Existing GPGPU benchmark suites used by academics have a bottom-up approach by focusing on essential computational kernels and algorithm implementations. Cactus, on the other hand, focuses on a top-down approach by using modern GPGPU applications in HPC domain where multiple kernels and algorithms are used in order to run an application. More information about the philosophy of Cactus and its methodology is described in the following paper:

```
Mahmood Naderan-Tahan and Lieven Eeckhout, 
"Cactus: Top-Down GPU-Compute Benchmarking using Real-Life Applications", 
IEEE International Symposium on Workload Characterization (IISWC), 2021.
```

This is a repository page of Cactus that contains scripts to fetch program sources, build them and run the workloads. As of version 1.0, Cactus contains the following applications and workloads:

1) Molecular simulations: [Gromacs](http://www.gromacs.org/) (GMS) and [LAMMPS](https://www.lammps.org/) with two inputs, Rhodo and Colloid (LMR and LMC)
2) Graph analytics: [Gunrock](https://gunrock.github.io) with BFS traversal on road and social networks (GRU and GST)
3) Machine learning: [PyTorch](https://pytorch.org/) with DCGAN modeling (DCG), neural style transformer (NST), reinforcement learning (RFL), spatial transformer (SPT) and language translation (LGT)

We have tested Cactus on [Nvidia RTX 3080](https://www.nvidia.com/fr-be/geforce/graphics-cards/30-series/rtx-3080-3080ti/) platform with [Ampere](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf) architecture and the following softwares

* OS                      ->      Ubuntu 20.04.1 with kernel 5.4.0
* CUDA Toolkit path       ->      /usr/local/cuda
* CUDA driver             ->      460.27
* CUDA                    ->      11.2 
* CUDA ARCH               ->      86
* CUDNN                   ->      8.1.0
* Gromacs                 ->      2021.1
* Lammps                  ->      19Oct2020
* Gunrock                 ->      1.1
* Pytorch                 ->      1.7.1

Generally, Cactus workloads can be run on different platforms not limited to Nvidia RTX 3080. However, it is important to note that a program itself may have restrictions on hardware or software versions and such information is mentioned in their documentation. The following notes are useful:

* Gromacs: Supports CUDA>4 and OpenCL>1.1
* LAMMPS: Supports CUDA>3.2 and OpenCL>1.1
* Gunrock: Supports CUDA>10.2
* Pytorch: Supports CUDA>3.7 (For OpenCL see [here](https://github.com/pytorch/pytorch/issues/488))



# Building Cactus

With respect to the default configurations mentioned earlier, if you have fresh Ubuntu 20.04.1 installation with RTX 3080 device, you can run the following three commands:

```
git clone https://github.com/gpubench/cactus
./setup.sh
```

Otherwise, for a custom configuration, open `./scripts/common` in a text editor prior to running `./setup` and review the following variables:

* `CACTUS_HOME`: The root folder of Cactus
* `ARCH`: This is the value of `-arch=compute_`. You can see this [page](https://arnon.dk/tag/cuda-arch/) if you don't know the correct number of your Nvidia device.
* `INSTALL_CUDA`: Set it to `YES` if you want the setup to automatically install CUDA-11.2 with driver v460.27. The installation procedure is 1) blacklist the nouveau driver, 2) update initramfs, 3) reboot and 4) execute cuda .run file with `--driver and --toolkit` switches. If you have a working CUDA driver and toolkit, you may want to set this variablet to `NO`.
* `INSTALL_CUDNN`: Set it to `YES` if you want the setup to automatically install CuDNN-8.1.0 for Pytorch workloads. If you have a working CuDNN, you may set this variable to `NO`.

Generally, `setup.sh` file runs the scripts in `./scripts` folder where each of them download and build a program. If all things go well, the setup will report that and exit normally. Cactus scripts use standard installation procedures described in the program documentations. Therefore, in case of a compilation error, it is recommended first to check the documets of that program.

# Running workloads

Upon a successfull installation, navigate to `./workloads/` and you see 10 workloads. Inside each folder, execute `./runme.sh` command and it will automatically fetch the necessary infput files and run the workload.

# Reporting problems

If you have question or problem with running Cactus workloads, open an issue in the repository page.

# Roadmap and future works

Since the main target of Cactus is academia and GPU simulators, the future work of Cactus is summarized below:

* Create and release RTX 3080 traces compatible with [Accel-Sim](https://accel-sim.github.io/).
* Test and expand the hardware platforms beyond Nvidia devices.
