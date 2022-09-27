# CLIPSep
This is a anonymus code for the paper "CLIPSep: Learning text-queried sound separation with noisy unlabeled videos", submitted to ICLR 2023.  
All source code, including training code and data preprocessing, will be made publicly available upon acceptance.

### Setup environment
```
conda env create -f environment.yml
```

### Run visualization
```
OMP_NUM_THREADS=1 python inference.py -o checkpoints/clipsep_nit/ -t data/MUSIC/test.csv -t2 data/vggsound/test.csv
```
### Run evaluation
```
OMP_NUM_THREADS=1 python evaluate.py -o checkpoints/clipsep_nit/ -t data/MUSIC/test.csv -t2 data/vggsound/test.csv --vis_dir visualization
```
