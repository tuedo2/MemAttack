# MemAttack

Dependencies:
```
torch torchvision numpy matplotlib pot 
```

Example usage:
To get curvature scores for CIFAR10, run
```
mkdir cifar10_curv_scores
python cifar10_curv.py
```

Then make plots with the first few cells provided in `curv_plots.ipynb`. Similar procedures can be done with other datasets and memorization proxies.