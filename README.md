This repository is the official `neuromorphic Mosaic` code repository provided by [Yigit Demirag](https://yigit.ai).

### Getting Started

To install the dependencies, run:

```
conda env create -f mosaic-env.yml
conda activate mosaic-env
```

To train neuromorphic Mosaic architecture (with 8x8 distributed neuron tiles with 32 neurons on each tile) on Spiking Heidelberg Digits (SHD) dataset, run:

```
python train.py --lambda_con=0.1 --lr=0.001 --n_core=64 --n_epochs=500 --n_rec=2048 --noise_std=0.05 --noise_str=0
```

This will train a particular spiking recurrent neural network architecture for 500 epochs and exports the hardware-aware small-world connectivity matrix `connectivity_matrix.png` during the training.

PS: We used seeds `{42,52,62,72,82}` for our experiments.

### Acknowledgement

Our preprocessing of Spiking Heidelberg Digits (SHD) and Spiking Google Speech Commands (SSC) datasets is based on the code from [https://github.com/fzenke/spytorch](SpyTorch from Friedemann Zenke).


### Citation

If you find this code useful, feel free to cite our paper:

```
@article{Dalgaty2024,
  title = {Mosaic: in-memory computing and routing for small-world spike-based neuromorphic systems},
  volume = {15},
  ISSN = {2041-1723},
  url = {http://dx.doi.org/10.1038/s41467-023-44365-x},
  DOI = {10.1038/s41467-023-44365-x},
  number = {1},
  journal = {Nature Communications},
  publisher = {Springer Science and Business Media LLC},
  author = {Dalgaty,  Thomas and Moro,  Filippo and Demirağ,  Yiğit and De Pra,  Alessio and Indiveri,  Giacomo and Vianello,  Elisa and Payvand,  Melika},
  year = {2024},
  month = jan 
}
```

### License

MIT
