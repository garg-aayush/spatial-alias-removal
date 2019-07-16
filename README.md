# spatial-alias-removal
Removing the spatial aliasing from the seismic data

## Blind Test
Input data with spatial aliasing |  Output data without spatial aliasing
:-------------------------:|:-------------------------:
![In shot domain](./figures_results/data_20m_true.png "In shot domain")  |  ![ In shot domain](./figures_results/data_10m_blind.png "In shot domain")
![In fk domain](./figures_results/data_20m_true_fk.png "In f-k domain")  |  ![ In shot domain](./figures_results/data_10m_blind_fk.png "In f-k domain")

### Influence of spatial aliasing in imaging
<p align="center"><strong> Imaging with spatially aliased data </strong></p>

![In shot domain](./figures_results/image_20m_true1.png "In shot domain") 


<p align="center"><strong> Imaging with spatially aliased removed data </strong></p>

![ In shot domain](./figures_results/image_10m_blind1.png "In shot domain")
