# HyperspectralGraphNetwork

A multi-stream network based on graph k-NN for hyperspectral point cloud segmentation in geological application. Find our paper [here](https://www.mdpi.com/2072-4292/16/13/2336).

![GraphicalAbstract](https://github.com/user-attachments/assets/0dd362d8-3b48-435c-aa5f-52748c532ca7)

<img width="576" alt="Screenshot 2024-11-09 at 03 20 44" src="https://github.com/user-attachments/assets/7be482ff-8f78-485b-a68c-8a5e98443c55">

Download Tinto data from [RODARE](https://rodare.hzdr.de/record/2256).

First preprocess data:
```
python gen_h5.py
```
Then train and test:
```
python train.py
python test.py
```
Return prediction points with correct coordinates:
```
python return_coords.py
```

Cite the paper here:
> Rizaldy, A.; Afifi, A.J.; Ghamisi, P.; Gloaguen, R. Improving Mineral Classification Using Multimodal Hyperspectral Point Cloud Data and Multi-Stream Neural Network. Remote Sens. 2024, 16, 2336. https://doi.org/10.3390/rs16132336
