# Geometric-metrics-for-perceptual-manifolds-in-deep-neural-networks
## Perceptual Manifold in Deep Neural Network
In the neural system, when neurons receive stimuli from the same category with different physical features, a perceptual manifold is formed. The formation of perceptual manifolds helps the neural system to perceive and process objects of the same category with different features distinctly. Recent studies have shown that the response of deep neural networks to images is similar to human vision and follows the manifold distribution law. Specifically, embeddings of natural images are distributed near a low-dimensional manifold embedded in a high-dimensional space.

Given a set of data <span>$X=[x_1, \dots, x_m]$</span>, and a trained deep neural network, <span>$Model = \{f(x, \theta_1), g(z, \theta_2)\}$</span>, where <span>$f(x, \theta_1)$</span> and <span>$g(z, \theta_2)$</span> represent the representation network and classifier of the model, respectively. The representation network extracts <span>$p$</span>-dimensional embeddings <span>$Z=[z_1, \dots, z_m] \in \mathbb{R}^{p \times m}$</span> for <span>$X$</span>, where <span>$z_i = f(x_i, \theta_1) \in \mathbb{R}^p$</span>. The point cloud manifold formed by the set of embeddings <span>$Z$</span> is referred to as the perceptual manifold in deep neural networks.

Natural datasets exhibit inherent patterns that can be generalized under the manifold distribution principle: the distribution of a class of data is close to a low-dimensional perceptual manifold. As illustrated, data classification can be viewed as the unraveling and separation of perceptual manifolds. The difficulty of classifying a manifold increases when it is entangled with other perceptible manifolds. Typically, a deep neural network consists of a feature extractor and a classifier. Feature learning can be considered as unfolding perceptual manifolds, where a well-learned feature extractor can often unfold multiple manifolds for the classifier to decode. From this perspective, all factors related to manifold complexity may impact the classification performance of the model. We will provide metrics for the curvature, volume, and separability of perceptual manifolds.

![fig24](https://github.com/mayanbiao1234/Geometric-metrics-for-perceptual-manifolds-in-deep-neural-networks/assets/31196857/5b5d4ee3-cab2-4078-a7eb-09d52648121b)

## 1. Curvature metrics for perceptual manifolds in deep neural networks (CVPR 2023)
The curvature metric of the perceptual manifold in deep neural networks allows analyzing the fairness of the model from a geometric point of view.
For related conclusions on the curvature and model preferences of perceptual manifolds please refer to the paper, [Curvature-Balanced Feature Manifold Learning for Long-Tailed Classification](https://arxiv.org/abs/2303.12307)

The citation format is: 

```
@inproceedings{ma2023curvature,
  title={Curvature-Balanced Feature Manifold Learning for Long-Tailed Classification},
  author={Ma, Yanbiao and Jiao, Licheng and Liu, Fang and Yang, Shuyuan and Liu, Xu and Li, Lingling},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15824--15835},
  year={2023}
}
```

### The following is the code to calculate the average Gaussian curvature of the perceptual manifold.

```python
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 14:14:15 2024

@author: Yanbiao Ma
"""
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import svd


def estimate_manifold_curvature(X, k=10):
    N, D = X.shape  # N is the number of samples, D is the dimensionality
    local_curvatures = np.zeros(N)

    # Initialize the NearestNeighbors model
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    
    for i in range(N):
        # Find the k+1 nearest neighbors to include the point itself
        indices = nbrs.kneighbors(X[i].reshape(1, -1), n_neighbors=k+1, return_distance=False)
        # Exclude the point itself
        indices = indices[0][1:]
        # Center the points
        M_i = X[indices] - np.mean(X[indices], axis=0)
        # Perform SVD
        U_i, Sigma_i, _ = svd(M_i, full_matrices=False)
        
        # Create diagonal matrix for Sigma_i
        Sigma_i_matrix = np.diag(Sigma_i)
        
        # Initialize sum of angles for the current point
        sum_angles = 0
        
        # Iterate over each neighbor
        for index in indices:
            # Center the neighbor's points
            neighbor_indices = nbrs.kneighbors(X[index].reshape(1, -1), n_neighbors=k+1, return_distance=False)
            neighbor_indices = neighbor_indices[0][1:]
            M_j = X[neighbor_indices] - np.mean(X[neighbor_indices], axis=0)
            # Perform SVD
            U_j, Sigma_j, _ = svd(M_j, full_matrices=False)
            
            # Create diagonal matrix for Sigma_j
            Sigma_j_matrix = np.diag(Sigma_j)
            
            # Compute Q using the provided formula
            #Q = np.dot(np.dot(U_i, Sigma_i_matrix).T, np.dot(U_j, Sigma_j_matrix))
            Q = np.dot(U_i.T, U_j)
            
            # Perform SVD of Q
            _, Sigma_Q, _ = svd(Q)
            
            # Compute the angle between local subspaces
            angle = np.arccos(np.clip(np.sum(Sigma_Q) / (np.linalg.norm(Sigma_i) * np.linalg.norm(Sigma_j)), -1.0, 1.0))
            #angle = np.arccos(np.clip(np.sum(Sigma_Q) / np.sum(np.dot(Sigma_i_matrix.T, Sigma_j_matrix)), -1.0, 1.0))
            sum_angles += angle
        
        # Calculate the average curvature for the current point
        local_curvatures[i] = sum_angles / k

    # Calculate the overall curvature of the manifold
    overall_curvature = np.mean(local_curvatures)
    
    return overall_curvature, local_curvatures


# Example usage
# Suppose DATA is your data matrix, with the number of rows indicating the number of samples and the number of columns indicating the sample dimensions.
DATA = np.load(r"...\MLP feature\resnet50_cifar10\Linear_output_10\0.npy")
print(data.shape)
curvatures, _ = estimate_manifold_curvature(data, k=20)
print(curvatures)
```

![图片1](https://github.com/mayanbiao1234/Geometric-metrics-for-perceptual-manifolds/assets/31196857/d5f4a764-1ac6-4e30-bf21-294319bcc7c5)


## 2. Volume metrics for perceptual manifolds in deep neural networks (ICLR 2023)
The volume of the perceptual manifold measures the richness of the distribution. See the paper, [Delving into Semantic Scale Imbalance](https://openreview.net/pdf?id=07tc5kKRIo), for how to use multiscale volumetric metrics for perceptual manifolds, and for more conclusions.

The citation format is: 

```
@inproceedings{
ma2023delving,
title={Delving into Semantic Scale Imbalance},
author={Yanbiao Ma and Licheng Jiao and Fang Liu and Yuxin Li and Shuyuan Yang and Xu Liu},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=07tc5kKRIo}
}
```

### The following is the code to calculate the volume of the perceptual manifold.

```
import numpy as np

def calculate_volume(Z, d, Z_mean):
    # Calculate (Z - Z_mean)
    diff = Z - Z_mean

    # Calculate (Z - Z_mean)(Z - Z_mean)^T
    outer_product = np.dot(diff.T, diff)

    # Calculate \frac{d}{m}(Z - Z_mean)(Z - Z_mean)^T
    scaled_outer_product = (d / Z.shape[0]) * outer_product

    # Calculate I + \frac{d}{m}(Z - Z_mean)(Z - Z_mean)^T
    matrix_sum = np.eye(Z.shape[1]) + scaled_outer_product

    # Calculate \frac{1}{2} \log_2(I + \frac{d}{m}(Z - Z_mean)(Z - Z_mean)^T)
    volume = 0.5 * np.log2(np.linalg.det(matrix_sum))

    return volume

# Example usage
# Assume Z is a matrix of size 5000x10
Z = np.random.rand(5000, 10)
# Assume d is a hyperparameter
d = 1.0
# Calculate the mean Z_mean of Z
Z_mean = np.mean(Z, axis=0)

# Calculate the volume of the perceptual manifold
volume = calculate_volume(Z, d, Z_mean)
print("Perceptual manifold volume:", volume)
```


## 3. Intrinsic Dimensions for perceptual manifolds in deep neural networks (Submitted to TPAMI)
The intrinsic dimensionality of perceptual manifolds can predict the fairness of models. Specifically, the larger the intrinsic dimensionality of the perceptual manifold corresponding to a class, the poorer the model performs on that class. Below, we provide two estimation methods for intrinsic dimensionality.

![image](https://github.com/mayanbiao1234/Geometric-metrics-for-perceptual-manifolds/assets/31196857/5d5d5e71-db81-40ea-899a-42215464c391)


See the paper, [Unveiling and Mitigating Generalized Biases of DNNs through the Intrinsic Dimensions of Perceptual Manifolds](https://arxiv.org/abs/2404.13859)

The citation format is: 

```
@misc{ma2024unveiling,
      title={Unveiling and Mitigating Generalized Biases of DNNs through the Intrinsic Dimensions of Perceptual Manifolds}, 
      author={Yanbiao Ma and Licheng Jiao and Fang Liu and Lingling Li and Wenping Ma and Shuyuan Yang and Xu Liu and Puhua Chen},
      year={2024},
      eprint={2404.13859},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

### (1) Estimation of intrinsic dimensions using TLE

```
from skdim.id import TLE
dim_estimator = TLE()
intrinsic_dim = dim_estimator.fit(augmented_data).dimension_
print("Intrinsic Dimensions:", intrinsic_dim)
```
### (2) Covariance estimation methods for intrinsic dimensions
```
import numpy as np
# 5000 denotes the sample number and 10 denotes the dimension.
data = np.ones(5000,10)
intrinsic_dim = (np.trace(np.dot(data.T, data)))**2/np.trace(np.dot(data.T, data)**2)
print("Intrinsic Dimensions:", intrinsic_dim)
```

## 4. The geometric shape of the perceptual manifold in deep neural networks (IJCV 2024)

We found that if two categories are highly similar, the geometric shapes of their corresponding embedding distributions are also highly similar. This discovery demonstrates for the first time that the geometric shape of the perceptual manifold can also serve as prior knowledge to help rare categories recover their true distribution. For specific details, please refer to the paper: [Geometric Prior Guided Feature Representation Learning for Long-Tailed Classification](https://link.springer.com/article/10.1007/s11263-024-01983-2)

![figure3](https://github.com/mayanbiao1234/Geometric-metrics-for-perceptual-manifolds/assets/31196857/b195d4aa-48db-42ba-80f3-c5a76f7a20db)

The citation format is: 

```
@article{ma2024geometric,
  title={Geometric Prior Guided Feature Representation Learning for Long-Tailed Classification},
  author={Ma, Yanbiao and Jiao, Licheng and Liu, Fang and Yang, Shuyuan and Liu, Xu and Chen, Puhua},
  journal={International Journal of Computer Vision},
  pages={1--18},
  year={2024},
  publisher={Springer}
}
```

### The following code calculates the similarity of the geometric shapes between perceptual manifolds.
```
import matplotlib.pyplot as plt
import numpy as np

#Assume the number of samples is 5000, and each sample is a (1, 10) vector.
data_matrix1 = np.random.rand(5000, 10)
data_matrix2 = np.random.rand(5000, 10)

# Calculate the covariance matrix
covariance_matrix1 = np.cov(data_matrix1, rowvar=False)

# Perform eigenvalue decomposition on the covariance matrix
eigenvalues1, eigenvectors1 = np.linalg.eigh(covariance_matrix1)

# Sort the eigenvalues
sorted_indices = np.argsort(eigenvalues1)[::-1]
eigenvalues1 = eigenvalues1[sorted_indices]
eigenvectors1 = eigenvectors1[:, sorted_indices]


# Calculate the covariance matrix
covariance_matrix2 = np.cov(data_matrix2, rowvar=False)

# Perform eigenvalue decomposition on the covariance matrix
eigenvalues2, eigenvectors2 = np.linalg.eigh(covariance_matrix2)

# Sort the eigenvalues
sorted_indices = np.argsort(eigenvalues2)[::-1]
eigenvalues2 = eigenvalues2[sorted_indices]
eigenvectors2 = eigenvectors2[:, sorted_indices]

similarity = 0
for i in range(len(eigenvectors2)):
    similarity += np.abs(np.dot(eigenvectors1[:,i].T,eigenvectors2[:,i]))

print("Similarity of the geometric shapes of the two perceptual manifolds:", similarity)
```



