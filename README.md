# Geometric-metrics-for-perceptual-manifolds-in-deep-neural-networks
Natural datasets exhibit inherent patterns that can be generalized under the manifold distribution principle: the distribution of a class of data is close to a low-dimensional perceptual manifold. As illustrated, data classification can be viewed as the unraveling and separation of perceptual manifolds. The difficulty of classifying a manifold increases when it is entangled with other perceptible manifolds. Typically, a deep neural network consists of a feature extractor and a classifier. Feature learning can be considered as unfolding perceptual manifolds, where a well-learned feature extractor can often unfold multiple manifolds for the classifier to decode. From this perspective, all factors related to manifold complexity may impact the classification performance of the model. We will provide metrics for the curvature, volume, and separability of perceptual manifolds.

![fig24](https://github.com/mayanbiao1234/Geometric-metrics-for-perceptual-manifolds-in-deep-neural-networks/assets/31196857/5b5d4ee3-cab2-4078-a7eb-09d52648121b)

## Curvature metrics for perceptual manifolds in deep neural networks
The curvature metric of the perceptual manifold in deep neural networks allows analyzing the fairness of the model from a geometric point of view.
For related conclusions on the curvature and model preferences of perceptual manifolds please refer to the paper, [Curvature-Balanced Feature Manifold Learning for Long-Tailed Classification](https://arxiv.org/abs/2303.12307)
The citation format is: 

@inproceedings{ma2023curvature,

  title={Curvature-Balanced Feature Manifold Learning for Long-Tailed Classification},
  
  author={Ma, Yanbiao and Jiao, Licheng and Liu, Fang and Yang, Shuyuan and Liu, Xu and Li, Lingling},
  
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  
  pages={15824--15835},
  
  year={2023}
  
}

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
[fig2_1.pdf](https://github.com/mayanbiao1234/Geometric-metrics-for-perceptual-manifolds/files/14945910/fig2_1.pdf)

## Volume metrics for perceptual manifolds in deep neural networks
The volume of the perceptual manifold measures the richness of the distribution. See the paper, [Delving into Semantic Scale Imbalance](https://openreview.net/pdf?id=07tc5kKRIo), for how to use multiscale volumetric metrics for perceptual manifolds, and for more conclusions.

The citation format is: 

@inproceedings{

ma2023delving,

title={Delving into Semantic Scale Imbalance},

author={Yanbiao Ma and Licheng Jiao and Fang Liu and Yuxin Li and Shuyuan Yang and Xu Liu},

booktitle={The Eleventh International Conference on Learning Representations },

year={2023},

url={https://openreview.net/forum?id=07tc5kKRIo}

}

```
import numpy as np

def calculate_volume(Z, d, Z_mean):
    # Calculate (Z - Z_mean)
    diff = Z - Z_mean

    # Calculate (Z - Z_mean)(Z - Z_mean)^T
    outer_product = np.outer(diff, diff)

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
