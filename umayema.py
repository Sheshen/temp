#Mehmet Abdullah Şeşen 150220029 BLG202E Project, 2024 Spring

from visualize import plot_point_sets
import numpy as np

def my_svd(M):
    
    # Compute M^T * M
    MtM = np.dot(M.T, M)
    
    # Eigen decomposition of M^T * M
    eigenvalues, eigenvectors = np.linalg.eig(MtM)
    
    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Compute singular values
    singular_values = np.sqrt(eigenvalues)
    
    # Compute matrix W
    W = eigenvectors
    
    # Compute matrix S
    S = np.diag(singular_values)
    
    # Compute matrix V
    V = np.dot(M, np.dot(W, np.linalg.inv(S)))
    
    return V, S, W.T


def kabsch_umeyama(Q, P):
    # Q and P are sets of points to be aligned

    # Compute d × d matrix M = QP^T
    M = np.dot(Q, P.T)
    
    # Compute SVD of M, identify d × d matrices V , S, W, so that M = VSWT intheSVDsense.
    V, S, Wt = my_svd(M)
    
    # Initialize scaling factors so that: set s1 = s2 = s3 = ... = sd-1 = 1
    d = M.shape[0]
    scaling_factors = np.ones(d-1)
    
    #Ifdet(VW)>0,thensetsd =1,elsesetsd =−1,
    # Compute determinant of VW
    det_VW = np.linalg.det(np.dot(V, Wt.T))
    if det_VW > 0:
        scaling_factors = np.append(scaling_factors, 1)
    else:
        scaling_factors = np.append(scaling_factors, -1)
    
    #Set S ̃ = diag{s1,...,sd}.
    S_tilde = np.diag(scaling_factors)
    
    #Return d × d rotation matrix U = WS ̃V T .
    U = np.dot(Wt.T, np.dot(S_tilde, Wt))

    return U

