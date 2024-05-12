#Mehmet Abdullah Şeşen 150220029 BLG202E Project, 2024 Spring

from visualize import plot_point_sets
import numpy as np
import sys

#function that performs the power method to find the largest eigenvalue and its corresponding eigenvector
def power_method(A, num_iterations=1000, tol=1e-6):
    n = A.shape[0]
    
    # Initialize a random vector
    x = np.random.rand(n)
    
    # Power iteration
    for _ in range(num_iterations):
        x_new = np.dot(A, x)
        eigenvalue = np.linalg.norm(x_new)
        x_new = x_new / eigenvalue
        if np.linalg.norm(x - x_new) < tol:
            break
        x = x_new
    
    # Calculate eigenvector corresponding to the dominant eigenvalue
    eigenvector = x_new
    
    return eigenvalue, eigenvector

#function that calculates the eigenvalues using the powermethod function
def eigen(A):
    n = A.shape[0]
    
    eigenvalues = []
    eigenvectors = []
    
    for _ in range(n):
        eigenvalue, eigenvector = power_method(A)
        eigenvalues.append(eigenvalue)
        eigenvectors.append(eigenvector)
        
        # Deflate matrix
        A = A - eigenvalue * np.outer(eigenvector, eigenvector)
    
    return eigenvalues, eigenvectors

#Singular Value Decomposition function, M = V S Wt
def my_svd(M):
    
    # Compute M^T * M
    MtM = np.dot(M.T, M)
    
    # Eigen decomposition of M^T * M
    eigenvalues, eigenvectors = eigen(MtM)
    
    
    
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

#main function that performs the kabshc umayema alogrithm based on the pseudo code from the essay of Lawrance et. al
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



if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python main.py <mat1.txt> <mat2.txt> <correspondences.txt>")
        sys.exit(1)

    mat1_filename = sys.argv[1]
    mat2_filename = sys.argv[2]
    correspondences_filename = sys.argv[3]

    mat1 = np.loadtxt(mat1_filename)
    mat2 = np.loadtxt(mat2_filename)
    correspondences = np.loadtxt(correspondences_filename)

    if mat1.shape[0] != correspondences.shape[0] or mat2.shape[0] != correspondences.shape[0]:
        print("Number of correspondences doesn't match the number of points.")
        sys.exit(1)

    Q = mat1[correspondences[:, 0]]
    P = mat2[correspondences[:, 1]]

    print("Q matrix (from mat1.txt using correspondences):")
    print(Q)
    print("P matrix (from mat2.txt using correspondences):")
    print(P)

