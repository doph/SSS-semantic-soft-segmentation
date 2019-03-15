import numpy as np
from imguidedfilter import imguidedfilter
from scipy.io import loadmat
from scipy import misc
import cv2

def feature_PCA(features, dim):
    features = features.astype(float)
    h, w, d = features.shape
    features = features.reshape((-1, d))
    #print(features.shape)
    featmean = np.mean(features, axis=0).reshape((1,-1)) #not 0
    #featmean = np.mean(features, axis=0) #not 0
    #print(featmean.shape)
    #temp = np.matmul(np.ones((h * w)), featmean)
    features = features - np.dot(np.ones((h * w)).reshape((h * w,1)), featmean)#(dim 0) !=  (dim 0)
    #features = features - np.matmul(np.ones((h * w)), featmean) #(dim 1) != 1 (dim 0)
    covar = np.matmul(features.T, features)
    eigen_values, eigen_vectors = np.linalg.eig(covar)
    idx = eigen_values.argsort()
    eigen_vectors = eigen_vectors[:, idx[:-dim-1:-1]]
    #eigen_vectors = eigen_vectors[:, idx][:dim]
    print(features.shape)
    print(eigen_vectors.shape)
    pcafeat = np.matmul(features, eigen_vectors)
    pcafeat = pcafeat.reshape((h, w, dim))

    return pcafeat

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def preprocess_features(features, img=None):
    features[features < -5] = -5
    features[features > 5] = 5

    if img is not None:
        fd = features.shape[2]
        #fd = features.shape[3]
        maxfd = fd - fd % 3
        for i in range(0, maxfd, 3):
            #  features(:, :, i : i+2) = imguidedfilter(features(:, :, i : i+2), image, 'NeighborhoodSize', 10);
            features[:, :, i : i + 3] = imguidedfilter(features[:, :, i : i + 3], img, (10, 10), 0.01)
        for i in range(maxfd, fd):
            # features(:, :, i) = imguidedfilter(features(:, :, i), image, 'NeighborhoodSize', 10);
            features[:, :, i] = imguidedfilter(features[:, :, i], img, (10, 10), 0.01)

    simp = feature_PCA(features, 3)
    for i in range(0, 3):
        # simp(:,:,i) = simp(:,:,i) - min(min(simp(:,:,i)));
        simp[:, :, i] = simp[:, :, i] - simp[:, :, i].min()
        # simp(:,:,i) = simp(:,:,i) / max(max(simp(:,:,i)));
        simp[:, :, i] = simp[:, :, i] / simp[:, :, i].max()

    return simp

ori=loadmat('/Users/yangxuemeng/mygithub/SemanticSoftSegmentation/af.mat')
feat = ori['embedmap']
img = misc.imread('../af.png',mode='RGB')
simp = preprocess_features(feat,img)
misc.imsave('./res-9.jpg',simp)
