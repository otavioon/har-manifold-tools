from functools import partial

import numpy as np
#import umap
from sklearn import manifold
from sklearn.decomposition import PCA

def get_model(model_name: str, **kwargs):
    LLE = partial(manifold.LocallyLinearEmbedding)
    model_name = model_name.lower()
    if model_name == 'pca':
        return PCA(**kwargs)
    elif model_name == "lle":
        return LLE(method='standard', **kwargs)
    elif model_name =="ltsa":
        return LLE(method='ltsa', **kwargs) 
    elif model_name == "hlle":
        return LLE(method='hessian', **kwargs)
    elif model_name == "mlle":
        return LLE(method='modified', **kwargs)
    elif model_name== "isomap":
        return manifold.Isomap(**kwargs)
    elif model_name == "mds":
        return manifold.MDS(**kwargs)
    elif model_name == "se":
        return manifold.SpectralEmbedding(**kwargs)
    elif model_name == "t-sne":
        return manifold.TSNE(**kwargs)
    elif model_name == "umap":
        return umap.UMAP(**kwargs)
    raise ValueError(f"Invalid method {model_name}")

def manifold_fit_transform(X: np.ndarray, model_name: str, **model_kwargs):
    model = get_model(model_name, **model_kwargs)
    return model.fit_transform(X)

class ManifoldFitTransform:
    def __init__(self, model_name: str, **model_kwargs) -> None:
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.model = get_model(model_name=model_name, **model_kwargs)

    def __call__(self, X):
        return self.model.fit_transform(X)