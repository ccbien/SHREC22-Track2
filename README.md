# SHREC 2022 - Track 2: Fitting and recognition of simple geometric primitives on point clouds

```HONV.ipynb```: Evaluates kNN models that use Histogram of Oriented Normal Vectors (HONV) feature

```SP.ipynb```: Evaluates kNN models that use Surflet-Pair feature

```feature_extractor.py```: Contains functions that extract the 2 features above

```fit.ipynb```: Fits parameters for planes, cylinders, spheres, toruses

```fit_cone.ipynb```: Fits parameters for cones

```infer_runA.ipynb``` and ```infer_runB.ipynb```: Complete pipeline for inference on the test set combining kNN models and PCA method.

