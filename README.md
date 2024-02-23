# ML From Scratch
## Table of contents
* [Introduction](#Introduction)
* [KNN](#KNN)

## Introduction
This repo is a collection of simple projects I tackled to familiarize myself with ML algorithms.

Most of these codes are insipred by [AssemblyAI videos](https://www.youtube.com/@AssemblyAI).

## KNN
This is a simple [K Nearest Neighbours algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm).
There are two scripts, both running on [sklearn iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html).

This implementation is quite basic, using the euclidean distance and no weights.

* **KNN.py**
    * Defines a `euclidean_distance(x1,x2)`
    * For each x evaluates distances from the training 
    * Sorts and assigns on Majority vote of the first `k`
* **KNN_fancyAnimation.py**
    * Same core functioning as KNN.py
    * Adds a silly animation of the decision process with `matplotlib.animation`
    ![Alt Text](./knn_classification_animation.gif)
