# ML From Scratch
## Table of contents
* [Introduction](#Introduction)
* [KNN](#KNN)
* [LinearRegression](#LinearRegression)
* [LogicRegression](#LogicRegression)
## Introduction
This repo is a collection of simple projects I tackled to familiarize myself with ML algorithms.

Most of these codes are insipred by [AssemblyAI videos](https://www.youtube.com/watch?v=p1hGz0w_OCo&list=PLcWfeUsAys2k_xub3mHks85sBHZvg24Jd&pp=iAQB).

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

## LinearRegression
We assume a linear relation between variables and we try to [find this line](https://en.wikipedia.org/wiki/Linear_regression).

$$y = wx+b$$

$$
\chi^2 = \sum_{i=1}^{n} \frac{(y_i - (wx_i + b))^2}{n}
$$

where $y_i$ are the observed values, $x_i$ are the independent variables, and $n$ is the number of data points.

The Jacobian matrix $J$ for the derivatives with respect to $w$ and $b$ gives us how to change the parameters:

$$
J = \begin{bmatrix}
\frac{\partial \chi^2}{\partial w} \\
\frac{\partial \chi^2}{\partial b}
\end{bmatrix} = 
\begin{bmatrix}
\sum \frac{-2x_i(y_i - (wx_i + b))}{n}\\
\sum \frac{-2(y_i - (wx_i + b))}{n}
\end{bmatrix} = 
-\frac{2}{n}\begin{bmatrix}
\sum x_i(\hat{y} - y_i)\\
\sum (\hat{y} - y_i)
\end{bmatrix}
$$

The whole thing is done in a matrices form

* LinearRegression.py
    * Initialize *weight* $w$ and *bias* $b$ to zero
    * Repeat q.b. :
        * Predict the result
        * Evaluate the error
        * Gradient descent

## LogicRegression
We want to decide, based on features, in which of two classes the tests are.

We take the linear regression and we convolute it with a **sigmoid**

The result is an output between 0 and 1 with a `fast' transition

Once we found the prediction for each test entry we just cut at 0.5

    class_pred      = [0 if y<=0.5 else 1 for y in y_pred]

