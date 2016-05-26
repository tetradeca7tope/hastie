Boosted Binary Regression Trees v2.0

-----------------
Version history
-----------------
v2.0 08/05/2013 Speed up training by using mex for finding the best splitting function. OpenMP is supported in the mex file.
v1.0 06/07/2013

-----------------
Author
-----------------
Kota Hara
Center for Automation Research (CfAR), 4436 A.V.Williams Bldg, University of Maryland, College Park, MD 20742 USA
E-mail: kotahara@umd.edu
www: http://www.kotahara.com/

-----------------
Disclaimer and copyright notice
-----------------
(C) Kota Hara, 2013
You are free to use, modify, or redistribute this code in any way you want for non-commercial purposes provided that this copyright notice appear on all copies and supporting documentation.
The programs are provided on an 'as is' basis without any express or implied warranty of any kind including warranties of merchantability, noninfringement of intellectual property,
or fitness for any particular purpose. In no event shall the authors be liable for any damages whatsoever (including, without limitation, damages for loss of profits, business 
interruption, loss of information) arising out of the use of or inability to use these programs, even if the author has been advised of the possibility of such damages.
If you have any bugs, questions, suggestions, or modifications, please contact the author.

-----------------
Citation
-----------------
If you use this code in your work, we recommend you to cite [1] which first proposed the method.
It would appreciated if you could also cite our paper [2] where we use boosted binary regression trees for image based human pose estimation.

-----------------
Introduction
-----------------
Boosted Binary Regression Trees (BBRT) is a powerful regression method proposed in [1]. BBRT combines binary regression trees [3] using a gradient boosting technique.
There are several variants proposed in [1]. In [1], it is assumed that the target is scalar value. However, it is trivial to extend the method to vector target case by proper modifications.
This code is based on "LS_Boost" described in [1] but it can also handle vector target case. In other words, you do not need to train an independent regressor for each target dimension, unlike Support Vector Regression.
The detail of the algorithm this code implements can be found in [2].

-----------------
Usage
-----------------
First you need to compile C++ code to get mex file. To compile with OpenMP support, do the following:
mex findBestSplit.cpp COMPFLAGS="/openmp $COMPFLAGS"

For training, use 
brtModel = brtTrain( X, T, leafNum, treeNum, nu )
X: N by D input data matrix where N is the number of training data and D is the dimensionality of the input space.
T: N by K target data matrix where K is the dimensionality of the target space.
leafNum: The number of leaf nodes for each binary regression tree. The tree is grown until the number of leaf nodes becomes this value.
treeNum: The number of binary regression trees that will be added to the model. We recommend 1000 or less.
nu: Shrinkage parameter. We recommend 0.1 or less.
brtModel: BBRT model.

For testing, use
output = brtTest( input, brtModel, varargin )
input: 1 by D input vector
brtModel: A model returned by brtTrain.
varargin: If specified, the number of trees to be used. Otherwise, all the trees in the model are used for prediction.
output: 1 by K predicted target vector

-----------------
References
-----------------
[1] J. H. Friedman. Greedy Function Approximation: a Gradient Boosting Machine. Annals of Statistics, 2001.
[2] Kota Hara and Rama Chellappa, Computationally Efficient Regression on a Dependency Graph for Human Pose Estimation, CVPR 2013. 
[3] Breiman, Leo; Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984). Classification and regression trees. Monterey, CA: Wadsworth & Brooks/Cole Advanced Books & Software. ISBN 978-0-412-04841-8.

