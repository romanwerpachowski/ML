#pragma once

/** @mainpage MLpp

C++ library with efficient implementations of selected machine learning algorithms.

(C) 2020 <a href="mailto:roman.werpachowski@gmail.com">Roman Werpachowski</a>.

Project webpage: https://romanwerpachowski.github.io/ML/

@section provalg Provided algorithms

@subsection Clustering

We provide the Gaussian E-M algorithm. Initial means can be initialised in three different ways:
- <a href="https://en.wikipedia.org/wiki/K-means_clustering#Initialization_methods">Forgy and Random partition</a>
- <a href="https://en.wikipedia.org/wiki/K-means%2B%2B">K++</a>

Implemented in ml::Clustering namespace and ml::EM class.

@subsection decision_trees Decision trees

Supported decision trees:
- multinomial classification
- multivariate regression with a scalar dependent variable

With and without cost-complexity pruning.

Implemented in ml::DecisionTrees namespace.

@subsection linreg Linear regression

Only Ordinary Least Squares for now:
- univariate with and without intercept
- multivariate
- <a href="https://cpb-us-w2.wpmucdn.com/sites.gatech.edu/dist/2/436/files/2017/07/22-notes-6250-f16.pdf">recursive multivariate</a>
- ridge regression

Implemented in ml::LinearRegression namespace.

@subsection Cross-validation

Methods:
- <a href="https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation">k-fold</a>
- <a href="https://en.wikipedia.org/wiki/Cross-validation_(statistics)#Leave-one-out_cross-validation">leave-one-out</a>

Implemented in ml::Crossvalidation namespace.

@subsection Statistics

A set of standard statistical functions used by other modules. Implemented in ml::Statistics namespace.

*/