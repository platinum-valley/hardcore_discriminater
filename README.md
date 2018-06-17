# hardcore_discriminater

This system is discriminating between "Hardcore" music and "not Hardcore" music.

## dataset 
Dataset contain 545 music data (Hardcore: 254, not Hardcore: 291)

Not Hardcore set are "Rock", "HipHop", "Jazz", "Trance", "Rap", "Pops", "Trap", "Electro", "EDM", "classic", "drum'n'step", "Dubstep"

These music are transform MFCC(Mel-Frequency Cepstrum Coefficients) feature and compute statistics(mean, median, etc)

## model
This system has simple machine learning model.

・Logistic_regression(normal, PCA, norm)

・SVM(rbf)

・Dicision Tree

・Random Forest

・KNN

## experiment
In experiment, dataset segment 10-fold.
Evaluation measure is mean accuracy for each testset after each training.   
