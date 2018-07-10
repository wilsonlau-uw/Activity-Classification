# Activity-Classification
This project aims to automatically classify activities in free-text structure to a list of pre-defined categories using natural language processing techniques.   The collected data consist of roughly 76000 records written in 3 languages (English, French, Spanish) and 3 diseases (Malaria, TB, HIV).      We employ an active learning approach where an initial 200 records are manually curated to create a training dataset.  We then train a classifier using statistical NLP to predict the categories of the test data.   The predicated results will then be sampled for validation to help us understand the sensitivity and specificity of the model.  The validated results will then be added to the original training data to enhance the prediction.    The iterations will go on until an optimal predictive performance is reached.  

To select the best model, we use 5-fold cross validation on various machine learning algorithms.    The best model is based on linear SVM with data balancing.   Given that the activity text is relatively short, we decide to reinforce the prediction model by using the unlabeled data for self-training.    In addition, by first translating the text data into English, we are able to build a single model that can predict three languages.  Our experimentation has shown that this single model has better performance than each language specific model.   However, since categories are strongly specific to the disease type, we decide to build a separate model for each disease.  

The code is implemented in Python 3.  Libraries include pandas, argparse, csv, sklearn, imblearn, config, numpy, pickle, etc…. 
Data files are not included in the project since they contain sensitive information.  

## Results (up to 3 iterations now):  

./iterations/iteration1/iteration1.csv
sensitivity - 0.842696629213
specificity - 0.999358797618

./iterations/iteration2/iteration2.csv
sensitivity - 0.859756097561
specificity - 0.999662406434

./iterations/iteration3/iteration3.csv
sensitivity - 0.8875
specificity - 0.999735665945

## STEPS

### == Training and Prediction ==
Each iteration we will train a new model using the previous validated data and predicted results. 
Replace iteration<num> with the folder name of last iteration.

`python Main.py --semisupervised  'iterations/iteration<num>/test_output.csv'`

A test_output.csv will be created which has the new predicted categories.

### == Generate sample data for a new iteration ==
Using the test_output.csv we will create a new sample for validation

`python NextIteration.py –iteration <num>`

A new file iteration<num>.csv will be created 

### == Validations ==
This process requires a person manually going through the records and entering the corrected categories(module and intervention) in the last 2 columns.
Once done, create a new subfolder  “iteration<num>”  (e.g. iteration4) under the “iterations” folder.   Move the validated csv file into the new folder.  

### == Scoring ==
Compute the latest score (sensitivity/specificity).   

`python iterationScores.py`

You should see something like this 

./iterations/iteration1/iteration1.csv
sensitivity - 0.842696629213
specificity - 0.999358797618

./iterations/iteration2/iteration2.csv
sensitivity - 0.859756097561
specificity - 0.999662406434

./iterations/iteration3/iteration3.csv
sensitivity - 0.8875
specificity - 0.999735665945

…
