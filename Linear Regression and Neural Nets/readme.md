## Feature Preprocessing
1. Based on Diabetes_clean.csv, make all columns to be numeric. Besides that, some columns such as 'gender' and 'race' are desigend to be one-hot coding. For some other columns such as 'age', I used a mapping. For example, [0-10) maps to 1 and [10-20) maps to 2. The details could be referred from mapping.txt

2. I remove glimepiride.pioglitazone because all its values are same. Besides that, I removed encounter_id and patient_nbr.

3. After processes above, the data is stored in hospital_ready.csv. There are 74 columns and the last one is the label. Besides that, I also shuffled the rows and created hospital_ready_shuffle.csv. I used the shuffled version for later experiments. The benefit of doing shuffle now is that I don't need to shuffle for each experiment. If I shuffle in each experiment, their training data and test data may be different. I hope they could use the same train and test data for a more fair comparison. 

## Training
### Linear regression (Logistic Regression)
* Experiment
	1. I used MinMax Normalization for the data. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html. In this way, all data are in the range of (0,1). Otherwise, some columns with bigger values will be assumed to be more important which is not appropriate. 
	2. 1:60000 as training set, 60001:71518 as test set
	3. The model is multiclass logistic regression. It's implemented in sklearn https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression. Since there are three classes (No readmitted, readmitted < 30, readmitted > 30), the problem is considered to be 3 binary classifier. For each classifier, there are 73 coefficients. 

* Result
	1. Classification Accuracy on test data: 0.6092203507553394
	
	2. Look into coefficients. For each classifier, I choosed 5 most important factors that have highest abstract coefficients. Red means negative coefficients and blue means positive coefficients. 
	
	   
	
	   1. class: no readmission 
	
	      important factors & coefficients:
	
	    number_emergency :  -6.17924362697192
	    number_inpatient :  -5.287297946139257
	    number_outpatient :  -3.109589625795336
	    number_diagnoses :  -1.8200520816600017
	    miglitol :  -0.7928093545529517
	
	![](LR_class1.jpg)
	
	​		2. class:  readmitted < 30
	
	​			important factors & coefficients:
	
	```
	number_inpatient :  3.8989570904297075
	number_emergency :  1.9616037290918948
	chlorpropamide :  -0.896685456127739
	diabetesMed_No :  -0.8648345034283673
	number_diagnoses :  0.8238872938405993
	```

![](LR_class2.jpg)

​			3. class: readmitted > 30

​				important factors & coefficients:

```
number_emergency :  4.285109442436811
number_outpatient :  2.8359583259919097
number_inpatient :  2.5582402067721772
miglitol :  1.606332132975167
number_diagnoses :  1.2851296546923137
```

![](LR_class3.jpg)





### Neural Network

Experiment 

1. I used MinMax Normalization for the data. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html. In this way, all data are in the range of (0,1). Otherwise, some columns with bigger values will be assumed to be more important which is not appropriate. 
2. 1:60000 as training set, 60001:71518 as test set (The first and second steps are the same as in linear regression so that their results could be compared.)
3. The model is neural networks. I used pytorch to implement it. The size of input layer is 73 and the output layer has 3 outputs because we have three classes to predict.  I tried two structures. The first one has one hidden layer whose size is 300. So the layers for the network is (73, 300, 3). The second one has another hidden layer. The structure is (73, 300, 300, 3).  
4. The batch size is 256. I used a big batch size for stable training. The learning rate is 0.0001. A small learning rate helps us to track the process of learning. It's a simple task, if the learning rate is too high, after one epoch of learning, the accuracy will already around 0.7. 



Result

1. Accuracy : Both of them achieved a test accuracy of 0.6014933148115993. It's interesting that their accuracy are exactly the same. This could be the maximum performance neural networks can achieve. 

2. Loss during training

   ![](loss.jpg)

3. Test accuracy during trianing

   ![](acc.jpg)

4. Comparison with linear regression

   Surprisingly, linear regression has a little better performance. The reason could be that, first of all, the task is relatively simple and the number of data is also not big. Neural networks could handle complicated task with more data. As a result, its advantage could not fully shown in this task. Besides that, in linear regression, it's a classifier composed of 3 binary classifier. This is like ensemble? That's probably the reason it works better than neural networks. 