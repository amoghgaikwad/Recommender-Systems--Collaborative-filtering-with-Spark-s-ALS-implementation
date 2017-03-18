# Cross Validation - 5 Folds

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.recommendation import ALS, Rating
from pyspark.mllib.Evalution import RankingMetrics
import math

conf = SparkConf().setAppName("Assignment 4")
sc = SparkContext(conf=conf)
sqlCtx = SQLContext(sc)

# Defining lists (For MAP Calculation)
predictedRatingList = []
groundTruthList = []

#Full ratings data
f_data = sc.textFile("ratings.csv")
f_data_header = f_data.take(1)[0]

f_data_rdd = f_data.filter(lambda line: line!=f_data_header)\
    .map(lambda line: line.split(",")).map(lambda r: (r[0],r[1],r[2])).cache()

# List of all the training files - which are got after executing the shell script from movie lens dataset
list_train = []
for i in range(1,5):
    list_train.append("r"+str(i)+".train")

# List of all the test files - which are got after executing the shell script from movie lens dataset
list_test = []
for i in range(1,5):
    list_test.append("r"+str(i)+".test")


# Parameters for ALS Tuning
t=0
sum_mse =0
sum_rmse =0
sum_map =0
iterations = 10
reg_param = 0.1
rank_chosen = 12 # this rank is chosen based on output from recommendation.py (The best model rank)

def map_calc(x):
    for t in x:
        if(t[1]>=3):
            groundTruthList.append(t[1])
        if(t[2]>=3):
            predictedRatingList.append(t[2])
    predictedRatingList = sorted(predictedRatingList)
    groundTruthList = sorted(groundTruthList)
    pair = (predictedRatingList, groundTruthList)
    return pair
# Start Cross Validation
for file in list_train:
	#get the training data
    tr_data = sc.textFile(file)
    tr_data_header = tr_data.take(1)[0]
    tr_data_rdd = tr_data.filter(lambda line: line!=tr_data_header)\
    .map(lambda line: line.split(",")).map(lambda r: (r[0],r[1],r[2])) 
    training_data = f_data_rdd.subtract(tr_data_rdd) # this is the training data got from each fold

    # model the training data 
    model = ALS.train(training_data, rank_chosen, seed=5L, iterations=iterations, lambda_=reg_param)

    #get the test data
    t_data = sc.textFile(list_test[t])
    t_data_header = t_data.take(1)[0]
    test_data_rdd = t_data.filter(lambda line: line!=t_data_header)\
    .map(lambda line: line.split(",")).map(lambda r: (r[0],r[1],r[2]))
    test_data = test_data_rdd.map(lambda x: (x[0], x[1]))
    t=t+1

    # make predictions on the test data, with the model trained with the training data above.
    predictions = model.predictAll(test_data).map(lambda r: ((r[0], r[1]), r[2]))
    #predictions= predictions.filter(lambda x: x[1] != float('nan'))
    rates_and_preds = test_data_rdd.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    #rates_and_preds is of the form =>[(userID, movieID),(actualRating, predictedRating)]

    # steps followed to get the an RDD of (predicted ranking, ground truth set) pairs. Which is need for MAP calculation
    step1 = rates_and_preds.map(lambda (x,v): (x[0],(x[1],x[2],x[3]))).groupByKey().mapValues(list)
    step2 = step1.map(lambda x: x[1]).map(lambda x: map_calc(x))
    MAP_eval = RankingMetrics(step2)
    MAP = MAP_eval.meanAveragePrecision
    MSE = (rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    RMSE = math.sqrt(MSE)
    # Sum up all the RMSE values to find the average later
    sum_mse = sum_mse + MSE
    sum_rmse = sum_rmse + RMSE
    sum_map = sum_map + MAP
# Find the average RMSE value after cross validation
avg_mse = sum_mse/5
avg_rmse = sum_rmse/5
avg_map = sum_map/5
print ("The average MSE value after Cross Validation is : " avg_mse)
print ("The average RMSE value after Cross Validation is : " avg_rmse)
print ("The average MAP value after Cross Validation is : " avg_map)