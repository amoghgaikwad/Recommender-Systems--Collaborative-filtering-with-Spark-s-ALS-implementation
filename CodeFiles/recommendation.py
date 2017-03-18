from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.recommendation import ALS, Rating
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import math

conf = SparkConf().setAppName("Assignment 4")
sc = SparkContext(conf=conf)
sqlCtx = SQLContext(sc)

# Full Ratings data
r_data = sc.textFile("ratings.csv")
r_data_header = r_data.take(1)[0]

r_data_rdd = r_data.filter(lambda line: line!=r_data_header)\
    .map(lambda line: line.split(",")).map(lambda r: (r[0],r[1],r[2])).cache()

# Full movie data
m_data = sc.textFile("movies.csv")
m_data_header = m_data.take(1)[0]

m_data_rdd = m_data.filter(lambda line: line!=m_data_header)\
    .map(lambda line: line.split(",")).map(lambda r: (r[0],r[1])).cache()

# Split the data into Training, Test and Validation and Tune the loss function using ALS to get the best model
training_rdd, validation_rdd, test_rdd = r_data_rdd.randomSplit([6, 2, 2], seed=0L)
vpredict_rdd = validation_rdd.map(lambda x: (x[0], x[1]))
Final_test_rdd = test_rdd.map(lambda x: (x[0], x[1]))


# Parameters for ALS tuning (Loss function) - to get best model
seed = 5L
iterations = 10
r_param = 0.1
ranks = [5, 8, 12]
l_rmse = [0, 0, 0]
irr = 0
emin = float('inf')
rank_chosen = -1
best_iteration = -1

for rank in ranks:
    model = ALS.train(training_rdd, rank, seed=seed, iterations=iterations, lambda_=r_param)
    predictions = model.predictAll(vpredict_rdd).map(lambda r: ((r[0], r[1]), r[2]))
    rp = validation_rdd.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    MSE = (rp.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    RMSE = math.sqrt(MSE)
    l_rmse[irr] = RMSE
    irr += 1
    print 'For rank %s' % (rank)
    print 'MSE is %s' % (MSE)
    print 'RMSE is %s' % (RMSE)
    if RMSE < emin:
        emin = RMSE
        rank_chosen = rank

print 'The best model was trained with rank %s' % rank_chosen

#Testing the TEST data on the best model
model = ALS.train(training_rdd, rank_chosen, seed=seed, iterations=iterations,
                      lambda_=r_param)
predictions = model.predictAll(Final_test_rdd).map(lambda r: ((r[0], r[1]), r[2]))
rp = test_rdd.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
MSE = (rp.map(lambda r: (r[1][0] - r[1][1])**2).mean())
RMSE = math.sqrt(MSE)

print 'For testing data'
print 'MSE : %s' % (MSE)
print 'RMSE is %s' % (RMSE)

#Adding user and predicting rating for 5 movies
user_ID = 0

# The format of each line is (userID, movieID, rating)
additional_user = [
     (user_ID,  666, 4.0), 
     (user_ID, 1024, 3.0), 
     (user_ID, 1606, 4.0), 
     (user_ID, 1485, 5.0), 
     (user_ID,   32, 5.0), 
     (user_ID,  335, 4.0), 
     (user_ID,  379, 3.0), 
     (user_ID,  296, 4.0), 
     (user_ID,  858, 5.0), 
     (user_ID,   50, 4.5),
     (user_ID,  318, 5.0),
     (user_ID,  858, 5.0),
     (user_ID,   50, 5.0),  
     (user_ID,  527, 4.5),    
     (user_ID, 1221, 5.0),      
     (user_ID,  912, 5.0), 
     (user_ID, 1193, 4.0)
    ]

additional_user_rdd = sc.parallelize(additional_user)
N_ratings = r_data_rdd.union(additional_user_rdd)
Nr_model = ALS.train(N_ratings, rank_chosen, seed=seed, 
                              iterations=iterations, lambda_=r_param)
n_movie_id = map(lambda x: x[1], additional_user) # get just movie IDs
movie_unrated = (m_data_rdd.filter(lambda x: x[0] not in n_movie_id).map(lambda x: (user_ID, x[0])))
# Use the input rdd, movie_unrated, with Nr_model.predictAll() to predict new ratings for the movies
user_recommendation_result = Nr_model.predictAll(movie_unrated)

print user_recommendation_result.take(5)