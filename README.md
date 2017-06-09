# Recommender-Systems--Collaborative-filtering-with-Spark-s-ALS-implementation


-Movie recommender system using collaborative filtering with Spark's Alternating Least Squares implementation. 
It is organized in two parts:

--The first module is about getting and parsing movies and ratings data into Spark RDDs. We use the Products of Factors technique for the system and optimize the loss function with ALS.
To further evaluate the parameters thus found as the best parameters with the lowest error, we perform 5-fold Cross Validation and compute the metrics RMSE, MSE and MAP

--The second module is about using the recommender system. Add one user to the database, by adding recommendations to a few 20 movies. Then use the recommendation system we have built to output the predicted ratings for 5 movies the user did not evaluate.


--New user added and Utilizing our recommendation system to rank the unrated movies


--optimized the loss function using ALS and got the best parameters for building our recommender system.;80% accurate in recommending the ratings or rankings for the unrated movies by the users.
