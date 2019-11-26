import psycopg2
import os
import pandas as pd
from pyspark import SparkContext
from pyspark.sql import SparkSession,SQLContext
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import *

# Connection to Database Server
def get_connection():
	# Get DatabaseURL from environment variable
	dsn = os.environ['DATABASE_URL']
	return psycopg2.connect(dsn)

# open SQL Context
def get_sqlcontext():
	sc = SparkContext("local", "App Name")
	sql_context = SQLContext(sc)

	return sql_context

# Creating DataFrame with a review
def create_dataframe_reviews(sql,con):
	# Create DataFrame on pandas
	data = pd.read_sql(sql='SELECT * from "Reviews";',con=con)

	# Create DataFrame on Spark
	sdf = sql.createDataFrame(data)

	return sdf

#Creating DataFrame with a users
def create_dataframe_users(con):
	# Create DataFrame on pandas
	data = pd.read_sql(sql='select distinct(user_name_id) from "Reviews";',con=con)

	user_list = data.values.tolist()

	return user_list

# ALS処理
def learning_model(sql,schema,user_list):
	# # Build the recommendation model using ALS on the training data
	als = ALS(
		rank=20,
		maxIter=10,
		regParam=0.1,
		userCol='user_name_id',
		itemCol='contents_id',
		ratingCol='review',
		seed=0
	)
	model = als.fit(schema)


	for user in user_list:
		df1 = model.recommendForAllUsers(
			numItems=10
		).filter(
			'user_name_id = %d' % user[0]
		).select(
			'recommendations.contents_id',
			'recommendations.rating'
		).first()

		print(df1[0])
		print(df1[1])

		recommends = pd.DataFrame({
			'user' : user[0],
			'content' : df1[0],
			'rating' : df1[1]
		})

		print(recommends)
		print(type(recommends))

	#df1.show()








	#recommends = pd.DataFrame({
	#	'item':tmp[0],
	#	'rating' : tmp[1]
	#})


# Connection to DatabaseServer
conn = get_connection()
# Create Spark Session
sqlcontext = get_sqlcontext()
# Create DataFrame
schema = create_dataframe_reviews(sqlcontext,conn)
user_list = create_dataframe_users(conn)
learning_model(sqlcontext,schema,user_list)

