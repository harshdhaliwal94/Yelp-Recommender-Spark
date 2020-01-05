package ca.uwaterloo.cs451.project

import org.apache.log4j._
import org.apache.hadoop.fs._
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.Partitioner
import org.rogach.scallop._
import org.apache.spark.sql.SparkSession

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.ml.feature.IDF
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions
import org.apache.spark.sql.types

import org.apache.spark.sql.Row
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.ml.tuning.{TrainValidationSplit, ParamGridBuilder, CrossValidator}
import org.apache.spark.ml.evaluation.RegressionEvaluator

class CollabConf(args: Seq[String]) extends ScallopConf(args) {
    val numExecutors = opt[Int](descr = "number of executors", required = false, default = Some(1))
    val executorCores = opt[Int](descr = "number of cores", required = false, default = Some(1))
    verify()
}

object Collab {
    val log = Logger.getLogger(getClass().getName())
    def main(argv: Array[String]){
        val conf = new SparkConf().setAppName("Collab")
        val ss = SparkSession.builder.getOrCreate
        
        val businessDF = ss.read.parquet("Data/yelp_business.parquet")
        val userDF = ss.read.parquet("Data/yelp_users.parquet")
        val reviewDF = ss.read.parquet("Data/review_train.parquet")
        val rating_df = ss.read.parquet("Data/rating.parquet").repartition(8)

        val model_path = "ALS_Train/"
        val Array(train, test) = rating_df.randomSplit(Array[Double](0.8, 0.2), seed=123)
        val als = new ALS().setUserCol("userId").setItemCol("businessId").setRatingCol("rating")

        val param_grid = new ParamGridBuilder()
            .addGrid(als.rank,Array[Int](10, 15, 20))
            .addGrid(als.maxIter,Array[Int](10, 15, 20))
            .addGrid(als.coldStartStrategy,Array[String]("drop"))
            .build()

        var evaluator = new RegressionEvaluator().setMetricName("rmse").setLabelCol("rating")
        val cv = new CrossValidator()
            .setEstimator(als)
            .setEstimatorParamMaps(param_grid)
            .setEvaluator(evaluator)
            .setNumFolds(5)
            .setSeed(123)
        val cv_als_model = cv.fit(train)

        // Evaluate the model by computing the RMSE on the test data
        val als_predictions = cv_als_model.bestModel.transform(test)
        evaluator = new RegressionEvaluator().setMetricName("rmse").setLabelCol("rating").setPredictionCol("prediction")
        var rmse = evaluator.evaluate(als_predictions)
        println("Root-mean-square error = " + rmse.toString)
        val best_model = cv_als_model.bestModel

        //best_rank is 20
        //println(best_model.rank)

        //best_maxIter is 20
        //println(best_model
        //       ._java_obj     // Get Java object
        //       .parent()      // Get parent (ALS estimator)
        //       .getMaxIter()) // Get maxIter

        // Root-mean-square error is 1.3383152747968081

        val alsb = new ALS()
            .setRank(20)
            .setMaxIter(20)
            .setRegParam(0.3)
            .setUserCol("userId").setItemCol("businessId")
            .setRatingCol("rating")

        val alsb_model = alsb.fit(train)
        alsb_model.setColdStartStrategy("drop")

        val alsb_pred = alsb_model.transform(test)
        evaluator = new RegressionEvaluator().setMetricName("rmse").setLabelCol("rating").setPredictionCol("prediction")
        rmse = evaluator.evaluate(alsb_pred)
        println("Root-mean-square error = " + rmse.toString)
        
        alsb_model.write.overwrite().save(model_path + "alsb")
    }
}
