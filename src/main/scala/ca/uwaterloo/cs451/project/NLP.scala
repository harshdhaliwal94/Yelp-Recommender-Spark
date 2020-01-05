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
import org.apache.hadoop.fs.{FileSystem,Path}

class NLPConf(args: Seq[String]) extends ScallopConf(args) {
    val numExecutors = opt[Int](descr = "number of executors", required = false, default = Some(1))
    val executorCores = opt[Int](descr = "number of cores", required = false, default = Some(1))
    verify()
}

object NLP {
    val log = Logger.getLogger(getClass().getName())
    def main(argv: Array[String]){
        val conf = new SparkConf().setAppName("NLP")
        val ss = SparkSession.builder.getOrCreate
        
        val businessDF = ss.read.parquet("Data/yelp_business.parquet")
        val userDF = ss.read.parquet("Data/yelp_users.parquet")
        val reviewDF = ss.read.parquet("Data/review_train.parquet")

        businessDF.createOrReplaceTempView("business")
        userDF.createOrReplaceTempView("user")
        reviewDF.createOrReplaceTempView("review")
        
        val review_text = ss.sql("SELECT business_id, review_text FROM review");
        var review_text_rdd = review_text.rdd
        
        var review_by_business_rdd = review_text_rdd.map(row => (row.getString(0),row.getString(1))).reduceByKey(_+_)
        var review_by_business_df = ss.createDataFrame(review_by_business_rdd)
        review_by_business_df = review_by_business_df.withColumnRenamed("_1", "business_id").withColumnRenamed("_2", "text")
        
        val model_path="modelTrain/"
        val regexTokenizer = new RegexTokenizer().setGaps(false).setInputCol("text").setOutputCol("token").setPattern("\\w+")
        val stopWordsRemover = new StopWordsRemover().setInputCol("token").setOutputCol("nostopwrd")
        val countVectorizer = new CountVectorizer().setInputCol("nostopwrd").setOutputCol("rawFeature")
        val iDF = new IDF().setInputCol("rawFeature").setOutputCol("idf_vec")
        val word2Vec = new Word2Vec().setNumPartitions(4).setMaxIter(1).setVectorSize(100).setMinCount(5).setInputCol("nostopwrd").setOutputCol("word_vec").setSeed(123)
        val vectorAssembler = new VectorAssembler().setInputCols(Array("idf_vec", "word_vec")).setOutputCol("comb_vec")
        val pipeline = new Pipeline().setStages(Array(regexTokenizer, stopWordsRemover, countVectorizer, iDF, word2Vec, vectorAssembler))

        // fit the model
        val pipeline_mdl = pipeline.fit(review_by_business_df)

        //save the pipeline model
        pipeline_mdl.write.overwrite().save(model_path + "pipe_txt")
    }
}
