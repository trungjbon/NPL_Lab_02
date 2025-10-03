import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer, StopWordsRemover, Tokenizer, Normalizer}
import org.apache.spark.sql.functions._
import org.apache.log4j.{Level, Logger}
import java.io.{File, PrintWriter}

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.{Vector => MLVector}

object Lab17_NLPPipeline {
  def main(args: Array[String]): Unit = {
    // Tắt log INFO
    Logger.getRootLogger.setLevel(Level.WARN)

    val spark = SparkSession.builder
      .appName("NLP Pipeline Example")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._
    println("Spark Session created successfully.")
    println("Spark UI available at http://localhost:4040")
    println("Pausing for 5 seconds to allow you to open the Spark UI...")
    Thread.sleep(5000)

    val limitDocuments = if (args.length > 0) args(0).toInt else 1000
    println(s"--> Using limitDocuments = $limitDocuments")

    // 1. --- Read Dataset ---
    val dataPath = "../data/c4-train.00000-of-01024-30K.json.gz"
    val readStartTime = System.nanoTime()
    val initialDF = spark.read.json(dataPath).limit(limitDocuments) // Limit for faster processing during lab
    val readDuration = (System.nanoTime() - readStartTime) / 1e9d
    println(f"--> Successfully read ${initialDF.count()} records in $readDuration%.2f seconds.")
    initialDF.printSchema()
    println("\nSample of initial DataFrame:")
    initialDF.show(5, truncate = true) // Show content for better understanding

    // --- Pipeline Stages Definition ---

    // 2. --- Tokenization ---
    val tokenizer = new RegexTokenizer()
      .setInputCol("text")
      .setOutputCol("tokens")
      .setPattern("\\s+|[.,;!?()\"']") // Fix: Use \\s for regex, and \" for double quote

    /*
    // Alternative Tokenizer: A simpler, whitespace-based tokenizer.
    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("tokens")
    */

    // 3. --- Stop Words Removal ---
    val stopWordsRemover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("filtered_tokens")

    // 4. --- Vectorization (Term Frequency) ---
    // Convert tokens to feature vectors using HashingTF (a fast way to do count vectorization).
    // setNumFeatures defines the size of the feature vector. This is the maximum number of features
    // (dimensions) in the output vector. Each word is hashed to an index within this range.
    //
    // If setNumFeatures is smaller than the actual vocabulary size (number of unique words),
    // hash collisions will occur. This means different words will map to the same feature index.
    // While this leads to some loss of information, it allows for a fixed, manageable vector size
    // regardless of how large the vocabulary grows, saving memory and computation for very large datasets.
    // 20,000 is a common starting point for many NLP tasks.
    val hashingTF = new HashingTF()
      .setInputCol(stopWordsRemover.getOutputCol)
      .setOutputCol("raw_features")
      .setNumFeatures(20000) // Set the size of the feature vector

    // 5. --- Vectorization (Inverse Document Frequency) ---
    val idf = new IDF()
      .setInputCol(hashingTF.getOutputCol)
      .setOutputCol("features")

    // 6. --- Normalize vectors ---
    val normalizer = new Normalizer()
      .setInputCol("features")
      .setOutputCol("normFeatures")
      .setP(2.0)   // 2.0 = L2 norm, 1.0 = L1 norm

    // 7. --- Assemble the Pipeline ---
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, stopWordsRemover, hashingTF, idf, normalizer))

    // --- Time the main operations ---

    println("\nFitting the NLP pipeline...") // Fix: Ensure single-line string literal
    val fitStartTime = System.nanoTime()
    val pipelineModel = pipeline.fit(initialDF)
    val fitDuration = (System.nanoTime() - fitStartTime) / 1e9d
    println(f"--> Pipeline fitting took $fitDuration%.2f seconds.")
    

    println("\nTransforming data with the fitted pipeline...") // Fix: Ensure single-line string literal
    val transformStartTime = System.nanoTime()
    val transformedDF = pipelineModel.transform(initialDF)
    transformedDF.cache() // Cache the result for efficiency
    val transformCount = transformedDF.count() // Force an action to trigger the transformation
    val transformDuration = (System.nanoTime() - transformStartTime) / 1e9d
    println(f"--> Data transformation of $transformCount records took $transformDuration%.2f seconds.")
    

    // Calculate actual vocabulary size after tokenization and stop word removal
    val actualVocabSize = transformedDF
      .select(explode($"filtered_tokens").as("word"))
      .filter(length($"word") > 1) // Filter out single-character tokens
      .distinct()
      .count()
    println(s"--> Actual vocabulary size after tokenization and stop word removal: $actualVocabSize unique terms.")

    // --- Show and Save Results ---
    println("\nSample of transformed data:") // Fix: Ensure single-line string literal
    transformedDF.select("text", "features").show(5, truncate = 50)

    val n_results = 20
    val results = transformedDF.select("text", "features").take(n_results)


    // 7. --- Write Metrics and Results to Separate Files ---
    
    // Write data results to the results folder
    val result_path = "../results/lab17_pipeline_output.txt" // Corrected path
    new File(result_path).getParentFile.mkdirs() // Ensure directory exists
    val resultWriter = new PrintWriter(new File(result_path))
    val writeStartTime = System.nanoTime()
    try {
      resultWriter.println(s"--- NLP Pipeline Output (First $n_results results) ---")
      resultWriter.println(s"Output file generated at: ${new File(result_path).getAbsolutePath}\n")
      results.foreach { row =>
        val text = row.getAs[String]("text")
        val features = row.getAs[org.apache.spark.ml.linalg.Vector]("features")
        resultWriter.println("="*80)
        resultWriter.println(s"Original Text: ${text.substring(0, Math.min(text.length, 100))}...")
        resultWriter.println(s"TF-IDF Vector: ${features.toString}")
        resultWriter.println("="*80)
        resultWriter.println()
      }
      println(s"Successfully wrote $n_results results to $result_path")
    } finally {
      resultWriter.close()
    }
    val writeDuration = (System.nanoTime() - writeStartTime) / 1e9d
    println(f"--> Writing results took $writeDuration%.2f seconds.")

    // Write metrics to the log folder
    val log_path = "../log/lab17_metrics.log" // Corrected path
    new File(log_path).getParentFile.mkdirs() // Ensure directory exists
    val logWriter = new PrintWriter(new File(log_path))
    try {
      logWriter.println("--- Performance Metrics ---")
      logWriter.println(f"Read dataset duration: $readDuration%.2f seconds")
      logWriter.println(f"Pipeline fitting duration: $fitDuration%.2f seconds")
      logWriter.println(f"Data transformation duration: $transformDuration%.2f seconds")
      logWriter.println(f"Writing results duration: $writeDuration%.2f seconds")
      logWriter.println(s"Actual vocabulary size (after preprocessing): $actualVocabSize unique terms")
      logWriter.println(s"HashingTF numFeatures set to: 20000")
      if (20000 < actualVocabSize) {
        logWriter.println(s"Note: numFeatures (20000) is smaller than actual vocabulary size ($actualVocabSize). Hash collisions are expected.")
      }
      logWriter.println(s"Metrics file generated at: ${new File(log_path).getAbsolutePath}")
      logWriter.println("\nFor detailed stage-level metrics, view the Spark UI at http://localhost:4040 during execution.")
      println(s"\nSuccessfully wrote metrics to $log_path")
    } finally {
      logWriter.close()
    }    



    // ========= Cosine Similarity =========
    def cosineSimilarity(v1: Vector, v2: Vector): Double = {
      val dot = v1.toArray.zip(v2.toArray).map { case (a, b) => a * b }.sum
      val norm1 = math.sqrt(v1.toArray.map(x => x * x).sum)
      val norm2 = math.sqrt(v2.toArray.map(x => x * x).sum)
      if (norm1 == 0.0 || norm2 == 0.0) 0.0 else dot / (norm1 * norm2)
    }

    val docIndex = if (args.length > 1) args(1).toInt else 0
    val k = if (args.length > 2) args(2).toInt else 5
    println(s"--> Using docIndex = $docIndex as query, topK = $k")

    val queryText = initialDF.collect()(docIndex).getAs[String]("text")
    val firstLine = queryText.split("\n").headOption.getOrElse("")
    val preview = firstLine.substring(0, math.min(firstLine.length, 100))
    println(s"Query Document [$docIndex]: $preview...")
    

    val queryDF = Seq((0, queryText)).toDF("id", "text")
    val queryVec = pipelineModel.transform(queryDF).select("normFeatures").head().getAs[Vector]("normFeatures") 

    // UDF tính dot product
    val dotProductUDF = udf((v: MLVector) => cosineSimilarity(v, queryVec))

    val scoredDF = transformedDF.withColumn("similarity", dotProductUDF(col("normFeatures")))
    val topK = scoredDF.orderBy(col("similarity").desc).limit(k)

    println(s"\nTop 5 similar documents to doc #$docIndex:")
    topK.select("text", "similarity").show(truncate = true)

    spark.stop()
    println("Spark Session stopped.")
  }
}
