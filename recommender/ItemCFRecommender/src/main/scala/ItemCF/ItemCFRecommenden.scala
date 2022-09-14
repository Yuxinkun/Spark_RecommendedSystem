package ItemCF

import org.apache.spark.SparkConf
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.sql.SparkSession

case class Rating(userId: Int, productId: Int, score: Double)

case class MongoConfig(uri:String, db:String)

// 标准推荐对象，productId,score
case class Recommendation(productId: Int, score:Double)

// 用户推荐列表
case class UserRecs(userId: Int, recs: Seq[content.Recommendation])

// 商品相似度（商品推荐）
case class ProductRecs(productId: Int, recs: Seq[content.Recommendation])


object ItemCFRecommenden {
  // 定义常量
  val MONGODB_RATING_COLLECTION = "Rating"

  // 推荐表的名称
  val ITEM_CF_PRODUCT_RECS = "ItemCFProductRecs"

  val MAX_RECOMMENDATION = 10

  def main(args: Array[String]): Unit = {
    // 定义配置
    val config = Map(
      "spark.cores" -> "local[*]",
      "mongo.uri" -> "mongodb://localhost:27017/recommender",
      "mongo.db" -> "recommender"
    )

    // 创建spark session
    val sparkConf = new SparkConf().setMaster(config("spark.cores")).setAppName("ItemCFRecommender")
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    implicit val mongoConfig = MongoConfig(config("mongo.uri"),config("mongo.db"))

    import spark.implicits._

    val ratingDF = spark.read
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_RATING_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[Rating]
      .map(x=> (x.userId, x.productId, x.score) )
      .toDF("userId", "productId", "rating")

    // 统计每个商品的评分个数，并通过内连接添加到 ratingDF 中
    val numRatersPerProduct = ratingDF.groupBy("productId").count()
    val ratingWithCountDF = ratingDF.join(numRatersPerProduct, "productId")

    // 将商品评分按 userId 两两配对，可以统计两个商品被同一用户做出评分的次数
    val joinedDF = ratingWithCountDF.join(ratingWithCountDF, "userId")
      .toDF("userId", "product1", "rating1", "count1", "product2", "rating2", "count2")
      .select("userId", "product1", "count1", "product2", "count2")
    joinedDF.createOrReplaceTempView("joined")
    val cooccurrenceDF = spark.sql(
      """
        |select product1
        |, product2
        |, count(userId) as coocount
        |, first(count1) as count1
        |, first(count2) as count2
        |from joined
        |group by product1, product2
      """.stripMargin
    ).cache()

    val simDF = cooccurrenceDF.map{ row =>
      // 用同现的次数和各自的次数，计算同现相似度
      val coocSim = cooccurrenceSim( row.getAs[Long]("coocount"), row.getAs[Long]("count1"), row.getAs[Long]("count2") )
      ( row.getAs[Int]("product1"), ( row.getAs[Int]("product2"), coocSim ) )
    }
      .rdd
      .groupByKey()
      .map{
        case (productId, recs) =>
          ProductRecs( productId,
            recs.toList
              .filter(x=>x._1 != productId)
              .sortWith(_._2>_._2)
              .map(x=>Recommendation(x._1,x._2))
              .take(MAX_RECOMMENDATION)
          )
      }
      .toDF()

    simDF.write
      .option("uri",mongoConfig.uri)
      .option("collection",ITEM_CF_PRODUCT_RECS)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()
  }

  def cooccurrenceSim(cooCount: Long, count1: Long, count2: Long): Double ={
    cooCount / math.sqrt( count1 * count2 )
  }

}
