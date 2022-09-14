package content

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.linalg.SparseVector
//import org.apache.spark.mllib.feature.IDF
//import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.sql.SparkSession
import org.jblas.DoubleMatrix


case class Product(productId:Int,name:String,imageUrl:String, categories:String,tags:String)

case class MongoConfig(uri:String,db:String)

// 标准推荐对象，productId,score
case class Recommendation(productId: Int, score:Double)

// 用户推荐列表
case class UserRecs(userId: Int, recs: Seq[Recommendation])

// 商品相似度（商品推荐）
case class ProductRecs(productId: Int, recs: Seq[Recommendation])


object ContentRecommender {
  //表名
  val MONGODB_PRODUCT_COLLECTION="Product"
  val CONTENT_PRODUCT_RECS = "ContentBasedProductRecs"

  def main(args: Array[String]): Unit = {
    // 定义配置
    val config = Map(
      "spark.cores" -> "local[*]",
      "mongo.uri" -> "mongodb://localhost:27017/recommender",
      "mongo.db" -> "recommender"
    )

    // 创建spark session
    val sparkConf = new SparkConf().setMaster(config("spark.cores")).setAppName("ContentRecommender")
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    implicit val mongoConfig = MongoConfig(config("mongo.uri"),config("mongo.db"))

    import spark.implicits._

    // 载入商品数据集
    val productTagsDF = spark
      .read
      .option("uri",mongoConfig.uri)
      .option("collection",MONGODB_PRODUCT_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[Product]
      .map(x => (x.productId, x.name, x.tags.map(c => if(c == '|') ' ' else c)))
      .toDF("productId", "name", "tags").cache()


    // 实例化一个分词器，默认按空格分
    val tokenizer = new Tokenizer().setInputCol("tags").setOutputCol("words")

    // 用分词器做转换
    val wordsData = tokenizer.transform(productTagsDF)

    // 定义一个HashingTF工具
    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(200)

    // 用 HashingTF 做处理
    val featurizedData = hashingTF.transform(wordsData)

    // 定义一个IDF工具
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")

    // 将词频数据传入，得到idf模型（统计文档）
    val idfModel = idf.fit(featurizedData)

    // 用tf-idf算法得到新的特征矩阵
    val rescaledData = idfModel.transform(featurizedData)

    // 从计算得到的 rescaledData 中提取特征向量
    val productFeatures = rescaledData.map{
      case row => ( row.getAs[Int]("productId"),row.getAs[SparseVector]("features").toArray )
    }
      .rdd
      .map(x => {
        (x._1, new DoubleMatrix(x._2) )
      })

    // 计算笛卡尔积并过滤合并
    val productRecs = productFeatures.cartesian(productFeatures)
      .filter{case (a,b) => a._1 != b._1}
      .map{case (a,b) =>
        val simScore = this.consinSim(a._2,b._2) // 求余弦相似度
        (a._1,(b._1,simScore))
      }.filter(_._2._2 > 0.6)
      .groupByKey()
      .map{case (productId,items) =>
        ProductRecs(productId,items.toList.map(x => Recommendation(x._1,x._2)))
      }.toDF()

    productRecs
      .write
      .option("uri", mongoConfig.uri)
      .option("collection",CONTENT_PRODUCT_RECS)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()

    // 关闭spark
    spark.stop()

  }
  //计算两个商品之间的余弦相似度
  def consinSim(product1: DoubleMatrix, product2:DoubleMatrix) : Double ={
    product1.dot(product2) / ( product1.norm2()  * product2.norm2() )
  }

}
