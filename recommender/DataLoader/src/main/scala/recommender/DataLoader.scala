package recommender

import com.mongodb.casbah.Imports.{MongoClientURI, MongoDBObject}
import com.mongodb.casbah.MongoClient
import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}


case class Product(productId:Int,name:String,imageUrl:String, categories:String,tags:String)

case class Rating(userId:Int,productId:Int,score:Double,timestamp:Int)

case class MongoConfig(uri:String,db:String)

object DataLoader {
  //路径
  val PRODUCT_DATA_PATH="/Users/yuxinkun/IdeaProjects/ECommerceRecommendSystem/recommender/DataLoader/src/main/resources/products.csv"
  val RATINGS_DATA_PATH="/Users/yuxinkun/IdeaProjects/ECommerceRecommendSystem/recommender/DataLoader/src/main/resources/ratings.csv"
  //表名
  val MONGODB_PRODUCT_COLLECTION="Product"
  val MONGODB_RATING_COLLECTION="Rating"

  def main(args: Array[String]): Unit = {
    val config=Map(
      "spark.cores"->"local[*]",
      "mongo.uri"->"mongodb://localhost:27017/recommender",
      "mongo.db"->"remommender"
    )
    val sparkConf= new SparkConf().setMaster(config("spark.cores")).setAppName("Dataloader")
    val spark=SparkSession.builder().config(sparkConf).getOrCreate()

    import spark.implicits._

    //加载数据
    val productRdd=spark.sparkContext.textFile(PRODUCT_DATA_PATH)
    val productDF=productRdd.map(item=>{
      val attr=item.split("\\^")
      Product(attr(0).toInt,attr(1).trim,attr(4).trim,attr(5).trim,attr(6).trim)
    }).toDF()
    val ratingRdd=spark.sparkContext.textFile(RATINGS_DATA_PATH)
    val ratingDF=ratingRdd.map(item=>{
      val attr=item.split(",")
      Rating(attr(0).toInt,attr(1).toInt,attr(2).toDouble,attr(3).toInt)
    }).toDF()

    implicit val mongoConfig=MongoConfig(config("mongo.uri"),config("mongo.db"))
    storeDataInMongoDB(productDF,ratingDF)

    spark.stop()
  }
  def storeDataInMongoDB(productDF:DataFrame,ratingDF:DataFrame)(implicit mongoConfig: MongoConfig): Unit={
    // 新建一个mongodb的连接，客户端
    val mongoClient = MongoClient( MongoClientURI(mongoConfig.uri) )
    // 定义要操作的mongodb表，可以理解为 db.Product
    val productCollection = mongoClient( mongoConfig.db )( MONGODB_PRODUCT_COLLECTION )
    val ratingCollection = mongoClient( mongoConfig.db )( MONGODB_RATING_COLLECTION )

    // 如果表已经存在，则删掉
    productCollection.dropCollection()
    ratingCollection.dropCollection()

    // 将当前数据存入对应的表中
    productDF.write
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_PRODUCT_COLLECTION)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()

    ratingDF.write
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_RATING_COLLECTION)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()

    // 对表创建索引
    productCollection.createIndex( MongoDBObject( "productId" -> 1 ) )
    ratingCollection.createIndex( MongoDBObject( "productId" -> 1 ) )
    ratingCollection.createIndex( MongoDBObject( "userId" -> 1 ) )

    mongoClient.close()

  }

}
