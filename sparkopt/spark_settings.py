import pyspark
from pyspark import StorageLevel

persistence_level_dictionary = {"MEMORY_ONLY": StorageLevel.MEMORY_ONLY,
                                "MEMORY_AND_DISK": StorageLevel.MEMORY_AND_DISK,
                                "DISK_ONLY": StorageLevel.DISK_ONLY,
                                "MEMORY_AND_DISK_DESER": StorageLevel.MEMORY_AND_DISK_DESER}


def set_spark_configurations(configurations: dict) -> pyspark.conf.SparkConf:
    spark_conf = pyspark.conf.SparkConf()
    for conf, conf_value in configurations.items():
        if conf not in ["spark.driver.cores"]:
            spark_conf.set(conf, conf_value)
    return spark_conf


def build_spark_session(master_conf: str,
                        spark_conf: pyspark.conf.SparkConf) -> pyspark.sql.session.SparkSession:
    spark = pyspark.sql.SparkSession.builder \
        .master(master_conf) \
        .appName("Calcul des variations historiques de température par commune") \
        .config(conf=spark_conf) \
        .getOrCreate()
    print("Information about Spark cluster")
    print("Application Name : " + spark.sparkContext.appName)
    print("Web URI for UI : " + spark.sparkContext.uiWebUrl)
    print("PySpark version : " + spark.sparkContext.version)
    return spark


def persist_dataframe(dataframe: pyspark.sql.dataframe.DataFrame,
                      persistence_level: str) -> None:
    dataframe.persist(persistence_level_dictionary[persistence_level])


def write_dataframe_to_disk(dataframe: pyspark.sql.dataframe.DataFrame,
                            saved_file_path: str,
                            is_parquet_file: int) -> None:
    if is_parquet_file:
        dataframe.write.parquet(path=saved_file_path + ".parquet",
                                mode="overwrite",
                                compression="uncompressed")
    else:
        dataframe.write.csv(path=saved_file_path + ".csv",
                            mode="overwrite",
                            sep=",",
                            header=True)
