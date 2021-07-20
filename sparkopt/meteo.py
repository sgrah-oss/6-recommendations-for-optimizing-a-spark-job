import pyspark
from pyspark.sql.functions import col, year, month


def read_meteo_file(spark: pyspark.sql.session.SparkSession,
                    meteo_file_path: str,
                    is_parquet_file: int) -> pyspark.sql.dataframe.DataFrame:
    if is_parquet_file:
        dataframe_result = spark.read.format("parquet").load(path=meteo_file_path + ".parquet")
    else:
        dataframe_result = spark.read.csv(path=meteo_file_path + ".csv",
                                          header=True,
                                          sep=";")
    return dataframe_result


def clean_meteo_dataset(meteo_dataframe: pyspark.sql.dataframe.DataFrame,
                        is_parquet_file: bool) -> pyspark.sql.dataframe.DataFrame:
    if is_parquet_file:
        dataframe_result = meteo_dataframe.select(col("Date").cast("DATE"),
                                                  col("communes__name_").cast("STRING").alias("Communes"),
                                                  col("communes__code_").cast("STRING").alias("CodePostal"),
                                                  col("Température").cast("INT").alias("Temperature"))
    else:
        dataframe_result = meteo_dataframe.select(col("Date").cast("DATE"),
                                                  col("communes (name)").cast("STRING").alias("Communes"),
                                                  col("communes (code)").cast("STRING").alias("CodePostal"),
                                                  col("Température").cast("INT").alias("Temperature"))
    dataframe_result = dataframe_result.withColumn("year", year("Date"))
    dataframe_result = dataframe_result.withColumn("month", month("Date"))
    return dataframe_result
