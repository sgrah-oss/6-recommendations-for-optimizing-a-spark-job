import pyspark
from pyspark.sql.functions import col


def read_insee_file(spark: pyspark.sql.session.SparkSession,
                    insee_file_path: str,
                    is_parquet_file: int) -> pyspark.sql.dataframe.DataFrame:
    if is_parquet_file:
        dataframe_result = spark.read.format("parquet").load(path=insee_file_path + ".parquet")
    else:
        dataframe_result = spark.read.csv(path=insee_file_path + ".csv",
                                          header=True,
                                          sep=";")
    return dataframe_result


def clean_insee_dataset(insee_dataframe: pyspark.sql.dataframe.DataFrame) -> pyspark.sql.dataframe.DataFrame:
    dataframe_result = insee_dataframe.select(col("CODGEO").cast("INT").alias("CodePostal"),
                                              col("Population").cast("INT"))
    return dataframe_result
