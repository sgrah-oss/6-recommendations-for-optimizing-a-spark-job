import pyspark
from pyspark.sql import functions as F


def filter_communes_not_null(meteo_dataframe: pyspark.sql.dataframe.DataFrame) -> pyspark.sql.dataframe.DataFrame:
    dataframe_result = meteo_dataframe.where(F.col("Communes").isNotNull())
    return dataframe_result


def join_meteo_and_insee(meteo_dataframe: pyspark.sql.dataframe.DataFrame,
                         insee_dataframe: pyspark.sql.dataframe.DataFrame) -> pyspark.sql.dataframe.DataFrame:
    dataframe_result = meteo_dataframe.join(insee_dataframe,
                                            on="CodePostal",
                                            how='left')
    return dataframe_result


def filter_communes_based_on_population(joined_dataframe: pyspark.sql.dataframe.DataFrame,
                                        population_threshold: float) -> pyspark.sql.dataframe.DataFrame:
    dataframe_result = joined_dataframe.where(F.col("Population") >= population_threshold)
    return dataframe_result


def compute_temperature_features_by_communes_and_month_over_years(joined_dataframe: pyspark.sql.dataframe.DataFrame) \
        -> pyspark.sql.dataframe.DataFrame:
    dataframe_result = joined_dataframe.groupby("Communes", "year", "month") \
        .agg(F.avg("Temperature"), F.min("Temperature"), F.max("Temperature"))
    dataframe_result = dataframe_result.groupby("Communes", "month") \
        .agg(F.stddev_samp("avg(Temperature)").alias("std_avg_temp"),
             F.stddev_samp("min(Temperature)").alias("std_min_temp"),
             F.stddev_samp("max(Temperature)").alias("std_max_temp"))
    return dataframe_result
