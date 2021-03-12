import json
import time
import pyspark
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, year, month, max, min, avg, stddev_samp
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark import StorageLevel
import mlflow


METEO_FILE_PATH: str = "dataset/meteo"
INSEE_FILE_PATH: str = "dataset/insee"
POPULATION_THRESHOLD: int = 1000
OUTPUT_CLUSTERING_PATH: str = "output/result_clustering"
CONFIGURATION_FILE_PATH: str = '/Users/simon.grah/Documents/Spark/BBL/code_optim_spark/sparkopt/configurations.json'
MLFLOW_EXPERIMENT_NAME = "Spark Job for clustering cities based on their temperature changes"

with open(CONFIGURATION_FILE_PATH) as json_file:
    configurations = json.load(json_file)

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


def filter_communes_not_null(meteo_dataframe: pyspark.sql.dataframe.DataFrame) -> pyspark.sql.dataframe.DataFrame:
    dataframe_result = meteo_dataframe.where(col("Communes").isNotNull())
    return dataframe_result


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


def join_meteo_and_insee(meteo_dataframe: pyspark.sql.dataframe.DataFrame,
                         insee_dataframe: pyspark.sql.dataframe.DataFrame) -> pyspark.sql.dataframe.DataFrame:
    dataframe_result = meteo_dataframe.join(insee_dataframe,
                                            on="CodePostal",
                                            how='left')
    return dataframe_result


def filter_communes_based_on_population(joined_dataframe: pyspark.sql.dataframe.DataFrame,
                                        population_threshold: float) -> pyspark.sql.dataframe.DataFrame:
    dataframe_result = joined_dataframe.where(col("Population") >= population_threshold)
    return dataframe_result


def compute_temperature_features_by_communes_and_month_over_years(joined_dataframe: pyspark.sql.dataframe.DataFrame) \
        -> pyspark.sql.dataframe.DataFrame:
    dataframe_result = joined_dataframe.groupby("Communes", "year", "month")\
        .agg(avg("Temperature"), min("Temperature"), max("Temperature"))
    dataframe_result = dataframe_result.groupby("Communes", "month")\
        .agg(stddev_samp("avg(Temperature)").alias("std_avg_temp"),
             stddev_samp("min(Temperature)").alias("std_min_temp"),
             stddev_samp("max(Temperature)").alias("std_max_temp"))
    return dataframe_result


def persist_dataframe(dataframe: pyspark.sql.dataframe.DataFrame,
                      persistence_level: str) -> None:
    dataframe.persist(persistence_level_dictionary[persistence_level])


def compute_cluster(dataframe: pyspark.sql.dataframe.DataFrame) -> pyspark.sql.dataframe.DataFrame:
    assembler = VectorAssembler(inputCols=["std_avg_temp", "std_min_temp", "std_max_temp"],
                                outputCol="features")
    kmeans = KMeans(k=10,
                    maxIter=100,
                    initSteps=10,
                    featuresCol="features",
                    seed=1)
    pipeline = Pipeline(stages=[assembler, kmeans])
    model = pipeline.fit(dataframe)
    return model.transform(dataframe)\
        .select(["Communes", "month", "std_avg_temp", "std_min_temp", "std_max_temp", "prediction"])


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


def data_treatment_pipeline(spark: pyspark.sql.session.SparkSession) -> None:
    meteo = read_meteo_file(spark,
                            meteo_file_path=METEO_FILE_PATH,
                            is_parquet_file=configurations['is_parquet_file'])
    meteo = clean_meteo_dataset(meteo,
                                is_parquet_file=configurations['is_parquet_file'])
    meteo = filter_communes_not_null(meteo)
    insee = read_insee_file(spark,
                            INSEE_FILE_PATH,
                            is_parquet_file=configurations['is_parquet_file'])
    insee = clean_insee_dataset(insee)
    meteo_and_insee_joined = join_meteo_and_insee(meteo, insee)
    meteo_and_insee_joined = filter_communes_based_on_population(joined_dataframe=meteo_and_insee_joined,
                                                                 population_threshold=POPULATION_THRESHOLD)
    temp_features_by_communes = compute_temperature_features_by_communes_and_month_over_years(meteo_and_insee_joined)
    if configurations['persistence_level']:
        persist_dataframe(dataframe=temp_features_by_communes,
                          persistence_level=configurations['persistence_level'])
    output_clustering = compute_cluster(temp_features_by_communes)
    write_dataframe_to_disk(dataframe=output_clustering,
                            saved_file_path=OUTPUT_CLUSTERING_PATH,
                            is_parquet_file=configurations['is_parquet_file'])


try:
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id
except:
    print("Create a new mlflow experiment")
    experiment_id = mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)


if __name__ == "__main__":
    with mlflow.start_run(experiment_id=experiment_id):
        for conf, conf_value in configurations.items():
            mlflow.log_param(conf, conf_value)
        # Spark initiation
        spark_conf = set_spark_configurations(configurations)
        spark = build_spark_session(master_conf="local[" + str(configurations['spark.driver.cores']) + "]",
                                    spark_conf=spark_conf)
        print(spark.sparkContext._conf.getAll())
        # Data Pipeline
        start_time = time.time()
        data_treatment_pipeline(spark)
        end_time = time.time()
        mlflow.log_metric("elapsed time", round(end_time - start_time, 2))