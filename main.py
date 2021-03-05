import json
import time
import pyspark
from pyspark.sql.functions import col, year, month, max, stddev_samp
import mlflow


#METEO_FILE_PATH: str = "dataset/extract_meteo.csv"
METEO_FILE_PATH: str = "dataset/donnees-synop-essentielles-omm.csv"
INSEE_FILE_PATH: str = "dataset/MDB-INSEE-V2.csv"
POPULATION_THRESHOLD: int = 1000
FILE_SAVED_PATH: str = "output/result_pipeline.csv"
CONFIGURATION_FILE_PATH: str = 'configurations.json'
MLFLOW_EXPERIMENT_NAME = "pipeline spark temperature"

with open(CONFIGURATION_FILE_PATH) as json_file:
    configurations = json.load(json_file)


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


def read_meteo_file(meteo_file_path: str) -> pyspark.sql.dataframe.DataFrame:
    dataframe_result = spark.read.csv(path=meteo_file_path,
                                      header=True,
                                      sep=";")
    return dataframe_result


def clean_meteo_dataset(meteo_dataframe: pyspark.sql.dataframe.DataFrame) -> pyspark.sql.dataframe.DataFrame:
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


def read_insee_file(insee_file_path: str) -> pyspark.sql.dataframe.DataFrame:
    dataframe_result = spark.read.csv(path=insee_file_path,
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


def compute_stddev_of_max_temperature_by_communes_and_month_over_years(joined_dataframe: pyspark.sql.dataframe.DataFrame) \
        -> pyspark.sql.dataframe.DataFrame:
    dataframe_result = joined_dataframe.groupby("Communes", "year", "month").agg(max("Temperature"))
    dataframe_result = dataframe_result.groupby("Communes", "month").agg(stddev_samp("max(Temperature)"))
    return dataframe_result

def write_dataframe_to_disk(dataframe: pyspark.sql.dataframe.DataFrame,
                            file_saved_path: str) -> None:
    dataframe.write.csv(path=file_saved_path,
                        mode="overwrite",
                        sep=",",
                        header=True)

try:
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id
except:
    print("Create a new mlflow experiment")
    experiment_id = mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)


if __name__ == "__main__":
    with mlflow.start_run(run_name="essai simon final",
                          experiment_id=experiment_id):
        # Spark configurations
        spark_conf = pyspark.conf.SparkConf()
        for conf, conf_value in configurations.items():
            mlflow.log_param(conf, conf_value)
            if conf not in ["spark.driver.cores"]:
                spark_conf.set(conf, conf_value)

        spark = build_spark_session(master_conf="local[" + str(configurations['spark.driver.cores']) + "]",
                                    spark_conf=spark_conf)
        print(spark.sparkContext._conf.getAll())
        # Data Pipeline
        start_time = time.time()
        meteo = read_meteo_file(METEO_FILE_PATH)
        meteo = clean_meteo_dataset(meteo)
        meteo = filter_communes_not_null(meteo)
        insee = read_insee_file(INSEE_FILE_PATH)
        insee = clean_insee_dataset(insee)
        meteo_and_insee_joined = join_meteo_and_insee(meteo, insee)
        meteo_and_insee_joined = filter_communes_based_on_population(meteo_and_insee_joined,
                                                                     population_threshold=POPULATION_THRESHOLD)
        stddev_of_max_temp_by_communes = compute_stddev_of_max_temperature_by_communes_and_month_over_years(meteo_and_insee_joined)
        write_dataframe_to_disk(stddev_of_max_temp_by_communes, FILE_SAVED_PATH)

        end_time = time.time()
        mlflow.log_metric("elapsed time", round(end_time - start_time, 2))
