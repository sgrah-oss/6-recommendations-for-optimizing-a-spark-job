import json
import time
import pyspark
import mlflow

from sparkopt.clustering import compute_cluster
from sparkopt.feature_engineering import filter_communes_not_null, join_meteo_and_insee, \
    filter_communes_based_on_population, compute_temperature_features_by_communes_and_month_over_years
from sparkopt.insee import read_insee_file, clean_insee_dataset
from sparkopt.meteo import read_meteo_file, clean_meteo_dataset
from sparkopt.spark_settings import persist_dataframe, write_dataframe_to_disk, set_spark_configurations, \
    build_spark_session

METEO_FILE_PATH: str = "dataset/meteo"
INSEE_FILE_PATH: str = "dataset/insee"
POPULATION_THRESHOLD: int = 1000
OUTPUT_CLUSTERING_PATH: str = "output/result_clustering"
CONFIGURATION_FILE_PATH: str = 'configurations.json'
MLFLOW_EXPERIMENT_NAME = "Spark Job for clustering cities based on their temperature changes"

with open(CONFIGURATION_FILE_PATH) as json_file:
    configurations = json.load(json_file)


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
