import time
from tqdm import tqdm
import mlflow
from sklearn.model_selection import ParameterGrid

from main import build_spark_session, data_treatment_pipeline, set_spark_configurations


METEO_FILE_PATH: str = "dataset/meteo"
INSEE_FILE_PATH: str = "dataset/insee"
POPULATION_THRESHOLD: int = 1000
OUTPUT_CLUSTERING_PATH: str = "output/result_clustering"
MLFLOW_EXPERIMENT_NAME = "Grid Search for configuration parameters"

conf_parameter_grid = {
    "spark.driver.cores": ['6'],
    "spark.driver.memory": ["4g"],
    "spark.executor.memory": ["4g"],
    "spark.sql.shuffle.partitions": ['10', '100'],
    "spark.shuffle.file.buffer": ["32k", "1m"],
    "spark.sql.files.maxPartitionBytes": ["32m", "128m"],
    "spark.sql.autoBroadcastJoinThreshold": ['-1', "1g"],
    "is_parquet_file": [0, 1],
    "persistence_level": [0, "MEMORY_AND_DISK_DESER"]
}
list_of_all_conf_parameter_combinations = list(ParameterGrid(conf_parameter_grid))

try:
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id
except:
    print("Create a new mlflow experiment")
    experiment_id = mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)


if __name__ == "__main__":
    for configurations in tqdm(list_of_all_conf_parameter_combinations):
        print()
        print(configurations)
        print()
        with mlflow.start_run(experiment_id=experiment_id):
            for conf, conf_value in configurations.items():
                mlflow.log_param(conf, conf_value)
            # Spark initiation
            spark_conf = set_spark_configurations(configurations)
            spark = build_spark_session(master_conf="local[" + str(configurations['spark.driver.cores']) + "]",
                                        spark_conf=spark_conf)
            # Data Pipeline
            start_time = time.time()
            data_treatment_pipeline(spark)
            end_time = time.time()
            mlflow.log_metric("elapsed time", round(end_time - start_time, 2))
