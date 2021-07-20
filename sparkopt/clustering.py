import pyspark
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline


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
    return model.transform(dataframe) \
        .select(["Communes", "month", "std_avg_temp", "std_min_temp", "std_max_temp", "prediction"])
