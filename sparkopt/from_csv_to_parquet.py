import pyspark
import click


@click.command()
@click.option('--input-csv-file-path', help='path of the csv file')
@click.option('--output-parquet-file-path', help='path of the saved parquet file')
def from_csv_to_parquet(input_csv_file_path: str = "",
                        output_parquet_file_path: str = "") -> None:
    spark = pyspark.sql.SparkSession.builder \
        .master("local[*]") \
        .appName("Transform a CSV into Parquet format") \
        .config("spark.sql.files.maxPartitionBytes", "128m") \
        .config("spark.sql.debug.maxToStringFields", 1000) \
        .getOrCreate()
    dataframe = spark.read.csv(path=input_csv_file_path,
                               header=True,
                               sep=";")
    for column_name in dataframe.columns:
        new_column_name = column_name
        for special_char in [" ", ",", ";", "{", "}", "(", ")", "\n", "\t", "="]:
            new_column_name = new_column_name.replace(special_char, "_")
        dataframe = dataframe.withColumnRenamed(column_name, new_column_name)
    dataframe.write.parquet(path=output_parquet_file_path,
                            mode="overwrite",
                            compression="uncompressed")


if __name__ == "__main__":
    from_csv_to_parquet()
