from sparkopt.main import build_spark_session, set_spark_configurations


def test_setting_configurations_for_a_spark_job():
    # Given
    configurations = {
        "spark.driver.memory": "4g",
        "spark.executor.memory": "4g",
        "spark.sql.shuffle.partitions": '10',
        "spark.executor.fraction": '0.6',
        "spark.shuffle.file.buffer": "32k",
        "spark.file.transferTo": "false",
        "spark.sql.files.maxPartitionBytes": "10m",
        "spark.sql.autoBroadcastJoinThreshold": '-1',
    }
    spark_conf = set_spark_configurations(configurations)
    # When
    spark = build_spark_session(master_conf="local[*]",
                                spark_conf=spark_conf)
    # Then
    for conf in configurations.keys():
        assert spark.conf.get(conf) == configurations[conf]