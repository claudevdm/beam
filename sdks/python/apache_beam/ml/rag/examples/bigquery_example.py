import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

# Create some test data as beam.Rows
def create_test_data():
    return [
        beam.Row(name='Alice', age=25, score=90.5),
        beam.Row(name='Bob', age=30, score=85.0),
        beam.Row(name='Charlie', age=35, score=95.5)
    ]

# BigQuery write config
write_config = {
    "table": "dataflow-twest:claude_test.managed_transform_write",
    "create_disposition": "CREATE_IF_NEEDED",
    "write_disposition": "WRITE_TRUNCATE",
}

# Create and run pipeline
with beam.Pipeline(options=PipelineOptions([
    '--runner=DirectRunner',
    '--temp_location=gs://cvandermerwe/managed',
    '--expansion_service_port=8888'
    ])) as p:
    
    # Create test data
    rows = p | "Create Data" >> beam.Create(create_test_data())
    
    # Write to BigQuery using managed transform
    write_out = (rows 
         | "Write to BigQuery" >> beam.managed.Write(
             beam.managed.BIGQUERY,
             config=write_config
             )
        )
    
    print(write_out)

# java -jar sdks/java/io/google-cloud-platform/expansion-service/build/libs/beam-sdks-java-io-google-cloud-platform-expansion-service-2.62.0-SNAPSHOT.jar 8888