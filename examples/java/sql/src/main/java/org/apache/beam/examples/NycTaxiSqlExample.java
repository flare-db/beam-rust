/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.beam.examples;

import java.util.Locale;
import org.apache.beam.sdk.Pipeline;
import org.apache.beam.sdk.extensions.sql.SqlTransform;
import org.apache.beam.sdk.extensions.sql.meta.provider.parquet.ParquetTableProvider;
import org.apache.beam.sdk.io.TextIO;
import org.apache.beam.sdk.options.Default;
import org.apache.beam.sdk.options.Description;
import org.apache.beam.sdk.options.PipelineOptions;
import org.apache.beam.sdk.options.PipelineOptionsFactory;
import org.apache.beam.sdk.schemas.Schema;
import org.apache.beam.sdk.transforms.Create;
import org.apache.beam.sdk.transforms.MapElements;
import org.apache.beam.sdk.values.PCollection;
import org.apache.beam.sdk.values.PCollectionTuple;
import org.apache.beam.sdk.values.Row;
import org.apache.beam.sdk.values.TupleTag;
import org.apache.beam.sdk.values.TypeDescriptors;

/**
 * Joins NYC TLC trip Parquet data with a license-to-service table using Beam SQL.
 *
 * <p>This is the Java SDK equivalent of the Rust {@code sql_nyc_taxi} example. Beam SQL reads
 * Parquet directly in Java, so the Rust example's batching and Arrow conversion steps are not
 * needed. ./gradlew :examples:java:sql:execute \
 * -PmainClass=org.apache.beam.examples.NycTaxiSqlExample \ -Pexec.args="--runner=DirectRunner"
 */
public class NycTaxiSqlExample {
  private static final String DEFAULT_INPUT =
      "gs://apache-beam-samples/nyc_trip/parquet/fhvhv_tripdata_2023-02.parquet";
  private static final String DEFAULT_OUTPUT = "/tmp/sql_nyc_taxi.txt";

  private static final Schema SERVICES_SCHEMA =
      Schema.builder().addStringField("license").addStringField("service").build();

  private static final String QUERY =
      "SELECT services.service AS service,"
          + " trips.hvfhs_license_num AS hvfhs_license_num,"
          + " trips.trip_miles AS trip_miles,"
          + " trips.base_passenger_fare AS base_passenger_fare"
          + " FROM trips"
          + " JOIN services ON trips.hvfhs_license_num = services.license"
          + " WHERE services.service = 'Uber' AND trips.trip_miles > 10";

  /** Pipeline options for the NYC taxi SQL example. */
  public interface Options extends PipelineOptions {
    @Description("Parquet file or file pattern containing NYC TLC trips")
    @Default.String(DEFAULT_INPUT)
    String getInput();

    void setInput(String value);

    @Description("Text file to write the matching trips to")
    @Default.String(DEFAULT_OUTPUT)
    String getOutput();

    void setOutput(String value);
  }

  public static void main(String[] args) {
    Options options = PipelineOptionsFactory.fromArgs(args).withValidation().as(Options.class);
    Pipeline pipeline = Pipeline.create(options);

    PCollection<Row> services =
        pipeline
            .apply(
                "CreateServices",
                Create.of(
                    Row.withSchema(SERVICES_SCHEMA).addValues("HV0002", "Juno").build(),
                    Row.withSchema(SERVICES_SCHEMA).addValues("HV0003", "Uber").build(),
                    Row.withSchema(SERVICES_SCHEMA).addValues("HV0004", "Via").build(),
                    Row.withSchema(SERVICES_SCHEMA).addValues("HV0005", "Lyft").build()))
            .setRowSchema(SERVICES_SCHEMA);

    String tripsDdl =
        String.format(
            "CREATE EXTERNAL TABLE trips ("
                + "hvfhs_license_num VARCHAR,"
                + " trip_miles DOUBLE,"
                + " base_passenger_fare DOUBLE"
                + ") TYPE parquet LOCATION '%s'",
            sqlLiteral(options.getInput()));

    PCollection<Row> result =
        PCollectionTuple.of(new TupleTag<>("services"), services)
            .apply(
                "SqlJoinFilter",
                SqlTransform.query(QUERY)
                    .withDdlString(tripsDdl)
                    .withTableProvider("parquet", new ParquetTableProvider()));

    result
        .apply(
            "FormatRows",
            MapElements.into(TypeDescriptors.strings()).via(NycTaxiSqlExample::formatRow))
        .apply("WriteLines", TextIO.write().to(options.getOutput()).withoutSharding());

    pipeline.run().waitUntilFinish();
    System.out.printf("Wrote %s%n", options.getOutput());
  }

  private static String formatRow(Row row) {
    String fare =
        row.getValue("base_passenger_fare") == null
            ? "null"
            : String.format(Locale.ROOT, "%.2f", row.getDouble("base_passenger_fare"));
    return String.format(
        Locale.ROOT,
        "%s: license %s, %.2f miles, fare %s",
        row.getString("service"),
        row.getString("hvfhs_license_num"),
        row.getDouble("trip_miles"),
        fare);
  }

  private static String sqlLiteral(String value) {
    return value.replace("'", "''");
  }
}
