/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.sql.execution.python

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, IntegratedUDFTestUtils}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.Attribute
import org.apache.spark.sql.execution.{ApplyColumnarRulesAndInsertTransitions, SparkPlan, UnaryExecNode}
import org.apache.spark.sql.execution.convention.{BatchType, Convention, ConventionReq, RowType}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.test.SharedSparkSession
import org.apache.spark.sql.types.StringType
import org.apache.spark.sql.vectorized.{ArrowColumnVector, ColumnarBatch}

/**
 * End-to-end tests for the Arrow columnar Python UDF input path.
 *
 * Verifies that when reading from an Arrow-backed DataSource V2 connector,
 * ArrowEvalPythonExec uses the columnar path (no ColumnarToRowExec) and
 * produces correct results.
 */
class ArrowColumnarPythonUDFSuite extends SharedSparkSession {

  import ArrowColumnarPythonUDFSuite._
  import IntegratedUDFTestUtils._

  private val arrowSource =
    "org.apache.spark.sql.execution.python.ArrowBackedDataSourceV2"

  private def readArrowSource(numRows: Int = 100, numPartitions: Int = 1) = {
    spark.read
      .format(arrowSource)
      .option("numRows", numRows.toString)
      .option("numPartitions", numPartitions.toString)
      .load()
  }

  private def collectNodes[T <: SparkPlan](
      plan: SparkPlan)(implicit tag: reflect.ClassTag[T]): Seq[T] = {
    plan.collect {
      case p if tag.runtimeClass.isInstance(p) => p.asInstanceOf[T]
    }
  }

  test("Arrow-backed source: no ColumnarToRowExec before ArrowEvalPythonExec") {
    assume(shouldTestPandasUDFs)
    withSQLConf(SQLConf.ARROW_PYSPARK_EXECUTION_ENABLED.key -> "true") {
      val pandasUDF = TestScalarPandasUDF(
        name = "arrow_test_udf", returnType = Some(StringType))
      registerTestUDF(pandasUDF, spark)

      val df = readArrowSource()
      // Select all columns to avoid column pruning inserting a
      // ProjectExec (which does not support columnar) between the
      // scan and ArrowEvalPythonExec.
      val result = df.selectExpr(
        "id", "name", "value", "data",
        "arrow_test_udf(id) as udf_id")
      val plan = result.queryExecution.executedPlan

      // ArrowEvalPythonExec should be present.
      val arrowExecs = collectNodes[ArrowEvalPythonExec](plan)
      assert(arrowExecs.nonEmpty,
        s"Expected ArrowEvalPythonExec in plan:\n$plan")

      // The direct child of ArrowEvalPythonExec should support
      // columnar (the scan), with no ColumnarToRowExec in between.
      val arrowExec = arrowExecs.head
      assert(arrowExec.child.supportsColumnar,
        "ArrowEvalPythonExec child should support columnar " +
          s"when reading from Arrow-backed source:\n$plan")
      // The scan declares vanilla batches: Arrow vectors are only detected at runtime.
      assert(arrowExec.convention.batchType == BatchType.VanillaBatchType)
    }
  }

  test("Arrow-backed source: correctness of scalar pandas UDF") {
    assume(shouldTestPandasUDFs)
    withSQLConf(SQLConf.ARROW_PYSPARK_EXECUTION_ENABLED.key -> "true") {
      val pandasUDF = TestScalarPandasUDF(
        name = "arrow_str_udf", returnType = Some(StringType))
      registerTestUDF(pandasUDF, spark)

      val df = readArrowSource(numRows = 10)
      // The test pandas UDF converts input to string.
      val result = df.selectExpr("id", "arrow_str_udf(id) as udf_id")
      val rows = result.collect()

      assert(rows.length == 10)
      rows.zipWithIndex.foreach { case (row, i) =>
        assert(row.getInt(0) == i, s"id mismatch at row $i")
        assert(row.getString(1) == i.toString,
          s"UDF result mismatch at row $i")
      }
    }
  }

  test("Arrow-backed source: multiple UDF columns") {
    assume(shouldTestPandasUDFs)
    withSQLConf(SQLConf.ARROW_PYSPARK_EXECUTION_ENABLED.key -> "true") {
      val udf1 = TestScalarPandasUDF(
        name = "arrow_udf1", returnType = Some(StringType))
      val udf2 = TestScalarPandasUDF(
        name = "arrow_udf2", returnType = Some(StringType))
      registerTestUDF(udf1, spark)
      registerTestUDF(udf2, spark)

      val df = readArrowSource(numRows = 10)
      val result = df.selectExpr(
        "id", "arrow_udf1(id) as u1", "arrow_udf2(name) as u2")
      val rows = result.collect()

      assert(rows.length == 10)
      rows.zipWithIndex.foreach { case (row, i) =>
        assert(row.getInt(0) == i)
        assert(row.getString(1) == i.toString)
        assert(row.getString(2) == s"row_$i")
      }
    }
  }

  test("Arrow-backed source: scan supportsColumnar is true") {
    val df = readArrowSource()
    val plan = df.queryExecution.executedPlan
    // The top-level plan may be ColumnarToRowExec (for Spark's row output),
    // but the scan underneath should support columnar.
    val scans = plan.collect {
      case p if p.supportsColumnar => p
    }
    assert(scans.nonEmpty,
      s"Arrow-backed source should have a columnar-capable scan:\n$plan")
  }

  test("Row-based source: child does not support columnar (baseline)") {
    assume(shouldTestPandasUDFs)
    withSQLConf(SQLConf.ARROW_PYSPARK_EXECUTION_ENABLED.key -> "true") {
      val pandasUDF = TestScalarPandasUDF(
        name = "row_test_udf", returnType = Some(StringType))
      registerTestUDF(pandasUDF, spark)

      // spark.range does not produce Arrow-backed columnar output.
      val df = spark.range(100).toDF("id")
      val result = df.selectExpr("row_test_udf(id)")
      val plan = result.queryExecution.executedPlan

      val arrowExecs = collectNodes[ArrowEvalPythonExec](plan)
      arrowExecs.foreach { exec =>
        assert(!exec.child.supportsColumnar,
          "Row-based source child should not support columnar:" +
            s"\n$plan")
      }
    }
  }

  test("Arrow-backed source: multiple partitions") {
    assume(shouldTestPandasUDFs)
    withSQLConf(SQLConf.ARROW_PYSPARK_EXECUTION_ENABLED.key -> "true") {
      val pandasUDF = TestScalarPandasUDF(
        name = "arrow_mp_udf", returnType = Some(StringType))
      registerTestUDF(pandasUDF, spark)

      val df = readArrowSource(numRows = 100, numPartitions = 4)
      val result = df.selectExpr("id", "arrow_mp_udf(id) as udf_id")
      val rows = result.collect().sortBy(_.getInt(0))

      assert(rows.length == 100)
      rows.zipWithIndex.foreach { case (row, i) =>
        assert(row.getInt(0) == i)
        assert(row.getString(1) == i.toString)
      }
    }
  }

  /**
   * Plans the `ArrowEvalPythonExec` of `df` again, with its Arrow-backed scan child declared as
   * [[BatchType.ArrowBatchType]].
   */
  private def planWithArrowBatchTypeChild(df: DataFrame): SparkPlan = {
    val exec = collectNodes[ArrowEvalPythonExec](df.queryExecution.executedPlan).head
    assert(exec.child.supportsColumnar, s"Expected the Arrow-backed scan as child:\n$exec")
    ApplyColumnarRulesAndInsertTransitions(Nil, outputsColumnar = false)
      .apply(exec.copy(child = ArrowBatchTypeExec(exec.child)))
  }

  test("Arrow batch type child: Arrow batches in and out") {
    assume(shouldTestPandasUDFs)
    withSQLConf(
        SQLConf.ARROW_PYSPARK_EXECUTION_ENABLED.key -> "true",
        SQLConf.ADAPTIVE_EXECUTION_ENABLED.key -> "false") {
      val pandasUDF = TestScalarPandasUDF(
        name = "arrow_type_udf", returnType = Some(StringType))
      registerTestUDF(pandasUDF, spark)

      val df = readArrowSource(numRows = 3000, numPartitions = 3)
        .selectExpr("id", "name", "value", "data", "arrow_type_udf(name) as u")
      val plan = planWithArrowBatchTypeChild(df)
      // Arrow batches are required from the child, with no transition in between.
      plan match {
        case p: ArrowEvalPythonExec =>
          assert(p.convention == Convention(RowType.VanillaRowType, BatchType.ArrowBatchType))
          assert(p.child.isInstanceOf[ArrowBatchTypeExec], plan)
        case _ => fail(s"Unexpected plan:\n$plan")
      }

      val allArrow = plan.executeColumnar().mapPartitionsInternal { iter =>
        iter.map(b => (0 until b.numCols()).forall(b.column(_).isInstanceOf[ArrowColumnVector]))
      }.collect()
      assert(allArrow.nonEmpty && allArrow.forall(identity))
    }
  }

  test("Arrow batch type child: non-column UDF inputs keep vanilla batches") {
    assume(shouldTestPandasUDFs)
    withSQLConf(
        SQLConf.ARROW_PYSPARK_EXECUTION_ENABLED.key -> "true",
        SQLConf.ADAPTIVE_EXECUTION_ENABLED.key -> "false") {
      val pandasUDF = TestScalarPandasUDF(
        name = "arrow_expr_udf", returnType = Some(StringType))
      registerTestUDF(pandasUDF, spark)

      val df = readArrowSource(numRows = 10)
        .selectExpr("id", "name", "value", "data", "arrow_expr_udf(id + 1) as u")
      val plan = planWithArrowBatchTypeChild(df)
      val exec = collectNodes[ArrowEvalPythonExec](plan).head
      assert(exec.convention.batchType == BatchType.VanillaBatchType)
      val rows = plan.executeCollect().map(r => (r.getInt(0), r.getUTF8String(4).toString))
      assert(rows.sorted.toSeq == (0 until 10).map(i => (i, (i + 1).toString)))
    }
  }
}

object ArrowColumnarPythonUDFSuite {
  /** Declares the Arrow-backed batches of its vanilla columnar child as Arrow batch type. */
  case class ArrowBatchTypeExec(child: SparkPlan) extends UnaryExecNode {
    override def output: Seq[Attribute] = child.output
    override def supportsColumnar: Boolean = true
    override def supportsRowBased: Boolean = false
    override def convention: Convention = Convention(RowType.None, BatchType.ArrowBatchType)
    override def requiredChildConventions(outputsColumnar: Boolean): Seq[ConventionReq] =
      Seq(ConventionReq.vanillaBatch)

    override protected def doExecute(): RDD[InternalRow] =
      throw new UnsupportedOperationException
    override protected def doExecuteColumnar(): RDD[ColumnarBatch] = child.executeColumnar()
    override protected def withNewChildInternal(newChild: SparkPlan): SparkPlan =
      copy(child = newChild)
  }
}
