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

package org.apache.spark.sql.execution.convention

import org.apache.spark.SparkException
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.Attribute
import org.apache.spark.sql.execution._
import org.apache.spark.sql.test.SharedSparkSession
import org.apache.spark.sql.vectorized.ColumnarBatch

class ConventionSuite extends SharedSparkSession {
  import ConventionSuite._

  private def insert(plan: SparkPlan, outputsColumnar: Boolean = false): SparkPlan =
    ApplyColumnarRulesAndInsertTransitions(Nil, outputsColumnar).apply(plan)

  test("vanilla plans keep the existing transitions") {
    assert(insert(RowLeaf(), outputsColumnar = true) == RowToColumnarExec(RowLeaf()))
    assert(insert(VanillaBatchLeaf()) == ColumnarToRowExec(VanillaBatchLeaf()))
    assert(insert(RowUnary(VanillaBatchLeaf())) ==
      RowUnary(ColumnarToRowExec(VanillaBatchLeaf())))
    assert(insert(VanillaBatchUnary(RowLeaf()), outputsColumnar = true) ==
      VanillaBatchUnary(RowToColumnarExec(RowLeaf())))
    // A dual-mode plan follows its parent.
    assert(insert(RowUnary(DualUnary(VanillaBatchLeaf()))) ==
      RowUnary(DualUnary(ColumnarToRowExec(VanillaBatchLeaf()))))
    assert(insert(DualUnary(RowLeaf()), outputsColumnar = true) ==
      DualUnary(RowToColumnarExec(RowLeaf())))
    // A plan supporting neither row-based nor columnar execution is executed row-based.
    assert(insert(MixedBinary(RowLeaf(), VanillaBatchLeaf())) ==
      MixedBinary(RowLeaf(), ColumnarToRowExec(VanillaBatchLeaf())))
    assert(insert(MixedBinary(RowLeaf(), FooLeaf()), outputsColumnar = true) ==
      RowToColumnarExec(MixedBinary(RowLeaf(), ColumnarToRowExec(FooToVanillaExec(FooLeaf())))))
    // Existing transitions are kept.
    val c2r = ColumnarToRowExec(VanillaBatchLeaf())
    assert(insert(c2r) eq c2r)
  }

  test("custom batch type: transitions from / to vanilla") {
    assert(insert(RowUnary(FooLeaf())) ==
      RowUnary(ColumnarToRowExec(FooToVanillaExec(FooLeaf()))))
    assert(insert(FooUnary(VanillaBatchLeaf())) ==
      ColumnarToRowExec(FooToVanillaExec(FooUnary(VanillaToFooExec(VanillaBatchLeaf())))))
    assert(insert(FooUnary(RowLeaf()), outputsColumnar = true) ==
      FooToVanillaExec(FooUnary(VanillaToFooExec(RowToColumnarExec(RowLeaf())))))
    // Adjacent plans with the same custom type need no transition.
    assert(insert(FooUnary(FooLeaf()), outputsColumnar = true) ==
      FooToVanillaExec(FooUnary(FooLeaf())))
  }

  test("Arrow batch type is compatible with vanilla batch type") {
    assert(insert(RowUnary(ArrowLeaf())) == RowUnary(ColumnarToRowExec(ArrowLeaf())))
    assert(insert(VanillaBatchUnary(ArrowLeaf()), outputsColumnar = true) ==
      VanillaBatchUnary(ArrowLeaf()))
  }

  test("cheapest transition path") {
    // Foo -> Vanilla -> Bar costs 2, Foo -> Bar costs 1.
    assert(TransitionGraph.findPathOrThrow(FooBatchType, BarBatchType).size == 1)
    assert(insert(BarUnary(FooLeaf()), outputsColumnar = true) ==
      BarToVanillaExec(BarUnary(FooToBarExec(FooLeaf()))))
  }

  test("no transition path") {
    val e = intercept[SparkException](insert(RowUnary(IsolatedLeaf())))
    assert(e.getMessage.contains("No transition"))
  }
}

object ConventionSuite {
  case object FooBatchType extends BatchType {
    override protected def registerTransitions(): Unit = {
      fromBatch(BatchType.VanillaBatchType, Transition(VanillaToFooExec(_)))
      toBatch(BatchType.VanillaBatchType, Transition(FooToVanillaExec(_)))
    }
  }

  case object BarBatchType extends BatchType {
    override protected def registerTransitions(): Unit = {
      fromBatch(BatchType.VanillaBatchType, Transition(VanillaToBarExec(_)))
      toBatch(BatchType.VanillaBatchType, Transition(BarToVanillaExec(_)))
      fromBatch(FooBatchType, Transition(FooToBarExec(_)))
    }
  }

  case object IsolatedBatchType extends BatchType

  trait MockExec extends SparkPlan {
    override def output: Seq[Attribute] = Nil
    override protected def doExecute(): RDD[InternalRow] = throw new UnsupportedOperationException
    override protected def doExecuteColumnar(): RDD[ColumnarBatch] =
      throw new UnsupportedOperationException
  }

  trait MockLeaf extends LeafExecNode with MockExec

  abstract class MockUnary extends UnaryExecNode with MockExec {
    override protected def withNewChildInternal(newChild: SparkPlan): SparkPlan =
      getClass.getConstructors.head.newInstance(newChild).asInstanceOf[SparkPlan]
  }

  trait BatchOnly extends SparkPlan {
    def batchType: BatchType
    override def supportsColumnar: Boolean = true
    override def supportsRowBased: Boolean = false
    override def convention: Convention = Convention(RowType.None, batchType)
  }

  case class RowLeaf() extends MockLeaf
  case class VanillaBatchLeaf() extends MockLeaf {
    override def supportsColumnar: Boolean = true
  }
  case class FooLeaf() extends MockLeaf with BatchOnly {
    override def batchType: BatchType = FooBatchType
  }
  case class ArrowLeaf() extends MockLeaf with BatchOnly {
    override def batchType: BatchType = BatchType.ArrowBatchType
  }
  case class IsolatedLeaf() extends MockLeaf with BatchOnly {
    override def batchType: BatchType = IsolatedBatchType
  }

  case class MixedBinary(left: SparkPlan, right: SparkPlan) extends BinaryExecNode with MockExec {
    override def supportsColumnar: Boolean = false
    override def supportsRowBased: Boolean = false
    override protected def withNewChildrenInternal(
        newLeft: SparkPlan, newRight: SparkPlan): SparkPlan = copy(left = newLeft, right = newRight)
  }

  case class RowUnary(child: SparkPlan) extends MockUnary
  case class VanillaBatchUnary(child: SparkPlan) extends MockUnary {
    override def supportsColumnar: Boolean = true
  }
  case class DualUnary(child: SparkPlan) extends MockUnary {
    override def supportsColumnar: Boolean = true
    override def supportsRowBased: Boolean = true
  }
  case class FooUnary(child: SparkPlan) extends MockUnary with BatchOnly {
    override def batchType: BatchType = FooBatchType
  }
  case class BarUnary(child: SparkPlan) extends MockUnary with BatchOnly {
    override def batchType: BatchType = BarBatchType
  }

  case class VanillaToFooExec(child: SparkPlan) extends MockUnary with BatchOnly {
    override def batchType: BatchType = FooBatchType
    override def requiredChildConventions(outputsColumnar: Boolean): Seq[ConventionReq] =
      Seq(ConventionReq.vanillaBatch)
  }
  case class FooToVanillaExec(child: SparkPlan) extends MockUnary with BatchOnly {
    override def batchType: BatchType = BatchType.VanillaBatchType
    override def requiredChildConventions(outputsColumnar: Boolean): Seq[ConventionReq] =
      Seq(ConventionReq.Batch(FooBatchType))
  }
  case class VanillaToBarExec(child: SparkPlan) extends MockUnary with BatchOnly {
    override def batchType: BatchType = BarBatchType
    override def requiredChildConventions(outputsColumnar: Boolean): Seq[ConventionReq] =
      Seq(ConventionReq.vanillaBatch)
  }
  case class BarToVanillaExec(child: SparkPlan) extends MockUnary with BatchOnly {
    override def batchType: BatchType = BatchType.VanillaBatchType
    override def requiredChildConventions(outputsColumnar: Boolean): Seq[ConventionReq] =
      Seq(ConventionReq.Batch(BarBatchType))
  }
  case class FooToBarExec(child: SparkPlan) extends MockUnary with BatchOnly {
    override def batchType: BatchType = BarBatchType
    override def requiredChildConventions(outputsColumnar: Boolean): Seq[ConventionReq] =
      Seq(ConventionReq.Batch(FooBatchType))
  }
}
