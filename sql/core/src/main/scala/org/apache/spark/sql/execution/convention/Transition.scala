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

import scala.collection.mutable

import org.apache.spark.SparkException
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.sql.execution.SparkPlan

/**
 * :: DeveloperApi ::
 * Converts the output of a plan from one [[ConventionType]] to another, usually by wrapping the
 * plan with a transition operator.
 */
@DeveloperApi
abstract class Transition {
  def apply(plan: SparkPlan): SparkPlan

  /** Cost used to pick the cheapest transition path. */
  def cost: Int = 1
}

object Transition {
  /** A no-op transition between compatible types. */
  val empty: Transition = new Transition {
    override def apply(plan: SparkPlan): SparkPlan = plan
    override def cost: Int = 0
    override def toString: String = "Transition.empty"
  }

  def apply(f: SparkPlan => SparkPlan): Transition = new Transition {
    override def apply(plan: SparkPlan): SparkPlan = f(plan)
  }
}

/**
 * :: DeveloperApi ::
 * The global graph of registered [[ConventionType]]s and [[Transition]]s between them.
 */
@DeveloperApi
object TransitionGraph {
  private val vertices = mutable.LinkedHashSet[ConventionType]()
  private val edges = mutable.LinkedHashMap[ConventionType, mutable.LinkedHashMap[
    ConventionType, Transition]]()
  private val pathCache =
    mutable.HashMap[(ConventionType, ConventionType), Option[Seq[Transition]]]()

  private[convention] def register(t: ConventionType): Unit = synchronized {
    if (t != RowType.None && t != BatchType.None && !vertices.contains(t)) {
      // Mark first: registerTransitions() may reference types that reference this one back.
      vertices += t
      t.doRegisterTransitions()
    }
  }

  private[convention] def addEdge(
      from: ConventionType,
      to: ConventionType,
      transition: Transition): Unit = synchronized {
    require(from != to, s"Transition from $from to itself")
    require(from != RowType.None && from != BatchType.None && to != RowType.None &&
      to != BatchType.None, s"Transition from $from to $to")
    val out = edges.getOrElseUpdate(from, mutable.LinkedHashMap())
    require(!out.contains(to), s"Transition from $from to $to is already registered")
    out(to) = transition
    pathCache.clear()
    register(from)
    register(to)
  }

  def registeredTypes: Seq[ConventionType] = synchronized(vertices.toSeq)

  /**
   * The cheapest sequence of transitions from `from` to `to`, applied in order. Empty when
   * `from == to`, `None` if `to` is unreachable.
   */
  def findPath(from: ConventionType, to: ConventionType): Option[Seq[Transition]] = synchronized {
    from.ensureRegistered()
    to.ensureRegistered()
    pathCache.getOrElseUpdate((from, to), dijkstra(from, to))
  }

  def findPathOrThrow(from: ConventionType, to: ConventionType): Seq[Transition] = {
    findPath(from, to).getOrElse {
      throw SparkException.internalError(
        s"No transition from $from to $to. Registered types: ${registeredTypes.mkString(", ")}")
    }
  }

  private def dijkstra(from: ConventionType, to: ConventionType): Option[Seq[Transition]] = {
    val dist = mutable.HashMap[ConventionType, Int](from -> 0)
    val prev = mutable.HashMap[ConventionType, (ConventionType, Transition)]()
    val done = mutable.HashSet[ConventionType]()
    var current: Option[ConventionType] = Some(from)
    while (current.isDefined && current.get != to) {
      val u = current.get
      done += u
      edges.get(u).foreach(_.foreach { case (v, t) =>
        val d = dist(u) + t.cost
        if (!done.contains(v) && dist.get(v).forall(d < _)) {
          dist(v) = d
          prev(v) = (u, t)
        }
      })
      current = dist.iterator.filter { case (v, _) => !done.contains(v) }
        .minByOption(_._2).map(_._1)
    }
    if (current.isEmpty) {
      None
    } else {
      val path = mutable.ArrayBuffer[Transition]()
      var v = to
      while (v != from) {
        val (u, t) = prev(v)
        path.prepend(t)
        v = u
      }
      Some(path.toSeq)
    }
  }

  // Spark's built-in types.
  register(RowType.VanillaRowType)
  register(BatchType.VanillaBatchType)
  register(BatchType.ArrowBatchType)
}
