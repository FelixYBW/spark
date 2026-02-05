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
package org.apache.spark.sql.execution.datasources

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

import org.apache.spark.Partition
import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.connector.read.InputPartition

/**
 * A collection of file blocks that should be read as a single task
 * (possibly from multiple partitioned directories).
 */
case class FilePartition(index: Int, files: Array[PartitionedFile])
  extends Partition with InputPartition {
  override def preferredLocations(): Array[String] = {
    // Computes total number of bytes can be retrieved from each host.
    val hostToNumBytes = mutable.HashMap.empty[String, Long]
    files.foreach { file =>
      file.locations.filter(_ != "localhost").foreach { host =>
        hostToNumBytes(host) = hostToNumBytes.getOrElse(host, 0L) + file.length
      }
    }

    // Takes the first 3 hosts with the most data to be retrieved
    hostToNumBytes.toSeq.sortBy {
      case (host, numBytes) => numBytes
    }.reverse.take(3).map {
      case (host, numBytes) => host
    }.toArray
  }
}

object FilePartition extends Logging {

  private def getFilePartitions(
      partitionedFiles: Seq[PartitionedFile],
      maxSplitBytes: Long,
      openCostInBytes: Long): Seq[FilePartition] = {
    // Calculate total bytes and number of partitions
    val totalBytes = partitionedFiles.map(_.length + openCostInBytes).sum
    val numPartitions = Math.max(1, (totalBytes / maxSplitBytes).toInt)
    
    // Initialize partitions with ArrayBuffers for files and size tracking
    val partitionFiles = Array.fill(numPartitions)(new ArrayBuffer[PartitionedFile])
    val partitionSizes = Array.fill(numPartitions)(0L)
    
    // Sort files from large to small
    val sortedFiles = partitionedFiles.sortBy(_.length)(Ordering[Long].reverse)
    
    // Assign files to partitions using round-robin with size check
    var currentPartitionIndex = 0
    sortedFiles.foreach { file =>
      val fileSize = file.length + openCostInBytes
      var attempts = 0
      var placed = false
      
      // Try to place file in partitions round-robin style
      while (attempts < numPartitions && !placed) {
        val partIndex = (currentPartitionIndex + attempts) % numPartitions
        if (partitionSizes(partIndex) + fileSize <= maxSplitBytes) {
          partitionFiles(partIndex) += file
          partitionSizes(partIndex) += fileSize
          placed = true
          currentPartitionIndex = (partIndex + 1) % numPartitions
        } else {
          attempts += 1
        }
      }
      
      // If file couldn't be placed in any partition without exceeding maxSplitBytes,
      // place it in the partition with the smallest current size
      if (!placed) {
        val minSizePartIndex = partitionSizes.zipWithIndex.minBy(_._1)._2
        partitionFiles(minSizePartIndex) += file
        partitionSizes(minSizePartIndex) += fileSize
        currentPartitionIndex = (minSizePartIndex + 1) % numPartitions
      }
    }
    
    // Create FilePartition objects from non-empty partitions
    partitionFiles.zipWithIndex.filter(_._1.nonEmpty).map { case (files, index) =>
      FilePartition(index, files.toArray)
    }.toSeq
  }

  def getFilePartitions(
      sparkSession: SparkSession,
      partitionedFiles: Seq[PartitionedFile],
      maxSplitBytes: Long): Seq[FilePartition] = {
    val openCostBytes = sparkSession.sessionState.conf.filesOpenCostInBytes
    val maxPartNum = sparkSession.sessionState.conf.filesMaxPartitionNum
    val partitions = getFilePartitions(partitionedFiles, maxSplitBytes, openCostBytes)
    partitions
  }

  def maxSplitBytes(
      sparkSession: SparkSession,
      selectedPartitions: Seq[PartitionDirectory]): Long = {
    val defaultMaxSplitBytes = sparkSession.sessionState.conf.filesMaxPartitionBytes
    val openCostInBytes = sparkSession.sessionState.conf.filesOpenCostInBytes
    val maxPartNum = sparkSession.sessionState.conf.filesMaxPartitionNum
    var minPartitionNum = sparkSession.sessionState.conf.filesMinPartitionNum
      .getOrElse(sparkSession.leafNodeDefaultParallelism)
    val totalBytes = selectedPartitions.flatMap(_.files.map(_.getLen + openCostInBytes)).sum

    // If totalBytes/maxPartNum < defaultMaxSplitBytes, return maxPartNum
    if (maxPartNum.exists(totalBytes / _ < defaultMaxSplitBytes)) {
      return totalBytes / maxPartNum.get
    }

    // Calculate splitBytes and adjust minPartitionNum
    var splitBytes = totalBytes / minPartitionNum
    while (splitBytes > defaultMaxSplitBytes) {
      minPartitionNum *= 2
      splitBytes = totalBytes / minPartitionNum
    }

    // Return based on comparison with maxPartNum
    if (maxPartNum.exists(minPartitionNum > _)) {
      totalBytes / maxPartNum.get
    } else {
      splitBytes
    }
  }
}
