package jp.axinc.ailia_kotlin

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class InferencePostProcessTest {
    @Test
    fun detrThresholdUsesBestObjectClassProbability() {
        // The combined object probability is above 0.7, but neither object class is.
        // DETR's threshold must be applied to the best individual class.
        val detections = decodeDetrOutputs(
            logits = floatArrayOf(0.4f, 0.3f, 0.0f),
            boxes = floatArrayOf(0.5f, 0.5f, 0.4f, 0.2f),
            numQueries = 1,
            numClasses = 3,
            threshold = 0.7f,
        )

        assertTrue(detections.isEmpty())
    }

    @Test
    fun detrDecodesConfidentCenterBoxToNormalizedCorners() {
        val detection = decodeDetrOutputs(
            logits = floatArrayOf(0.0f, 5.0f, -2.0f),
            boxes = floatArrayOf(0.5f, 0.5f, 0.4f, 0.2f),
            numQueries = 1,
            numClasses = 3,
            threshold = 0.7f,
        ).single()

        assertEquals(1, detection.category)
        assertTrue(detection.confidence > 0.98f)
        assertEquals(0.3f, detection.x, 0.0001f)
        assertEquals(0.4f, detection.y, 0.0001f)
        assertEquals(0.4f, detection.width, 0.0001f)
        assertEquals(0.2f, detection.height, 0.0001f)
    }

    @Test
    fun quantizedClassificationHandlesSignedInt8() {
        val results = decodeQuantizedClassifications(
            outputShape = intArrayOf(1, 3),
            outputBuffer = byteArrayOf((-128).toByte(), 10, 127),
            signed = true,
            quantScale = 0.01f,
            quantZeroPoint = 0,
            labels = arrayOf("first", "second", "third"),
        )

        assertEquals(listOf(2, 1, 0), results.map { it.category })
        assertEquals(listOf("third", "second", "first"), results.map { it.label })
        assertEquals(1.27f, results[0].confidence, 0.0001f)
        assertEquals(0.10f, results[1].confidence, 0.0001f)
        assertEquals(-1.28f, results[2].confidence, 0.0001f)
    }

    @Test
    fun quantizedClassificationHandlesUnsignedInt8() {
        val results = decodeQuantizedClassifications(
            outputShape = intArrayOf(3),
            outputBuffer = byteArrayOf(1, 0xFE.toByte(), 3),
            signed = false,
            quantScale = 0.5f,
            quantZeroPoint = 2,
            labels = arrayOf("first", "second", "third"),
        )

        assertEquals(listOf(1, 2, 0), results.map { it.category })
        assertEquals(126.0f, results[0].confidence, 0.0001f)
        assertEquals(0.5f, results[1].confidence, 0.0001f)
        assertEquals(-0.5f, results[2].confidence, 0.0001f)
    }

    @Test
    fun vitPreprocessingConvertsRgbaToNormalizedRgbChw() {
        val input = preprocessVitRgba(
            img = byteArrayOf(0, 127, 0xFF.toByte(), 0xFF.toByte()),
            width = 1,
            height = 1,
        )
        val planeSize = 224 * 224

        assertEquals(3 * planeSize, input.size)
        assertEquals(-1f, input[0], 0.0001f)
        assertEquals(127f / 127.5f - 1f, input[planeSize], 0.0001f)
        assertEquals(1f, input[2 * planeSize], 0.0001f)
        assertEquals(1f, input.last(), 0.0001f)
    }

    @Test
    fun vitClassificationAppliesSoftmaxToLogits() {
        val results = decodeVitClassifications(
            logits = floatArrayOf(1f, 3f, 2f),
            labels = arrayOf("first", "second", "third"),
        )

        assertEquals(listOf(1, 2, 0), results.map { it.category })
        assertEquals(listOf("second", "third", "first"), results.map { it.label })
        assertEquals(0.66524f, results[0].confidence, 0.0001f)
        assertEquals(0.24473f, results[1].confidence, 0.0001f)
        assertEquals(0.09003f, results[2].confidence, 0.0001f)
        assertEquals(
            "1. second (0.67)\n2. third (0.24)\n3. first (0.09)",
            formatClassificationResults(results),
        )
    }

    @Test
    fun downloadProgressDisplaysMegabytes() {
        assertEquals(
            "Downloading 1.0 / 2.0 MB",
            formatDownloadProgress(1024L * 1024L, 2L * 1024L * 1024L),
        )
        assertEquals(
            "Downloading 1.5 MB",
            formatDownloadProgress(1536L * 1024L, -1L),
        )
    }

    @Test(expected = IllegalArgumentException::class)
    fun segmentationMaskRejectsMismatchedDimensions() {
        SegmentationMask(2, 2, FloatArray(3))
    }
}
