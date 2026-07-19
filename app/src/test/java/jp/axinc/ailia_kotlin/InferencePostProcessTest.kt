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
        val result = decodeQuantizedClassification(
            outputShape = intArrayOf(1, 3),
            outputBuffer = byteArrayOf((-128).toByte(), 10, 127),
            signed = true,
            quantScale = 0.01f,
            quantZeroPoint = 0,
            labels = arrayOf("first", "second", "third"),
        )

        assertEquals(2, result.category)
        assertEquals("third", result.label)
        assertEquals(1.27f, result.confidence, 0.0001f)
    }

    @Test
    fun quantizedClassificationHandlesUnsignedInt8() {
        val result = decodeQuantizedClassification(
            outputShape = intArrayOf(3),
            outputBuffer = byteArrayOf(1, 0xFE.toByte(), 3),
            signed = false,
            quantScale = 0.5f,
            quantZeroPoint = 2,
            labels = arrayOf("first", "second", "third"),
        )

        assertEquals(1, result.category)
        assertEquals(126.0f, result.confidence, 0.0001f)
    }

    @Test(expected = IllegalArgumentException::class)
    fun segmentationMaskRejectsMismatchedDimensions() {
        SegmentationMask(2, 2, FloatArray(3))
    }
}
