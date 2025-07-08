package jp.axinc.ailia_kotlin

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Color
import android.util.Log
import android.widget.ImageView
import axip.ailia_tflite.AiliaTFLite
import java.io.File
import kotlin.math.exp
import kotlin.math.pow

class AiliaTFLiteObjectDetectionSample {
    companion object {
        private const val TAG = "AILIA_Main"
    }

    private var tflite: AiliaTFLite? = null
    private var isInitialized = false
    private var inputShape: IntArray? = null
    private var inputTensorIndex: Int = -1
    private var outputTensorIndex: Int = -1
    private var outputShape: IntArray? = null
    private var outputType: Int = -1
    private var quantScale: Float = 1.0f
    private var quantZeroPoint: Long = 0L
    private var lastDetectionResults: List<AiliaTrackerSample.DetectionResult> = emptyList()

    private fun loadImage(inputTensorType: Int, inputBuffer: ByteArray, inputShape: IntArray, bitmap : Bitmap): ByteArray {
        Log.i(TAG, ""+inputShape[0].toString()+" "+inputShape[1].toString()+ " "+inputShape[2].toString()+ " "+inputShape[3].toString())
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, inputShape[2], inputShape[1], true)
        val channels = inputShape[3]
        val buffer = ByteArray(inputShape[1] * inputShape[2] * channels)

        val pixels = IntArray(inputShape[1] * inputShape[2])
        scaledBitmap.getPixels(pixels, 0, inputShape[2], 0, 0, inputShape[2], inputShape[1])

        for (y in 0 until inputShape[1]) {
            for (x in 0 until inputShape[2]) {
                val pixel = pixels[y * inputShape[2] + x]
                val r = (pixel shr 16) and 0xFF
                val g = (pixel shr 8) and 0xFF
                val b = pixel and 0xFF
                if (inputTensorType == AiliaTFLite.AILIA_TFLITE_TENSOR_TYPE_INT8) {
                    buffer[(y * inputShape[2] + x) * channels + 0] = (b - 128).toByte()
                    buffer[(y * inputShape[2] + x) * channels + 1] = (g - 128).toByte()
                    buffer[(y * inputShape[2] + x) * channels + 2] = (r - 128).toByte()
                } else {
                    buffer[(y * inputShape[2] + x) * channels + 0] = b.toByte()
                    buffer[(y * inputShape[2] + x) * channels + 1] = g.toByte()
                    buffer[(y * inputShape[2] + x) * channels + 2] = r.toByte()
                }
            }
        }
        return buffer
    }

    private fun dequantUint8(
        value: Byte, quantScale: Float, quantZeroPoint: Long, tensorType: Int
    ): Float {
        return if (tensorType == AiliaTFLite.AILIA_TFLITE_TENSOR_TYPE_INT8) {
            ((value.toInt() - quantZeroPoint).toFloat() * quantScale)
        } else {
            (((value.toInt() and 0xFF) - quantZeroPoint).toFloat() * quantScale)
        }
    }

    fun initializeObjectDetection(modelData: ByteArray?, env: Int = AiliaTFLite.AILIA_TFLITE_ENV_REFERENCE): Boolean {
        if (modelData == null || modelData.isEmpty()) {
            Log.e(TAG, "Model data is null or empty")
            return false
        }

        if (isInitialized) {
            releaseObjectDetection()
        }

        return try {
            tflite = AiliaTFLite()
            if (!tflite!!.open(modelData, env)) {
                Log.e(TAG, "Failed to open TFLite model")
                releaseObjectDetection()
                return false
            }

            if (!tflite!!.allocateTensors()) {
                Log.e(TAG, "Failed to allocate tensors")
                releaseObjectDetection()
                return false
            }

            inputTensorIndex = tflite!!.getInputTensorIndex(0)
            if (inputTensorIndex < 0) {
                Log.e(TAG, "Invalid input tensor index: $inputTensorIndex")
                releaseObjectDetection()
                return false
            }

            inputShape = tflite!!.getInputTensorShape(0) ?: run {
                Log.e(TAG, "Failed to get input tensor shape")
                releaseObjectDetection()
                return false
            }

            if (inputShape!!.size != 4 || inputShape!!.any { it <= 0 }) {
                Log.e(TAG, "Invalid input shape: ${inputShape!!.contentToString()}")
                releaseObjectDetection()
                return false
            }

            outputTensorIndex = tflite!!.getOutputTensorIndex(0)
            if (outputTensorIndex < 0) {
                Log.e(TAG, "Invalid output tensor index: $outputTensorIndex")
                releaseObjectDetection()
                return false
            }

            outputShape = tflite!!.getOutputTensorShape(0) ?: run {
                Log.e(TAG, "Failed to get output tensor shape")
                releaseObjectDetection()
                return false
            }

            if (outputShape!!.isEmpty() || outputShape!!.any { it <= 0 }) {
                Log.e(TAG, "Invalid output shape: ${outputShape!!.contentToString()}")
                releaseObjectDetection()
                return false
            }

            outputType = tflite!!.getOutputTensorType(0)

            val quantCount = tflite!!.getTensorQuantizationCount(outputTensorIndex)
            if (quantCount != 1) {
                Log.e(TAG, "Unexpected quantization count: $quantCount")
                releaseObjectDetection()
                return false
            }

            quantScale = tflite!!.getTensorQuantizationScale(outputTensorIndex)?.get(0) ?: 1.0f
            quantZeroPoint = tflite!!.getTensorQuantizationZeroPoint(outputTensorIndex)?.get(0) ?: 0L

            isInitialized = true
            Log.i(TAG, "Object detection initialized successfully - Input: ${inputShape!!.contentToString()}, Output: ${outputShape!!.contentToString()}")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize object detection: ${e.javaClass.name}: ${e.message}")
            e.printStackTrace()
            releaseObjectDetection()
            false
        }
    }

    fun processObjectDetectionWithoutDrawing(bitmap: Bitmap, w: Int, h: Int, threshold: Float = 0.25f, iou: Float = 0.45f): Long {
        if (!isInitialized || tflite == null || inputShape == null || outputShape == null) {
            Log.e(TAG, "Object detection not initialized properly")
            return -1
        }

        return try {
            val inputTensorType = tflite!!.getInputTensorType(0)
            val dummyBuffer = ByteArray(inputShape!![1] * inputShape!![2] * inputShape!![3])
            val inputBuffer = loadImage(inputTensorType, dummyBuffer, inputShape!!, bitmap)

            if (inputBuffer.isEmpty()) {
                Log.e(TAG, "Failed to load image data")
                return -1
            }

            if (!tflite!!.setTensorData(inputTensorIndex, inputBuffer)) {
                Log.e(TAG, "Failed to set input tensor data")
                return -1
            }

            val startTime = System.nanoTime()
            if (!tflite!!.predict()) {
                Log.e(TAG, "Predict failed")
                return -1
            }
            val endTime = System.nanoTime()

            val outputData = tflite!!.getTensorData(outputTensorIndex) ?: run {
                Log.e(TAG, "Failed to get output tensor data")
                return -1
            }

            if (outputData.isEmpty()) {
                Log.e(TAG, "Output data is empty")
                return -1
            }

            // Create dummy canvas for processing without drawing
            val dummyBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            val dummyCanvas = Canvas(dummyBitmap)
            val dummyPaint = Paint()
            val dummyTextPaint = Paint()

            lastDetectionResults = postProcessYolox(inputShape!!, outputShape!!, outputData, outputType, quantScale, quantZeroPoint, dummyCanvas, dummyPaint, dummyTextPaint, w, h, threshold, iou)

            return (endTime - startTime) / 1000000
        } catch (e: Exception) {
            Log.e(TAG, "Failed to process object detection: ${e.javaClass.name}: ${e.message}")
            e.printStackTrace()
            -1
        }
    }

    fun processObjectDetection(bitmap: Bitmap, canvas: Canvas, paint: Paint, text: Paint, w: Int, h: Int, threshold: Float = 0.25f, iou: Float = 0.45f): Long {
        if (!isInitialized || tflite == null || inputShape == null || outputShape == null) {
            Log.e(TAG, "Object detection not initialized properly")
            return -1
        }

        return try {
            val inputTensorType = tflite!!.getInputTensorType(0)
            val dummyBuffer = ByteArray(inputShape!![1] * inputShape!![2] * inputShape!![3])
            val inputBuffer = loadImage(inputTensorType, dummyBuffer, inputShape!!, bitmap)

            if (inputBuffer.isEmpty()) {
                Log.e(TAG, "Failed to load image data")
                return -1
            }

            if (!tflite!!.setTensorData(inputTensorIndex, inputBuffer)) {
                Log.e(TAG, "Failed to set input tensor data")
                return -1
            }

            val startTime = System.nanoTime()
            if (!tflite!!.predict()) {
                Log.e(TAG, "Predict failed")
                return -1
            }
            val endTime = System.nanoTime()

            val outputData = tflite!!.getTensorData(outputTensorIndex) ?: run {
                Log.e(TAG, "Failed to get output tensor data")
                return -1
            }

            if (outputData.isEmpty()) {
                Log.e(TAG, "Output data is empty")
                return -1
            }

            lastDetectionResults = postProcessYolox(inputShape!!, outputShape!!, outputData, outputType, quantScale, quantZeroPoint, canvas, paint, text, w, h, threshold, iou)

            return (endTime - startTime) / 1000000
        } catch (e: Exception) {
            Log.e(TAG, "Failed to process object detection: ${e.javaClass.name}: ${e.message}")
            e.printStackTrace()
            -1
        }
    }

    fun releaseObjectDetection() {
        try {
            tflite?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing object detection: ${e.javaClass.name}: ${e.message}")
        } finally {
            tflite = null
            isInitialized = false
            inputShape = null
            inputTensorIndex = -1
            outputTensorIndex = -1
            outputShape = null
            outputType = -1
            quantScale = 1.0f
            quantZeroPoint = 0L
            Log.i(TAG, "Object detection released")
        }
    }

    data class RectF(var left: Float, var top: Float, var right: Float, var bottom: Float) {
        fun width() = right - left
        fun height() = bottom - top
        fun area() = width() * height()
    }

    private fun postProcessYolox(
        inputShape: IntArray,
        outputShape: IntArray,
        outputBuffer: ByteArray,
        outputTensorType: Int,
        quantScale: Float,
        quantZeroPoint: Long,
        canvas: Canvas,
        paint: Paint,
        text: Paint,
        originalW: Int,
        originalH: Int,
        threshold: Float,
        iou: Float
    ): List<AiliaTrackerSample.DetectionResult> {
        val ih = inputShape[1]
        val iw = inputShape[2]
        val oh = arrayOf(ih / 8, ih / 16, ih / 32)
        val ow = arrayOf(iw / 8, iw / 16, iw / 32)
        val numCells = oh[0] * ow[0] + oh[1] * ow[1] + oh[2] * ow[2]
        val numElements = 5 + CocoAndImageNetLabels.COCO_CATEGORY.size
        if (numCells != outputShape[1] || numElements != outputShape[2]) {
            Log.e(TAG, "Error! YOLOX output_shape[1,2] mismatch")
            return mutableListOf<AiliaTrackerSample.DetectionResult>()
        }

        val boxes = mutableListOf<RectF>()
        val scores = mutableListOf<Float>()
        val categories = mutableListOf<Int>()
        val detectionResults = mutableListOf<AiliaTrackerSample.DetectionResult>()

        var bufIndex = 0
        for (s in 0..2) {
            val stride = 2f.pow(3 + s)
            for (y in 0 until oh[s]) {
                for (x in 0 until ow[s]) {
                    var maxScore = 0
                    var maxClass = 0

                    for (cls in 0 until CocoAndImageNetLabels.COCO_CATEGORY.size) {
                        var score = 0;
                        if (outputTensorType == AiliaTFLite.AILIA_TFLITE_TENSOR_TYPE_INT8) {
                            score =
                                outputBuffer[bufIndex + 5 + cls].toInt() // Byte -> Int
                        } else {
                            score =
                                outputBuffer[bufIndex + 5 + cls].toInt() and 0xFF // Byte -> Int -> UByte
                        }
                        if (score > maxScore) {
                            maxScore = score
                            maxClass = cls
                        }
                    }

                    var score = dequantUint8(maxScore.toByte(), quantScale, quantZeroPoint, outputTensorType)
                    val c = dequantUint8(outputBuffer[bufIndex + 4], quantScale, quantZeroPoint, outputTensorType)
                    score *= c

                    if (score >= threshold) {
                        val cx = dequantUint8(outputBuffer[bufIndex + 0], quantScale, quantZeroPoint, outputTensorType)
                        val cy = dequantUint8(outputBuffer[bufIndex + 1], quantScale, quantZeroPoint, outputTensorType)
                        val w = dequantUint8(outputBuffer[bufIndex + 2], quantScale, quantZeroPoint, outputTensorType)
                        val h = dequantUint8(outputBuffer[bufIndex + 3], quantScale, quantZeroPoint, outputTensorType)

                        val bbCx = (cx + x) * stride
                        val bbCy = (cy + y) * stride
                        val bbW = exp(w) * stride + 1f
                        val bbH = exp(h) * stride + 1f

                        val bbox = RectF((bbCx - bbW/2) / iw, (bbCy - bbH/2) / ih, (bbCx + bbW/2) / iw, (bbCy + bbH/2) / ih)
                        boxes.add(bbox)
                        scores.add(score)
                        categories.add(maxClass)
                    }

                    bufIndex += numElements
                }
            }
        }

        // Apply NMS
        val selectedIndices = applyNMS(boxes, scores, threshold, iou)  // threshold and IOU threshold values

        for (i in selectedIndices) {
            val bbox = boxes[i]
            canvas.drawRect(
                bbox.left * originalW,
                bbox.top * originalH,
                bbox.right * originalW,
                bbox.bottom * originalH,
                paint
            )
            canvas.drawText(CocoAndImageNetLabels.COCO_CATEGORY[categories[i]] + " " + scores[i].toString(), bbox.left * originalW, bbox.top * originalH, text)

            detectionResults.add(AiliaTrackerSample.DetectionResult(
                category = categories[i],
                confidence = scores[i],
                x = bbox.left,
                y = bbox.top,
                width = bbox.width(),
                height = bbox.height()
            ))

            Log.i(TAG, "x=${bbox.left}, y=${bbox.top}, w=${bbox.width()}, h=${bbox.height()}, class=[${categories[i]}, ${CocoAndImageNetLabels.COCO_CATEGORY[categories[i]]}], score=${scores[i]}")
        }
        
        return detectionResults
    }
    
    fun getDetectionResults(bitmap: Bitmap): List<AiliaTrackerSample.DetectionResult> {
        // Return cached results from the last processObjectDetectionWithoutDrawing call
        return lastDetectionResults
    }

    private fun applyNMS(boxes: List<RectF>, scores: List<Float>, scoreThreshold: Float, iouThreshold: Float): List<Int> {
        val indices = scores.mapIndexed { index, score -> index to score }
            .filter { it.second > scoreThreshold }
            .sortedByDescending { it.second }
            .map { it.first }

        val selectedIndices = mutableListOf<Int>()
        val active = BooleanArray(indices.size) { true }

        for (i in indices.indices) {
            if (!active[i]) continue
            val index = indices[i]
            selectedIndices.add(index)
            for (j in i + 1 until indices.size) {
                if (active[j]) {
                    val otherIndex = indices[j]
                    val iou = computeIOU(boxes[index], boxes[otherIndex])
                    if (iou > iouThreshold) {
                        active[j] = false
                    }
                }
            }
        }
        return selectedIndices
    }

    private fun computeIOU(box1: RectF, box2: RectF): Float {
        val intersectionLeft = maxOf(box1.left, box2.left)
        val intersectionTop = maxOf(box1.top, box2.top)
        val intersectionRight = minOf(box1.right, box2.right)
        val intersectionBottom = minOf(box1.bottom, box2.bottom)

        val intersectionWidth = maxOf(0f, intersectionRight - intersectionLeft)
        val intersectionHeight = maxOf(0f, intersectionBottom - intersectionTop)
        val intersectionArea = intersectionWidth * intersectionHeight

        val box1Area = (box1.right - box1.left) * (box1.bottom - box1.top)
        val box2Area = (box2.right - box2.left) * (box2.bottom - box2.top)

        val unionArea = box1Area + box2Area - intersectionArea
        return if (unionArea <= 0) 0f else intersectionArea / unionArea
    }

}
