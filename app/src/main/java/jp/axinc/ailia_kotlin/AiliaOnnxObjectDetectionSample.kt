package jp.axinc.ailia_kotlin

import android.graphics.Canvas
import android.graphics.Paint
import android.util.Log
import axip.ailia.*
import java.io.File
import java.util.EnumSet

class AiliaOnnxObjectDetectionSample(private val modelDirectory: File) {
    companion object {
        private const val TAG = "AILIA_Main"
        init { System.loadLibrary("ailia") }
        private const val MODEL_URL = "https://storage.googleapis.com/ailia-models/yolox/yolox_s.opt.onnx"
        private const val MODEL_FILE = "yolox_s.opt.onnx"
        private const val PROTO_URL = "https://storage.googleapis.com/ailia-models/yolox/yolox_s.opt.onnx.prototxt"
        private const val PROTO_FILE = "yolox_s.opt.onnx.prototxt"
    }

    private var ailia: AiliaModel? = null
    private var detector: AiliaDetectorModel? = null
    private var isInitialized = false
    private var lastDetectionResults: List<DetectionResult> = emptyList()

    fun downloadModel(listener: ModelDownloadListener? = null): Boolean {
        return try {
            Log.i(TAG, "Starting ONNX model download/check...")
            check(ModelDownloader.downloadFile(modelDirectory, ModelFileSpec(PROTO_URL, PROTO_FILE), listener) != null)
            check(ModelDownloader.downloadFile(modelDirectory, ModelFileSpec(MODEL_URL, MODEL_FILE), listener) != null)
            listener?.onComplete()
            Log.i(TAG, "ONNX model download/check complete")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Model Download Failed: $MODEL_FILE", e)
            listener?.onError(e.message ?: "Download failed")
            false
        }
    }

    fun initializeObjectDetection(envId: Int): Boolean {
        if (isInitialized) {
            releaseObjectDetection()
        }

        return try {
            val protoPath = File(modelDirectory, PROTO_FILE).absolutePath
            val modelPath = File(modelDirectory, MODEL_FILE).absolutePath

            ailia = AiliaModel(envId, Ailia.MULTITHREAD_AUTO, protoPath, modelPath)

            detector = AiliaDetectorModel(
                ailia!!.handle,
                AiliaNetworkImageFormat.RGB,
                AiliaNetworkImageChannel.FIRST,
                AiliaNetworkImageRange.UNSIGNED_INT8,
                AiliaDetectorAlgorithm.YOLOX,
                CocoLabels.CATEGORY.size,
                EnumSet.noneOf(AiliaDetectorFlags::class.java)
            )

            isInitialized = true
            Log.i(TAG, "ONNX Object detection initialized successfully with envId=$envId")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize ONNX object detection: ${e.javaClass.name}: ${e.message}")
            releaseObjectDetection()
            false
        }
    }

    fun processObjectDetection(img: ByteArray, canvas: Canvas, paint: Paint, text: Paint, w: Int, h: Int, threshold: Float = 0.25f, iou: Float = 0.45f): Long {
        val result = detect(img, w, h, threshold, iou) ?: return -1
        lastDetectionResults = result.value
        drawDetections(result.value, canvas, paint, text, w, h)
        return result.processingTimeMs
    }

    fun processObjectDetectionWithoutDrawing(img: ByteArray, w: Int, h: Int, threshold: Float = 0.25f, iou: Float = 0.45f): Long {
        val result = detect(img, w, h, threshold, iou) ?: return -1
        lastDetectionResults = result.value
        return result.processingTimeMs
    }

    /** Runs detection without depending on Canvas or View classes. */
    fun detect(img: ByteArray, w: Int, h: Int, threshold: Float = 0.25f, iou: Float = 0.45f): ModelInferenceResult<List<DetectionResult>>? {
        val currentDetector = detector
        if (!isInitialized || currentDetector == null) return null
        return try {
            val startTime = System.nanoTime()
            currentDetector.compute(img, w * 4, w, h, AiliaImageFormat.RGBA, threshold, iou)
            val endTime = System.nanoTime()
            val count = currentDetector.objectCount
            val detectionResults = mutableListOf<DetectionResult>()
            for (i in 0 until count) {
                val obj = currentDetector.getObject(i)
                detectionResults.add(
                    DetectionResult(
                        category = obj.category,
                        confidence = obj.prob,
                        x = obj.x,
                        y = obj.y,
                        width = obj.w,
                        height = obj.h
                    )
                )
            }
            ModelInferenceResult(detectionResults, (endTime - startTime) / 1_000_000)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to process ONNX object detection: ${e.javaClass.name}: ${e.message}", e)
            null
        }
    }

    fun drawDetections(
        detections: List<DetectionResult>,
        canvas: Canvas,
        paint: Paint,
        text: Paint,
        w: Int,
        h: Int,
    ) {
        for (detection in detections) {
            val catColor = CategoryColors.forCategory(detection.category)
            val left = detection.x * w
            val top = detection.y * h
            canvas.drawRect(left, top, (detection.x + detection.width) * w, (detection.y + detection.height) * h, Paint(paint).apply { color = catColor })
            val label = CocoLabels.CATEGORY.getOrElse(detection.category) { "class${detection.category}" }
            val labelText = "$label ${String.format(java.util.Locale.ROOT, "%.2f", detection.confidence)}"
            canvas.drawRect(left, top - text.textSize, left + text.measureText(labelText), top + text.textSize * 0.2f, Paint().apply { style = Paint.Style.FILL; color = catColor })
            canvas.drawText(labelText, left, top, Paint(text).apply { color = android.graphics.Color.WHITE })
        }
    }

    fun getDetectionResults(): List<DetectionResult> {
        return lastDetectionResults
    }

    fun releaseObjectDetection() {
        try {
            detector?.close()
            ailia?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing ONNX object detection: ${e.javaClass.name}: ${e.message}")
        } finally {
            detector = null
            ailia = null
            isInitialized = false
            Log.i(TAG, "ONNX Object detection released")
        }
    }
}
