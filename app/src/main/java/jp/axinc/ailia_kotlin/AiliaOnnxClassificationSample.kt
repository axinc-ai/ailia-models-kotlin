package jp.axinc.ailia_kotlin

import android.util.Log
import axip.ailia.*
import java.io.File

/**
 * ONNXの画像分類モデル定義。
 * range はailia-models(Python版)のClassifier設定に合わせる
 * (MobileNetV2: IMAGENET, resnet50.opt: SIGNED_INT8)。
 */
enum class OnnxClassificationModelType(
    val displayName: String,
    val weightUrl: String,
    val weightFile: String,
    val protoUrl: String,
    val protoFile: String,
    val range: AiliaNetworkImageRange
) {
    MOBILENETV2(
        "MobileNetV2",
        "https://storage.googleapis.com/ailia-models/mobilenetv2/mobilenetv2_1.0.onnx",
        "mobilenetv2_1.0.onnx",
        "https://storage.googleapis.com/ailia-models/mobilenetv2/mobilenetv2_1.0.onnx.prototxt",
        "mobilenetv2_1.0.onnx.prototxt",
        AiliaNetworkImageRange.IMAGENET
    ),
    RESNET50(
        "ResNet50",
        "https://storage.googleapis.com/ailia-models/resnet50/resnet50.opt.onnx",
        "resnet50.opt.onnx",
        "https://storage.googleapis.com/ailia-models/resnet50/resnet50.opt.onnx.prototxt",
        "resnet50.opt.onnx.prototxt",
        AiliaNetworkImageRange.SIGNED_INT8
    ),
}

class AiliaOnnxClassificationSample(private val modelDirectory: File) {
    companion object {
        private const val TAG = "AILIA_Main"
        init { System.loadLibrary("ailia") }
    }

    var modelType: OnnxClassificationModelType = OnnxClassificationModelType.MOBILENETV2

    private var ailia: AiliaModel? = null
    private var classifier: AiliaClassifierModel? = null
    private var isInitialized = false
    private var lastClassificationResult: String = ""

    fun downloadModel(listener: ModelDownloadListener? = null): Boolean {
        return try {
            Log.i(TAG, "Starting ONNX classification model download/check (${modelType.displayName})...")
            check(ModelDownloader.downloadFile(modelDirectory, ModelFileSpec(modelType.protoUrl, modelType.protoFile), listener) != null)
            check(ModelDownloader.downloadFile(modelDirectory, ModelFileSpec(modelType.weightUrl, modelType.weightFile), listener) != null)
            listener?.onComplete()
            Log.i(TAG, "ONNX classification model download/check complete")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Model Download Failed: ${modelType.weightFile}", e)
            listener?.onError(e.message ?: "Download failed")
            false
        }
    }

    fun initializeClassification(envId: Int): Boolean {
        if (isInitialized) {
            releaseClassification()
        }

        return try {
            val protoPath = File(modelDirectory, modelType.protoFile).absolutePath
            val modelPath = File(modelDirectory, modelType.weightFile).absolutePath

            ailia = AiliaModel(envId, Ailia.MULTITHREAD_AUTO, protoPath, modelPath)

            classifier = AiliaClassifierModel(
                ailia!!.handle,
                AiliaNetworkImageFormat.RGB,
                AiliaNetworkImageChannel.FIRST,
                modelType.range
            )

            isInitialized = true
            Log.i(TAG, "ONNX Classification (${modelType.displayName}) initialized successfully with envId=$envId")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize ONNX classification: ${e.javaClass.name}: ${e.message}")
            releaseClassification()
            false
        }
    }

    /** Compatibility wrapper for the demo UI. Use [classify] for typed output. */
    fun processClassification(img: ByteArray, w: Int, h: Int): Long =
        classify(img, w, h)?.processingTimeMs ?: -1

    /** Runs classification and returns a runtime-independent result. */
    fun classify(img: ByteArray, w: Int, h: Int): ModelInferenceResult<ClassificationResult>? {
        val currentClassifier = classifier
        if (!isInitialized || currentClassifier == null) {
            Log.e(TAG, "ONNX Classification not initialized")
            return null
        }

        return try {
            val startTime = System.nanoTime()
            currentClassifier.compute(img, w * 4, w, h, AiliaImageFormat.RGBA, 5)
            val endTime = System.nanoTime()

            val count = currentClassifier.classCount
            if (count > 0) {
                val topClass = currentClassifier.getClass(0)
                val label = ImageNetLabels.CATEGORY.getOrElse(topClass.category) { "class${topClass.category}" }
                val result = ClassificationResult(topClass.category, label, topClass.prob)
                lastClassificationResult = result.displayText()
                Log.i(TAG, "class ${topClass.category} $label confidence ${topClass.prob}")
                ModelInferenceResult(result, (endTime - startTime) / 1_000_000)
            } else {
                lastClassificationResult = "No classification result"
                null
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to process ONNX classification: ${e.javaClass.name}: ${e.message}")
            null
        }
    }

    fun getLastClassificationResult(): String {
        return lastClassificationResult
    }

    fun releaseClassification() {
        try {
            classifier?.close()
            ailia?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing ONNX classification: ${e.javaClass.name}: ${e.message}")
        } finally {
            classifier = null
            ailia = null
            isInitialized = false
            Log.i(TAG, "ONNX Classification released")
        }
    }
}
