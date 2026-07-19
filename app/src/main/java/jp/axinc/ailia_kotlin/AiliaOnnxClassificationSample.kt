package jp.axinc.ailia_kotlin

import android.graphics.Bitmap
import android.util.Log
import axip.ailia.*
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

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

class AiliaOnnxClassificationSample {
    companion object {
        private const val TAG = "AILIA_Main"
    }

    var modelType: OnnxClassificationModelType = OnnxClassificationModelType.MOBILENETV2

    private var ailia: AiliaModel? = null
    private var classifier: AiliaClassifierModel? = null
    private var isInitialized = false
    private var lastClassificationResult: String = ""
    var modelDir: String = ""

    private fun downloadFile(urlStr: String, fileName: String, listener: ModelDownloadListener? = null): Boolean {
        val dir = modelDir
        val path = "$dir/$fileName"
        val file = File(path)
        if (file.exists()) {
            if (file.canRead()) {
                Log.i(TAG, "Model file already exists and readable: $path (${file.length()} bytes)")
                return true
            } else {
                Log.w(TAG, "Model file exists but not readable, re-downloading: $path")
                file.delete()
            }
        }
        File(path).parentFile?.mkdirs()
        val tmpFile = File("$path.tmp")
        val url = URL(urlStr)
        val connection = url.openConnection() as HttpURLConnection
        connection.connectTimeout = 30000
        connection.readTimeout = 60000
        connection.connect()
        val totalBytes = connection.contentLengthLong
        connection.inputStream.use { input ->
            FileOutputStream(tmpFile).use { output ->
                val buffer = ByteArray(8192)
                var bytesDownloaded: Long = 0
                var bytesRead: Int
                while (input.read(buffer).also { bytesRead = it } != -1) {
                    output.write(buffer, 0, bytesRead)
                    bytesDownloaded += bytesRead
                    listener?.onProgress(fileName, bytesDownloaded, totalBytes)
                }
            }
        }
        tmpFile.renameTo(File(path))
        return true
    }

    fun downloadModel(listener: ModelDownloadListener? = null): Boolean {
        try {
            Log.i(TAG, "Starting ONNX classification model download/check (${modelType.displayName})...")
            downloadFile(modelType.protoUrl, modelType.protoFile, listener)
            downloadFile(modelType.weightUrl, modelType.weightFile, listener)
            listener?.onComplete()
            Log.i(TAG, "ONNX classification model download/check complete")
            return true
        } catch (e: Exception) {
            Log.e(TAG, "Model Download Failed: ${modelType.weightFile}", e)
            listener?.onError(e.message ?: "Download failed")
            return false
        }
    }

    fun initializeClassification(envId: Int): Boolean {
        if (isInitialized) {
            releaseClassification()
        }

        return try {
            val dir = modelDir
            val protoPath = "$dir/${modelType.protoFile}"
            val modelPath = "$dir/${modelType.weightFile}"

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

    fun processClassification(img: ByteArray, w: Int, h: Int): Long {
        if (!isInitialized || classifier == null) {
            Log.e(TAG, "ONNX Classification not initialized")
            return -1
        }

        return try {
            val startTime = System.nanoTime()
            classifier!!.compute(img, w * 4, w, h, AiliaImageFormat.RGBA, 5)
            val endTime = System.nanoTime()

            val count = classifier!!.classCount
            if (count > 0) {
                val topClass = classifier!!.getClass(0)
                val label = if (topClass.category < CocoAndImageNetLabels.IMAGENET_CATEGORY.size) {
                    CocoAndImageNetLabels.IMAGENET_CATEGORY[topClass.category]
                } else {
                    "class${topClass.category}"
                }
                lastClassificationResult = "$label (${String.format("%.2f", topClass.prob)})"
                Log.i(TAG, "class ${topClass.category} $label confidence ${topClass.prob}")
            } else {
                lastClassificationResult = "No classification result"
            }

            (endTime - startTime) / 1000000
        } catch (e: Exception) {
            Log.e(TAG, "Failed to process ONNX classification: ${e.javaClass.name}: ${e.message}")
            -1
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
