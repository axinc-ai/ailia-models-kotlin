package jp.axinc.ailia_kotlin

import android.graphics.Bitmap
import android.util.Log
import axip.ailia_tflite.AiliaTFLite
import java.io.File

/**
 * TFLiteの画像分類モデル定義。
 * modelUrlがnullの場合はraw resourceのMobileNetV2再キャリブレーション版を使用する。
 * ResNet50はailia-models-tfliteのint8量子化モデル(recalibrated)を使用する。
 */
enum class TFLiteClassificationModelType(
    val displayName: String,
    val modelUrl: String?,
    val modelFile: String?,
) {
    MOBILENETV2("MobileNetV2", null, null),
    RESNET50(
        "ResNet50",
        "https://storage.googleapis.com/ailia-models-tflite/resnet50/resnet50_quant_recalib.tflite",
        "resnet50_quant_recalib.tflite"
    ),
}

class AiliaTFLiteClassificationSample(private val modelDirectory: File) {
    companion object {
        private const val TAG = "AILIA_Main"
    }

    var modelType: TFLiteClassificationModelType = TFLiteClassificationModelType.MOBILENETV2

    /** modelUrlを持つモデル(ResNet50)をmodelDirへダウンロードする */
    fun downloadModel(listener: ModelDownloadListener? = null): Boolean {
        val url = modelType.modelUrl ?: return true
        val fileName = modelType.modelFile!!
        return try {
            check(ModelDownloader.downloadFile(modelDirectory, ModelFileSpec(url, fileName), listener) != null)
            listener?.onComplete()
            true
        } catch (e: Exception) {
            Log.e(TAG, "Model Download Failed: $fileName", e)
            listener?.onError(e.message ?: "Download failed")
            false
        }
    }

    /** ダウンロード済みモデルファイルから初期化する */
    fun initializeFromFile(env: Int = AiliaTFLite.AILIA_TFLITE_ENV_REFERENCE): Boolean {
        val fileName = modelType.modelFile ?: return false
        return try {
            val modelData = File(modelDirectory, fileName).readBytes()
            initializeClassification(modelData, env)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to read model file: ${e.message}")
            false
        }
    }

    private fun loadImage(
        inputShape: IntArray,
        bitmap: Bitmap,
        inputType: Int,
        quantScale: Float,
        quantZeroPoint: Long,
    ): ByteArray {
        Log.i(TAG, ""+inputShape[0].toString()+" "+inputShape[1].toString()+ " "+inputShape[2].toString()+ " "+inputShape[3].toString())
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, inputShape[2], inputShape[1], true)
        val channels = inputShape[3]
        val buffer = ByteArray(inputShape[1] * inputShape[2] * channels)

        val pixels = IntArray(inputShape[1] * inputShape[2])
        scaledBitmap.getPixels(pixels, 0, inputShape[2], 0, 0, inputShape[2], inputShape[1])
        if (scaledBitmap !== bitmap) scaledBitmap.recycle()
        val mobileNetInputSigned = if (modelType == TFLiteClassificationModelType.MOBILENETV2) {
            when (inputType) {
                AiliaTFLite.AILIA_TFLITE_TENSOR_TYPE_INT8 -> true
                AiliaTFLite.AILIA_TFLITE_TENSOR_TYPE_UINT8 -> false
                else -> error("Unsupported classification input type: $inputType")
            }
        } else {
            false
        }

        for (y in 0 until inputShape[1]) {
            for (x in 0 until inputShape[2]) {
                val pixel = pixels[y * inputShape[2] + x]
                val r : Int = (pixel shr 16) and 0xFF
                val g : Int = (pixel shr 8) and 0xFF
                val b : Int = pixel and 0xFF
                if (modelType == TFLiteClassificationModelType.RESNET50) {
                    // ResNet50: Caffe前処理(BGR順・mean減算)+int8量子化
                    // (ailia-models-tfliteのnormalize_image(normalize_type='Caffe')と同じ)
                    val bq = maxOf(-128, minOf((((b - 103.939f) / quantScale) + quantZeroPoint).toInt(), 127))
                    val gq = maxOf(-128, minOf((((g - 116.779f) / quantScale) + quantZeroPoint).toInt(), 127))
                    val rq = maxOf(-128, minOf((((r - 123.68f) / quantScale) + quantZeroPoint).toInt(), 127))
                    buffer[(y * inputShape[2] + x) * channels + 0] = bq.toByte()
                    buffer[(y * inputShape[2] + x) * channels + 1] = gq.toByte()
                    buffer[(y * inputShape[2] + x) * channels + 2] = rq.toByte()
                } else {
                    // MobileNetV2 recalibrated: RGB [-1, 1]を入力テンソル型に合わせて量子化する。
                    // 現行モデルはint8のため、uint8用の0..255クランプを行うと負値が失われる。
                    buffer[(y * inputShape[2] + x) * channels + 0] = quantizeClassificationInput(
                        r / 127.5f - 1.0f,
                        quantScale,
                        quantZeroPoint,
                        mobileNetInputSigned,
                    )
                    buffer[(y * inputShape[2] + x) * channels + 1] = quantizeClassificationInput(
                        g / 127.5f - 1.0f,
                        quantScale,
                        quantZeroPoint,
                        mobileNetInputSigned,
                    )
                    buffer[(y * inputShape[2] + x) * channels + 2] = quantizeClassificationInput(
                        b / 127.5f - 1.0f,
                        quantScale,
                        quantZeroPoint,
                        mobileNetInputSigned,
                    )
                }
            }
        }
        return buffer
    }

    private var tflite: AiliaTFLite? = null
    private var isInitialized = false
    private var inputShape: IntArray? = null
    private var inputTensorIndex: Int = -1
    private var inputType: Int = -1
    private var outputTensorIndex: Int = -1
    private var outputShape: IntArray? = null
    private var outputType: Int = -1
    private var quantScale: Float = 1.0f
    private var quantZeroPoint: Long = 0L
    private var lastClassificationResult: String = ""

    fun initializeClassification(modelData: ByteArray?, env: Int = AiliaTFLite.AILIA_TFLITE_ENV_REFERENCE): Boolean {
        if (modelData == null || modelData.isEmpty()) {
            Log.e(TAG, "Model data is null or empty")
            return false
        }

        if (isInitialized) {
            releaseClassification()
        }

        return try {
            tflite = AiliaTFLite()
            if (!tflite!!.open(modelData, env)) {
                Log.e(TAG, "Failed to open TFLite model")
                releaseClassification()
                return false
            }

            if (!tflite!!.allocateTensors()) {
                Log.e(TAG, "Failed to allocate tensors")
                releaseClassification()
                return false
            }

            inputTensorIndex = tflite!!.getInputTensorIndex(0)
            if (inputTensorIndex < 0) {
                Log.e(TAG, "Invalid input tensor index: $inputTensorIndex")
                releaseClassification()
                return false
            }

            inputShape = tflite!!.getInputTensorShape(0) ?: run {
                Log.e(TAG, "Failed to get input tensor shape")
                releaseClassification()
                return false
            }
            inputType = tflite!!.getInputTensorType(0)

            outputTensorIndex = tflite!!.getOutputTensorIndex(0)
            if (outputTensorIndex < 0) {
                Log.e(TAG, "Invalid output tensor index: $outputTensorIndex")
                releaseClassification()
                return false
            }

            outputShape = tflite!!.getOutputTensorShape(0) ?: run {
                Log.e(TAG, "Failed to get output tensor shape")
                releaseClassification()
                return false
            }

            outputType = tflite!!.getOutputTensorType(0)

            val quantCount = tflite!!.getTensorQuantizationCount(outputTensorIndex)
            if (quantCount != 1) {
                Log.e(TAG, "Unexpected quantization count: $quantCount")
                releaseClassification()
                return false
            }

            quantScale = tflite!!.getTensorQuantizationScale(outputTensorIndex)?.get(0) ?: 1.0f
            quantZeroPoint = tflite!!.getTensorQuantizationZeroPoint(outputTensorIndex)?.get(0) ?: 0L

            isInitialized = true
            Log.i(TAG, "Classification initialized successfully")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize classification: ${e.javaClass.name}: ${e.message}")
            releaseClassification()
            false
        }
    }

    /** Compatibility wrapper for the demo UI. Use [classify] for typed output. */
    fun processClassification(bitmap: Bitmap): Long = classify(bitmap)?.processingTimeMs ?: -1

    /** Runs preprocessing, inference and postprocessing and returns the top ranked results. */
    fun classify(bitmap: Bitmap): ModelInferenceResult<List<ClassificationResult>>? {
        if (!isInitialized || tflite == null || inputShape == null || outputShape == null) {
            Log.e(TAG, "Classification not initialized properly")
            return null
        }

        return try {
            val inputQuantScale = tflite!!.getTensorQuantizationScale(inputTensorIndex)?.get(0) ?: 1.0f
            val inputQuantZeroPoint = tflite!!.getTensorQuantizationZeroPoint(inputTensorIndex)?.get(0) ?: 0L

            val inputBuffer = loadImage(
                inputShape!!,
                bitmap,
                inputType,
                inputQuantScale,
                inputQuantZeroPoint,
            )

            if (!tflite!!.setTensorData(inputTensorIndex, inputBuffer)) {
                Log.e(TAG, "Failed to set input tensor data")
                return null
            }

            val startTime = System.nanoTime()
            if (!tflite!!.predict()) {
                Log.e(TAG, "Predict failed")
                return null
            }
            val endTime = System.nanoTime()

            val outputData = tflite!!.getTensorData(outputTensorIndex) ?: run {
                Log.e(TAG, "Failed to get output tensor data")
                return null
            }

            val results = decodeQuantizedClassifications(
                outputShape!!,
                outputData,
                outputType == AiliaTFLite.AILIA_TFLITE_TENSOR_TYPE_INT8,
                quantScale,
                quantZeroPoint,
                ImageNetLabels.CATEGORY,
            )
            lastClassificationResult = formatClassificationResults(results)
            results.forEachIndexed { rank, result ->
                Log.i(TAG, "rank ${rank + 1} class ${result.category} ${result.label} confidence ${result.confidence}")
            }
            ModelInferenceResult(results, (endTime - startTime) / 1_000_000)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to process classification: ${e.javaClass.name}: ${e.message}")
            null
        }
    }

    fun releaseClassification() {
        try {
            tflite?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing classification: ${e.javaClass.name}: ${e.message}")
        } finally {
            tflite = null
            isInitialized = false
            inputShape = null
            inputTensorIndex = -1
            inputType = -1
            outputTensorIndex = -1
            outputShape = null
            outputType = -1
            quantScale = 1.0f
            quantZeroPoint = 0L
            Log.i(TAG, "Classification released")
        }
    }

    fun getLastClassificationResult(): String {
        return lastClassificationResult
    }

}

/** Quantizes a normalized classification input using the tensor's signedness. */
internal fun quantizeClassificationInput(
    value: Float,
    quantScale: Float,
    quantZeroPoint: Long,
    signed: Boolean,
): Byte {
    require(quantScale > 0f) { "Input quantization scale must be positive" }
    val quantized = (value / quantScale + quantZeroPoint).toInt()
    val minValue = if (signed) -128 else 0
    val maxValue = if (signed) 127 else 255
    return quantized.coerceIn(minValue, maxValue).toByte()
}

/** Pure quantized classification postprocessing for host-side unit tests. */
internal fun decodeQuantizedClassifications(
    outputShape: IntArray,
    outputBuffer: ByteArray,
    signed: Boolean,
    quantScale: Float,
    quantZeroPoint: Long,
    labels: Array<String>,
    maxResults: Int = CLASSIFICATION_TOP_COUNT,
): List<ClassificationResult> {
    val classCount = outputShape.fold(1) { total, dimension -> total * dimension }
    require(classCount == labels.size) { "Unexpected classification output count: $classCount" }
    require(outputBuffer.size >= classCount) { "Classification output buffer is too small" }
    require(maxResults > 0) { "maxResults must be positive" }
    return (0 until classCount)
        .map { category ->
            val quantized = if (signed) {
                outputBuffer[category].toInt()
            } else {
                outputBuffer[category].toInt() and 0xFF
            }
            ClassificationResult(
                category = category,
                label = labels[category],
                confidence = (quantized - quantZeroPoint).toFloat() * quantScale,
            )
        }
        .sortedWith(compareByDescending<ClassificationResult> { it.confidence }.thenBy { it.category })
        .take(minOf(maxResults, classCount))
}
