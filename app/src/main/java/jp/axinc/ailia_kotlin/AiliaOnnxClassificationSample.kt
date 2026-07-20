package jp.axinc.ailia_kotlin

import android.util.Log
import axip.ailia.*
import java.io.File
import kotlin.math.exp

/**
 * ONNXの画像分類モデル定義。
 * range はailia-models(Python版)のClassifier設定に合わせる
 * (MobileNetV2: IMAGENET, resnet50.opt: SIGNED_INT8,
 * ViT-B/16: SIGNED_FP32[-1, 1])。
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
    VIT_B16(
        "ViT-B/16",
        "https://storage.googleapis.com/ailia-models/vit/ViT-B_16-224.onnx",
        "ViT-B_16-224.onnx",
        "https://storage.googleapis.com/ailia-models/vit/ViT-B_16-224.onnx.prototxt",
        "ViT-B_16-224.onnx.prototxt",
        AiliaNetworkImageRange.SIGNED_FP32
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

            // ViT exposes logits plus one attention tensor per Transformer block.
            // AiliaClassifierModel assumes a conventional single-output classifier,
            // so ViT is executed directly through AiliaModel in classifyVit().
            classifier = if (modelType == OnnxClassificationModelType.VIT_B16) {
                null
            } else {
                AiliaClassifierModel(
                    ailia!!.handle,
                    AiliaNetworkImageFormat.RGB,
                    AiliaNetworkImageChannel.FIRST,
                    modelType.range
                )
            }

            isInitialized = true
            lastClassificationResult = ""
            Log.i(TAG, "ONNX Classification (${modelType.displayName}) initialized successfully with envId=$envId")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize ONNX classification: ${e.javaClass.name}: ${e.message}", e)
            releaseClassification()
            false
        }
    }

    /** Compatibility wrapper for the demo UI. Use [classify] for typed output. */
    fun processClassification(img: ByteArray, w: Int, h: Int): Long =
        classify(img, w, h)?.processingTimeMs ?: -1

    /** Runs classification and returns the top ranked runtime-independent results. */
    fun classify(img: ByteArray, w: Int, h: Int): ModelInferenceResult<List<ClassificationResult>>? {
        val currentModel = ailia
        if (!isInitialized || currentModel == null) {
            Log.e(TAG, "ONNX Classification not initialized")
            lastClassificationResult = "Error: ONNX Classification is not initialized"
            return null
        }

        return try {
            if (modelType == OnnxClassificationModelType.VIT_B16) {
                classifyVit(currentModel, img, w, h)
            } else {
                classifyWithClassifier(checkNotNull(classifier), img, w, h)
            }
        } catch (e: Exception) {
            val message = e.message ?: e.javaClass.simpleName
            lastClassificationResult = "Error: $message"
            Log.e(TAG, "Failed to process ONNX classification: ${e.javaClass.name}: ${e.message}", e)
            null
        }
    }

    private fun classifyWithClassifier(
        currentClassifier: AiliaClassifierModel,
        img: ByteArray,
        w: Int,
        h: Int,
    ): ModelInferenceResult<List<ClassificationResult>>? {
        val startTime = System.nanoTime()
        currentClassifier.compute(img, w * 4, w, h, AiliaImageFormat.RGBA, CLASSIFICATION_TOP_COUNT)
        val processingTimeMs = (System.nanoTime() - startTime) / 1_000_000
        if (currentClassifier.classCount <= 0) {
            lastClassificationResult = "No classification result"
            return null
        }

        val results = (0 until minOf(CLASSIFICATION_TOP_COUNT, currentClassifier.classCount)).map { rank ->
            val item = currentClassifier.getClass(rank)
            val label = ImageNetLabels.CATEGORY.getOrElse(item.category) { "class${item.category}" }
            ClassificationResult(item.category, label, item.prob)
        }
        setLastResults(results)
        return ModelInferenceResult(results, processingTimeMs)
    }

    private fun classifyVit(
        currentModel: AiliaModel,
        img: ByteArray,
        w: Int,
        h: Int,
    ): ModelInferenceResult<List<ClassificationResult>> {
        val startTime = System.nanoTime()
        val input = preprocessVitRgba(img, w, h)
        val inputIndex = currentModel.getBlobIndexByInputIndex(0)
        currentModel.setInputBlobShapeND(
            intArrayOf(1, 3, VIT_INPUT_SIZE, VIT_INPUT_SIZE),
            inputIndex,
        )
        currentModel.setInputBlobData(input, input.size * Float.SIZE_BYTES, inputIndex)
        currentModel.update()

        // Output 0 is [1, 1000] logits. Remaining outputs are attention tensors.
        val logitsIndex = currentModel.getBlobIndexByOutputIndex(0)
        val logitsSize = currentModel.getBlobShapeND(logitsIndex).fold(1) { size, dimension ->
            Math.multiplyExact(size, dimension)
        }
        val logits = FloatArray(logitsSize)
        currentModel.getBlobData(logits, logits.size * Float.SIZE_BYTES, logitsIndex)
        val results = decodeVitClassifications(logits)
        setLastResults(results)
        return ModelInferenceResult(results, (System.nanoTime() - startTime) / 1_000_000)
    }

    private fun setLastResults(results: List<ClassificationResult>) {
        lastClassificationResult = formatClassificationResults(results)
        results.forEachIndexed { rank, result ->
            Log.i(TAG, "rank ${rank + 1} class ${result.category} ${result.label} confidence ${result.confidence}")
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

private const val VIT_INPUT_SIZE = 224

/** ViT preprocessing used by ailia-models: bilinear resize, RGB [-1, 1], and BCHW. */
internal fun preprocessVitRgba(img: ByteArray, width: Int, height: Int): FloatArray {
    require(width > 0 && height > 0) { "Image dimensions must be positive" }
    require(img.size >= width * height * 4) { "RGBA image buffer is too small" }
    val planeSize = VIT_INPUT_SIZE * VIT_INPUT_SIZE
    val input = FloatArray(3 * planeSize)

    for (dstY in 0 until VIT_INPUT_SIZE) {
        val sourceY = ((dstY + 0.5f) * height / VIT_INPUT_SIZE - 0.5f).coerceIn(0f, height - 1f)
        val y0 = sourceY.toInt()
        val y1 = minOf(y0 + 1, height - 1)
        val yWeight = sourceY - y0
        for (dstX in 0 until VIT_INPUT_SIZE) {
            val sourceX = ((dstX + 0.5f) * width / VIT_INPUT_SIZE - 0.5f).coerceIn(0f, width - 1f)
            val x0 = sourceX.toInt()
            val x1 = minOf(x0 + 1, width - 1)
            val xWeight = sourceX - x0
            val dstOffset = dstY * VIT_INPUT_SIZE + dstX

            for (channel in 0 until 3) {
                fun component(x: Int, y: Int): Float =
                    (img[(y * width + x) * 4 + channel].toInt() and 0xFF).toFloat()

                val top = component(x0, y0) * (1f - xWeight) + component(x1, y0) * xWeight
                val bottom = component(x0, y1) * (1f - xWeight) + component(x1, y1) * xWeight
                val value = top * (1f - yWeight) + bottom * yWeight
                input[channel * planeSize + dstOffset] = value / 127.5f - 1f
            }
        }
    }
    return input
}

/** Applies softmax to ViT logits and returns the most probable ImageNet classes. */
internal fun decodeVitClassifications(
    logits: FloatArray,
    labels: Array<String> = ImageNetLabels.CATEGORY,
    maxResults: Int = CLASSIFICATION_TOP_COUNT,
): List<ClassificationResult> {
    require(logits.isNotEmpty()) { "ViT logits are empty" }
    require(logits.size <= labels.size) { "ViT output has more classes than labels" }
    require(maxResults > 0) { "maxResults must be positive" }
    val maxLogit = logits.maxOrNull() ?: error("ViT logits are empty")
    var probabilitySum = 0.0
    val exponentials = DoubleArray(logits.size)
    logits.forEachIndexed { category, logit ->
        val value = exp((logit - maxLogit).toDouble())
        exponentials[category] = value
        probabilitySum += value
    }
    require(probabilitySum.isFinite() && probabilitySum > 0.0) { "Invalid ViT logits" }
    return logits.indices
        .sortedWith(compareByDescending<Int> { exponentials[it] }.thenBy { it })
        .take(minOf(maxResults, logits.size))
        .map { category ->
            ClassificationResult(
                category = category,
                label = labels[category],
                confidence = (exponentials[category] / probabilitySum).toFloat(),
            )
        }
}
