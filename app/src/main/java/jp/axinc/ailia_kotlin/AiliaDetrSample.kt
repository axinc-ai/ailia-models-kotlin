package jp.axinc.ailia_kotlin

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.util.Log
import axip.ailia.Ailia
import axip.ailia.AiliaModel
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import kotlin.math.exp

/**
 * DETR (DEtection TRansformer, ResNet50)による物体検出サンプル。
 * ailia-models(Python版 object_detection/detr)を移植:
 * ImageNet正規化(RGB, CHW)入力、出力logits/boxesをsoftmax+閾値0.7で検出に変換する。
 */
class AiliaDetrSample {
    companion object {
        private const val TAG = "AILIA_Main"
        private const val MODEL_URL = "https://storage.googleapis.com/ailia-models/detr/detr-r50-e632da11.onnx"
        private const val MODEL_FILE = "detr-r50-e632da11.onnx"
        private const val PROTO_URL = "https://storage.googleapis.com/ailia-models/detr/detr-r50-e632da11.onnx.prototxt"
        private const val PROTO_FILE = "detr-r50-e632da11.onnx.prototxt"

        // 入力サイズ(Python版は短辺800。本サンプルは正方形入力のため800x800)
        private const val INPUT_SIZE = 800
        private const val THRESHOLD = 0.7f

        // ImageNet正規化
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)

        // DETRはCOCOの91クラスID(欠番は'N/A')+末尾にno-objectクラスを持つ
        val DETR_CATEGORY = arrayOf(
            "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
            "train", "truck", "boat", "traffic light", "fire hydrant", "N/A",
            "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
            "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack",
            "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis",
            "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass",
            "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
            "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A",
            "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
            "N/A", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        )
    }

    private var ailia: AiliaModel? = null
    private var isInitialized = false
    var modelDir: String = ""

    private fun downloadFile(urlStr: String, fileName: String, listener: ModelDownloadListener? = null): Boolean {
        val path = "$modelDir/$fileName"
        val file = File(path)
        if (file.exists() && file.canRead()) {
            Log.i(TAG, "Model file already exists: $path (${file.length()} bytes)")
            return true
        }
        file.parentFile?.mkdirs()
        val tmpFile = File("$path.tmp")
        val connection = URL(urlStr).openConnection() as HttpURLConnection
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
        tmpFile.renameTo(file)
        return true
    }

    fun downloadModel(listener: ModelDownloadListener? = null): Boolean {
        return try {
            Log.i(TAG, "Starting DETR model download/check...")
            downloadFile(PROTO_URL, PROTO_FILE, listener)
            downloadFile(MODEL_URL, MODEL_FILE, listener)
            listener?.onComplete()
            true
        } catch (e: Exception) {
            Log.e(TAG, "DETR model download failed", e)
            listener?.onError(e.message ?: "Download failed")
            false
        }
    }

    fun initialize(envId: Int): Boolean {
        if (isInitialized) {
            release()
        }
        return try {
            ailia = AiliaModel(envId, Ailia.MULTITHREAD_AUTO, "$modelDir/$PROTO_FILE", "$modelDir/$MODEL_FILE")
            try {
                ailia!!.setInputShapeND(intArrayOf(1, 3, INPUT_SIZE, INPUT_SIZE))
                // DETRのONNXは2番目の入力としてパディングマスク(1,H,W)を持つ
                if (ailia!!.inputBlobCount >= 2) {
                    val maskIdx = ailia!!.getBlobIndexByInputIndex(1)
                    ailia!!.setInputBlobShapeND(intArrayOf(1, INPUT_SIZE, INPUT_SIZE), maskIdx)
                }
            } catch (e: Exception) {
                Log.w(TAG, "DETR: input shape setup failed, using model default shape: ${e.message}")
            }
            isInitialized = true
            Log.i(TAG, "DETR initialized successfully with envId=$envId")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize DETR: ${e.javaClass.name}: ${e.message}")
            release()
            false
        }
    }

    /** 物体検出を実行し、検出枠とラベルをcanvasに描画する */
    fun processObjectDetection(bitmap: Bitmap, canvas: Canvas, paint: Paint, text: Paint, w: Int, h: Int): Long {
        val model = ailia
        if (!isInitialized || model == null) {
            Log.e(TAG, "DETR not initialized")
            return -1
        }

        return try {
            // 前処理: 800x800リサイズ + ImageNet正規化(RGB, CHW)
            val scaled = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
            val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
            scaled.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
            val plane = INPUT_SIZE * INPUT_SIZE
            val input = FloatArray(3 * plane)
            for (i in 0 until plane) {
                val p = pixels[i]
                val r = ((p shr 16) and 0xFF) / 255.0f
                val g = ((p shr 8) and 0xFF) / 255.0f
                val b = (p and 0xFF) / 255.0f
                input[i] = (r - MEAN[0]) / STD[0]
                input[plane + i] = (g - MEAN[1]) / STD[1]
                input[2 * plane + i] = (b - MEAN[2]) / STD[2]
            }

            val startTime = System.nanoTime()
            model.setInputBlobData(input, input.size * 4, model.getBlobIndexByInputIndex(0))
            // 2番目の入力(パディングマスク)には全ゼロ(=パディングなし)を与える
            if (model.inputBlobCount >= 2) {
                val maskInput = FloatArray(plane)
                model.setInputBlobData(maskInput, maskInput.size * 4, model.getBlobIndexByInputIndex(1))
            }
            model.update()

            // 出力はlogits(1,100,92)とboxes(1,100,4)。最終次元で判別する
            var logits: FloatArray? = null
            var boxes: FloatArray? = null
            var numQueries = 0
            var numClasses = 0
            for (outIdx in 0 until model.outputBlobCount) {
                val blobIndex = model.getBlobIndexByOutputIndex(outIdx)
                val shape = model.getBlobShapeND(blobIndex)
                val size = shape.fold(1) { acc, v -> acc * v }
                val data = FloatArray(size)
                model.getBlobData(data, size * 4, blobIndex)
                if (shape.last() == 4) {
                    boxes = data
                } else {
                    logits = data
                    numQueries = shape[shape.size - 2]
                    numClasses = shape.last()
                }
            }
            val endTime = System.nanoTime()

            if (logits == null || boxes == null) {
                Log.e(TAG, "DETR: unexpected outputs")
                return -1
            }

            // 後処理: softmax → score = 1 - P(no-object) → 閾値0.7
            for (q in 0 until numQueries) {
                val offset = q * numClasses
                var maxLogit = Float.NEGATIVE_INFINITY
                for (c in 0 until numClasses) {
                    if (logits[offset + c] > maxLogit) maxLogit = logits[offset + c]
                }
                var sum = 0.0f
                val probs = FloatArray(numClasses)
                for (c in 0 until numClasses) {
                    probs[c] = exp((logits[offset + c] - maxLogit).toDouble()).toFloat()
                    sum += probs[c]
                }
                for (c in 0 until numClasses) probs[c] /= sum

                val score = 1.0f - probs[numClasses - 1]
                if (score <= THRESHOLD) continue

                var label = 0
                var maxProb = 0.0f
                for (c in 0 until numClasses - 1) {
                    if (probs[c] > maxProb) {
                        maxProb = probs[c]
                        label = c
                    }
                }

                // box: cxcywh(正規化) → xyxy(ピクセル)
                val cx = boxes[q * 4 + 0]
                val cy = boxes[q * 4 + 1]
                val bw = boxes[q * 4 + 2]
                val bh = boxes[q * 4 + 3]
                val x1 = ((cx - bw / 2).coerceIn(0f, 1f)) * w
                val y1 = ((cy - bh / 2).coerceIn(0f, 1f)) * h
                val x2 = ((cx + bw / 2).coerceIn(0f, 1f)) * w
                val y2 = ((cy + bh / 2).coerceIn(0f, 1f)) * h

                val catColor = CocoAndImageNetLabels.categoryColor(label, DETR_CATEGORY.size)
                val boxPaint = Paint(paint).apply { color = catColor }
                canvas.drawRect(x1, y1, x2, y2, boxPaint)

                val name = if (label < DETR_CATEGORY.size) DETR_CATEGORY[label] else "class$label"
                val labelText = "$name ${String.format("%.2f", maxProb)}"
                val bgPaint = Paint().apply { style = Paint.Style.FILL; color = catColor }
                canvas.drawRect(x1, y1 - text.textSize, x1 + text.measureText(labelText), y1 + text.textSize * 0.2f, bgPaint)
                val labelPaint = Paint(text).apply { color = android.graphics.Color.WHITE }
                canvas.drawText(labelText, x1, y1, labelPaint)

                Log.i(TAG, "DETR: $name score=$score box=($x1,$y1)-($x2,$y2)")
            }

            (endTime - startTime) / 1000000
        } catch (e: Exception) {
            Log.e(TAG, "Failed to process DETR: ${e.javaClass.name}: ${e.message}")
            -1
        }
    }

    fun release() {
        try {
            ailia?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing DETR: ${e.message}")
        } finally {
            ailia = null
            isInitialized = false
            Log.i(TAG, "DETR released")
        }
    }
}
