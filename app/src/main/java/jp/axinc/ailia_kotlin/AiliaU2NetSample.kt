package jp.axinc.ailia_kotlin

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Rect
import android.util.Log
import axip.ailia.Ailia
import axip.ailia.AiliaModel
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

/**
 * U-2-Netによる背景除去サンプル(ailia-models-flutterのu2net.dartを移植)。
 * 320x320にリサイズしてImageNet正規化した画像を入力し、
 * 出力マスクを黒の半透明オーバーレイとして元画像に重ねる。
 */
class AiliaU2NetSample {
    companion object {
        private const val TAG = "AILIA_Main"
        private const val MODEL_URL = "https://storage.googleapis.com/ailia-models/u2net/u2net_opset11.onnx"
        private const val MODEL_FILE = "u2net_opset11.onnx"
        private const val PROTO_URL = "https://storage.googleapis.com/ailia-models/u2net/u2net_opset11.onnx.prototxt"
        private const val PROTO_FILE = "u2net_opset11.onnx.prototxt"
        private const val IMAGE_SIZE = 320

        // ImageNet正規化(Flutter版imageToAiliaTensorと同じ)
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    interface DownloadListener {
        fun onProgress(fileName: String, bytesDownloaded: Long, totalBytes: Long)
        fun onComplete()
        fun onError(error: String)
    }

    private var ailia: AiliaModel? = null
    private var isInitialized = false
    var modelDir: String = ""

    private fun downloadFile(urlStr: String, fileName: String, listener: DownloadListener? = null): Boolean {
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

    fun downloadModel(listener: DownloadListener? = null): Boolean {
        return try {
            Log.i(TAG, "Starting U2Net model download/check...")
            downloadFile(PROTO_URL, PROTO_FILE, listener)
            downloadFile(MODEL_URL, MODEL_FILE, listener)
            listener?.onComplete()
            true
        } catch (e: Exception) {
            Log.e(TAG, "U2Net model download failed", e)
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
            isInitialized = true
            Log.i(TAG, "U2Net initialized successfully with envId=$envId")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize U2Net: ${e.javaClass.name}: ${e.message}")
            release()
            false
        }
    }

    /**
     * 背景除去を実行し、背景部分を黒の半透明オーバーレイとしてcanvasに描画する。
     * canvasには元画像が描画済みであること。
     */
    fun process(bitmap: Bitmap, canvas: Canvas, w: Int, h: Int): Long {
        val model = ailia
        if (!isInitialized || model == null) {
            Log.e(TAG, "U2Net not initialized")
            return -1
        }

        return try {
            // 前処理: 320x320リサイズ + ImageNet正規化(RGB, CHW)
            val scaled = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)
            val pixels = IntArray(IMAGE_SIZE * IMAGE_SIZE)
            scaled.getPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)
            val plane = IMAGE_SIZE * IMAGE_SIZE
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
            model.update()

            // 出力d0(マスク, 1x1x320x320)を取得
            val outIdx = model.getBlobIndexByOutputIndex(0)
            val mask = FloatArray(plane)
            model.getBlobData(mask, mask.size * 4, outIdx)
            val endTime = System.nanoTime()

            // マスクをアルファに変換した黒オーバーレイ(Flutter版のreverse表示と同じ)
            val overlayPixels = IntArray(plane)
            for (i in 0 until plane) {
                val m = mask[i].coerceIn(0f, 1f)
                val alpha = ((1.0f - m) * 255).toInt()
                overlayPixels[i] = Color.argb(alpha, 0, 0, 0)
            }
            val overlay = Bitmap.createBitmap(overlayPixels, IMAGE_SIZE, IMAGE_SIZE, Bitmap.Config.ARGB_8888)
            canvas.drawBitmap(overlay, null, Rect(0, 0, w, h), null)

            (endTime - startTime) / 1000000
        } catch (e: Exception) {
            Log.e(TAG, "Failed to process U2Net: ${e.javaClass.name}: ${e.message}")
            -1
        }
    }

    fun release() {
        try {
            ailia?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing U2Net: ${e.message}")
        } finally {
            ailia = null
            isInitialized = false
            Log.i(TAG, "U2Net released")
        }
    }
}
