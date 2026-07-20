package jp.axinc.ailia_kotlin

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import axip.ailia.*
import axip.ailia_tflite.AiliaTFLite
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.io.File
import java.util.EnumSet

/**
 * Integration tests for ONNX (ailia SDK) and TFLite inference.
 * Tests ObjectDetection, Classification, and Tracking with both runtimes.
 *
 * Model files are downloaded at runtime and cached in app-specific storage.
 */
@RunWith(AndroidJUnit4::class)
class OnnxTFLiteInstrumentedTest {
    companion object {
        private const val TAG = "OnnxTFLiteTest"

        // ONNX models (downloaded at runtime)
        private const val ONNX_YOLOX_URL = "https://storage.googleapis.com/ailia-models/yolox/yolox_s.opt.onnx"
        private const val ONNX_YOLOX_FILE = "yolox_s.opt.onnx"
        private const val ONNX_YOLOX_PROTO_URL = "https://storage.googleapis.com/ailia-models/yolox/yolox_s.opt.onnx.prototxt"
        private const val ONNX_YOLOX_PROTO_FILE = "yolox_s.opt.onnx.prototxt"
        private const val ONNX_MOBILENET_URL = "https://storage.googleapis.com/ailia-models/mobilenetv2/mobilenetv2_1.0.onnx"
        private const val ONNX_MOBILENET_FILE = "mobilenetv2_1.0.onnx"
        private const val ONNX_MOBILENET_PROTO_URL = "https://storage.googleapis.com/ailia-models/mobilenetv2/mobilenetv2_1.0.onnx.prototxt"
        private const val ONNX_MOBILENET_PROTO_FILE = "mobilenetv2_1.0.onnx.prototxt"

        init {
            System.loadLibrary("ailia")
        }
    }

    private val context by lazy { InstrumentationRegistry.getInstrumentation().targetContext }

    private fun modelDirectory(): File = context.filesDir

    // =====================================================================
    // Helper: Download file to context.filesDir and return file path
    // =====================================================================
    private fun downloadModel(urlStr: String, fileName: String): String {
        val dir = modelDirectory()
        val file = ModelDownloader.downloadFile(dir, ModelFileSpec(urlStr, fileName))
        assertNotNull("Model download failed: $fileName", file)
        return file!!.absolutePath
    }

    private fun initializedTFLiteObjectDetectionSample(): AiliaTFLiteObjectDetectionSample {
        val sample = AiliaTFLiteObjectDetectionSample(modelDirectory())
        assertTrue("TFLite ObjectDetection model should download", sample.downloadModel())
        assertTrue(
            "TFLite ObjectDetection should initialize",
            sample.initializeFromFile(AiliaTFLite.AILIA_TFLITE_ENV_REFERENCE),
        )
        return sample
    }

    private fun initializedTFLiteClassificationSample(): AiliaTFLiteClassificationSample {
        val sample = AiliaTFLiteClassificationSample(modelDirectory())
        assertTrue("TFLite Classification model should download", sample.downloadModel())
        assertTrue(
            "TFLite Classification should initialize",
            sample.initializeFromFile(AiliaTFLite.AILIA_TFLITE_ENV_REFERENCE),
        )
        return sample
    }

    // =====================================================================
    // Helper: Load person.jpg as Bitmap
    // =====================================================================
    private fun loadPersonBitmap(): Bitmap {
        val resId = context.resources.getIdentifier("person", "raw", context.packageName)
        assertTrue("person.jpg resource should exist", resId != 0)
        val options = BitmapFactory.Options()
        options.inScaled = false
        return BitmapFactory.decodeResource(context.resources, resId, options)
    }

    // =====================================================================
    // Helper: Convert Bitmap to RGBA ByteArray (same as ImageUtil.loadRawImage)
    // =====================================================================
    private fun bitmapToRgbaBytes(bmp: Bitmap): ByteArray {
        val w = bmp.width
        val h = bmp.height
        val pixels = IntArray(w * h)
        bmp.getPixels(pixels, 0, w, 0, 0, w, h)
        val bout = ByteArrayOutputStream()
        val out = DataOutputStream(bout)
        for (pixel in pixels) {
            out.writeByte(pixel shr 16 and 0xff) // R
            out.writeByte(pixel shr 8 and 0xff)  // G
            out.writeByte(pixel shr 0 and 0xff)  // B
            out.writeByte(pixel shr 24 and 0xff) // A
        }
        return bout.toByteArray()
    }

    // =====================================================================
    // Helper: Get ailia environment ID (GPU preferred)
    // =====================================================================
    private fun getAiliaEnvId(): Int {
        Ailia.SetTemporaryCachePath(context.cacheDir.absolutePath)
        val environments = AiliaModel.getEnvironments()
        var envId = 0
        for (env in environments) {
            Log.i(TAG, "Environment ${env.id}: type=${env.type}, name=${env.name}, props=${env.props}")
            if (env.type == AiliaEnvironment.TYPE_GPU && env.props and AiliaEnvironment.PROPERTY_FP16 == 0) {
                envId = env.id
            }
        }
        Log.i(TAG, "Using ailia envId=$envId")
        return envId
    }

    // =====================================================================
    // TFLite ObjectDetection
    // =====================================================================
    @Test
    fun testTFLiteObjectDetection() {
        Log.i(TAG, "=== Test TFLite ObjectDetection ===")

        val sample = initializedTFLiteObjectDetectionSample()

        try {
            val personBmp = loadPersonBitmap()
            val w = personBmp.width
            val h = personBmp.height

            val bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(bitmap)
            val paint = Paint().apply { style = Paint.Style.STROKE; color = Color.RED; strokeWidth = 3f }
            val textPaint = Paint().apply { color = Color.BLACK; textSize = 30f; isAntiAlias = true }

            val time = sample.processObjectDetection(personBmp, canvas, paint, textPaint, w, h)
            Log.i(TAG, "TFLite ObjectDetection time: ${time}ms")
            assertTrue("TFLite ObjectDetection should return valid time", time >= 0)

            // Also test without drawing (for tracking use case)
            val time2 = sample.processObjectDetectionWithoutDrawing(personBmp, threshold = 0.1f, iou = 1.0f)
            val results = sample.getDetectionResults()
            Log.i(TAG, "TFLite ObjectDetection (no draw) time: ${time2}ms, detections: ${results.size}")
            assertTrue("TFLite should detect at least one object in person.jpg", results.size > 0)

            // Verify person is detected (category 0 = person in COCO)
            val personDetected = results.any { it.category == 0 }
            assertTrue("TFLite should detect a person (category=0)", personDetected)
            Log.i(TAG, "TFLite ObjectDetection PASSED")
        } finally {
            sample.releaseObjectDetection()
        }
    }

    // =====================================================================
    // ONNX ObjectDetection (using ailia SDK with byte array loading)
    // =====================================================================
    @Test
    fun testOnnxObjectDetection() {
        Log.i(TAG, "=== Test ONNX ObjectDetection ===")

        val protoPath = downloadModel(ONNX_YOLOX_PROTO_URL, ONNX_YOLOX_PROTO_FILE)
        val modelPath = downloadModel(ONNX_YOLOX_URL, ONNX_YOLOX_FILE)
        val envId = getAiliaEnvId()

        Log.i(TAG, "Creating AiliaModel with protoPath=$protoPath, modelPath=$modelPath, envId=$envId")
        val ailia = AiliaModel(envId, Ailia.MULTITHREAD_AUTO, protoPath, modelPath)
        Log.i(TAG, "AiliaModel created successfully")

        val detector = AiliaDetectorModel(
            ailia.handle,
            AiliaNetworkImageFormat.RGB,
            AiliaNetworkImageChannel.FIRST,
            AiliaNetworkImageRange.UNSIGNED_INT8,
            AiliaDetectorAlgorithm.YOLOX,
            CocoLabels.CATEGORY.size,
            EnumSet.noneOf(AiliaDetectorFlags::class.java)
        )
        Log.i(TAG, "AiliaDetectorModel created successfully")

        try {
            val personBmp = loadPersonBitmap()
            val img = bitmapToRgbaBytes(personBmp)
            val w = personBmp.width
            val h = personBmp.height
            Log.i(TAG, "Image: ${w}x${h}, RGBA bytes: ${img.size}, stride: ${w * 4}")
            Log.i(TAG, "First 16 RGBA bytes: ${img.take(16).map { it.toInt() and 0xff }}")

            val startTime = System.nanoTime()
            detector.compute(img, w * 4, w, h, AiliaImageFormat.RGBA, 0.25f, 0.45f)
            val endTime = System.nanoTime()
            val time = (endTime - startTime) / 1000000

            val count = detector.objectCount
            Log.i(TAG, "ONNX ObjectDetection time: ${time}ms, detections: $count")
            assertTrue("ONNX ObjectDetection should return valid time", time >= 0)
            assertTrue("ONNX should detect at least one object", count > 0)

            var personDetected = false
            for (i in 0 until count) {
                val obj = detector.getObject(i)
                val label = if (obj.category < CocoLabels.CATEGORY.size) {
                    CocoLabels.CATEGORY[obj.category]
                } else {
                    "class${obj.category}"
                }
                Log.i(TAG, "  ONNX: $label (${obj.prob}) @ [${obj.x}, ${obj.y}, ${obj.w}, ${obj.h}]")
                if (obj.category == 0) personDetected = true
            }

            assertTrue("ONNX should detect a person (category=0)", personDetected)
            Log.i(TAG, "ONNX ObjectDetection PASSED")
        } finally {
            detector.close()
            ailia.close()
        }
    }

    // =====================================================================
    // TFLite Classification
    // =====================================================================
    @Test
    fun testTFLiteClassification() {
        Log.i(TAG, "=== Test TFLite Classification ===")

        val sample = initializedTFLiteClassificationSample()

        try {
            val personBmp = loadPersonBitmap()

            val time = sample.processClassification(personBmp)
            val result = sample.getLastClassificationResult()
            Log.i(TAG, "TFLite Classification time: ${time}ms, result: $result")
            assertTrue("TFLite Classification should return valid time", time >= 0)
            assertTrue("TFLite Classification result should not be empty", result.isNotEmpty())
            Log.i(TAG, "TFLite Classification PASSED")
        } finally {
            sample.releaseClassification()
        }
    }

    // =====================================================================
    // ONNX Classification (using ailia SDK with byte array loading)
    // =====================================================================
    @Test
    fun testOnnxClassification() {
        Log.i(TAG, "=== Test ONNX Classification ===")

        val protoPath = downloadModel(ONNX_MOBILENET_PROTO_URL, ONNX_MOBILENET_PROTO_FILE)
        val modelPath = downloadModel(ONNX_MOBILENET_URL, ONNX_MOBILENET_FILE)
        val envId = getAiliaEnvId()

        Log.i(TAG, "Creating AiliaModel with protoPath=$protoPath, modelPath=$modelPath, envId=$envId")
        val ailia = AiliaModel(envId, Ailia.MULTITHREAD_AUTO, protoPath, modelPath)
        Log.i(TAG, "AiliaModel created successfully")

        val classifier = AiliaClassifierModel(
            ailia.handle,
            AiliaNetworkImageFormat.RGB,
            AiliaNetworkImageChannel.FIRST,
            AiliaNetworkImageRange.IMAGENET
        )
        Log.i(TAG, "AiliaClassifierModel created successfully")

        try {
            val personBmp = loadPersonBitmap()
            val img = bitmapToRgbaBytes(personBmp)
            val w = personBmp.width
            val h = personBmp.height

            val startTime = System.nanoTime()
            classifier.compute(img, w * 4, w, h, AiliaImageFormat.RGBA, 5)
            val endTime = System.nanoTime()
            val time = (endTime - startTime) / 1000000

            val count = classifier.classCount
            Log.i(TAG, "ONNX Classification time: ${time}ms, classCount: $count")
            assertTrue("ONNX Classification should return valid time", time >= 0)
            assertTrue("ONNX should return at least one class", count > 0)

            val topClass = classifier.getClass(0)
            val label = if (topClass.category < ImageNetLabels.CATEGORY.size) {
                ImageNetLabels.CATEGORY[topClass.category]
            } else {
                "class${topClass.category}"
            }
            Log.i(TAG, "  ONNX Top-1: $label (category=${topClass.category}, prob=${topClass.prob})")
            assertTrue("ONNX classification probability should be positive", topClass.prob > 0)
            Log.i(TAG, "ONNX Classification PASSED")
        } finally {
            classifier.close()
            ailia.close()
        }
    }

    // =====================================================================
    // TFLite Tracking (ObjectDetection + Tracker)
    // =====================================================================
    @Test
    fun testTFLiteTracking() {
        Log.i(TAG, "=== Test TFLite Tracking ===")

        val objDetSample = initializedTFLiteObjectDetectionSample()
        val trackerSample = AiliaTrackerSample()

        val trackerInitialized = trackerSample.initializeTracker()
        assertTrue("Tracker should initialize", trackerInitialized)

        try {
            val personBmp = loadPersonBitmap()
            val w = personBmp.width
            val h = personBmp.height

            val bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(bitmap)
            val paint = Paint().apply { style = Paint.Style.STROKE; color = Color.RED; strokeWidth = 3f }

            // Run detection
            val detTime = objDetSample.processObjectDetectionWithoutDrawing(personBmp, threshold = 0.1f, iou = 1.0f)
            val detResults = objDetSample.getDetectionResults()
            Log.i(TAG, "TFLite Detection time: ${detTime}ms, detections: ${detResults.size}")
            assertTrue("TFLite should detect objects for tracking", detResults.isNotEmpty())

            // Run tracking
            val trackTime = trackerSample.processTrackingWithDetections(canvas, paint, w, h, detResults)
            val trackResult = trackerSample.getLastTrackingResult()
            Log.i(TAG, "TFLite Tracking time: ${trackTime}ms, result: $trackResult")
            assertTrue("TFLite Tracking should return valid time", trackTime >= 0)
            Log.i(TAG, "TFLite Tracking PASSED")
        } finally {
            objDetSample.releaseObjectDetection()
            trackerSample.releaseTracker()
        }
    }

    // =====================================================================
    // ONNX Tracking (ONNX ObjectDetection + Tracker)
    // =====================================================================
    @Test
    fun testOnnxTracking() {
        Log.i(TAG, "=== Test ONNX Tracking ===")

        val protoPath = downloadModel(ONNX_YOLOX_PROTO_URL, ONNX_YOLOX_PROTO_FILE)
        val modelPath = downloadModel(ONNX_YOLOX_URL, ONNX_YOLOX_FILE)
        val envId = getAiliaEnvId()

        val ailia = AiliaModel(envId, Ailia.MULTITHREAD_AUTO, protoPath, modelPath)
        val detector = AiliaDetectorModel(
            ailia.handle,
            AiliaNetworkImageFormat.RGB,
            AiliaNetworkImageChannel.FIRST,
            AiliaNetworkImageRange.UNSIGNED_INT8,
            AiliaDetectorAlgorithm.YOLOX,
            CocoLabels.CATEGORY.size,
            EnumSet.noneOf(AiliaDetectorFlags::class.java)
        )
        val trackerSample = AiliaTrackerSample()
        val trackerInitialized = trackerSample.initializeTracker()
        assertTrue("Tracker should initialize", trackerInitialized)

        try {
            val personBmp = loadPersonBitmap()
            val img = bitmapToRgbaBytes(personBmp)
            val w = personBmp.width
            val h = personBmp.height

            // Run ONNX detection
            val startTime = System.nanoTime()
            detector.compute(img, w * 4, w, h, AiliaImageFormat.RGBA, 0.1f, 1.0f)
            val detTime = (System.nanoTime() - startTime) / 1000000

            val count = detector.objectCount
            val detResults = mutableListOf<DetectionResult>()
            for (i in 0 until count) {
                val obj = detector.getObject(i)
                detResults.add(
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
            Log.i(TAG, "ONNX Detection time: ${detTime}ms, detections: ${detResults.size}")
            assertTrue("ONNX should detect objects for tracking", detResults.isNotEmpty())

            // Run tracking
            val bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(bitmap)
            val paint = Paint().apply { style = Paint.Style.STROKE; color = Color.RED; strokeWidth = 3f }

            val trackTime = trackerSample.processTrackingWithDetections(canvas, paint, w, h, detResults)
            val trackResult = trackerSample.getLastTrackingResult()
            Log.i(TAG, "ONNX Tracking time: ${trackTime}ms, result: $trackResult")
            assertTrue("ONNX Tracking should return valid time", trackTime >= 0)
            Log.i(TAG, "ONNX Tracking PASSED")
        } finally {
            detector.close()
            ailia.close()
            trackerSample.releaseTracker()
        }
    }

    // =====================================================================
    // Cross-runtime comparison: Both should detect person in person.jpg
    // =====================================================================
    @Test
    fun testObjectDetectionCrossRuntimeComparison() {
        Log.i(TAG, "=== Test ObjectDetection Cross-Runtime Comparison ===")

        // --- TFLite ---
        val tfliteSample = initializedTFLiteObjectDetectionSample()

        // --- ONNX ---
        val protoPath = downloadModel(ONNX_YOLOX_PROTO_URL, ONNX_YOLOX_PROTO_FILE)
        val modelPath = downloadModel(ONNX_YOLOX_URL, ONNX_YOLOX_FILE)
        val envId = getAiliaEnvId()
        val ailiaModel = AiliaModel(envId, Ailia.MULTITHREAD_AUTO, protoPath, modelPath)
        val onnxDetector = AiliaDetectorModel(
            ailiaModel.handle,
            AiliaNetworkImageFormat.RGB,
            AiliaNetworkImageChannel.FIRST,
            AiliaNetworkImageRange.UNSIGNED_INT8,
            AiliaDetectorAlgorithm.YOLOX,
            CocoLabels.CATEGORY.size,
            EnumSet.noneOf(AiliaDetectorFlags::class.java)
        )

        try {
            val personBmp = loadPersonBitmap()
            val img = bitmapToRgbaBytes(personBmp)
            val w = personBmp.width
            val h = personBmp.height

            // TFLite detection
            val tfliteTime = tfliteSample.processObjectDetectionWithoutDrawing(personBmp, threshold = 0.25f, iou = 0.45f)
            val tfliteResults = tfliteSample.getDetectionResults()
            Log.i(TAG, "TFLite: ${tfliteResults.size} detections, ${tfliteTime}ms")
            for (r in tfliteResults) {
                val label = if (r.category < CocoLabels.CATEGORY.size) CocoLabels.CATEGORY[r.category] else "class${r.category}"
                Log.i(TAG, "  TFLite: $label (${r.confidence}) @ [${r.x}, ${r.y}, ${r.width}, ${r.height}]")
            }

            // ONNX detection
            val startTime = System.nanoTime()
            onnxDetector.compute(img, w * 4, w, h, AiliaImageFormat.RGBA, 0.25f, 0.45f)
            val onnxTime = (System.nanoTime() - startTime) / 1000000
            val onnxCount = onnxDetector.objectCount
            Log.i(TAG, "ONNX: $onnxCount detections, ${onnxTime}ms")
            var onnxPersonDetected = false
            for (i in 0 until onnxCount) {
                val obj = onnxDetector.getObject(i)
                val label = if (obj.category < CocoLabels.CATEGORY.size) CocoLabels.CATEGORY[obj.category] else "class${obj.category}"
                Log.i(TAG, "  ONNX: $label (${obj.prob}) @ [${obj.x}, ${obj.y}, ${obj.w}, ${obj.h}]")
                if (obj.category == 0) onnxPersonDetected = true
            }

            // Both should detect person
            val tflitePerson = tfliteResults.any { it.category == 0 }
            assertTrue("TFLite should detect person", tflitePerson)
            assertTrue("ONNX should detect person", onnxPersonDetected)

            Log.i(TAG, "Cross-runtime ObjectDetection comparison PASSED")
        } finally {
            tfliteSample.releaseObjectDetection()
            onnxDetector.close()
            ailiaModel.close()
        }
    }

    // =====================================================================
    // Cross-runtime comparison: Classification
    // =====================================================================
    @Test
    fun testClassificationCrossRuntimeComparison() {
        Log.i(TAG, "=== Test Classification Cross-Runtime Comparison ===")

        // --- TFLite ---
        val tfliteSample = initializedTFLiteClassificationSample()

        // --- ONNX ---
        val protoPath = downloadModel(ONNX_MOBILENET_PROTO_URL, ONNX_MOBILENET_PROTO_FILE)
        val modelPath = downloadModel(ONNX_MOBILENET_URL, ONNX_MOBILENET_FILE)
        val envId = getAiliaEnvId()
        val ailiaModel = AiliaModel(envId, Ailia.MULTITHREAD_AUTO, protoPath, modelPath)
        val onnxClassifier = AiliaClassifierModel(
            ailiaModel.handle,
            AiliaNetworkImageFormat.RGB,
            AiliaNetworkImageChannel.FIRST,
            AiliaNetworkImageRange.IMAGENET
        )

        try {
            val personBmp = loadPersonBitmap()
            val img = bitmapToRgbaBytes(personBmp)
            val w = personBmp.width
            val h = personBmp.height

            // TFLite classification
            val tfliteTime = tfliteSample.processClassification(personBmp)
            val tfliteResult = tfliteSample.getLastClassificationResult()
            Log.i(TAG, "TFLite Classification: $tfliteResult (${tfliteTime}ms)")

            // ONNX classification
            val startTime = System.nanoTime()
            onnxClassifier.compute(img, w * 4, w, h, AiliaImageFormat.RGBA, 5)
            val onnxTime = (System.nanoTime() - startTime) / 1000000
            val onnxCount = onnxClassifier.classCount
            assertTrue("ONNX should return classes", onnxCount > 0)
            val topClass = onnxClassifier.getClass(0)
            val onnxLabel = if (topClass.category < ImageNetLabels.CATEGORY.size) {
                ImageNetLabels.CATEGORY[topClass.category]
            } else {
                "class${topClass.category}"
            }
            Log.i(TAG, "ONNX Classification: $onnxLabel (prob=${topClass.prob}) (${onnxTime}ms)")

            // Both should return valid results
            assertTrue("TFLite result should not be empty", tfliteResult.isNotEmpty())
            assertTrue("ONNX should have positive probability", topClass.prob > 0)

            Log.i(TAG, "Cross-runtime Classification comparison PASSED")
        } finally {
            tfliteSample.releaseClassification()
            onnxClassifier.close()
            ailiaModel.close()
        }
    }
}
