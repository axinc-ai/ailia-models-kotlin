package jp.axinc.ailia_kotlin

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Log
import android.widget.ImageView
import axip.ailia.Ailia
import axip.ailia.AiliaEnvironment
import axip.ailia.AiliaImageFormat
import axip.ailia.AiliaModel
import axip.ailia.AiliaPoseEstimatorAlgorithm
import axip.ailia.AiliaPoseEstimatorModel
import axip.ailia.AiliaPoseEstimatorObjectPose
import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.io.IOException
import java.nio.ByteBuffer

class AiliaPoseEstimatorSample {
    private var ailia: AiliaModel? = null
    private var poseEstimator: AiliaPoseEstimatorModel? = null
    private var isInitialized = false

    companion object {
        // Python版(lightweight-human-pose-estimation.py)と同じ閾値。
        // スコアがこれ以下のキーポイント/ラインは描画しない
        private const val KEYPOINT_THRESHOLD = 0.3f

        // Python版 display_result() と同じキーポイントの接続
        private val LINE_PAIRS = arrayOf(
            intArrayOf(AiliaPoseEstimatorObjectPose.KEYPOINT_NOSE, AiliaPoseEstimatorObjectPose.KEYPOINT_SHOULDER_CENTER),
            intArrayOf(AiliaPoseEstimatorObjectPose.KEYPOINT_SHOULDER_LEFT, AiliaPoseEstimatorObjectPose.KEYPOINT_SHOULDER_CENTER),
            intArrayOf(AiliaPoseEstimatorObjectPose.KEYPOINT_SHOULDER_RIGHT, AiliaPoseEstimatorObjectPose.KEYPOINT_SHOULDER_CENTER),
            intArrayOf(AiliaPoseEstimatorObjectPose.KEYPOINT_EYE_LEFT, AiliaPoseEstimatorObjectPose.KEYPOINT_NOSE),
            intArrayOf(AiliaPoseEstimatorObjectPose.KEYPOINT_EYE_RIGHT, AiliaPoseEstimatorObjectPose.KEYPOINT_NOSE),
            intArrayOf(AiliaPoseEstimatorObjectPose.KEYPOINT_EAR_LEFT, AiliaPoseEstimatorObjectPose.KEYPOINT_EYE_LEFT),
            intArrayOf(AiliaPoseEstimatorObjectPose.KEYPOINT_EAR_RIGHT, AiliaPoseEstimatorObjectPose.KEYPOINT_EYE_RIGHT),
            intArrayOf(AiliaPoseEstimatorObjectPose.KEYPOINT_ELBOW_LEFT, AiliaPoseEstimatorObjectPose.KEYPOINT_SHOULDER_LEFT),
            intArrayOf(AiliaPoseEstimatorObjectPose.KEYPOINT_ELBOW_RIGHT, AiliaPoseEstimatorObjectPose.KEYPOINT_SHOULDER_RIGHT),
            intArrayOf(AiliaPoseEstimatorObjectPose.KEYPOINT_WRIST_LEFT, AiliaPoseEstimatorObjectPose.KEYPOINT_ELBOW_LEFT),
            intArrayOf(AiliaPoseEstimatorObjectPose.KEYPOINT_WRIST_RIGHT, AiliaPoseEstimatorObjectPose.KEYPOINT_ELBOW_RIGHT),
            intArrayOf(AiliaPoseEstimatorObjectPose.KEYPOINT_BODY_CENTER, AiliaPoseEstimatorObjectPose.KEYPOINT_SHOULDER_CENTER),
            intArrayOf(AiliaPoseEstimatorObjectPose.KEYPOINT_HIP_LEFT, AiliaPoseEstimatorObjectPose.KEYPOINT_BODY_CENTER),
            intArrayOf(AiliaPoseEstimatorObjectPose.KEYPOINT_HIP_RIGHT, AiliaPoseEstimatorObjectPose.KEYPOINT_BODY_CENTER),
            intArrayOf(AiliaPoseEstimatorObjectPose.KEYPOINT_KNEE_LEFT, AiliaPoseEstimatorObjectPose.KEYPOINT_HIP_LEFT),
            intArrayOf(AiliaPoseEstimatorObjectPose.KEYPOINT_ANKLE_LEFT, AiliaPoseEstimatorObjectPose.KEYPOINT_KNEE_LEFT),
            intArrayOf(AiliaPoseEstimatorObjectPose.KEYPOINT_KNEE_RIGHT, AiliaPoseEstimatorObjectPose.KEYPOINT_HIP_RIGHT),
            intArrayOf(AiliaPoseEstimatorObjectPose.KEYPOINT_ANKLE_RIGHT, AiliaPoseEstimatorObjectPose.KEYPOINT_KNEE_RIGHT),
        )

        /**
         * Python版の hsv_to_rgb(255*point1/KEYPOINT_CNT, 255, 255) と同じ色。
         * OpenCVのHSV(H:0-179)相当の値から色相(0-360°)に変換する。
         */
        private fun lineColor(point1: Int): Int {
            val hCv = 255f * point1 / AiliaPoseEstimatorObjectPose.KEYPOINT_COUNT
            val hueDeg = (hCv % 180f) * 2f
            return Color.HSVToColor(floatArrayOf(hueDeg, 1f, 1f))
        }
    }

    fun ailia_environment(cache_dir: String) : AiliaEnvironment {
        Ailia.SetTemporaryCachePath(cache_dir)
        val envList = AiliaModel.getEnvironments()
        var selectedEnv = envList[0]
        for (env in envList) {
            Log.i(
                "AILIA_Main",
                "Environment " + env.id + ": ( type: " + env.type + ", name: " + env.name + ")"
            )
            if (env.type == AiliaEnvironment.TYPE_GPU && env.props and AiliaEnvironment.PROPERTY_FP16 == 0) {
                selectedEnv = env
                break
            }
        }
        Log.i(
            "AILIA_Main",
            "Selected environment id: " + selectedEnv.id + " (" + selectedEnv.name + ")"
        )
        return selectedEnv;
    }

    fun initializePoseEstimator(envId: Int, proto: ByteArray?, model: ByteArray?): Boolean {
        return try {
            if (isInitialized) {
                releasePoseEstimator()
            }

            Log.i("AILIA_Main", "Pose estimator: initializing with envId=$envId")

            ailia = AiliaModel(
                envId,
                Ailia.MULTITHREAD_AUTO,
                proto,
                model
            )
            poseEstimator = AiliaPoseEstimatorModel(ailia!!.handle, AiliaPoseEstimatorAlgorithm.LW_HUMAN_POSE)
            isInitialized = true
            Log.i("AILIA_Main", "Pose estimator initialized successfully")
            true
        } catch (e: Exception) {
            Log.e("AILIA_Error", "Failed to initialize pose estimator: ${e.javaClass.name}: ${e.message}")
            releasePoseEstimator()
            false
        }
    }

    fun processPoseEstimation(img: ByteArray, canvas: Canvas, paint: Paint, w: Int, h: Int): Long {
        if (!isInitialized || poseEstimator == null) {
            Log.e("AILIA_Error", "Pose estimator not initialized")
            return -1
        }

        return try {
            val startTime = System.nanoTime()
            
            poseEstimator!!.compute(img, w * 4, w, h, AiliaImageFormat.RGBA)
            val objCount = poseEstimator!!.objectCount
            Log.i("AILIA_Main", "objCount (human count) = $objCount")

            val linePaint = Paint().apply {
                strokeWidth = 5.0f
                isAntiAlias = true
                strokeCap = Paint.Cap.ROUND
            }

            for (objIdx in 0 until objCount) {
                val pose = poseEstimator!!.getObjectPose(objIdx)
                Log.i("AILIA_Main", "person[$objIdx] total score = " + pose.totalScore)

                // キーポイント間のライン(Python版と同じ接続・HSVカラー・閾値0.3)
                for (pair in LINE_PAIRS) {
                    val p1 = pose.points[pair[0]]
                    val p2 = pose.points[pair[1]]
                    if (p1.score > KEYPOINT_THRESHOLD && p2.score > KEYPOINT_THRESHOLD) {
                        linePaint.color = lineColor(pair[0])
                        canvas.drawLine(
                            (w * p1.x).toFloat(), (h * p1.y).toFloat(),
                            (w * p2.x).toFloat(), (h * p2.y).toFloat(),
                            linePaint
                        )
                    }
                }

                // キーポイント(スコアの低いものは非表示)
                for (i in 0 until AiliaPoseEstimatorObjectPose.KEYPOINT_COUNT) {
                    val p = pose.points[i]
                    if (p.score <= KEYPOINT_THRESHOLD) {
                        continue
                    }
                    canvas.drawCircle(
                        (w * p.x).toFloat(),
                        (h * p.y).toFloat(),
                        8.0f,
                        paint
                    )
                }
            }
            if (objCount == 0) {
                Log.i("AILIA_Main", "No object detected.")
            }

            val endTime = System.nanoTime()
            return (endTime - startTime) / 1000000
        } catch (e: Exception) {
            Log.e("AILIA_Error", "Failed to process pose estimation: ${e.javaClass.name}: ${e.message}")
            -1
        }
    }

    fun releasePoseEstimator() {
        try {
            poseEstimator?.close()
            ailia?.close()
        } catch (e: Exception) {
            Log.e("AILIA_Error", "Error releasing pose estimator: ${e.javaClass.name}: ${e.message}")
        } finally {
            poseEstimator = null
            ailia = null
            isInitialized = false
            Log.i("AILIA_Main", "Pose estimator released")
        }
    }
}
