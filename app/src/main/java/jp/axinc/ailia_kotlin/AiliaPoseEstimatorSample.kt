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
            
            if (objCount != 0) {
                val pose = poseEstimator!!.getObjectPose(0)
                Log.i("AILIA_Main", "total score = " + pose.totalScore)
                Log.i(
                    "AILIA_Main",
                    "angle[3] = {" + pose.angle[0] + ", " + pose.angle[1] + ", " + pose.angle[2] + "}"
                )
                for (i in 0 until AiliaPoseEstimatorObjectPose.KEYPOINT_COUNT) {
                    val p = pose.points[i]
                    Log.i(
                        "AILIA_Main",
                        "keypoint[" + i + "] = {x: " + p.x + ", y: " + p.y + ", z_local: " + p.z_local + ", score: " + p.score + ", interpolated: " + p.interpolated + "}"
                    )

                    var r:Float = 8.0f
                    canvas.drawCircle(
                        (w * p.x).toFloat(),
                        (h * p.y).toFloat(),
                        r,
                        paint
                    )
                }
            } else {
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
