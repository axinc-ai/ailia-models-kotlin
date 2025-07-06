package jp.axinc.ailia_kotlin

import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PointF
import android.util.Log
import axip.ailia_tracker.*

class AiliaTrackerSample {
    companion object {
        private const val TAG = "AILIA_Main"
    }

    private var tracker: AiliaTracker? = null
    private var isInitialized = false
    private var lastTrackingResult: String = ""
    private val trajectoryPoints = mutableMapOf<Int, MutableList<PointF>>()
    private val maxTrajectoryPoints = 50

    fun initializeTracker(): Boolean {
        return try {
            if (isInitialized) {
                releaseTracker()
            }

            var settings: AiliaTrackerSettings = AiliaTrackerSettings(score_threshold = 0.1f, nms_threshold = 0.7f, track_threshold = 0.5f, track_buffer = 30, match_threshold = 0.8f)
            tracker = AiliaTracker(algorithm = AiliaTracker.AILIA_TRACKER_ALGORITHM_BYTE_TRACK, settings = settings)
            isInitialized = true
            Log.i(TAG, "Tracker initialized successfully")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize tracker: ${e.javaClass.name}: ${e.message}")
            releaseTracker()
            false
        }
    }

     fun releaseTracker() {
        try {
            tracker?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing tracker: ${e.javaClass.name}: ${e.message}")
        } finally {
            tracker = null
            isInitialized = false
            trajectoryPoints.clear()
            Log.i(TAG, "Tracker released")
        }
    }
    
    fun processTrackingWithDetections(canvas: Canvas, paint: Paint, w: Int, h: Int, detectionResults: List<DetectionResult>): Long {
        if (!isInitialized || tracker == null) {
            Log.e(TAG, "Tracker not initialized")
            return -1
        }

        return try {
            val startTime = System.nanoTime()
            
            for (detection in detectionResults) {
                val result = tracker!!.addTarget(
                    category = detection.category,
                    prob = detection.confidence,
                    x = detection.x,
                    y = detection.y,
                    w = detection.width,
                    h = detection.height
                )
                
                if (result != 0) {
                    Log.e(TAG, "Failed to add target: $result")
                }
            }

            val computeResult = tracker!!.compute()
            if (computeResult == 0) {
                Log.d(TAG, "Successfully computed tracking")

                val objectCount = tracker!!.getObjectCount()
                Log.d(TAG, "Object count: $objectCount")

                var trackingInfo = "Objects: $objectCount"
                for (i in 0 until objectCount) {
                    val obj = tracker!!.getObject(i)
                    obj?.let {
                        Log.d(TAG, "Object $i: id=${it.id}, category=${it.category}, prob=${it.prob}, x=${it.x}, y=${it.y}, w=${it.w}, h=${it.h}")
                        trackingInfo += "\nID:${it.id} Cat:${it.category} Conf:${String.format("%.2f", it.prob)}"
                        
                        // Calculate center point of bounding box
                        val centerX = w * (it.x + it.w / 2)
                        val centerY = h * (it.y + it.h / 2)
                        val centerPoint = PointF(centerX, centerY)
                        
                        // Update trajectory points for this ID
                        val trajectoryList = trajectoryPoints.getOrPut(it.id) { mutableListOf() }
                        trajectoryList.add(centerPoint)
                        
                        // Limit trajectory points to prevent memory issues
                        if (trajectoryList.size > maxTrajectoryPoints) {
                            trajectoryList.removeAt(0)
                        }
                        
                        // Generate color based on ID
                        val trajectoryColor = generateColorForId(it.id)
                        
                        // Draw trajectory lines
                        if (trajectoryList.size > 1) {
                            val trajectoryPaint = Paint().apply {
                                color = trajectoryColor
                                strokeWidth = 3f
                                style = Paint.Style.STROKE
                                isAntiAlias = true
                            }
                            
                            for (j in 1 until trajectoryList.size) {
                                val prevPoint = trajectoryList[j - 1]
                                val currPoint = trajectoryList[j]
                                canvas.drawLine(
                                    prevPoint.x, prevPoint.y,
                                    currPoint.x, currPoint.y,
                                    trajectoryPaint
                                )
                            }
                        }
                        
                        // Draw bounding box
                        canvas.drawRect(
                            (w * it.x).toFloat(),
                            (h * it.y).toFloat(),
                            (w * (it.x + it.w)).toFloat(),
                            (h * (it.y + it.h)).toFloat(),
                            paint
                        )
                        
                        // Draw tracking ID
                        val textPaint = Paint().apply {
                            color = android.graphics.Color.WHITE
                            textSize = 24f
                            isAntiAlias = true
                        }
                        canvas.drawText("ID:${it.id}", (w * it.x).toFloat(), (h * it.y).toFloat() - 5, textPaint)
                    }
                }
                lastTrackingResult = trackingInfo
            } else {
                Log.e(TAG, "Failed to compute tracking: $computeResult")
                lastTrackingResult = "Tracking failed"
                return -1
            }

            val endTime = System.nanoTime()
            (endTime - startTime) / 1000000
        } catch (e: Exception) {
            Log.e(TAG, "Failed to process tracking: ${e.javaClass.name}: ${e.message}")
            lastTrackingResult = "Error: ${e.message}"
            -1
        }
    }
    
    fun getLastTrackingResult(): String {
        return lastTrackingResult
    }
    
    private fun generateColorForId(id: Int): Int {
        // Generate a unique color for each ID using HSV color space
        val hue = (id * 137.508f) % 360f // Golden angle approximation for good color distribution
        val saturation = 0.8f
        val value = 0.9f
        return Color.HSVToColor(floatArrayOf(hue, saturation, value))
    }
    
    data class DetectionResult(
        val category: Int,
        val confidence: Float,
        val x: Float,
        val y: Float,
        val width: Float,
        val height: Float
    )

}
