package jp.axinc.ailia_kotlin

/** SDK-neutral normalized detection result shared by detectors and trackers. */
data class DetectionResult(
    val category: Int,
    val confidence: Float,
    val x: Float,
    val y: Float,
    val width: Float,
    val height: Float,
)
