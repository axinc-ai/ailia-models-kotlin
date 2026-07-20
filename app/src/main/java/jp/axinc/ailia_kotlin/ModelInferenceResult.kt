package jp.axinc.ailia_kotlin

/** A typed inference output and the model execution time, excluding rendering. */
data class ModelInferenceResult<T>(
    val value: T,
    val processingTimeMs: Long,
)

/** Row-major foreground mask returned by an image-segmentation model. */
data class SegmentationMask(
    val width: Int,
    val height: Int,
    val values: FloatArray,
) {
    init {
        require(width > 0 && height > 0 && values.size == width * height)
    }
}
