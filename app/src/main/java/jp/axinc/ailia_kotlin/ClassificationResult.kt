package jp.axinc.ailia_kotlin

import java.util.Locale

/** SDK-neutral classification output that can be copied with either runtime sample. */
data class ClassificationResult(
    val category: Int,
    val label: String,
    val confidence: Float,
) {
    fun displayText(): String = "$label (${String.format(Locale.ROOT, "%.2f", confidence)})"
}
