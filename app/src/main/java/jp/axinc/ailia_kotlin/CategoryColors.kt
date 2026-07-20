package jp.axinc.ailia_kotlin

import android.graphics.Color

object CategoryColors {
    /** Matches ailia-models detector_utils.py category colors. */
    fun forCategory(category: Int, categoryCount: Int = CocoLabels.CATEGORY.size): Int {
        val hCv = 256f * category / (categoryCount + 1)
        val hueDegrees = (hCv % 180f) * 2f
        return Color.HSVToColor(floatArrayOf(hueDegrees, 1f, 1f))
    }
}
