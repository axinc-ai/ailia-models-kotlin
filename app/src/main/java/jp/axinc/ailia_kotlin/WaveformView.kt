package jp.axinc.ailia_kotlin

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.AttributeSet
import android.util.TypedValue
import android.view.View
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * ailia-models-flutter の WaveformView 相当のスクロール波形表示。
 * 約10msごとのピーク振幅を1ブロックとして最新[BLOCK_COUNT]ブロックを右詰めで描画する。
 * マイク入力は push() で逐次追加、TTS再生は startPlayback() で再生位置に追従して流し込む。
 */
class WaveformView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    companion object {
        const val BLOCK_COUNT = 300

        /** 約10ms(sampleRate/100サンプル)ごとのピーク振幅ブロックを計算する */
        fun peakBlocks(samples: FloatArray, count: Int, sampleRate: Int): FloatArray {
            val blockSize = max(1, sampleRate / 100)
            val n = count / blockSize
            val blocks = FloatArray(n)
            for (b in 0 until n) {
                var peak = 0f
                val start = b * blockSize
                for (j in start until start + blockSize) {
                    peak = max(peak, abs(samples[j]))
                }
                blocks[b] = peak
            }
            return blocks
        }
    }

    private val blocks = ArrayDeque<Float>()

    private val uiHandler = Handler(Looper.getMainLooper())
    private var playbackRunnable: Runnable? = null

    private val barPaint = Paint().apply {
        isAntiAlias = true
        strokeCap = Paint.Cap.ROUND
    }
    private val baselinePaint = Paint().apply {
        strokeWidth = 1f
    }
    private val borderPaint = Paint().apply {
        isAntiAlias = true
        style = Paint.Style.STROKE
        strokeWidth = dp(1f)
    }

    init {
        // Flutter版と同様にテーマのprimary/outline色を使う
        val primary = resolveColor(androidx.appcompat.R.attr.colorPrimary, 0xFF6200EE.toInt())
        barPaint.color = primary
        baselinePaint.color = (primary and 0x00FFFFFF) or 0x4D000000 // 30% alpha
        borderPaint.color = 0xFF888888.toInt()
    }

    private fun resolveColor(attr: Int, fallback: Int): Int {
        val tv = TypedValue()
        return if (context.theme.resolveAttribute(attr, tv, true)) tv.data else fallback
    }

    private fun dp(v: Float): Float = v * resources.displayMetrics.density

    /** ブロックを追加し、最新BLOCK_COUNT件だけ保持する */
    fun push(newBlocks: FloatArray) {
        for (b in newBlocks) {
            blocks.addLast(b)
        }
        while (blocks.size > BLOCK_COUNT) {
            blocks.removeFirst()
        }
        postInvalidateOnAnimation()
    }

    fun clear() {
        stopPlayback()
        blocks.clear()
        postInvalidateOnAnimation()
    }

    /**
     * 合成音声の波形を再生位置に追従して表示する(Flutter版 playFromWav 相当)。
     * 33ms周期で経過時間ぶんのブロックを流し込む。
     */
    fun startPlayback(samples: FloatArray, channels: Int, sampleRate: Int) {
        stopPlayback()
        blocks.clear()
        postInvalidateOnAnimation()

        // 波形表示はチャンネル0のみを使用
        val mono = if (channels <= 1) samples else FloatArray(samples.size / channels) { samples[it * channels] }
        val wavBlocks = peakBlocks(mono, mono.size, sampleRate)
        if (wavBlocks.isEmpty()) {
            return
        }

        val startMs = SystemClock.elapsedRealtime()
        var nextBlock = 0
        val runnable = object : Runnable {
            override fun run() {
                val elapsedMs = SystemClock.elapsedRealtime() - startMs
                val target = min((elapsedMs / 10).toInt(), wavBlocks.size)
                if (target > nextBlock) {
                    push(wavBlocks.copyOfRange(nextBlock, target))
                    nextBlock = target
                }
                if (nextBlock < wavBlocks.size) {
                    uiHandler.postDelayed(this, 33)
                } else {
                    playbackRunnable = null
                }
            }
        }
        playbackRunnable = runnable
        uiHandler.post(runnable)
    }

    fun stopPlayback() {
        playbackRunnable?.let { uiHandler.removeCallbacks(it) }
        playbackRunnable = null
    }

    override fun onDetachedFromWindow() {
        stopPlayback()
        super.onDetachedFromWindow()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val w = width.toFloat()
        val h = height.toFloat()
        val centerY = h / 2f

        // ベースライン
        canvas.drawLine(0f, centerY, w, centerY, baselinePaint)

        // 波形バー(最新が右端)
        val step = w / BLOCK_COUNT
        barPaint.strokeWidth = max(1f, step)
        val offset = BLOCK_COUNT - blocks.size
        for ((i, sample) in blocks.withIndex()) {
            val x = (offset + i + 0.5f) * step
            val amp = sample.coerceIn(0f, 1f) * (h / 2f - dp(2f))
            val half = max(amp, dp(0.5f))
            canvas.drawLine(x, centerY - half, x, centerY + half, barPaint)
        }

        // 枠線(角丸)
        val r = dp(8f)
        val inset = borderPaint.strokeWidth / 2f
        canvas.drawRoundRect(inset, inset, w - inset, h - inset, r, r, borderPaint)
    }
}
