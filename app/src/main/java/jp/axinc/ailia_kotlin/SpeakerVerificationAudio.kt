package jp.axinc.ailia_kotlin

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

data class VadSegment(val start: Int, val end: Int)

data class SpeakerVerificationMetrics(
    val cosineSimilarity: Float,
    val normalizedSimilarity: Float,
    val cosineDistance: Float,
    val isMatch: Boolean,
)

/** Pure audio preprocessing shared by WeSpeaker and its unit tests. */
object SpeakerVerificationAudio {
    const val SAMPLE_RATE = 16_000
    const val MATCH_THRESHOLD = 0.7f
    private const val FRAME_LENGTH = 400
    private const val FRAME_SHIFT = 160
    private const val FFT_SIZE = 512
    private const val MEL_BINS = 80
    private const val EPSILON = 1.1920929e-7f

    fun toMono16k(
        interleavedAudio: FloatArray,
        channels: Int,
        sampleRate: Int,
    ): FloatArray {
        require(channels > 0) { "Invalid channel count: $channels" }
        require(sampleRate > 0) { "Invalid sample rate: $sampleRate" }
        if (interleavedAudio.isEmpty()) return FloatArray(0)

        val frameCount = interleavedAudio.size / channels
        val mono = FloatArray(frameCount)
        for (frame in 0 until frameCount) {
            var sum = 0f
            val offset = frame * channels
            for (channel in 0 until channels) sum += interleavedAudio[offset + channel]
            mono[frame] = sum / channels
        }
        if (sampleRate == SAMPLE_RATE) return mono

        val outputSize = (mono.size.toDouble() * SAMPLE_RATE / sampleRate)
            .roundToInt()
            .coerceAtLeast(1)
        val result = FloatArray(outputSize)
        val sourceStep = sampleRate.toDouble() / SAMPLE_RATE
        for (i in result.indices) {
            val position = i * sourceStep
            val left = floor(position).toInt().coerceIn(0, mono.lastIndex)
            val right = (left + 1).coerceAtMost(mono.lastIndex)
            val fraction = (position - left).toFloat()
            result[i] = mono[left] + (mono[right] - mono[left]) * fraction
        }
        return result
    }

    /** Mirrors silero-vad get_speech_timestamps defaults used by the official sample. */
    fun speechSegments(
        probabilities: FloatArray,
        audioLength: Int,
        windowSize: Int = 512,
        threshold: Float = 0.5f,
        minSpeechMs: Int = 250,
        minSilenceMs: Int = 100,
        speechPadMs: Int = 30,
    ): List<VadSegment> {
        if (audioLength <= 0 || probabilities.isEmpty()) return emptyList()
        val negativeThreshold = threshold - 0.15f
        val minSpeechSamples = SAMPLE_RATE * minSpeechMs / 1000
        val minSilenceSamples = SAMPLE_RATE * minSilenceMs / 1000
        val speechPadSamples = SAMPLE_RATE * speechPadMs / 1000
        val rawSegments = mutableListOf<VadSegment>()
        var triggered = false
        var start = 0
        var temporaryEnd = 0

        probabilities.forEachIndexed { index, probability ->
            val sample = index * windowSize
            if (probability >= threshold && temporaryEnd != 0) temporaryEnd = 0
            if (probability >= threshold && !triggered) {
                triggered = true
                start = sample
            } else if (triggered && probability < negativeThreshold) {
                if (temporaryEnd == 0) temporaryEnd = sample
                if (sample - temporaryEnd >= minSilenceSamples) {
                    if (temporaryEnd - start > minSpeechSamples) {
                        rawSegments += VadSegment(start, temporaryEnd)
                    }
                    triggered = false
                    temporaryEnd = 0
                }
            }
        }
        if (triggered && audioLength - start > minSpeechSamples) {
            rawSegments += VadSegment(start, audioLength)
        }
        if (rawSegments.isEmpty()) return emptyList()

        val padded = rawSegments.toMutableList()
        for (index in padded.indices) {
            val current = padded[index]
            var paddedStart = current.start
            var paddedEnd = current.end
            if (index == 0) paddedStart = (paddedStart - speechPadSamples).coerceAtLeast(0)
            if (index < padded.lastIndex) {
                val next = padded[index + 1]
                val silence = next.start - current.end
                if (silence < speechPadSamples * 2) {
                    val half = silence / 2
                    paddedEnd += half
                    padded[index + 1] = next.copy(start = (next.start - half).coerceAtLeast(0))
                } else {
                    paddedEnd = (paddedEnd + speechPadSamples).coerceAtMost(audioLength)
                    padded[index + 1] = next.copy(start = (next.start - speechPadSamples).coerceAtLeast(0))
                }
            } else {
                paddedEnd = (paddedEnd + speechPadSamples).coerceAtMost(audioLength)
            }
            padded[index] = VadSegment(paddedStart, paddedEnd)
        }
        return padded
    }

    fun concatenateSegments(audio: FloatArray, segments: List<VadSegment>): FloatArray {
        val valid = segments.mapNotNull { segment ->
            val start = segment.start.coerceIn(0, audio.size)
            val end = segment.end.coerceIn(start, audio.size)
            if (end > start) VadSegment(start, end) else null
        }
        val result = FloatArray(valid.sumOf { it.end - it.start })
        var offset = 0
        valid.forEach { segment ->
            val length = segment.end - segment.start
            System.arraycopy(audio, segment.start, result, offset, length)
            offset += length
        }
        return result
    }

    /** Kaldi-compatible 25 ms/10 ms, Hamming, 80-bin log-power fbank with CMN. */
    fun computeFbank(audio: FloatArray): FloatArray {
        require(audio.size >= FRAME_LENGTH) { "At least 25 ms of speech is required" }
        val frameCount = 1 + (audio.size - FRAME_LENGTH) / FRAME_SHIFT
        val features = FloatArray(frameCount * MEL_BINS)
        val melBanks = buildMelBanks()
        val real = DoubleArray(FFT_SIZE)
        val imaginary = DoubleArray(FFT_SIZE)
        val power = DoubleArray(FFT_SIZE / 2 + 1)

        for (frameIndex in 0 until frameCount) {
            val frameOffset = frameIndex * FRAME_SHIFT
            var mean = 0.0
            for (i in 0 until FRAME_LENGTH) mean += audio[frameOffset + i] * 32768.0
            mean /= FRAME_LENGTH

            java.util.Arrays.fill(real, 0.0)
            java.util.Arrays.fill(imaginary, 0.0)
            var previous = audio[frameOffset] * 32768.0 - mean
            for (i in 0 until FRAME_LENGTH) {
                val sample = audio[frameOffset + i] * 32768.0 - mean
                val emphasized = if (i == 0) sample - 0.97 * sample else sample - 0.97 * previous
                previous = sample
                val hamming = 0.54 - 0.46 * cos(2.0 * PI * i / (FRAME_LENGTH - 1))
                real[i] = emphasized * hamming
            }
            fft(real, imaginary)
            for (i in power.indices) power[i] = real[i] * real[i] + imaginary[i] * imaginary[i]

            for (mel in 0 until MEL_BINS) {
                var energy = 0.0
                // The official kaldifeat implementation leaves the Nyquist bin at zero.
                for (bin in 0 until FFT_SIZE / 2) energy += power[bin] * melBanks[mel][bin]
                features[frameIndex * MEL_BINS + mel] = ln(energy.coerceAtLeast(EPSILON.toDouble())).toFloat()
            }
        }

        // Cepstral mean normalization only (no variance normalization).
        for (mel in 0 until MEL_BINS) {
            var mean = 0f
            for (frame in 0 until frameCount) mean += features[frame * MEL_BINS + mel]
            mean /= frameCount
            for (frame in 0 until frameCount) features[frame * MEL_BINS + mel] -= mean
        }
        return features
    }

    fun compare(reference: FloatArray, query: FloatArray): SpeakerVerificationMetrics {
        require(reference.size == query.size && reference.isNotEmpty()) {
            "Speaker embeddings must have the same non-zero size"
        }
        var dot = 0.0
        var referenceNorm = 0.0
        var queryNorm = 0.0
        for (i in reference.indices) {
            dot += reference[i] * query[i]
            referenceNorm += reference[i] * reference[i]
            queryNorm += query[i] * query[i]
        }
        require(referenceNorm > 0.0 && queryNorm > 0.0) { "Speaker embedding norm is zero" }
        val cosine = (dot / (sqrt(referenceNorm) * sqrt(queryNorm))).toFloat().coerceIn(-1f, 1f)
        val normalized = (cosine + 1f) / 2f
        return SpeakerVerificationMetrics(
            cosineSimilarity = cosine,
            normalizedSimilarity = normalized,
            cosineDistance = 1f - cosine,
            isMatch = normalized >= MATCH_THRESHOLD,
        )
    }

    private fun buildMelBanks(): Array<DoubleArray> {
        val banks = Array(MEL_BINS) { DoubleArray(FFT_SIZE / 2 + 1) }
        val melLow = melScale(20.0)
        val melHigh = melScale(SAMPLE_RATE / 2.0)
        val delta = (melHigh - melLow) / (MEL_BINS + 1)
        val binWidth = SAMPLE_RATE.toDouble() / FFT_SIZE
        for (mel in 0 until MEL_BINS) {
            val left = melLow + delta * mel
            val center = left + delta
            val right = center + delta
            for (bin in 0 until FFT_SIZE / 2) {
                val value = melScale(binWidth * bin)
                if (value > left && value < right) {
                    banks[mel][bin] = if (value <= center) {
                        (value - left) / (center - left)
                    } else {
                        (right - value) / (right - center)
                    }
                }
            }
        }
        return banks
    }

    private fun melScale(frequency: Double): Double = 1127.0 * ln(1.0 + frequency / 700.0)

    private fun fft(real: DoubleArray, imaginary: DoubleArray) {
        var j = 0
        for (i in 1 until FFT_SIZE) {
            var bit = FFT_SIZE shr 1
            while (j and bit != 0) {
                j = j xor bit
                bit = bit shr 1
            }
            j = j xor bit
            if (i < j) {
                val realTemp = real[i]
                real[i] = real[j]
                real[j] = realTemp
                val imaginaryTemp = imaginary[i]
                imaginary[i] = imaginary[j]
                imaginary[j] = imaginaryTemp
            }
        }

        var length = 2
        while (length <= FFT_SIZE) {
            val angle = -2.0 * PI / length
            val rootReal = cos(angle)
            val rootImaginary = sin(angle)
            var offset = 0
            while (offset < FFT_SIZE) {
                var twiddleReal = 1.0
                var twiddleImaginary = 0.0
                for (i in 0 until length / 2) {
                    val even = offset + i
                    val odd = even + length / 2
                    val oddReal = real[odd] * twiddleReal - imaginary[odd] * twiddleImaginary
                    val oddImaginary = real[odd] * twiddleImaginary + imaginary[odd] * twiddleReal
                    real[odd] = real[even] - oddReal
                    imaginary[odd] = imaginary[even] - oddImaginary
                    real[even] += oddReal
                    imaginary[even] += oddImaginary
                    val nextReal = twiddleReal * rootReal - twiddleImaginary * rootImaginary
                    twiddleImaginary = twiddleReal * rootImaginary + twiddleImaginary * rootReal
                    twiddleReal = nextReal
                }
                offset += length
            }
            length = length shl 1
        }
    }
}
