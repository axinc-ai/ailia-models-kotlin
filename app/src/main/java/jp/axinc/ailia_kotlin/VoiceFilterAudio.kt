package jp.axinc.ailia_kotlin

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

data class VoiceFilterSpectrum(
    /** Normalized magnitude in frame-major [frames, 601] order. */
    val magnitude: FloatArray,
    /** Phase in frame-major [frames, 601] order. */
    val phase: FloatArray,
    val frameCount: Int,
    val originalSampleCount: Int,
)

/**
 * VoiceFilter audio preprocessing and postprocessing translated from audio_utils.py.
 *
 * The implementation intentionally stays self-contained so the model sample can be
 * copied without adding an FFT or audio-processing dependency. Its zero-centered STFT,
 * periodic Hann window, Slaney mel bank, dB normalization, and overlap-add inverse
 * match the librosa defaults used by the official ailia-models sample.
 */
object VoiceFilterAudio {
    const val SAMPLE_RATE = 16_000
    const val FILTER_FFT_SIZE = 1_200
    const val FILTER_FREQUENCY_BINS = FILTER_FFT_SIZE / 2 + 1
    const val EMBEDDER_FFT_SIZE = 512
    const val MEL_BINS = 40

    private const val HOP_LENGTH = 160
    private const val WINDOW_LENGTH = 400
    private const val MIN_LEVEL_DB = -100.0
    private const val REFERENCE_LEVEL_DB = 20.0
    private const val AMPLITUDE_FLOOR = 1e-5
    private const val MEL_ENERGY_FLOOR = 1e-6

    private val filterPlan = FftPlan(FILTER_FFT_SIZE)
    private val embedderPlan = FftPlan(EMBEDDER_FFT_SIZE)
    private val hannWindow = DoubleArray(WINDOW_LENGTH) { index ->
        0.5 - 0.5 * cos(2.0 * PI * index / WINDOW_LENGTH)
    }
    private val melBank = createSlaneyMelBank()

    fun toMono16k(interleaved: FloatArray, channels: Int, sampleRate: Int): FloatArray {
        require(channels > 0) { "Channel count must be positive" }
        require(sampleRate > 0) { "Sample rate must be positive" }
        require(interleaved.size % channels == 0) { "Interleaved audio has incomplete frames" }
        if (interleaved.isEmpty()) return FloatArray(0)

        val frameCount = interleaved.size / channels
        val mono = FloatArray(frameCount) { frame ->
            var sum = 0f
            for (channel in 0 until channels) sum += interleaved[frame * channels + channel]
            sum / channels
        }
        if (sampleRate == SAMPLE_RATE) return mono

        val outputSize = max(1, (frameCount.toLong() * SAMPLE_RATE / sampleRate).toInt())
        return FloatArray(outputSize) { outputIndex ->
            val sourcePosition = outputIndex.toDouble() * sampleRate / SAMPLE_RATE
            val left = floor(sourcePosition).toInt().coerceIn(0, frameCount - 1)
            val right = (left + 1).coerceAtMost(frameCount - 1)
            val fraction = sourcePosition - left
            (mono[left] * (1.0 - fraction) + mono[right] * fraction).toFloat()
        }
    }

    /** Returns librosa-compatible log10 mel power in [40, frames] order. */
    fun melSpectrogram(audio: FloatArray): FloatArray {
        require(audio.isNotEmpty()) { "Reference audio is empty" }
        val padded = centerPad(audio, EMBEDDER_FFT_SIZE / 2)
        val frameCount = 1 + (padded.size - EMBEDDER_FFT_SIZE) / HOP_LENGTH
        val output = FloatArray(MEL_BINS * frameCount)
        val real = DoubleArray(EMBEDDER_FFT_SIZE)
        val imaginary = DoubleArray(EMBEDDER_FFT_SIZE)
        val power = DoubleArray(EMBEDDER_FFT_SIZE / 2 + 1)
        val windowOffset = (EMBEDDER_FFT_SIZE - WINDOW_LENGTH) / 2

        for (frame in 0 until frameCount) {
            real.fill(0.0)
            imaginary.fill(0.0)
            val inputOffset = frame * HOP_LENGTH + windowOffset
            for (i in 0 until WINDOW_LENGTH) {
                real[windowOffset + i] = padded[inputOffset + i] * hannWindow[i]
            }
            embedderPlan.forward(real, imaginary)
            for (bin in power.indices) {
                power[bin] = real[bin] * real[bin] + imaginary[bin] * imaginary[bin]
            }
            for (mel in 0 until MEL_BINS) {
                var energy = 0.0
                for (bin in power.indices) energy += melBank[mel][bin] * power[bin]
                output[mel * frameCount + frame] = log10(energy + MEL_ENERGY_FLOOR).toFloat()
            }
        }
        return output
    }

    fun analysisSpectrum(audio: FloatArray): VoiceFilterSpectrum {
        require(audio.isNotEmpty()) { "Input audio is empty" }
        val padded = centerPad(audio, FILTER_FFT_SIZE / 2)
        val frameCount = 1 + (padded.size - FILTER_FFT_SIZE) / HOP_LENGTH
        val magnitude = FloatArray(frameCount * FILTER_FREQUENCY_BINS)
        val phase = FloatArray(magnitude.size)
        val real = DoubleArray(FILTER_FFT_SIZE)
        val imaginary = DoubleArray(FILTER_FFT_SIZE)
        val windowOffset = (FILTER_FFT_SIZE - WINDOW_LENGTH) / 2

        for (frame in 0 until frameCount) {
            real.fill(0.0)
            imaginary.fill(0.0)
            val inputOffset = frame * HOP_LENGTH + windowOffset
            for (i in 0 until WINDOW_LENGTH) {
                real[windowOffset + i] = padded[inputOffset + i] * hannWindow[i]
            }
            filterPlan.forward(real, imaginary)
            val outputOffset = frame * FILTER_FREQUENCY_BINS
            for (bin in 0 until FILTER_FREQUENCY_BINS) {
                val amplitude = sqrt(real[bin] * real[bin] + imaginary[bin] * imaginary[bin])
                val db = 20.0 * log10(max(AMPLITUDE_FLOOR, amplitude)) - REFERENCE_LEVEL_DB
                magnitude[outputOffset + bin] =
                    ((db / -MIN_LEVEL_DB).coerceIn(-1.0, 0.0) + 1.0).toFloat()
                phase[outputOffset + bin] = kotlin.math.atan2(imaginary[bin], real[bin]).toFloat()
            }
        }
        return VoiceFilterSpectrum(magnitude, phase, frameCount, audio.size)
    }

    /** Applies the model mask in normalized-magnitude space and performs librosa-style ISTFT. */
    fun reconstruct(spectrum: VoiceFilterSpectrum, mask: FloatArray): FloatArray {
        require(mask.size == spectrum.magnitude.size) { "VoiceFilter mask shape does not match input" }
        val expectedLength = HOP_LENGTH * (spectrum.frameCount - 1)
        if (expectedLength <= 0) return FloatArray(0)
        val paddedLength = FILTER_FFT_SIZE + HOP_LENGTH * (spectrum.frameCount - 1)
        val overlap = DoubleArray(paddedLength)
        val windowEnvelope = DoubleArray(paddedLength)
        val real = DoubleArray(FILTER_FFT_SIZE)
        val imaginary = DoubleArray(FILTER_FFT_SIZE)
        val windowOffset = (FILTER_FFT_SIZE - WINDOW_LENGTH) / 2

        for (frame in 0 until spectrum.frameCount) {
            real.fill(0.0)
            imaginary.fill(0.0)
            val spectrumOffset = frame * FILTER_FREQUENCY_BINS
            for (bin in 0 until FILTER_FREQUENCY_BINS) {
                // The original model multiplies its mask by the normalized dB magnitude.
                val normalized = (spectrum.magnitude[spectrumOffset + bin] * mask[spectrumOffset + bin])
                    .coerceIn(0f, 1f)
                val db = (normalized - 1.0) * -MIN_LEVEL_DB + REFERENCE_LEVEL_DB
                val amplitude = 10.0.pow(db * 0.05)
                val angle = spectrum.phase[spectrumOffset + bin]
                real[bin] = amplitude * cos(angle.toDouble())
                imaginary[bin] = amplitude * sin(angle.toDouble())
            }
            // Restore the conjugate half required for a real-valued inverse transform.
            for (bin in 1 until FILTER_FFT_SIZE / 2) {
                real[FILTER_FFT_SIZE - bin] = real[bin]
                imaginary[FILTER_FFT_SIZE - bin] = -imaginary[bin]
            }
            filterPlan.inverse(real, imaginary)

            val frameOffset = frame * HOP_LENGTH + windowOffset
            for (i in 0 until WINDOW_LENGTH) {
                val window = hannWindow[i]
                overlap[frameOffset + i] += real[windowOffset + i] * window
                windowEnvelope[frameOffset + i] += window * window
            }
        }

        val centerOffset = FILTER_FFT_SIZE / 2
        val outputSize = minOf(expectedLength, spectrum.originalSampleCount)
        return FloatArray(outputSize) { index ->
            val paddedIndex = centerOffset + index
            val envelope = windowEnvelope[paddedIndex]
            if (envelope > 1e-10) (overlap[paddedIndex] / envelope).toFloat() else 0f
        }
    }

    private fun centerPad(audio: FloatArray, amount: Int): FloatArray {
        if (amount == 0) return audio.copyOf()
        return FloatArray(audio.size + amount * 2).also { padded ->
            System.arraycopy(audio, 0, padded, amount, audio.size)
        }
    }

    private fun createSlaneyMelBank(): Array<DoubleArray> {
        val frequencies = DoubleArray(EMBEDDER_FFT_SIZE / 2 + 1) { bin ->
            bin.toDouble() * SAMPLE_RATE / EMBEDDER_FFT_SIZE
        }
        val minMel = hzToSlaneyMel(0.0)
        val maxMel = hzToSlaneyMel(SAMPLE_RATE / 2.0)
        val melFrequencies = DoubleArray(MEL_BINS + 2) { index ->
            val mel = minMel + (maxMel - minMel) * index / (MEL_BINS + 1)
            slaneyMelToHz(mel)
        }
        return Array(MEL_BINS) { mel ->
            val lowerSpan = melFrequencies[mel + 1] - melFrequencies[mel]
            val upperSpan = melFrequencies[mel + 2] - melFrequencies[mel + 1]
            val normalization = 2.0 / (melFrequencies[mel + 2] - melFrequencies[mel])
            DoubleArray(frequencies.size) { bin ->
                val lower = (frequencies[bin] - melFrequencies[mel]) / lowerSpan
                val upper = (melFrequencies[mel + 2] - frequencies[bin]) / upperSpan
                max(0.0, minOf(lower, upper)) * normalization
            }
        }
    }

    private fun hzToSlaneyMel(frequency: Double): Double {
        val linearSpacing = 200.0 / 3.0
        if (frequency < 1_000.0) return frequency / linearSpacing
        val logStep = ln(6.4) / 27.0
        return 15.0 + ln(frequency / 1_000.0) / logStep
    }

    private fun slaneyMelToHz(mel: Double): Double {
        val linearSpacing = 200.0 / 3.0
        if (mel < 15.0) return mel * linearSpacing
        val logStep = ln(6.4) / 27.0
        return 1_000.0 * kotlin.math.exp(logStep * (mel - 15.0))
    }

    /** Arbitrary-size FFT using Bluestein convolution on the power-of-two kernel below. */
    private class FftPlan(private val size: Int) {
        private val convolutionSize: Int
        private val chirpCos = DoubleArray(size)
        private val chirpSin = DoubleArray(size)
        private val kernelReal: DoubleArray
        private val kernelImaginary: DoubleArray

        init {
            var length = 1
            while (length < size * 2 - 1) length = length shl 1
            convolutionSize = length
            kernelReal = DoubleArray(length)
            kernelImaginary = DoubleArray(length)
            for (index in 0 until size) {
                // index^2 is reduced first to avoid precision loss for larger transforms.
                val angle = PI * ((index.toLong() * index) % (size.toLong() * 2)) / size
                chirpCos[index] = cos(angle)
                chirpSin[index] = sin(angle)
                kernelReal[index] = chirpCos[index]
                kernelImaginary[index] = chirpSin[index]
                if (index != 0) {
                    kernelReal[length - index] = chirpCos[index]
                    kernelImaginary[length - index] = chirpSin[index]
                }
            }
            powerOfTwoFft(kernelReal, kernelImaginary, inverse = false)
        }

        fun forward(real: DoubleArray, imaginary: DoubleArray) {
            require(real.size == size && imaginary.size == size)
            val workReal = DoubleArray(convolutionSize)
            val workImaginary = DoubleArray(convolutionSize)
            for (index in 0 until size) {
                workReal[index] = real[index] * chirpCos[index] + imaginary[index] * chirpSin[index]
                workImaginary[index] = -real[index] * chirpSin[index] + imaginary[index] * chirpCos[index]
            }
            powerOfTwoFft(workReal, workImaginary, inverse = false)
            for (index in 0 until convolutionSize) {
                val productReal = workReal[index] * kernelReal[index] - workImaginary[index] * kernelImaginary[index]
                val productImaginary = workReal[index] * kernelImaginary[index] + workImaginary[index] * kernelReal[index]
                workReal[index] = productReal
                workImaginary[index] = productImaginary
            }
            powerOfTwoFft(workReal, workImaginary, inverse = true)
            for (index in 0 until size) {
                real[index] = workReal[index] * chirpCos[index] + workImaginary[index] * chirpSin[index]
                imaginary[index] = -workReal[index] * chirpSin[index] + workImaginary[index] * chirpCos[index]
            }
        }

        fun inverse(real: DoubleArray, imaginary: DoubleArray) {
            for (index in 0 until size) imaginary[index] = -imaginary[index]
            forward(real, imaginary)
            for (index in 0 until size) {
                real[index] = real[index] / size
                imaginary[index] = -imaginary[index] / size
            }
        }
    }

    private fun powerOfTwoFft(real: DoubleArray, imaginary: DoubleArray, inverse: Boolean) {
        val size = real.size
        require(size > 0 && size and (size - 1) == 0)
        var target = 0
        for (source in 1 until size) {
            var bit = size shr 1
            while (target and bit != 0) {
                target = target xor bit
                bit = bit shr 1
            }
            target = target xor bit
            if (source < target) {
                val realValue = real[source]
                real[source] = real[target]
                real[target] = realValue
                val imaginaryValue = imaginary[source]
                imaginary[source] = imaginary[target]
                imaginary[target] = imaginaryValue
            }
        }

        var length = 2
        while (length <= size) {
            val angle = (if (inverse) 2.0 else -2.0) * PI / length
            val baseReal = cos(angle)
            val baseImaginary = sin(angle)
            var start = 0
            while (start < size) {
                var twiddleReal = 1.0
                var twiddleImaginary = 0.0
                for (offset in 0 until length / 2) {
                    val even = start + offset
                    val odd = even + length / 2
                    val oddReal = real[odd] * twiddleReal - imaginary[odd] * twiddleImaginary
                    val oddImaginary = real[odd] * twiddleImaginary + imaginary[odd] * twiddleReal
                    real[odd] = real[even] - oddReal
                    imaginary[odd] = imaginary[even] - oddImaginary
                    real[even] += oddReal
                    imaginary[even] += oddImaginary
                    val nextReal = twiddleReal * baseReal - twiddleImaginary * baseImaginary
                    twiddleImaginary = twiddleReal * baseImaginary + twiddleImaginary * baseReal
                    twiddleReal = nextReal
                }
                start += length
            }
            length = length shl 1
        }
        if (inverse) {
            for (index in 0 until size) {
                real[index] = real[index] / size
                imaginary[index] = imaginary[index] / size
            }
        }
    }
}
