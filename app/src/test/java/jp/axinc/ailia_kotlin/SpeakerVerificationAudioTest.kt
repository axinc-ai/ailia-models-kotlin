package jp.axinc.ailia_kotlin

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.sin

class SpeakerVerificationAudioTest {
    @Test
    fun stereoAudioIsMixedAndResampledTo16k() {
        val stereo8k = FloatArray(8_000 * 2) { index -> if (index % 2 == 0) 1f else -0.5f }

        val result = SpeakerVerificationAudio.toMono16k(stereo8k, channels = 2, sampleRate = 8_000)

        assertEquals(16_000, result.size)
        assertEquals(0.25f, result[0], 1e-6f)
        assertEquals(0.25f, result.last(), 1e-6f)
    }

    @Test
    fun sileroProbabilitiesBecomePaddedSpeechSegments() {
        val probabilities = floatArrayOf(
            0.0f,
            0.7f, 0.8f, 0.9f, 0.8f, 0.9f, 0.8f, 0.9f, 0.8f, 0.9f, 0.8f,
            0.1f, 0.1f, 0.1f, 0.1f, 0.1f,
        )

        val segments = SpeakerVerificationAudio.speechSegments(probabilities, audioLength = 8_192)

        assertEquals(listOf(VadSegment(32, 6_112)), segments)
    }

    @Test
    fun fbankHasOfficialShapeAndCmn() {
        val tone = FloatArray(16_000) { index ->
            (0.25 * sin(2.0 * PI * 440.0 * index / 16_000.0)).toFloat()
        }

        val features = SpeakerVerificationAudio.computeFbank(tone)

        assertEquals(98 * 80, features.size)
        // Values produced by ailia-models/wespeaker/kaldifeat.py for the same signal.
        val expected = floatArrayOf(
            -1.5370512f, -1.4821329f, -1.1957836f, -1.1484060f,
            0.89075375f, 0.5943184f, 0.6663914f, -1.2421036f,
        )
        val indices = intArrayOf(0, 1, 2, 10, 79, 80, 1234, 7839)
        indices.forEachIndexed { index, featureIndex ->
            assertEquals(expected[index], features[featureIndex], 2e-3f)
        }
        for (mel in 0 until 80) {
            var mean = 0f
            for (frame in 0 until 98) mean += features[frame * 80 + mel]
            assertEquals(0f, mean / 98, 1e-4f)
        }
    }

    @Test
    fun cosineDistanceAndOfficialThresholdAreReported() {
        val same = SpeakerVerificationAudio.compare(floatArrayOf(1f, 2f), floatArrayOf(1f, 2f))
        val opposite = SpeakerVerificationAudio.compare(floatArrayOf(1f, 0f), floatArrayOf(-1f, 0f))

        assertEquals(0f, same.cosineDistance, 1e-6f)
        assertEquals(1f, same.normalizedSimilarity, 1e-6f)
        assertTrue(same.isMatch)
        assertEquals(2f, opposite.cosineDistance, 1e-6f)
        assertFalse(opposite.isMatch)
    }
}
