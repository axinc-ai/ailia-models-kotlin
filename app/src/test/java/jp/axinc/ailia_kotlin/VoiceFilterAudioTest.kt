package jp.axinc.ailia_kotlin

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

class VoiceFilterAudioTest {
    @Test
    fun monoMixAndResample() {
        val stereo = floatArrayOf(1f, -1f, 0.5f, 0.5f, -0.5f, 0.5f, 0f, 1f)
        val result = VoiceFilterAudio.toMono16k(stereo, channels = 2, sampleRate = 8_000)
        assertEquals(8, result.size)
        assertEquals(0f, result[0], 1e-6f)
        assertEquals(0.25f, result[1], 1e-6f)
        assertEquals(0.5f, result[2], 1e-6f)
    }

    @Test
    fun preprocessingMatchesOfficialLibrosaPipeline() {
        val audio = testTone()
        val mel = VoiceFilterAudio.melSpectrogram(audio)
        assertEquals(40 * 101, mel.size)
        val expectedMel = mapOf(
            0 to -0.96907985f,
            1 to -3.0641501f,
            50 to -5.1211414f,
            100 to -0.9690840f,
            505 to 0.76491946f,
            1000 to -3.4484468f,
            2020 to -2.8857892f,
            4039 to -4.3410068f,
        )
        expectedMel.forEach { (index, expected) -> assertEquals(expected, mel[index], 2e-3f) }

        val spectrum = VoiceFilterAudio.analysisSpectrum(audio)
        assertEquals(101, spectrum.frameCount)
        assertEquals(101 * 601, spectrum.magnitude.size)
        val expectedMagnitude = mapOf(
            0 to 0.85017735f,
            1 to 0.8504342f,
            100 to 0.71439695f,
            600 to 0.51366055f,
            601 to 0.6442987f,
            777 to 0.4081055f,
            1202 to 0.0f,
            60700 to 0.51366043f,
        )
        expectedMagnitude.forEach { (index, expected) ->
            assertEquals(expected, spectrum.magnitude[index], 2e-3f)
        }
    }

    @Test
    fun inverseSpectrogramMatchesOfficialLibrosaPipeline() {
        val spectrum = VoiceFilterAudio.analysisSpectrum(testTone())
        val output = VoiceFilterAudio.reconstruct(spectrum, FloatArray(spectrum.magnitude.size) { 1f })
        assertEquals(16_000, output.size)
        val expected = mapOf(
            0 to 0.09608239f,
            1 to 0.13470814f,
            10 to 0.15580751f,
            100 to -0.25811186f,
            15_999 to 0.054154415f,
        )
        expected.forEach { (index, value) -> assertEquals(value, output[index], 3e-3f) }
        assertTrue(output.all { it.isFinite() })
    }

    private fun testTone(): FloatArray = FloatArray(16_000) { index ->
        val time = index.toDouble() / 16_000
        (0.3 * sin(2.0 * PI * 440.0 * time) + 0.1 * cos(2.0 * PI * 880.0 * time)).toFloat()
    }
}
