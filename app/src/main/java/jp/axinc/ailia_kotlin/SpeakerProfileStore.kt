package jp.axinc.ailia_kotlin

import android.content.Context
import android.util.Base64
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.UUID

data class SpeakerProfile(
    val id: String,
    val name: String,
    val embedding: FloatArray,
    val hasAudio: Boolean = false,
)

data class SpeakerProfileAudio(
    val audioData: FloatArray,
    val sampleRate: Int,
)

/** Persists enrolled speaker embeddings and optional playback audio in app-private storage. */
class SpeakerProfileStore(
    context: Context,
    storageName: String = DEFAULT_STORAGE_NAME,
) {
    private val preferences = context.getSharedPreferences(storageName, Context.MODE_PRIVATE)
    private val audioDirectory = File(context.filesDir, "${storageName}_audio")

    fun list(): List<SpeakerProfile> = preferences.getStringSet(KEY_IDS, emptySet()).orEmpty()
        .mapNotNull { id ->
            val name = preferences.getString(nameKey(id), null) ?: return@mapNotNull null
            val encoded = preferences.getString(embeddingKey(id), null) ?: return@mapNotNull null
            runCatching {
                SpeakerProfile(
                    id = id,
                    name = name,
                    embedding = decodeEmbedding(encoded),
                    hasAudio = audioFile(id).isFile &&
                        preferences.getInt(audioSampleRateKey(id), 0) > 0,
                )
            }.getOrNull()
        }
        .sortedBy { it.name.lowercase() }

    fun save(
        name: String,
        embedding: FloatArray,
        audio: FloatArray? = null,
        sampleRate: Int = 0,
    ): SpeakerProfile {
        require(name.isNotBlank()) { "Speaker name must not be blank" }
        require(embedding.isNotEmpty()) { "Speaker embedding must not be empty" }
        require(audio == null || audio.isNotEmpty()) { "Speaker audio must not be empty" }
        require(audio == null || sampleRate > 0) { "Speaker audio sample rate must be positive" }
        val profile = SpeakerProfile(
            id = UUID.randomUUID().toString(),
            name = name.trim(),
            embedding = embedding.copyOf(),
            hasAudio = audio != null,
        )
        if (audio != null) writeAudio(profile.id, audio)
        val ids = preferences.getStringSet(KEY_IDS, emptySet()).orEmpty().toMutableSet()
        ids += profile.id
        val saved =
            preferences.edit()
                .putStringSet(KEY_IDS, ids)
                .putString(nameKey(profile.id), profile.name)
                .putString(embeddingKey(profile.id), encodeEmbedding(profile.embedding))
                .apply {
                    if (audio != null) putInt(audioSampleRateKey(profile.id), sampleRate)
                }
                .commit()
        if (!saved) audioFile(profile.id).delete()
        check(saved) { "Failed to save speaker profile" }
        return profile
    }

    fun loadAudio(profileId: String): SpeakerProfileAudio? {
        val sampleRate = preferences.getInt(audioSampleRateKey(profileId), 0)
        val file = audioFile(profileId)
        if (sampleRate <= 0 || !file.isFile) return null
        return runCatching {
            val bytes = file.readBytes()
            require(bytes.isNotEmpty() && bytes.size % Float.SIZE_BYTES == 0) {
                "Invalid stored speaker audio"
            }
            val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            SpeakerProfileAudio(
                audioData = FloatArray(bytes.size / Float.SIZE_BYTES) { buffer.float },
                sampleRate = sampleRate,
            )
        }.getOrNull()
    }

    private fun encodeEmbedding(embedding: FloatArray): String {
        val buffer = ByteBuffer.allocate(embedding.size * Float.SIZE_BYTES).order(ByteOrder.LITTLE_ENDIAN)
        embedding.forEach(buffer::putFloat)
        return Base64.encodeToString(buffer.array(), Base64.NO_WRAP)
    }

    private fun decodeEmbedding(encoded: String): FloatArray {
        val bytes = Base64.decode(encoded, Base64.NO_WRAP)
        require(bytes.size % Float.SIZE_BYTES == 0) { "Invalid stored speaker embedding" }
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(bytes.size / Float.SIZE_BYTES) { buffer.float }
    }

    private fun writeAudio(profileId: String, audio: FloatArray) {
        check(audioDirectory.exists() || audioDirectory.mkdirs()) {
            "Failed to create speaker audio directory"
        }
        val target = audioFile(profileId)
        val temporary = File(audioDirectory, "$profileId.tmp")
        try {
            val buffer = ByteBuffer.allocate(audio.size * Float.SIZE_BYTES).order(ByteOrder.LITTLE_ENDIAN)
            audio.forEach(buffer::putFloat)
            temporary.outputStream().use { it.write(buffer.array()) }
            check(temporary.renameTo(target)) { "Failed to save speaker audio" }
        } finally {
            temporary.delete()
        }
    }

    private fun nameKey(id: String) = "speaker.$id.name"
    private fun embeddingKey(id: String) = "speaker.$id.embedding"
    private fun audioSampleRateKey(id: String) = "speaker.$id.audio_sample_rate"
    private fun audioFile(id: String) = File(audioDirectory, "$id.pcm-f32le")

    companion object {
        private const val DEFAULT_STORAGE_NAME = "speaker_verification_profiles"
        private const val KEY_IDS = "speaker_ids"
    }
}
