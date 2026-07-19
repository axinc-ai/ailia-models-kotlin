package jp.axinc.ailia_kotlin

import android.content.Context
import android.util.Base64
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.UUID

data class SpeakerProfile(
    val id: String,
    val name: String,
    val embedding: FloatArray,
)

/** Persists user-enrolled speaker embeddings; raw microphone audio is never retained. */
class SpeakerProfileStore(context: Context) {
    private val preferences = context.getSharedPreferences(PREFERENCES_NAME, Context.MODE_PRIVATE)

    fun list(): List<SpeakerProfile> = preferences.getStringSet(KEY_IDS, emptySet()).orEmpty()
        .mapNotNull { id ->
            val name = preferences.getString(nameKey(id), null) ?: return@mapNotNull null
            val encoded = preferences.getString(embeddingKey(id), null) ?: return@mapNotNull null
            runCatching { SpeakerProfile(id, name, decodeEmbedding(encoded)) }.getOrNull()
        }
        .sortedBy { it.name.lowercase() }

    fun save(name: String, embedding: FloatArray): SpeakerProfile {
        require(name.isNotBlank()) { "Speaker name must not be blank" }
        require(embedding.isNotEmpty()) { "Speaker embedding must not be empty" }
        val profile = SpeakerProfile(UUID.randomUUID().toString(), name.trim(), embedding.copyOf())
        val ids = preferences.getStringSet(KEY_IDS, emptySet()).orEmpty().toMutableSet()
        ids += profile.id
        check(
            preferences.edit()
                .putStringSet(KEY_IDS, ids)
                .putString(nameKey(profile.id), profile.name)
                .putString(embeddingKey(profile.id), encodeEmbedding(profile.embedding))
                .commit()
        ) { "Failed to save speaker profile" }
        return profile
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

    private fun nameKey(id: String) = "speaker.$id.name"
    private fun embeddingKey(id: String) = "speaker.$id.embedding"

    companion object {
        private const val PREFERENCES_NAME = "speaker_verification_profiles"
        private const val KEY_IDS = "speaker_ids"
    }
}
