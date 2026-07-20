package jp.axinc.ailia_kotlin

import android.content.Context
import java.io.File
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

/**
 * Keeps enrolled speaker embeddings and playback audio in memory for this app process.
 * My Voice entries intentionally disappear when the app process is restarted.
 */
class SpeakerProfileStore(
    context: Context,
    storageName: String = DEFAULT_STORAGE_NAME,
) {
    private val session = synchronized(sessions) {
        sessions.getOrPut(storageName) { SessionProfiles() }
    }

    init {
        removeLegacyPersistentProfiles(context, storageName)
    }

    fun list(): List<SpeakerProfile> = synchronized(session) {
        session.profiles.values
            .map { it.copyForCaller() }
            .sortedBy { it.name.lowercase() }
    }

    fun add(
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
        synchronized(session) {
            session.profiles[profile.id] = profile
            if (audio != null) {
                session.audio[profile.id] = SpeakerProfileAudio(audio.copyOf(), sampleRate)
            }
        }
        return profile.copyForCaller()
    }

    fun loadAudio(profileId: String): SpeakerProfileAudio? = synchronized(session) {
        session.audio[profileId]?.let { saved ->
            SpeakerProfileAudio(saved.audioData.copyOf(), saved.sampleRate)
        }
    }

    private fun SpeakerProfile.copyForCaller() = copy(embedding = embedding.copyOf())

    private fun removeLegacyPersistentProfiles(context: Context, storageName: String) {
        context.getSharedPreferences(storageName, Context.MODE_PRIVATE)
            .edit()
            .clear()
            .apply()
        val audioDirectory = File(context.filesDir, "${storageName}_audio")
        audioDirectory.listFiles()?.forEach { file ->
            if (file.isFile) file.delete()
        }
        audioDirectory.delete()
    }

    private class SessionProfiles {
        val profiles = linkedMapOf<String, SpeakerProfile>()
        val audio = mutableMapOf<String, SpeakerProfileAudio>()
    }

    companion object {
        private const val DEFAULT_STORAGE_NAME = "speaker_verification_profiles"
        private val sessions = mutableMapOf<String, SessionProfiles>()
    }
}
