package jp.axinc.ailia_kotlin

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

import axip.ailia_speech.AiliaSpeech
import axip.ailia_speech.AiliaSpeechText
import axip.ailia_speech.IntermediateCallback

/**
 * Enum defining available speech recognition models.
 */
enum class SpeechModelType(
    val displayName: String,
    val encoderUrl: String,
    val encoderFileName: String,
    val decoderUrl: String,
    val decoderFileName: String,
    val modelTypeId: Int,
    val needsDecoder: Boolean = true
) {
    WHISPER_TINY(
        "Whisper Tiny",
        "https://storage.googleapis.com/ailia-models/whisper/encoder_tiny.opt3.onnx",
        "encoder_tiny.onnx",
        "https://storage.googleapis.com/ailia-models/whisper/decoder_tiny_fix_kv_cache.opt3.onnx",
        "decoder_tiny.onnx",
        AiliaSpeech.AILIA_SPEECH_MODEL_TYPE_WHISPER_MULTILINGUAL_TINY
    ),
    WHISPER_BASE(
        "Whisper Base",
        "https://storage.googleapis.com/ailia-models/whisper/encoder_base.opt3.onnx",
        "encoder_base.onnx",
        "https://storage.googleapis.com/ailia-models/whisper/decoder_base_fix_kv_cache.opt3.onnx",
        "decoder_base.onnx",
        AiliaSpeech.AILIA_SPEECH_MODEL_TYPE_WHISPER_MULTILINGUAL_BASE
    ),
    WHISPER_SMALL(
        "Whisper Small",
        "https://storage.googleapis.com/ailia-models/whisper/encoder_small.opt3.onnx",
        "encoder_small.onnx",
        "https://storage.googleapis.com/ailia-models/whisper/decoder_small_fix_kv_cache.opt3.onnx",
        "decoder_small.onnx",
        AiliaSpeech.AILIA_SPEECH_MODEL_TYPE_WHISPER_MULTILINGUAL_SMALL
    ),
    SENSEVOICE_SMALL(
        "SenseVoice Small",
        "https://storage.googleapis.com/ailia-models/sensevoice/sensevoice_small.onnx",
        "sensevoice_small.onnx",
        "https://storage.googleapis.com/ailia-models/sensevoice/sensevoice_small.model",
        "sensevoice_small.model",
        AiliaSpeech.AILIA_SPEECH_MODEL_TYPE_SENSEVOICE_SMALL
    )
}

class AiliaSpeechSample {
    /** マイク録音中にUIへ返すデータだけを定義する。各コールバックはバックグラウンドスレッドで呼ばれる。 */
    interface MicRecordingListener {
        fun onWaveform(samples: FloatArray, sampleRate: Int)
        fun onIntermediateResult(text: String)
        fun onResult(lines: List<String>, isFinal: Boolean)
        fun onError(error: String)
    }

    companion object {
        private const val TAG = "AILIA_Main"
        private const val VAD_URL = "https://storage.googleapis.com/ailia-models/silero-vad/silero_vad_v6_2.onnx"
        private const val VAD_FILE = "silero_vad_v6_2.onnx"
        private const val DIARIZATION_SEGMENTATION_URL = "https://storage.googleapis.com/ailia-models/pyannote-audio/segmentation.onnx"
        private const val DIARIZATION_EMBEDDING_URL = "https://storage.googleapis.com/ailia-models/pyannote-audio/speaker-embedding.onnx"
        private const val DIARIZATION_SEGMENTATION_FILE = "segmentation.onnx"
        private const val DIARIZATION_EMBEDDING_FILE = "speaker-embedding.onnx"
    }

    private var speech: AiliaSpeech? = null
    @Volatile
    private var isInitialized = false
    private var audioRecord: AudioRecord? = null
    private var micReadExecutor: ExecutorService? = null
    private var micRecognitionExecutor: ExecutorService? = null
    private val micRecording = AtomicBoolean(false)
    private val micSessionActive = AtomicBoolean(false)
    private val finalizeMicInput = AtomicBoolean(true)
    private var liveModeEnabled = false
    private var speechIntermediateCallback: IntermediateCallback? = null
    var modelDir: String = ""
    var currentModelType: SpeechModelType = SpeechModelType.SENSEVOICE_SMALL
    var diarizationEnabled: Boolean = false

    val isMicRecording: Boolean
        get() = micRecording.get()

    /** 認識言語("ja"/"en"など)。"auto"の場合は自動判定(setLanguageを呼ばない) */
    var language: String = "ja"

    private fun downloadFile(urlStr: String, fileName: String, listener: ModelDownloadListener? = null): String {
        val dir = modelDir
        if (dir.isEmpty()) throw IllegalStateException("modelDir not set")
        val path = "$dir/$fileName"
        val file = File(path)
        if (file.exists()) {
            if (file.canRead()) {
                Log.i(TAG, "Model file already exists and readable: $path (${file.length()} bytes)")
                return path
            } else {
                Log.w(TAG, "Model file exists but not readable, re-downloading: $path")
                file.delete()
            }
        }
        File(path).parentFile?.mkdirs()
        val tmpFile = File("$path.tmp")
        val url = URL(urlStr)
        val connection = url.openConnection() as HttpURLConnection
        connection.connectTimeout = 30000
        connection.readTimeout = 60000
        connection.connect()
        val totalBytes = connection.contentLengthLong
        connection.inputStream.use { input ->
            FileOutputStream(tmpFile).use { output ->
                val buffer = ByteArray(8192)
                var bytesDownloaded: Long = 0
                var bytesRead: Int
                while (input.read(buffer).also { bytesRead = it } != -1) {
                    output.write(buffer, 0, bytesRead)
                    bytesDownloaded += bytesRead
                    listener?.onProgress(fileName, bytesDownloaded, totalBytes)
                }
            }
        }
        tmpFile.renameTo(File(path))
        return path
    }

    /**
     * Downloads model files for the specified (or current) speech model type.
     * Always downloads Silero VAD model for all modes.
     * If diarizationEnabled is true, also downloads pyannote-audio segmentation and embedding models.
     */
    fun downloadModel(modelType: SpeechModelType = currentModelType, listener: ModelDownloadListener? = null): Boolean {
        currentModelType = modelType
        return try {
            Log.i(TAG, "Starting speech model download/check for ${modelType.displayName}...")
            downloadFile(
                modelType.encoderUrl,
                modelType.encoderFileName,
                listener
            )
            if (modelType.needsDecoder) {
                downloadFile(
                    modelType.decoderUrl,
                    modelType.decoderFileName,
                    listener
                )
            }
            // Always download VAD model
            Log.i(TAG, "Downloading VAD model...")
            downloadFile(VAD_URL, VAD_FILE, listener)
            if (diarizationEnabled) {
                Log.i(TAG, "Downloading diarization models...")
                downloadFile(DIARIZATION_SEGMENTATION_URL, DIARIZATION_SEGMENTATION_FILE, listener)
                downloadFile(DIARIZATION_EMBEDDING_URL, DIARIZATION_EMBEDDING_FILE, listener)
            }
            listener?.onComplete()
            Log.i(TAG, "Speech model download/check complete for ${modelType.displayName}")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Speech model download failed for ${modelType.displayName}", e)
            listener?.onError(e.message ?: "Download failed")
            false
        }
    }

    /**
     * Initializes the speech engine with the current model type.
     *
     * @param envId Environment ID (-1 for auto)
     * @param liveMode If true, initializes with AILIA_SPEECH_FLAG_LIVE for streaming mic input
     */
    fun initializeSpeech(envId: Int = -1, liveMode: Boolean = false): Boolean {
        if (isInitialized) {
            releaseSpeech()
        }

        return try {
            val dir = modelDir
            val encoderPath = "$dir/${currentModelType.encoderFileName}"
            val decoderPath = if (currentModelType.needsDecoder) {
                "$dir/${currentModelType.decoderFileName}"
            } else {
                ""
            }

            // When diarization is enabled, disable LIVE flag (diarization requires non-live mode)
            // Streaming still works without the LIVE flag
            val useLiveFlag = liveMode && !diarizationEnabled
            val flags = if (useLiveFlag) AiliaSpeech.AILIA_SPEECH_FLAG_LIVE else AiliaSpeech.AILIA_SPEECH_FLAG_NONE
            liveModeEnabled = useLiveFlag

            Log.i(TAG, "Initializing speech with envId=$envId, model=${currentModelType.displayName}, liveMode=$liveMode, diarization=$diarizationEnabled, useLiveFlag=$useLiveFlag")
            Log.i(TAG, "Encoder: $encoderPath")
            Log.i(TAG, "Decoder: $decoderPath")

            speech = AiliaSpeech(
                envId = envId,
                task = AiliaSpeech.AILIA_SPEECH_TASK_TRANSCRIBE,
                flags = flags
            )
            speech?.openModel(encoderPath, decoderPath, currentModelType.modelTypeId)

            // 言語設定("auto"は自動判定のためsetLanguageを呼ばない。Flutter版と同じ挙動)
            if (language != "auto") {
                val langResult = speech?.setLanguage(language)
                Log.i(TAG, "setLanguage($language) result=$langResult")
            }

            // Always open VAD (Silero VAD)
            val vadPath = "$dir/$VAD_FILE"
            Log.i(TAG, "Opening VAD: $vadPath")
            val vadResult = speech?.openVad(vadPath, AiliaSpeech.AILIA_SPEECH_VAD_TYPE_SILERO)
            Log.i(TAG, "VAD openVad result=$vadResult")

            // VADを有効化するにはsetSilentThresholdの設定が必要
            // (ailia-models-flutterと同じ閾値: threshold=0.5, speechSec=1.0, noSpeechSec=1.0)
            val thresholdResult = speech?.setSilentThreshold(0.5f, 1.0f, 1.0f)
            Log.i(TAG, "VAD setSilentThreshold result=$thresholdResult")

            // Open diarization if enabled
            if (diarizationEnabled) {
                val segmentationPath = "$dir/$DIARIZATION_SEGMENTATION_FILE"
                val embeddingPath = "$dir/$DIARIZATION_EMBEDDING_FILE"
                Log.i(TAG, "Opening diarization: segmentation=$segmentationPath, embedding=$embeddingPath")
                val diarResult = speech?.openDiarization(
                    segmentationPath, embeddingPath,
                    AiliaSpeech.AILIA_SPEECH_DIARIZATION_TYPE_PYANNOTE_AUDIO
                )
                Log.i(TAG, "Diarization openDiarization result=$diarResult")
            }

            isInitialized = true
            Log.i(TAG, "Speech initialized successfully with envId=$envId, model=${currentModelType.displayName}")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize speech: ${e.javaClass.name}: ${e.message}")
            releaseSpeech()
            false
        }
    }

    /**
     * Processes audio from a WAV file (non-live mode).
     * Calls pushInputData, finalizeInputData, transcribe, and returns transcript lines.
     */
    fun process(audio: FloatArray, channels: Int, sampleRate: Int): List<String> {
        Log.i(TAG, "Speech process: audio.size=${audio.size}, channels=$channels, sampleRate=$sampleRate, samples=${audio.size / channels}")
        val pushResult = speech?.pushInputData(audio, channels, audio.size / channels, sampleRate)
        Log.i(TAG, "Speech pushInputData result=$pushResult")
        val finalizeResult = speech?.finalizeInputData()
        Log.i(TAG, "Speech finalizeInputData result=$finalizeResult")
        val transcribeResult = speech?.transcribe()
        Log.i(TAG, "Speech transcribe result=$transcribeResult")
        if (transcribeResult != null && transcribeResult != 0) {
            val errorDetail = speech?.getErrorDetail()
            Log.e(TAG, "Speech transcribe error detail: $errorDetail")
        }
        val lines = collectTextLines()
        speech?.resetTranscribeState()
        return lines
    }

    /**
     * Pushes live audio data for streaming recognition (live mode).
     * Does NOT call finalizeInputData - use finalizeLiveAudio() when recording stops.
     * Returns transcript lines confirmed by this call.
     */
    fun pushLiveAudio(audio: FloatArray, channels: Int, sampleRate: Int): List<String> {
        val engine = speech ?: return emptyList()
        val pushResult = engine.pushInputData(audio, channels, audio.size / channels, sampleRate)
        Log.d(TAG, "Speech pushLiveAudio: pushInputData result=$pushResult, samples=${audio.size / channels}")

        val lines = mutableListOf<String>()
        while (engine.getBuffered() != 0) {
            val transcribeResult = engine.transcribe()
            Log.d(TAG, "Speech pushLiveAudio: transcribe result=$transcribeResult")
            if (transcribeResult != 0) {
                Log.e(TAG, "Speech pushLiveAudio transcribe error: ${engine.getErrorDetail()}")
                break
            }
            lines.addAll(collectTextLines())
        }
        return lines
    }

    /**
     * Finalizes live audio input and returns the remaining transcript lines.
     * Call this when mic recording stops.
     */
    fun finalizeLiveAudio(): List<String> {
        val engine = speech ?: return emptyList()
        val finalizeResult = engine.finalizeInputData()
        Log.i(TAG, "Speech finalizeLiveAudio: finalizeInputData result=$finalizeResult")

        val lines = mutableListOf<String>()
        while (engine.getComplete() == 0) {
            val transcribeResult = engine.transcribe()
            Log.i(TAG, "Speech finalizeLiveAudio: transcribe result=$transcribeResult")
            if (transcribeResult != 0) {
                Log.e(TAG, "Speech finalizeLiveAudio transcribe error: ${engine.getErrorDetail()}")
                break
            }
            lines.addAll(collectTextLines())
        }
        engine.resetTranscribeState()
        return lines
    }

    /**
     * 16kHz/mono/PCM_FLOATでマイク録音を開始する。
     * Flutter版と同様に短いチャンクを逐次投入し、SDKが処理可能になった時だけ認識する。
     */
    @SuppressLint("MissingPermission")
    fun startMicRecording(listener: MicRecordingListener): Boolean {
        if (!isInitialized) {
            listener.onError("Speech model not ready")
            return false
        }
        if (!micSessionActive.compareAndSet(false, true)) {
            listener.onError("Previous microphone session is still finalizing")
            return false
        }

        if (liveModeEnabled) {
            val callback = object : IntermediateCallback {
                override fun onIntermediateResult(text: String): Int {
                    listener.onIntermediateResult(text)
                    return 0
                }
            }
            val callbackResult = speech?.setIntermediateCallback(callback)
            Log.i(TAG, "Speech setIntermediateCallback result=$callbackResult")
            if (callbackResult != 0) {
                speechIntermediateCallback = null
                micSessionActive.set(false)
                listener.onError("Failed to register intermediate result callback: $callbackResult")
                return false
            }
            // Keep a strong reference while the native speech instance owns the callback.
            speechIntermediateCallback = callback
        } else {
            speechIntermediateCallback = null
        }

        val sampleRate = 16000
        val readChunkSize = sampleRate / 10
        val recognitionChunkSize = readChunkSize
        val audioRecordBufferBytes = sampleRate * Float.SIZE_BYTES * 2

        return try {
            val recorder = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_FLOAT,
                audioRecordBufferBytes
            )
            if (recorder.state != AudioRecord.STATE_INITIALIZED) {
                recorder.release()
                micSessionActive.set(false)
                listener.onError("Failed to initialize AudioRecord")
                return false
            }

            synchronized(this) {
                audioRecord = recorder
            }
            finalizeMicInput.set(true)
            micRecognitionExecutor = newMicExecutor("ailia-speech-recognition", android.os.Process.THREAD_PRIORITY_BACKGROUND)
            micReadExecutor = newMicExecutor("ailia-speech-recording", android.os.Process.THREAD_PRIORITY_AUDIO)
            recorder.startRecording()
            micRecording.set(true)
            micReadExecutor?.execute {
                runMicRecordingLoop(
                    recorder,
                    listener,
                    sampleRate,
                    readChunkSize,
                    recognitionChunkSize
                )
            }
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start microphone recording", e)
            releaseAudioRecord()
            shutdownMicExecutors()
            micSessionActive.set(false)
            listener.onError(e.message ?: "Failed to start microphone recording")
            false
        }
    }

    private fun newMicExecutor(name: String, priority: Int): ExecutorService {
        return Executors.newSingleThreadExecutor { runnable ->
            Thread({
                android.os.Process.setThreadPriority(priority)
                runnable.run()
            }, name)
        }
    }

    private fun runMicRecordingLoop(
        recorder: AudioRecord,
        listener: MicRecordingListener,
        sampleRate: Int,
        readChunkSize: Int,
        recognitionChunkSize: Int
    ) {
        val readBuffer = FloatArray(readChunkSize)
        val recognitionBuffer = FloatArray(recognitionChunkSize)
        var recognitionFill = 0

        try {
            while (micRecording.get()) {
                val readResult = recorder.read(
                    readBuffer,
                    0,
                    readBuffer.size,
                    AudioRecord.READ_BLOCKING
                )
                if (readResult < 0) {
                    micRecording.set(false)
                    listener.onError("AudioRecord read failed: $readResult")
                    break
                }
                if (readResult == 0) {
                    continue
                }

                listener.onWaveform(readBuffer.copyOf(readResult), sampleRate)

                var sourceOffset = 0
                while (sourceOffset < readResult) {
                    val copySize = minOf(
                        readResult - sourceOffset,
                        recognitionChunkSize - recognitionFill
                    )
                    System.arraycopy(
                        readBuffer,
                        sourceOffset,
                        recognitionBuffer,
                        recognitionFill,
                        copySize
                    )
                    sourceOffset += copySize
                    recognitionFill += copySize

                    if (recognitionFill == recognitionChunkSize) {
                        submitMicChunk(recognitionBuffer.copyOf(), listener, sampleRate)
                        recognitionFill = 0
                    }
                }
            }
        } catch (e: Exception) {
            if (micRecording.get()) {
                Log.e(TAG, "Microphone recording loop failed", e)
                listener.onError(e.message ?: "Microphone recording failed")
            }
        } finally {
            micRecording.set(false)
            releaseAudioRecord(recorder)
            micReadExecutor?.shutdown()
            micReadExecutor = null

            if (finalizeMicInput.get()) {
                submitMicFinalization(
                    recognitionBuffer.copyOf(recognitionFill),
                    listener,
                    sampleRate
                )
            } else {
                micRecognitionExecutor?.shutdownNow()
                micRecognitionExecutor = null
                micSessionActive.set(false)
            }
        }
    }

    private fun submitMicChunk(
        chunk: FloatArray,
        listener: MicRecordingListener,
        sampleRate: Int
    ) {
        micRecognitionExecutor?.execute {
            if (!isInitialized) return@execute
            try {
                val lines = pushLiveAudio(chunk, 1, sampleRate)
                if (lines.isNotEmpty()) {
                    listener.onResult(lines, false)
                }
            } catch (e: Exception) {
                Log.e(TAG, "pushLiveAudio failed", e)
                listener.onError(e.message ?: "Live speech recognition failed")
            }
        }
    }

    private fun submitMicFinalization(
        tail: FloatArray,
        listener: MicRecordingListener,
        sampleRate: Int
    ) {
        val executor = micRecognitionExecutor
        if (executor == null) {
            micSessionActive.set(false)
            return
        }
        executor.execute {
            try {
                if (isInitialized && tail.isNotEmpty()) {
                    val lines = pushLiveAudio(tail, 1, sampleRate)
                    if (lines.isNotEmpty()) {
                        listener.onResult(lines, false)
                    }
                }
                val finalLines = if (isInitialized) finalizeLiveAudio() else emptyList()
                listener.onResult(finalLines, true)
            } catch (e: Exception) {
                Log.e(TAG, "finalizeLiveAudio failed", e)
                listener.onError(e.message ?: "Failed to finalize live speech recognition")
            } finally {
                micSessionActive.set(false)
                synchronized(this) {
                    if (micRecognitionExecutor === executor) {
                        micRecognitionExecutor = null
                    }
                }
            }
        }
        executor.shutdown()
    }

    /** 録音を停止する。通常停止時は1秒未満の末尾も投入して認識を確定する。 */
    fun stopMicRecording(finalize: Boolean = true) {
        if (!micSessionActive.get()) return
        finalizeMicInput.set(finalize)
        micRecording.set(false)
        releaseAudioRecord()
    }

    private fun releaseAudioRecord(expected: AudioRecord? = null) {
        val recorder = synchronized(this) {
            if (expected != null && audioRecord !== expected) {
                null
            } else {
                val current = audioRecord
                audioRecord = null
                current
            }
        } ?: return
        try {
            if (recorder.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                recorder.stop()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to stop AudioRecord: ${e.message}")
        }
        try {
            recorder.release()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to release AudioRecord: ${e.message}")
        }
    }

    private fun shutdownMicExecutors() {
        micReadExecutor?.shutdownNow()
        micReadExecutor = null
        micRecognitionExecutor?.shutdownNow()
        micRecognitionExecutor = null
    }

    private fun formatTimeStamp(sec: Float): String {
        val total = sec.toInt()
        return "%02d:%02d".format(total / 60, total % 60)
    }

    /**
     * Collects transcript lines from the speech engine in the
     * meeting-minutes format used by ailia-models-flutter:
     * "[mm:ss - mm:ss] text". When diarization is enabled, each line
     * is prefixed with the speaker ID.
     */
    private fun collectTextLines(): List<String> {
        val count: Int? = speech?.getTextCount()
        Log.i(TAG, "Speech getTextCount=$count")
        if (count == null || count == 0) {
            return emptyList()
        }
        val lines = mutableListOf<String>()
        for (i in 0 until count) {
            val text: AiliaSpeechText? = speech?.getText(i)
            if (text == null) {
                continue
            }
            val stamp = "[${formatTimeStamp(text.timeStampBegin)} - ${formatTimeStamp(text.timeStampEnd)}]"
            if (diarizationEnabled && text.speakerId.toLong() and 0xFFFFFFFFL != AiliaSpeech.AILIA_SPEECH_SPEAKER_ID_UNKNOWN.toLong() and 0xFFFFFFFFL) {
                Log.i(TAG, "Speech text[$i]: speaker=#${text.speakerId} '${text.text}' confidence=${text.confidence}")
                lines.add("$stamp [Speaker ${text.speakerId}] ${text.text}")
            } else {
                Log.i(TAG, "Speech text[$i]: '${text.text}' confidence=${text.confidence}")
                lines.add("$stamp ${text.text}")
            }
        }
        Log.i(TAG, "Speech result lines: $lines")
        return lines
    }

    fun releaseSpeech() {
        isInitialized = false
        stopMicRecording(finalize = false)
        shutdownMicExecutors()
        micSessionActive.set(false)
        try {
            speech?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing speech: ${e.javaClass.name}: ${e.message}")
        } finally {
            speech = null
            speechIntermediateCallback = null
            liveModeEnabled = false
            isInitialized = false
            Log.i(TAG, "Speech released")
        }
    }
}
