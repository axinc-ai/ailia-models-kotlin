package jp.axinc.ailia_kotlin

import android.content.Context
import android.util.Log
import axip.ailia_llm.AiliaLLM
import axip.ailia_llm.AiliaLLMChatMessage
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Available LLM models (URLs follow ailia-models-flutter: /gemma/<fileName>).
 */
enum class LLMModelType(val displayName: String, val fileName: String) {
    GEMMA_4_E2B("Gemma 4 E2B", "gemma-4-E2B-it-Q4_K_M.gguf"),
    GEMMA_4_E4B("Gemma 4 E4B", "gemma-4-E4B-it-Q4_K_M.gguf"),
    GEMMA_2_2B("Gemma 2 2B", "gemma-2-2b-it-Q4_K_M.gguf"),
}

/**
 * Sample class demonstrating ailia LLM inference for text generation.
 */
class AiliaLLMSample {
    private var llm: AiliaLLM? = null
    private var isInitialized = false
    private var lastResult: String = ""
    private var modelPath: String? = null
    private val conversationHistory = mutableListOf<AiliaLLMChatMessage>()
    private val cancelRequested = AtomicBoolean(false)

    var modelType: LLMModelType = LLMModelType.GEMMA_4_E2B

    companion object {
        private const val TAG = "AiliaLLMSample"
        private const val N_CTX = 8192 // Context window size
        private const val MAX_GENERATION_STEPS = 4096
    }

    interface LLMListener {
        fun onToken(token: String)
        fun onComplete(fullResponse: String)
        fun onError(error: String)
    }

    /**
     * Downloads and initializes the selected LLM model ([modelType]).
     * This is a blocking operation that should be called on a background thread.
     *
     * @param context The Android context
     * @param progressListener Optional listener for download progress
     * @return true if initialization succeeded, false otherwise
     */
    fun initialize(
        context: Context,
        progressListener: ModelDownloader.DownloadListener? = null
    ): Boolean {
        return try {
            if (isInitialized) {
                release()
            }

            Log.i(TAG, "Downloading ${modelType.displayName} model (${modelType.fileName})...")
            val modelFile = ModelDownloader.downloadLLMModel(context, modelType.fileName, progressListener)
            if (modelFile == null) {
                Log.e(TAG, "Failed to download model")
                return false
            }
            modelPath = modelFile.absolutePath

            Log.i(TAG, "Creating AiliaLLM instance...")
            llm = AiliaLLM()

            Log.i(TAG, "Opening model file: $modelPath")
            llm!!.openModelFile(modelPath!!, N_CTX)

            // Set default sampling parameters
            llm!!.setSamplingParams(40, 0.9f, 0.4f, 1234)

            // Add system prompt
            conversationHistory.clear()
            conversationHistory.add(AiliaLLMChatMessage("system", "You are a helpful assistant. Keep your responses brief and concise."))

            isInitialized = true
            Log.i(TAG, "LLM initialized successfully. Context size: ${llm!!.getContextSize()}")
            true

        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize LLM: ${e.message}", e)
            release()
            false
        }
    }

    /**
     * Checks if the model is already downloaded.
     */
    fun isModelDownloaded(context: Context): Boolean {
        return ModelDownloader.isLLMModelDownloaded(context, modelType.fileName)
    }

    /**
     * Generates a response for the given user input.
     * This is a blocking operation that should be called on a background thread.
     *
     * @param userInput The user's message
     * @param listener Optional listener for streaming tokens
     * @return The processing time in milliseconds
     */
    fun chat(userInput: String, listener: LLMListener? = null): Long {
        if (!isInitialized || llm == null) {
            Log.e(TAG, "LLM not initialized")
            listener?.onError("LLM not initialized")
            return -1
        }

        val historySizeBeforeRequest = conversationHistory.size
        return try {
            cancelRequested.set(false)
            val startTime = System.nanoTime()

            // Add user message to conversation history
            conversationHistory.add(AiliaLLMChatMessage("user", userInput))

            // Set the prompt
            llm!!.setPrompt(conversationHistory.toTypedArray())

            // Generate response token by token
            val responseBuilder = StringBuilder()
            var done = false
            var generationSteps = 0

            while (!done && !cancelRequested.get() && generationSteps < MAX_GENERATION_STEPS) {
                done = llm!!.generate()
                generationSteps++
                val token = llm!!.getDeltaText()
                if (token.isNotEmpty()) {
                    responseBuilder.append(token)
                    listener?.onToken(token)
                }
            }

            if (cancelRequested.get()) {
                while (conversationHistory.size > historySizeBeforeRequest) conversationHistory.removeAt(conversationHistory.lastIndex)
                listener?.onError("Generation cancelled")
                return -1
            }
            if (!done) {
                while (conversationHistory.size > historySizeBeforeRequest) conversationHistory.removeAt(conversationHistory.lastIndex)
                listener?.onError("Generation stopped after $MAX_GENERATION_STEPS steps")
                return -1
            }

            val fullResponse = responseBuilder.toString()
            lastResult = fullResponse

            // Add assistant response to conversation history
            conversationHistory.add(AiliaLLMChatMessage("assistant", fullResponse))

            val endTime = System.nanoTime()
            val processingTime = (endTime - startTime) / 1000000

            listener?.onComplete(fullResponse)
            Log.i(TAG, "Chat completed in ${processingTime}ms. Response: $fullResponse")

            processingTime

        } catch (e: Exception) {
            while (conversationHistory.size > historySizeBeforeRequest) conversationHistory.removeAt(conversationHistory.lastIndex)
            Log.e(TAG, "Failed to generate response: ${e.message}", e)
            listener?.onError("Failed to generate: ${e.message}")
            -1
        }
    }

    /** Requests the blocking generation loop to stop at the next token boundary. */
    fun cancelGeneration() {
        cancelRequested.set(true)
    }

    /**
     * Clears the conversation history and resets the context.
     */
    fun clearHistory() {
        conversationHistory.clear()
        conversationHistory.add(AiliaLLMChatMessage("system", "You are a helpful assistant."))
        Log.i(TAG, "Conversation history cleared")
    }

    /**
     * Gets the last generated response.
     */
    fun getLastResult(): String {
        return lastResult
    }

    /**
     * Gets the number of backends available.
     */
    fun getBackendCount(): Int {
        return try {
            AiliaLLM.getBackendCount()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get backend count: ${e.message}")
            0
        }
    }

    /**
     * Gets the name of a backend by index.
     */
    fun getBackendName(index: Int): String {
        return try {
            AiliaLLM.getBackendName(index)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get backend name: ${e.message}")
            "Unknown"
        }
    }

    /**
     * Releases the LLM resources.
     */
    fun release() {
        cancelGeneration()
        try {
            llm?.destroy()
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing LLM: ${e.message}")
        } finally {
            llm = null
            isInitialized = false
            modelPath = null
            conversationHistory.clear()
            Log.i(TAG, "LLM released")
        }
    }
}
