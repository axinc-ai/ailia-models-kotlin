package jp.axinc.ailia_kotlin

import android.util.Log
import axip.ailia_speech.AiliaSpeech

class AiliaSpeechSample {
    private var speech: AiliaSpeech? = null
    private var isInitialized = false
    private var lastTokenizationResult: String = ""

    fun initializeSpeech(): Boolean {
        return try {
            if (isInitialized) {
                releaseSpeech()
            }

            speech = AiliaSpeech(AiliaSpeech.AILIA_SPEECH_TASK_TRANSCRIBE)
            isInitialized = true
            Log.i("AILIA_Main", "Speech initialized successfully")
            true
        } catch (e: Exception) {
            Log.e("AILIA_Error", "Failed to initialize speech: ${e.javaClass.name}: ${e.message}")
            releaseSpeech()
            false
        }
    }

    /*
    fun processTokenization(text: String): Long {
        if (!isInitialized || tokenizer == null) {
            Log.e("AILIA_Error", "Tokenizer not initialized")
            return -1
        }

        return try {
            val startTime = System.nanoTime()
            
            val tokens = tokenizer!!.encode(text)
            var tokensText = ""
            for (i in tokens.indices) {
                tokensText += "${tokens[i]}"
                if (i < tokens.size - 1) tokensText += ", "
            }
            lastTokenizationResult = tokensText
            Log.i("AILIA_Main", "Tokens: $tokensText")
            
            val endTime = System.nanoTime()
            (endTime - startTime) / 1000000
        } catch (e: Exception) {
            Log.e("AILIA_Error", "Failed to process tokenization: ${e.javaClass.name}: ${e.message}")
            -1
        }
    }
    */

    fun releaseSpeech() {
        try {
            speech?.close()
        } catch (e: Exception) {
            Log.e("AILIA_Error", "Error releasing speech: ${e.javaClass.name}: ${e.message}")
        } finally {
            speech = null
            isInitialized = false
            Log.i("AILIA_Main", "Speech released")
        }
    }

    /*
    fun getLastTokenizationResult(): String {
        return lastTokenizationResult
    }
    */
}
