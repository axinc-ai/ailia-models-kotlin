package jp.axinc.ailia_kotlin

import android.os.Environment
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.net.URL
import java.nio.file.Files

import axip.ailia_speech.AiliaSpeech

class AiliaSpeechSample {
    private var speech: AiliaSpeech? = null
    private var isInitialized = false
    private var lastTokenizationResult: String = ""

    fun download(link : String, name : String): String {
        val dir : String = Environment.getExternalStoragePublicDirectory(android.os.Environment.DIRECTORY_DOWNLOADS).absolutePath
        val path : String = dir + "/" + name
        try {
            if (File(path).exists()) {
                return path;
            }
            URL(link).openStream().copyTo(FileOutputStream(File(path)))
        } catch (e: Exception) {
            Log.e("AILIA", "Cancel", e)
        }
        return path
    }

    fun initializeSpeech(): Boolean {
        return try {
            var encoder_path : String = download("https://storage.googleapis.com/ailia-models/whisper/encoder_tiny.opt3.onnx", "encoder_tiny.onnx")
            var decoder_path : String  = download("https://storage.googleapis.com/ailia-models/whisper/decoder_tiny_fix_kv_cache.opt3.onnx", "decoder_tiny.onnx")

            if (isInitialized) {
                releaseSpeech()
            }

            speech = AiliaSpeech(AiliaSpeech.AILIA_SPEECH_TASK_TRANSCRIBE)
            speech?.openModel(encoder_path, decoder_path, AiliaSpeech.AILIA_SPEECH_MODEL_TYPE_WHISPER_MULTILINGUAL_TINY)
            isInitialized = true
            Log.i("AILIA_Main", "Speech initialized successfully")
            true
        } catch (e: Exception) {
            Log.e("AILIA_Error", "Failed to initialize speech: ${e.javaClass.name}: ${e.message}")
            releaseSpeech()
            false
        }
    }

    fun Process() {

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
