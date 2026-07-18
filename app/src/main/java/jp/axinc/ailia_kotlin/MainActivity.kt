package jp.axinc.ailia_kotlin

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.BitmapFactory.Options
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import axip.ailia.*
import axip.ailia_tflite.*
import axip.ailia_llm.AiliaLLM
import java.io.*
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity() {
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var imageView: ImageView
    private lateinit var cameraPreviewView: PreviewView
    private lateinit var modeRadioGroup: RadioGroup
    private lateinit var algorithmSpinner: Spinner
    private lateinit var envSpinner: Spinner
    private lateinit var processingTimeTextView: TextView
    private lateinit var resultScrollView: ScrollView
    private lateinit var classificationResultTextView: TextView
    private lateinit var tokenizerInputEditText: EditText
    private lateinit var tokenizerOutputTextView: TextView
    private lateinit var trackingResultTextView: TextView
    private lateinit var voiceInputEditText: EditText
    private lateinit var voiceEnvSpinner: Spinner
    private lateinit var voiceReplayButton: Button
    private lateinit var voiceStatusTextView: TextView
    private lateinit var voiceGenerateButton: Button
    private lateinit var voiceResultTextView: TextView
    private lateinit var llmInputLabel: TextView
    private lateinit var llmInputEditText: EditText
    private lateinit var llmSendButton: Button
    private lateinit var llmOutputLabel: TextView
    private lateinit var llmChatContainer: LinearLayout
    private lateinit var llmStatusTextView: TextView
    private lateinit var multimodalImageView: ImageView
    private lateinit var speechLanguageLabel: TextView
    private lateinit var speechLanguageSpinner: Spinner
    private lateinit var speechModeRadioGroup: RadioGroup
    private lateinit var diarizationCheckBox: CheckBox
    private lateinit var liveModeCheckBox: CheckBox
    private lateinit var cameraSpinner: Spinner
    private lateinit var speechRunButton: Button
    private lateinit var micRecordButton: Button
    private lateinit var waveformView: WaveformView
    private lateinit var waveformInfoTextView: TextView
    private lateinit var voiceWaveformView: WaveformView
    private lateinit var modelSpinner: Spinner
    private lateinit var transcriptTextView: TextView
    private lateinit var rootScrollView: ScrollView
    private lateinit var visionRunButton: Button

    // 画像系/Tokenizeアルゴリズムは、Runボタンが押されるまで
    // モデルのダウンロードと実行を行わない
    private var runRequested = false

    // 議事録風トランスクリプト([mm:ss - mm:ss] text の蓄積)
    private val speechTranscript = mutableListOf<String>()

    private var poseEstimatorSample = AiliaPoseEstimatorSample()
    private var objectDetectionSample = AiliaTFLiteObjectDetectionSample()
    private var classificationSample = AiliaTFLiteClassificationSample()
    private var miniLMv2Sample = AiliaMiniLMv2Sample()
    private var trackerSample = AiliaTrackerSample()
    private var speechSample = AiliaSpeechSample()
    private var voiceSample = AiliaVoiceSample()
    private var llmSample = AiliaLLMSample()
    private var multimodalLLMSample = AiliaMultimodalLLMSample()
    private var onnxObjectDetectionSample = AiliaOnnxObjectDetectionSample()
    private var onnxClassificationSample = AiliaOnnxClassificationSample()

    private var selectedEnvId: Int = 0
    private var selectedRuntime: String = "TFLite"
    private var ailiaEnvironments: List<AiliaEnvironment>? = null
    private var isInitialized = false
    private var currentAlgorithm = AlgorithmType.POSE_ESTIMATION
    private var pendingAlgorithmSwitch: AlgorithmType? = null
    private var pendingModeSwitch: Int? = null
    private var isProcessing = AtomicBoolean(false)
    private var isWaitAlgorithmSwitch = AtomicBoolean(false)
    private var isWaitModeSwitch = AtomicBoolean(false)
    private var isStopCamera = AtomicBoolean(false)
    private var isDownloadingModel = AtomicBoolean(false)

    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var latestCameraBitmap: Bitmap? = null

    // カメラ選択(リストボックスで切り替え)
    private data class CameraChoice(val name: String, val selector: CameraSelector)
    private val cameraChoices = listOf(
        CameraChoice("Back Camera", CameraSelector.DEFAULT_BACK_CAMERA),
        CameraChoice("Front Camera", CameraSelector.DEFAULT_FRONT_CAMERA),
    )
    private var selectedCameraIndex = 0

    // 音声認識処理用の低優先度Executor。カメラ/マイク読み出しと分離して
    // transcribe中もUI(波形表示等)がカクつかないようにする。
    private lateinit var speechExecutor: ExecutorService

    private var selectedVoiceModelType: VoiceModelType = VoiceModelType.GPT_SOVITS_V1
    private var selectedSpeechModelType: SpeechModelType = SpeechModelType.WHISPER_TINY
    private var selectedSpeechLanguage: String = "ja"
    private var selectedLLMModelType: LLMModelType = LLMModelType.GEMMA_4_E2B
    private var audioRecord: AudioRecord? = null
    private var isRecording = AtomicBoolean(false)

    // マイク録音のREC経過時間表示用
    private val recTimerHandler = android.os.Handler(android.os.Looper.getMainLooper())
    private var recStartMs: Long = 0
    private val recTimerRunnable = object : Runnable {
        override fun run() {
            if (isRecording.get()) {
                val elapsedSec = (android.os.SystemClock.elapsedRealtime() - recStartMs) / 1000
                waveformInfoTextView.text = "● REC %02d:%02d".format(elapsedSec / 60, elapsedSec % 60)
                recTimerHandler.postDelayed(this, 500)
            }
        }
    }

    enum class AlgorithmType {
        POSE_ESTIMATION,
        OBJECT_DETECTION,
        TRACKING,
        TOKENIZE,
        CLASSIFICATION,
        SPEECH_TO_TEXT,
        TEXT_TO_SPEECH,
        LLM,
        MULTIMODAL_LLM,
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)

        init {
            System.loadLibrary("ailia")
            System.loadLibrary("ailia_llm")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Android 15 (targetSdk 35) 以降はエッジツーエッジが強制されるため、
        // システムバー/ディスプレイカットアウトのインセットを上下左右のpaddingとして適用する。
        // ActionBar が消費した後の残りのインセット(ステータスバー分は消費済み)を
        // コンテンツのルートに反映することで、ナビゲーションバー/ノッチとの被りを防ぐ。
        val rootLayout = findViewById<View>(R.id.rootLayout)
        ViewCompat.setOnApplyWindowInsetsListener(rootLayout) { v, insets ->
            val bars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout()
            )
            v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }

        // cameraExecutorはsetupModeSelection()より先に初期化する必要がある
        // (SpinnerのonItemSelectedでinitializeAilia()が呼ばれるため)
        cameraExecutor = Executors.newSingleThreadExecutor()

        // 音声認識はバックグラウンド優先度の専用スレッドで実行し、
        // 推論中でもUIスレッドがCPUを確保できるようにする
        speechExecutor = Executors.newSingleThreadExecutor { r ->
            Thread {
                android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_BACKGROUND)
                r.run()
            }
        }

        val modelDir = (getExternalFilesDir(null) ?: filesDir).absolutePath
        onnxObjectDetectionSample.modelDir = modelDir
        onnxClassificationSample.modelDir = modelDir
        miniLMv2Sample.modelDir = modelDir
        speechSample.modelDir = modelDir
        voiceSample.modelDir = modelDir

        initializeViews()
        adjustContentSizeForScreen()
        setupModeSelection()
        updateUIVisibility()

        if (allPermissionsGranted()) {
            initializeAilia()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }
    }

    private fun initializeViews() {
        imageView = findViewById(R.id.imageView)
        cameraPreviewView = findViewById(R.id.cameraPreviewView)
        modeRadioGroup = findViewById(R.id.modeRadioGroup)
        algorithmSpinner = findViewById(R.id.algorithmSpinner)
        envSpinner = findViewById(R.id.envSpinner)
        processingTimeTextView = findViewById(R.id.processingTimeTextView)
        resultScrollView = findViewById(R.id.resultScrollView)
        classificationResultTextView = findViewById(R.id.classificationResultTextView)
        tokenizerInputEditText = findViewById(R.id.tokenizerInputEditText)
        tokenizerOutputTextView = findViewById(R.id.tokenizerOutputTextView)
        trackingResultTextView = findViewById(R.id.trackingResultTextView)
        voiceInputEditText = findViewById(R.id.voiceInputEditText)
        voiceEnvSpinner = findViewById(R.id.voiceEnvSpinner)
        voiceReplayButton = findViewById(R.id.voiceReplayButton)
        voiceStatusTextView = findViewById(R.id.voiceStatusTextView)
        voiceGenerateButton = findViewById(R.id.voiceGenerateButton)
        voiceResultTextView = findViewById(R.id.voiceResultTextView)
        llmInputLabel = findViewById(R.id.llmInputLabel)
        llmInputEditText = findViewById(R.id.llmInputEditText)
        llmSendButton = findViewById(R.id.llmSendButton)
        llmOutputLabel = findViewById(R.id.llmOutputLabel)
        llmChatContainer = findViewById(R.id.llmChatContainer)
        llmStatusTextView = findViewById(R.id.llmStatusTextView)
        multimodalImageView = findViewById(R.id.multimodalImageView)
        speechLanguageLabel = findViewById(R.id.speechLanguageLabel)
        speechLanguageSpinner = findViewById(R.id.speechLanguageSpinner)
        speechModeRadioGroup = findViewById(R.id.speechModeRadioGroup)
        diarizationCheckBox = findViewById(R.id.diarizationCheckBox)
        liveModeCheckBox = findViewById(R.id.liveModeCheckBox)
        cameraSpinner = findViewById(R.id.cameraSpinner)
        transcriptTextView = findViewById(R.id.transcriptTextView)
        speechRunButton = findViewById(R.id.speechRunButton)
        micRecordButton = findViewById(R.id.micRecordButton)
        waveformView = findViewById(R.id.waveformView)
        waveformInfoTextView = findViewById(R.id.waveformInfoTextView)
        voiceWaveformView = findViewById(R.id.voiceWaveformView)
        modelSpinner = findViewById(R.id.modelSpinner)
        rootScrollView = findViewById(R.id.rootScrollView)
        visionRunButton = findViewById(R.id.visionRunButton)
    }

    private fun setupModeSelection() {
        val algorithms = arrayOf(
            "PoseEstimation",
            "ObjectDetection",
            "Tracking",
            "Tokenize",
            "Classification",
            "Speech2Text",
            "Text2Speech",
            "LLM",
            "MultimodalLLM",
        )

        // 全体メニュー風の見た目にするため専用のitemレイアウト(白太字・中央寄せ)を使う
        val adapter = ArrayAdapter(this, R.layout.spinner_item_menu, algorithms)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        algorithmSpinner.adapter = adapter

        algorithmSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(
                parent: AdapterView<*>?,
                view: View?,
                position: Int,
                id: Long
            ) {
                val newAlgorithm = AlgorithmType.values()[position]
                updateEnvSpinner(newAlgorithm)
                updateModelSpinner(newAlgorithm)
                if (newAlgorithm != currentAlgorithm) {
                    switchAlgorithm(newAlgorithm)
                }
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        modeRadioGroup.setOnCheckedChangeListener { _, checkedId ->
            when (checkedId) {
                R.id.imageRadioButton -> {
                    switchToImageMode()
                }

                R.id.cameraRadioButton -> {
                    switchToCameraMode()
                }
            }
        }

        // 初回起動時、最初のモデル初期化より前にデフォルトのenv(GPU等)を確定させる。
        // これを行わないと selectedEnvId=0 のまま初期化され、後から遅延実行される
        // algorithmSpinner の onItemSelected が envスピナーの表示だけをデフォルト値に
        // 更新するため「表示env≠実際に使われるenv」の不一致が起きる。
        updateEnvSpinner(currentAlgorithm)
        updateModelSpinner()
        setupCameraSpinner()
        setupVisionRunButton()

        switchToImageMode()
    }

    /**
     * 画像/カメラの表示領域を画面サイズに合わせて調整する。
     * 横画面でも高さが画面内に収まるよう、正方形の一辺を
     * 「画面幅(マージン込み)」と「画面高さの60%」の小さい方にする。
     * 画面回転時はActivityが再生成されるためonCreateで再計算される。
     */
    private fun adjustContentSizeForScreen() {
        val dm = resources.displayMetrics
        val horizontalMargin = (32 * dm.density).toInt()
        val side = minOf(dm.widthPixels - horizontalMargin, (dm.heightPixels * 0.6f).toInt())
        for (view in listOf<View>(imageView, cameraPreviewView)) {
            val lp = view.layoutParams
            lp.width = side
            lp.height = side
            view.layoutParams = lp
        }
    }

    /** Runボタンが必要な(押すまで実行しない)アルゴリズムかどうか */
    private fun needsVisionRunButton(algorithm: AlgorithmType = currentAlgorithm): Boolean {
        return when (algorithm) {
            AlgorithmType.POSE_ESTIMATION,
            AlgorithmType.OBJECT_DETECTION,
            AlgorithmType.TRACKING,
            AlgorithmType.CLASSIFICATION,
            AlgorithmType.TOKENIZE -> true
            else -> false
        }
    }

    /** Run状態をリセットしてボタン表記をRunに戻す */
    private fun resetRunState() {
        runRequested = false
        visionRunButton.text = "Run"
    }

    private fun setupVisionRunButton() {
        visionRunButton.setOnClickListener {
            if (isDownloadingModel.get()) {
                return@setOnClickListener
            }
            val isCameraMode = modeRadioGroup.checkedRadioButtonId == R.id.cameraRadioButton &&
                currentAlgorithm != AlgorithmType.TOKENIZE
            if (isCameraMode) {
                if (runRequested) {
                    // Stop: フレーム処理を停止する(プレビューは継続)
                    resetRunState()
                    processingTimeTextView.text = "Stopped. Press Run to restart"
                } else {
                    runRequested = true
                    visionRunButton.text = "Stop"
                    // CameraFrameAnalyzerが次フレームから処理を開始する
                }
            } else {
                runRequested = true
                processImageMode()
            }
        }
    }

    private fun setupCameraSpinner() {
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, cameraChoices.map { it.name }.toTypedArray())
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        cameraSpinner.adapter = adapter
        cameraSpinner.setSelection(selectedCameraIndex)
        cameraSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (position != selectedCameraIndex) {
                    selectedCameraIndex = position
                    // カメラモード中なら選択したカメラで再バインドする
                    if (modeRadioGroup.checkedRadioButtonId == R.id.cameraRadioButton) {
                        startCamera()
                    }
                }
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
    }

    private fun setupOnnxEnvSpinner(useBlas: Boolean, target: Spinner = envSpinner) {
        try {
            if (ailiaEnvironments == null) {
                Ailia.SetTemporaryCachePath(cacheDir.absolutePath)
                ailiaEnvironments = AiliaModel.getEnvironments()
            }
            val envNames = ailiaEnvironments!!.map { "${it.name} (id:${it.id})" }.toTypedArray()
            val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, envNames)
            adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
            target.adapter = adapter

            var defaultIndex = 0
            if (useBlas) {
                // デフォルトはBLAS (CPU-OpenBlas)
                for ((index, env) in ailiaEnvironments!!.withIndex()) {
                    if (env.name.contains("OpenBlas", ignoreCase = true)) {
                        defaultIndex = index
                        break
                    }
                }
            } else {
                // デフォルトはGPU
                for ((index, env) in ailiaEnvironments!!.withIndex()) {
                    if (env.type == AiliaEnvironment.TYPE_GPU && env.props and AiliaEnvironment.PROPERTY_FP16 == 0) {
                        defaultIndex = index
                        break
                    }
                }
            }
            target.setSelection(defaultIndex)
            selectedEnvId = ailiaEnvironments!![defaultIndex].id

            target.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                    val newEnvId = ailiaEnvironments!![position].id
                    if (newEnvId != selectedEnvId) {
                        selectedEnvId = newEnvId
                        isInitialized = false
                        isDownloadingModel.set(false)
                        resetRunState()
                        val isImageMode = modeRadioGroup.checkedRadioButtonId == R.id.imageRadioButton
                        if (isImageMode) {
                            releaseCurrentAlgorithm()
                            processImageMode()
                        } else {
                            // Camera mode: cameraExecutorでリリース→再初期化
                            cameraExecutor.execute {
                                releaseCurrentAlgorithm()
                            }
                        }
                    }
                }
                override fun onNothingSelected(parent: AdapterView<*>?) {}
            }

            target.visibility = View.VISIBLE
        } catch (e: Exception) {
            Log.e("AILIA_Main", "Failed to get ailia environments: ${e.message}")
            target.visibility = View.GONE
        }
    }

    private fun updateEnvSpinner(algorithm: AlgorithmType) {
        when (algorithm) {
            AlgorithmType.POSE_ESTIMATION -> {
                setupOnnxEnvSpinner(useBlas = false)
            }

            AlgorithmType.SPEECH_TO_TEXT, AlgorithmType.TOKENIZE -> {
                setupOnnxEnvSpinner(useBlas = true)
            }

            AlgorithmType.TEXT_TO_SPEECH -> {
                // TTSのバックエンド選択はGenerateボタンの右側に表示する
                envSpinner.visibility = View.GONE
                setupOnnxEnvSpinner(useBlas = true, target = voiceEnvSpinner)
            }

            AlgorithmType.OBJECT_DETECTION, AlgorithmType.CLASSIFICATION, AlgorithmType.TRACKING -> {
                if (selectedRuntime == "ONNX") {
                    setupOnnxEnvSpinner(useBlas = true)
                } else {
                    // TFLite: Reference (CPU) と NNAPI を表示
                    val tfliteEnvNames = arrayOf("Reference (CPU)", "NNAPI")
                    val tfliteEnvIds = intArrayOf(AiliaTFLite.AILIA_TFLITE_ENV_REFERENCE, AiliaTFLite.AILIA_TFLITE_ENV_NNAPI)
                    val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, tfliteEnvNames)
                    adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
                    envSpinner.adapter = adapter

                    // デフォルトは NNAPI
                    envSpinner.setSelection(1)
                    selectedEnvId = AiliaTFLite.AILIA_TFLITE_ENV_NNAPI

                    envSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
                        override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                            val newEnvId = tfliteEnvIds[position]
                            if (newEnvId != selectedEnvId) {
                                selectedEnvId = newEnvId
                                isInitialized = false
                                isDownloadingModel.set(false)
                                resetRunState()
                                val isImageMode = modeRadioGroup.checkedRadioButtonId == R.id.imageRadioButton
                                if (isImageMode) {
                                    releaseCurrentAlgorithm()
                                    processImageMode()
                                } else {
                                    cameraExecutor.execute {
                                        releaseCurrentAlgorithm()
                                    }
                                }
                            }
                        }
                        override fun onNothingSelected(parent: AdapterView<*>?) {}
                    }

                    envSpinner.visibility = View.VISIBLE
                }
            }

            else -> {
                envSpinner.visibility = View.GONE
            }
        }
    }

    /**
     * モデル選択スピナーを現在のアルゴリズムに合わせて更新する。
     * S2T/TTS/LLMのモデル選択や、YOLOX等のTFLite/ONNX(ランタイム)選択もここに統合。
     */
    private fun updateModelSpinner(algorithm: AlgorithmType = currentAlgorithm) {
        var selectedIndex = 0
        val items: Array<String> = when (algorithm) {
            AlgorithmType.POSE_ESTIMATION -> {
                selectedRuntime = "ONNX"
                arrayOf("LightweightHumanPose")
            }
            AlgorithmType.OBJECT_DETECTION -> {
                selectedIndex = if (selectedRuntime == "ONNX") 1 else 0
                arrayOf("YOLOX-S (TFLite)", "YOLOX-S (ONNX)")
            }
            AlgorithmType.TRACKING -> {
                selectedIndex = if (selectedRuntime == "ONNX") 1 else 0
                arrayOf("YOLOX-S + ByteTrack (TFLite)", "YOLOX-S + ByteTrack (ONNX)")
            }
            AlgorithmType.CLASSIFICATION -> {
                selectedIndex = if (selectedRuntime == "ONNX") 1 else 0
                arrayOf("MobileNetV2 (TFLite)", "MobileNetV2 (ONNX)")
            }
            AlgorithmType.TOKENIZE -> arrayOf("Multilingual MiniLMv2 (L12)")
            AlgorithmType.SPEECH_TO_TEXT -> {
                selectedIndex = SpeechModelType.values().indexOf(selectedSpeechModelType)
                SpeechModelType.values().map { it.displayName }.toTypedArray()
            }
            AlgorithmType.TEXT_TO_SPEECH -> {
                selectedIndex = VoiceModelType.values().indexOf(selectedVoiceModelType)
                VoiceModelType.values().map { it.displayName }.toTypedArray()
            }
            AlgorithmType.LLM -> {
                selectedIndex = LLMModelType.values().indexOf(selectedLLMModelType)
                LLMModelType.values().map { "${it.displayName} (Q4_K_M)" }.toTypedArray()
            }
            AlgorithmType.MULTIMODAL_LLM -> arrayOf("Gemma-3 4B IT (Q4_K_M)")
        }

        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, items)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        modelSpinner.onItemSelectedListener = null
        modelSpinner.adapter = adapter
        if (selectedIndex in items.indices) {
            modelSpinner.setSelection(selectedIndex, false)
        }
        modelSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                onModelSelected(algorithm, position)
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
    }

    /** モデル選択スピナーの選択変更を各アルゴリズムに反映する */
    private fun onModelSelected(algorithm: AlgorithmType, position: Int) {
        when (algorithm) {
            AlgorithmType.OBJECT_DETECTION, AlgorithmType.CLASSIFICATION, AlgorithmType.TRACKING -> {
                val newRuntime = if (position == 1) "ONNX" else "TFLite"
                if (newRuntime != selectedRuntime) {
                    selectedRuntime = newRuntime
                    updateEnvSpinner(algorithm)
                    isInitialized = false
                    isDownloadingModel.set(false)
                    resetRunState()
                    val isImageMode = modeRadioGroup.checkedRadioButtonId == R.id.imageRadioButton
                    if (isImageMode) {
                        releaseCurrentAlgorithm()
                        processImageMode()
                    } else {
                        cameraExecutor.execute {
                            releaseCurrentAlgorithm()
                        }
                    }
                }
            }
            AlgorithmType.SPEECH_TO_TEXT -> {
                val newType = SpeechModelType.values()[position]
                if (newType != selectedSpeechModelType) {
                    selectedSpeechModelType = newType
                    stopMicRecording()
                    // モデルダウンロードはRun/Record押下時まで遅延する
                    speechSample.releaseSpeech()
                    isInitialized = false
                    isDownloadingModel.set(false)
                    clearTranscript()
                    classificationResultTextView.text = "Speech Result: --"
                }
            }
            AlgorithmType.TEXT_TO_SPEECH -> {
                val newType = VoiceModelType.values()[position]
                if (newType != selectedVoiceModelType) {
                    selectedVoiceModelType = newType
                    // モデル切り替え時は解放のみ(ダウンロードはGenerate押下時)
                    voiceSample.releaseVoice()
                    isInitialized = false
                    voiceWaveformView.clear()
                    voiceReplayButton.visibility = View.GONE
                    voiceResultTextView.text = ""
                    voiceInputEditText.setText(defaultVoiceText(newType))
                    voiceStatusTextView.text = "Status: Press Generate to download model and synthesize"
                }
            }
            AlgorithmType.LLM -> {
                val newType = LLMModelType.values()[position]
                if (newType != selectedLLMModelType) {
                    selectedLLMModelType = newType
                    // モデル切り替え時は解放のみ(ダウンロードはSend押下時)
                    llmSample.release()
                    isInitialized = false
                    llmChatContainer.removeAllViews()
                    llmSendButton.isEnabled = true
                    llmStatusTextView.text = "Status: Press Send to download model and run"
                }
            }
            else -> {}
        }
    }

    private fun processAlgorithm(
        img: ByteArray,
        bitmap: Bitmap,
        canvas: Canvas,
        w: Int,
        h: Int
    ): Long {
        val paint = Paint().apply {
            color = Color.WHITE
        }

        val paint2 = Paint().apply {
            style = Paint.Style.STROKE
            color = Color.RED
            strokeWidth = 3f
        }

        val textPaint = Paint().apply {
            color = Color.BLACK
            textSize = 30f
            isAntiAlias = true
        }

        return when (currentAlgorithm) {
            AlgorithmType.POSE_ESTIMATION -> {
                poseEstimatorSample.processPoseEstimation(img, canvas, paint, w, h)
            }

            AlgorithmType.OBJECT_DETECTION -> {
                if (selectedRuntime == "ONNX") {
                    onnxObjectDetectionSample.processObjectDetection(
                        img, bitmap, canvas, paint2, textPaint, w, h
                    )
                } else {
                    objectDetectionSample.processObjectDetection(
                        bitmap, canvas, paint2, textPaint, w, h
                    )
                }
            }

            AlgorithmType.CLASSIFICATION -> {
                if (selectedRuntime == "ONNX") {
                    val time = onnxClassificationSample.processClassification(img, w, h)
                    val result = onnxClassificationSample.getLastClassificationResult()
                    runOnUiThread {
                        classificationResultTextView.text = "Classification Result: $result"
                    }
                    time
                } else {
                    val time = classificationSample.processClassification(bitmap)
                    val result = classificationSample.getLastClassificationResult()
                    runOnUiThread {
                        classificationResultTextView.text = "Classification Result: $result"
                    }
                    time
                }
            }

            AlgorithmType.TOKENIZE -> {
                val inputText =
                    tokenizerInputEditText.text.toString().ifEmpty { "今日、新しいiPhoneが発売されました" }
                val labels = listOf("スマートフォン", "エンタメ", "スポーツ", "政治", "科学")
                val time = miniLMv2Sample.predict(inputText, labels)
                val result = miniLMv2Sample.getLastResult()
                runOnUiThread {
                    tokenizerOutputTextView.text = "Result:\n$result"
                }
                time
            }

            AlgorithmType.TRACKING -> {
                if (selectedRuntime == "ONNX") {
                    val detectionTime = onnxObjectDetectionSample.processObjectDetectionWithoutDrawing(
                        img, w, h, threshold = 0.1f, iou = 1.0f
                    )
                    val detectionResults = onnxObjectDetectionSample.getDetectionResults()
                    val trackingTime = trackerSample.processTrackingWithDetections(
                        canvas, paint2, w, h, detectionResults
                    )
                    val trackingInfo = trackerSample.getLastTrackingResult()
                    runOnUiThread {
                        trackingResultTextView.text = "Tracking Results: $trackingInfo"
                    }
                    detectionTime + trackingTime
                } else {
                    // First run object detection to get detection results without drawing
                    val detectionTime = objectDetectionSample.processObjectDetectionWithoutDrawing(
                        bitmap, w, h, threshold = 0.1f, iou = 1.0f
                    )
                    val detectionResults = objectDetectionSample.getDetectionResults(bitmap)
                    // Then run tracking with the detection results and draw the tracking results
                    val trackingTime = trackerSample.processTrackingWithDetections(
                        canvas, paint2, w, h, detectionResults
                    )
                    val trackingInfo = trackerSample.getLastTrackingResult()
                    runOnUiThread {
                        trackingResultTextView.text = "Tracking Results: $trackingInfo"
                    }
                    detectionTime + trackingTime
                }
            }

            AlgorithmType.SPEECH_TO_TEXT -> {
                // Speech is handled asynchronously via the Run/Record buttons
                0
            }

            AlgorithmType.TEXT_TO_SPEECH -> {
                // Voice is handled asynchronously via the generate button
                0
            }

            AlgorithmType.LLM, AlgorithmType.MULTIMODAL_LLM -> {
                // LLM modes are handled asynchronously via the send button
                0
            }
        }
    }

    private fun updateUIVisibility() {
        val isImageMode = modeRadioGroup.checkedRadioButtonId == R.id.imageRadioButton
        val isCameraMode = modeRadioGroup.checkedRadioButtonId == R.id.cameraRadioButton

        when (currentAlgorithm) {
            AlgorithmType.TOKENIZE -> {
                modeRadioGroup.visibility = View.GONE
                imageView.visibility = View.GONE
                cameraPreviewView.visibility = View.GONE
                resultScrollView.visibility = View.VISIBLE
                classificationResultTextView.visibility = View.GONE
                tokenizerInputEditText.visibility = View.VISIBLE
                tokenizerOutputTextView.visibility = View.VISIBLE
                trackingResultTextView.visibility = View.GONE
                transcriptTextView.visibility = View.GONE
                findViewById<TextView>(R.id.tokenizerInputLabel).visibility = View.VISIBLE
                findViewById<TextView>(R.id.tokenizerOutputLabel).visibility = View.VISIBLE
                multimodalImageView.visibility = View.GONE
                llmInputLabel.visibility = View.GONE
                llmInputEditText.visibility = View.GONE
                llmSendButton.visibility = View.GONE
                llmOutputLabel.visibility = View.GONE
                llmChatContainer.visibility = View.GONE
                llmStatusTextView.visibility = View.GONE
                voiceInputEditText.visibility = View.GONE
                voiceStatusTextView.visibility = View.GONE
                voiceGenerateButton.visibility = View.GONE
                voiceEnvSpinner.visibility = View.GONE
                voiceResultTextView.visibility = View.GONE
                speechLanguageLabel.visibility = View.GONE
                speechLanguageSpinner.visibility = View.GONE
                speechModeRadioGroup.visibility = View.GONE
                diarizationCheckBox.visibility = View.GONE
                liveModeCheckBox.visibility = View.GONE
                speechRunButton.visibility = View.GONE
                micRecordButton.visibility = View.GONE
                waveformView.visibility = View.GONE
                waveformInfoTextView.visibility = View.GONE
                voiceWaveformView.visibility = View.GONE
                voiceReplayButton.visibility = View.GONE
            }

            AlgorithmType.CLASSIFICATION -> {
                modeRadioGroup.visibility = View.VISIBLE
                if (isImageMode) {
                    imageView.visibility = View.VISIBLE
                    cameraPreviewView.visibility = View.GONE
                } else {
                    imageView.visibility = View.VISIBLE
                    cameraPreviewView.visibility = View.VISIBLE
                }
                resultScrollView.visibility = View.GONE
                classificationResultTextView.visibility = View.GONE
                tokenizerInputEditText.visibility = View.GONE
                tokenizerOutputTextView.visibility = View.GONE
                trackingResultTextView.visibility = View.GONE
                transcriptTextView.visibility = View.GONE
                findViewById<TextView>(R.id.tokenizerInputLabel).visibility = View.GONE
                findViewById<TextView>(R.id.tokenizerOutputLabel).visibility = View.GONE
                multimodalImageView.visibility = View.GONE
                llmInputLabel.visibility = View.GONE
                llmInputEditText.visibility = View.GONE
                llmSendButton.visibility = View.GONE
                llmOutputLabel.visibility = View.GONE
                llmChatContainer.visibility = View.GONE
                llmStatusTextView.visibility = View.GONE
                voiceInputEditText.visibility = View.GONE
                voiceStatusTextView.visibility = View.GONE
                voiceGenerateButton.visibility = View.GONE
                voiceEnvSpinner.visibility = View.GONE
                voiceResultTextView.visibility = View.GONE
                speechLanguageLabel.visibility = View.GONE
                speechLanguageSpinner.visibility = View.GONE
                speechModeRadioGroup.visibility = View.GONE
                diarizationCheckBox.visibility = View.GONE
                liveModeCheckBox.visibility = View.GONE
                speechRunButton.visibility = View.GONE
                micRecordButton.visibility = View.GONE
                waveformView.visibility = View.GONE
                waveformInfoTextView.visibility = View.GONE
                voiceWaveformView.visibility = View.GONE
                voiceReplayButton.visibility = View.GONE
            }

            AlgorithmType.TRACKING -> {
                modeRadioGroup.visibility = View.VISIBLE
                if (isImageMode) {
                    imageView.visibility = View.VISIBLE
                    cameraPreviewView.visibility = View.GONE
                } else {
                    imageView.visibility = View.VISIBLE
                    cameraPreviewView.visibility = View.VISIBLE
                }
                resultScrollView.visibility = View.GONE
                classificationResultTextView.visibility = View.GONE
                tokenizerInputEditText.visibility = View.GONE
                tokenizerOutputTextView.visibility = View.GONE
                trackingResultTextView.visibility = View.GONE
                transcriptTextView.visibility = View.GONE
                findViewById<TextView>(R.id.tokenizerInputLabel).visibility = View.GONE
                findViewById<TextView>(R.id.tokenizerOutputLabel).visibility = View.GONE
                multimodalImageView.visibility = View.GONE
                llmInputLabel.visibility = View.GONE
                llmInputEditText.visibility = View.GONE
                llmSendButton.visibility = View.GONE
                llmOutputLabel.visibility = View.GONE
                llmChatContainer.visibility = View.GONE
                llmStatusTextView.visibility = View.GONE
                voiceInputEditText.visibility = View.GONE
                voiceStatusTextView.visibility = View.GONE
                voiceGenerateButton.visibility = View.GONE
                voiceEnvSpinner.visibility = View.GONE
                voiceResultTextView.visibility = View.GONE
                speechLanguageLabel.visibility = View.GONE
                speechLanguageSpinner.visibility = View.GONE
                speechModeRadioGroup.visibility = View.GONE
                diarizationCheckBox.visibility = View.GONE
                liveModeCheckBox.visibility = View.GONE
                speechRunButton.visibility = View.GONE
                micRecordButton.visibility = View.GONE
                waveformView.visibility = View.GONE
                waveformInfoTextView.visibility = View.GONE
                voiceWaveformView.visibility = View.GONE
                voiceReplayButton.visibility = View.GONE
            }

            AlgorithmType.SPEECH_TO_TEXT -> {
                val isMicMode = speechModeRadioGroup.checkedRadioButtonId == R.id.micRadioButton
                modeRadioGroup.visibility = View.GONE
                imageView.visibility = View.GONE
                cameraPreviewView.visibility = View.GONE
                resultScrollView.visibility = View.VISIBLE
                classificationResultTextView.visibility = View.VISIBLE
                classificationResultTextView.text = "Speech Result: --"
                tokenizerInputEditText.visibility = View.GONE
                tokenizerOutputTextView.visibility = View.GONE
                trackingResultTextView.visibility = View.GONE
                transcriptTextView.visibility = View.VISIBLE
                findViewById<TextView>(R.id.tokenizerInputLabel).visibility = View.GONE
                findViewById<TextView>(R.id.tokenizerOutputLabel).visibility = View.GONE
                multimodalImageView.visibility = View.GONE
                llmInputLabel.visibility = View.GONE
                llmInputEditText.visibility = View.GONE
                llmSendButton.visibility = View.GONE
                llmOutputLabel.visibility = View.GONE
                llmChatContainer.visibility = View.GONE
                llmStatusTextView.visibility = View.GONE
                voiceInputEditText.visibility = View.GONE
                voiceStatusTextView.visibility = View.GONE
                voiceGenerateButton.visibility = View.GONE
                voiceEnvSpinner.visibility = View.GONE
                voiceResultTextView.visibility = View.GONE
                // Speech-specific UI
                // 言語選択はMicモード時のみ(Wavはauto固定)
                speechLanguageLabel.visibility = if (isMicMode) View.VISIBLE else View.GONE
                speechLanguageSpinner.visibility = if (isMicMode) View.VISIBLE else View.GONE
                speechModeRadioGroup.visibility = View.VISIBLE
                diarizationCheckBox.visibility = View.VISIBLE
                liveModeCheckBox.visibility = if (isMicMode) View.VISIBLE else View.GONE
                speechRunButton.visibility = if (isMicMode) View.GONE else View.VISIBLE
                micRecordButton.visibility = if (isMicMode) View.VISIBLE else View.GONE
                waveformView.visibility = if (isMicMode) View.VISIBLE else View.GONE
                waveformInfoTextView.visibility = if (isMicMode) View.VISIBLE else View.GONE
                voiceWaveformView.visibility = View.GONE
                voiceReplayButton.visibility = View.GONE
            }
            AlgorithmType.LLM -> {
                modeRadioGroup.visibility = View.GONE
                imageView.visibility = View.GONE
                cameraPreviewView.visibility = View.GONE
                resultScrollView.visibility = View.VISIBLE
                classificationResultTextView.visibility = View.GONE
                tokenizerInputEditText.visibility = View.GONE
                tokenizerOutputTextView.visibility = View.GONE
                trackingResultTextView.visibility = View.GONE
                transcriptTextView.visibility = View.GONE
                findViewById<TextView>(R.id.tokenizerInputLabel).visibility = View.GONE
                findViewById<TextView>(R.id.tokenizerOutputLabel).visibility = View.GONE
                multimodalImageView.visibility = View.GONE
                llmInputLabel.visibility = View.VISIBLE
                llmInputEditText.visibility = View.VISIBLE
                llmSendButton.visibility = View.VISIBLE
                llmOutputLabel.visibility = View.VISIBLE
                llmChatContainer.visibility = View.VISIBLE
                llmStatusTextView.visibility = View.VISIBLE
                voiceInputEditText.visibility = View.GONE
                voiceStatusTextView.visibility = View.GONE
                voiceGenerateButton.visibility = View.GONE
                voiceEnvSpinner.visibility = View.GONE
                voiceResultTextView.visibility = View.GONE
                speechLanguageLabel.visibility = View.GONE
                speechLanguageSpinner.visibility = View.GONE
                speechModeRadioGroup.visibility = View.GONE
                diarizationCheckBox.visibility = View.GONE
                liveModeCheckBox.visibility = View.GONE
                speechRunButton.visibility = View.GONE
                micRecordButton.visibility = View.GONE
                waveformView.visibility = View.GONE
                waveformInfoTextView.visibility = View.GONE
                voiceWaveformView.visibility = View.GONE
                voiceReplayButton.visibility = View.GONE
                // モード切り替え時にリセット(ダウンロードはSend押下時)
                llmInputEditText.setText("Hello!")
                llmChatContainer.removeAllViews()
                llmStatusTextView.text = "Status: Press Send to download model and run"
                llmSendButton.isEnabled = true
            }
            AlgorithmType.MULTIMODAL_LLM -> {
                modeRadioGroup.visibility = View.VISIBLE
                imageView.visibility = View.GONE
                if (isImageMode) {
                    cameraPreviewView.visibility = View.GONE
                } else {
                    cameraPreviewView.visibility = View.VISIBLE
                }
                resultScrollView.visibility = View.VISIBLE
                classificationResultTextView.visibility = View.GONE
                tokenizerInputEditText.visibility = View.GONE
                tokenizerOutputTextView.visibility = View.GONE
                trackingResultTextView.visibility = View.GONE
                transcriptTextView.visibility = View.GONE
                findViewById<TextView>(R.id.tokenizerInputLabel).visibility = View.GONE
                findViewById<TextView>(R.id.tokenizerOutputLabel).visibility = View.GONE
                multimodalImageView.visibility = View.VISIBLE
                llmInputLabel.visibility = View.VISIBLE
                llmInputEditText.visibility = View.VISIBLE
                llmSendButton.visibility = View.VISIBLE
                llmOutputLabel.visibility = View.VISIBLE
                llmChatContainer.visibility = View.VISIBLE
                llmStatusTextView.visibility = View.VISIBLE
                voiceInputEditText.visibility = View.GONE
                voiceStatusTextView.visibility = View.GONE
                voiceGenerateButton.visibility = View.GONE
                voiceEnvSpinner.visibility = View.GONE
                voiceResultTextView.visibility = View.GONE
                speechLanguageLabel.visibility = View.GONE
                speechLanguageSpinner.visibility = View.GONE
                speechModeRadioGroup.visibility = View.GONE
                diarizationCheckBox.visibility = View.GONE
                liveModeCheckBox.visibility = View.GONE
                speechRunButton.visibility = View.GONE
                micRecordButton.visibility = View.GONE
                waveformView.visibility = View.GONE
                waveformInfoTextView.visibility = View.GONE
                voiceWaveformView.visibility = View.GONE
                voiceReplayButton.visibility = View.GONE
                // モード切り替え時にリセット(ダウンロードはSend押下時)
                llmInputEditText.setText("What is in this image?")
                llmChatContainer.removeAllViews()
                llmStatusTextView.text = "Status: Press Send to download model and run"
                llmSendButton.isEnabled = true
            }

            AlgorithmType.TEXT_TO_SPEECH -> {
                modeRadioGroup.visibility = View.GONE
                imageView.visibility = View.GONE
                cameraPreviewView.visibility = View.GONE
                resultScrollView.visibility = View.VISIBLE
                classificationResultTextView.visibility = View.GONE
                tokenizerInputEditText.visibility = View.GONE
                tokenizerOutputTextView.visibility = View.GONE
                trackingResultTextView.visibility = View.GONE
                transcriptTextView.visibility = View.GONE
                findViewById<TextView>(R.id.tokenizerInputLabel).visibility = View.GONE
                findViewById<TextView>(R.id.tokenizerOutputLabel).visibility = View.GONE
                multimodalImageView.visibility = View.GONE
                llmInputLabel.visibility = View.GONE
                llmInputEditText.visibility = View.GONE
                llmSendButton.visibility = View.GONE
                llmOutputLabel.visibility = View.GONE
                llmChatContainer.visibility = View.GONE
                llmStatusTextView.visibility = View.GONE
                voiceInputEditText.visibility = View.VISIBLE
                voiceStatusTextView.visibility = View.VISIBLE
                voiceGenerateButton.visibility = View.VISIBLE
                voiceEnvSpinner.visibility = View.VISIBLE
                voiceResultTextView.visibility = View.VISIBLE
                speechLanguageLabel.visibility = View.GONE
                speechLanguageSpinner.visibility = View.GONE
                speechModeRadioGroup.visibility = View.GONE
                diarizationCheckBox.visibility = View.GONE
                liveModeCheckBox.visibility = View.GONE
                speechRunButton.visibility = View.GONE
                micRecordButton.visibility = View.GONE
                waveformView.visibility = View.GONE
                waveformInfoTextView.visibility = View.GONE
                voiceWaveformView.visibility = View.VISIBLE
                voiceWaveformView.clear()
                voiceReplayButton.visibility = View.GONE
                voiceGenerateButton.isEnabled = true
                voiceResultTextView.text = ""
                voiceStatusTextView.text = "Status: Press Generate to download model and synthesize"
            }

            else -> {
                modeRadioGroup.visibility = View.VISIBLE
                if (isImageMode) {
                    imageView.visibility = View.VISIBLE
                    cameraPreviewView.visibility = View.GONE
                } else {
                    imageView.visibility = View.VISIBLE
                    cameraPreviewView.visibility = View.VISIBLE
                }
                resultScrollView.visibility = View.GONE
                classificationResultTextView.visibility = View.GONE
                tokenizerInputEditText.visibility = View.GONE
                tokenizerOutputTextView.visibility = View.GONE
                trackingResultTextView.visibility = View.GONE
                transcriptTextView.visibility = View.GONE
                findViewById<TextView>(R.id.tokenizerInputLabel).visibility = View.GONE
                findViewById<TextView>(R.id.tokenizerOutputLabel).visibility = View.GONE
                multimodalImageView.visibility = View.GONE
                llmInputLabel.visibility = View.GONE
                llmInputEditText.visibility = View.GONE
                llmSendButton.visibility = View.GONE
                llmOutputLabel.visibility = View.GONE
                llmChatContainer.visibility = View.GONE
                llmStatusTextView.visibility = View.GONE
                voiceInputEditText.visibility = View.GONE
                voiceStatusTextView.visibility = View.GONE
                voiceGenerateButton.visibility = View.GONE
                voiceEnvSpinner.visibility = View.GONE
                voiceResultTextView.visibility = View.GONE
                speechLanguageLabel.visibility = View.GONE
                speechLanguageSpinner.visibility = View.GONE
                speechModeRadioGroup.visibility = View.GONE
                diarizationCheckBox.visibility = View.GONE
                liveModeCheckBox.visibility = View.GONE
                speechRunButton.visibility = View.GONE
                micRecordButton.visibility = View.GONE
                waveformView.visibility = View.GONE
                waveformInfoTextView.visibility = View.GONE
                voiceWaveformView.visibility = View.GONE
                voiceReplayButton.visibility = View.GONE
            }
        }

        // カメラモードでは処理結果(imageView)を表示領域いっぱいに拡大(centerCrop)して、
        // 上下の帯(正方形クロップのfitCenterでできる透明余白から背面のライブプレビューが
        // 透けて見える現象)を解消する。imageViewが不透明な結果画像で領域全体を覆うため、
        // 背面のcameraPreviewViewも隠れる。画像モードでは全体を欠けなく見せるためfitCenter。
        if (isCameraMode) {
            imageView.scaleType = ImageView.ScaleType.CENTER_CROP
        } else {
            imageView.scaleType = ImageView.ScaleType.FIT_CENTER
        }

        // カメラ選択スピナーはカメラモード時のみ表示
        cameraSpinner.visibility =
            if (modeRadioGroup.visibility == View.VISIBLE && isCameraMode) View.VISIBLE else View.GONE

        // 画像系/TokenizeにはRunボタンを表示(押して初めてダウンロード+実行)
        visionRunButton.visibility = if (needsVisionRunButton()) View.VISIBLE else View.GONE
    }

    private fun switchAlgorithm(newAlgorithm: AlgorithmType) {
        if (isProcessing.get()) {
            Log.i(
                "AILIA_Main",
                "Processing active, queuing algorithm switch to ${newAlgorithm.name}"
            )
            pendingAlgorithmSwitch = newAlgorithm
            return
        }

        executeAlgorithmSwitch(newAlgorithm)
    }

    private fun executeAlgorithmSwitch(newAlgorithm: AlgorithmType) {
        releaseCurrentAlgorithm()
        currentAlgorithm = newAlgorithm
        isInitialized = false
        isDownloadingModel.set(false)
        resetRunState()
        // アルゴリズム切り替え時にProcessing Timeと波形表示をリセット
        processingTimeTextView.text = "Processing Time: -- ms"
        waveformView.clear()
        voiceWaveformView.clear()
        waveformInfoTextView.text = ""
        clearTranscript()
        updateModelSpinner()
        updateUIVisibility()

        // STT/TTSはImage/Cameraモードの選択状態に関わらずUIセットアップが必要
        // (processImageModeはImageモード時しか呼ばれないため、ここで登録する)
        when (newAlgorithm) {
            AlgorithmType.SPEECH_TO_TEXT -> {
                setupSpeechLanguageSpinner()
                setupSpeechModeRadioGroup()
                setupDiarizationCheckBox()
                setupLiveModeCheckBox()
                setupSpeechRunButton()
                setupMicRecordButton()
            }
            AlgorithmType.TEXT_TO_SPEECH -> {
                setupVoiceGenerateButton()
                setupVoiceReplayButton()
                voiceInputEditText.setText(defaultVoiceText(selectedVoiceModelType))
            }
            AlgorithmType.LLM -> {
                setupLLMSendButton()
            }
            AlgorithmType.MULTIMODAL_LLM -> {
                setupMultimodalLLMSendButton()
            }
            else -> {}
        }

        if (modeRadioGroup.checkedRadioButtonId == R.id.imageRadioButton) {
            processImageMode()
        }
    }

    private fun releaseCurrentAlgorithm() {
        try {
            stopMicRecording()
            poseEstimatorSample.releasePoseEstimator()
            objectDetectionSample.releaseObjectDetection()
            classificationSample.releaseClassification()
            onnxObjectDetectionSample.releaseObjectDetection()
            onnxClassificationSample.releaseClassification()
            miniLMv2Sample.release()
            trackerSample.releaseTracker()
            speechSample.releaseSpeech()
            voiceSample.releaseVoice()
            llmSample.release()
            multimodalLLMSample.release()
        } catch (e: Exception) {
            Log.e("AILIA_Error", "Error releasing algorithms: ${e.message}")
        }
    }

    private fun switchToImageMode() {
        if (isProcessing.get()) {
            Log.i("AILIA_Main", "Processing active, queuing mode switch to Image")
            pendingModeSwitch = R.id.imageRadioButton
            return
        }

        executeModeSwitch(R.id.imageRadioButton)
    }

    private fun switchToCameraMode() {
        if (isProcessing.get()) {
            Log.i("AILIA_Main", "Processing active, queuing mode switch to Camera")
            pendingModeSwitch = R.id.cameraRadioButton
            return
        }

        executeModeSwitch(R.id.cameraRadioButton)
    }

    private fun executeModeSwitch(modeId: Int) {
        // モード切り替え時はRun押下からやり直す
        resetRunState()
        when (modeId) {
            R.id.imageRadioButton -> {
                updateUIVisibility()
                stopCamera()
                processImageMode()
            }

            R.id.cameraRadioButton -> {
                if (allPermissionsGranted()) {
                    updateUIVisibility()
                    imageView.setImageBitmap(null)
                    startCamera()
                } else {
                    Toast.makeText(this, "Camera permission required", Toast.LENGTH_SHORT).show()
                    modeRadioGroup.check(R.id.imageRadioButton)
                }
            }
        }
    }

    private fun initializeAilia() {
        try {
            when (currentAlgorithm) {
                AlgorithmType.POSE_ESTIMATION -> {
                    val proto: ByteArray? = loadRawFile(R.raw.lightweight_human_pose_proto)
                    val model: ByteArray? = loadRawFile(R.raw.lightweight_human_pose_weight)
                    isInitialized =
                        poseEstimatorSample.initializePoseEstimator(selectedEnvId, proto, model)
                }

                AlgorithmType.OBJECT_DETECTION -> {
                    if (selectedRuntime == "ONNX") {
                        if (isDownloadingModel.get()) return
                        isDownloadingModel.set(true)
                        runOnUiThread {
                            processingTimeTextView.text = "Downloading ONNX model..."
                        }
                        Log.i("AILIA_Main", "ONNX ObjDet: submitting download task to cameraExecutor")
                        cameraExecutor.execute {
                            Log.i("AILIA_Main", "ONNX ObjDet: cameraExecutor task started")
                            try {
                                val downloaded = onnxObjectDetectionSample.downloadModel(object : AiliaOnnxObjectDetectionSample.DownloadListener {
                                    override fun onProgress(fileName: String, bytesDownloaded: Long, totalBytes: Long) {
                                        val percent = if (totalBytes > 0) (bytesDownloaded * 100 / totalBytes) else 0
                                        runOnUiThread {
                                            processingTimeTextView.text = "Downloading $fileName... $percent%"
                                        }
                                    }
                                    override fun onComplete() {}
                                    override fun onError(error: String) {
                                        runOnUiThread {
                                            processingTimeTextView.text = "Download error: $error"
                                        }
                                    }
                                })
                                Log.i("AILIA_Main", "ONNX ObjDet: download result=$downloaded")
                                if (downloaded) {
                                    Log.i("AILIA_Main", "ONNX ObjDet: initializing with envId=$selectedEnvId")
                                    val success = onnxObjectDetectionSample.initializeObjectDetection(selectedEnvId)
                                    Log.i("AILIA_Main", "ONNX ObjDet: initialization result=$success")
                                    isInitialized = success
                                    isDownloadingModel.set(false)
                                    runOnUiThread {
                                        if (success) {
                                            processingTimeTextView.text = "ONNX model ready"
                                            if (modeRadioGroup.checkedRadioButtonId == R.id.imageRadioButton) {
                                                processImageMode()
                                            }
                                        } else {
                                            processingTimeTextView.text = "Failed to initialize ONNX model"
                                        }
                                    }
                                } else {
                                    isDownloadingModel.set(false)
                                }
                            } catch (e: Exception) {
                                Log.e("AILIA_Main", "ONNX ObjDet: exception in cameraExecutor", e)
                                isDownloadingModel.set(false)
                                runOnUiThread {
                                    processingTimeTextView.text = "Error: ${e.message}"
                                }
                            }
                        }
                        return
                    } else {
                        //val yoloxModel: ByteArray? = loadRawFile(R.raw.yolox_tiny)
                        val yoloxModel: ByteArray? = loadRawFile(R.raw.yolox_s)
                        isInitialized = objectDetectionSample.initializeObjectDetection(
                            yoloxModel,
                            env = selectedEnvId
                        )
                    }
                }

                AlgorithmType.CLASSIFICATION -> {
                    if (selectedRuntime == "ONNX") {
                        if (isDownloadingModel.get()) return
                        isDownloadingModel.set(true)
                        runOnUiThread {
                            processingTimeTextView.text = "Downloading ONNX model..."
                        }
                        Log.i("AILIA_Main", "ONNX Classification: submitting download task to cameraExecutor")
                        cameraExecutor.execute {
                            Log.i("AILIA_Main", "ONNX Classification: cameraExecutor task started")
                            try {
                                val downloaded = onnxClassificationSample.downloadModel(object : AiliaOnnxClassificationSample.DownloadListener {
                                    override fun onProgress(fileName: String, bytesDownloaded: Long, totalBytes: Long) {
                                        val percent = if (totalBytes > 0) (bytesDownloaded * 100 / totalBytes) else 0
                                        runOnUiThread {
                                            processingTimeTextView.text = "Downloading $fileName... $percent%"
                                        }
                                    }
                                    override fun onComplete() {}
                                    override fun onError(error: String) {
                                        runOnUiThread {
                                            processingTimeTextView.text = "Download error: $error"
                                        }
                                    }
                                })
                                Log.i("AILIA_Main", "ONNX Classification: download result=$downloaded")
                                if (downloaded) {
                                    Log.i("AILIA_Main", "ONNX Classification: initializing with envId=$selectedEnvId")
                                    val success = onnxClassificationSample.initializeClassification(selectedEnvId)
                                    Log.i("AILIA_Main", "ONNX Classification: initialization result=$success")
                                    isInitialized = success
                                    isDownloadingModel.set(false)
                                    runOnUiThread {
                                        if (success) {
                                            processingTimeTextView.text = "ONNX model ready"
                                            if (modeRadioGroup.checkedRadioButtonId == R.id.imageRadioButton) {
                                                processImageMode()
                                            }
                                        } else {
                                            processingTimeTextView.text = "Failed to initialize ONNX model"
                                        }
                                    }
                                } else {
                                    isDownloadingModel.set(false)
                                }
                            } catch (e: Exception) {
                                Log.e("AILIA_Main", "ONNX Classification: exception in cameraExecutor", e)
                                isDownloadingModel.set(false)
                                runOnUiThread {
                                    processingTimeTextView.text = "Error: ${e.message}"
                                }
                            }
                        }
                        return
                    } else {
                        val classificationModel: ByteArray? = loadRawFile(R.raw.mobilenetv2)
                        isInitialized = classificationSample.initializeClassification(
                            classificationModel,
                            env = selectedEnvId
                        )
                    }
                }

                AlgorithmType.TOKENIZE -> {
                    if (isDownloadingModel.get()) return
                    isDownloadingModel.set(true)
                    runOnUiThread {
                        processingTimeTextView.text = "Downloading MiniLMv2 model..."
                    }
                    Log.i("AILIA_Main", "MiniLMv2: submitting download task to cameraExecutor")
                    cameraExecutor.execute {
                        try {
                            val downloaded = miniLMv2Sample.downloadModel(object : AiliaMiniLMv2Sample.DownloadListener {
                                override fun onProgress(fileName: String, bytesDownloaded: Long, totalBytes: Long) {
                                    val percent = if (totalBytes > 0) (bytesDownloaded * 100 / totalBytes) else 0
                                    runOnUiThread {
                                        processingTimeTextView.text = "Downloading $fileName... $percent%"
                                    }
                                }
                                override fun onComplete() {}
                                override fun onError(error: String) {
                                    runOnUiThread {
                                        processingTimeTextView.text = "Download error: $error"
                                    }
                                }
                            })
                            Log.i("AILIA_Main", "MiniLMv2: download result=$downloaded")
                            if (downloaded) {
                                val success = miniLMv2Sample.initialize(selectedEnvId)
                                Log.i("AILIA_Main", "MiniLMv2: initialization result=$success")
                                isInitialized = success
                                isDownloadingModel.set(false)
                                runOnUiThread {
                                    if (success) {
                                        processingTimeTextView.text = "MiniLMv2 ready"
                                        processImageMode()
                                    } else {
                                        processingTimeTextView.text = "Failed to initialize MiniLMv2"
                                    }
                                }
                            } else {
                                isDownloadingModel.set(false)
                            }
                        } catch (e: Exception) {
                            Log.e("AILIA_Main", "MiniLMv2: exception in cameraExecutor", e)
                            isDownloadingModel.set(false)
                            runOnUiThread {
                                processingTimeTextView.text = "Error: ${e.message}"
                            }
                        }
                    }
                    return
                }

                AlgorithmType.TRACKING -> {
                    if (selectedRuntime == "ONNX") {
                        if (isDownloadingModel.get()) return
                        isDownloadingModel.set(true)
                        runOnUiThread {
                            processingTimeTextView.text = "Downloading ONNX model..."
                        }
                        Log.i("AILIA_Main", "ONNX Tracking: submitting download task to cameraExecutor")
                        cameraExecutor.execute {
                            Log.i("AILIA_Main", "ONNX Tracking: cameraExecutor task started")
                            try {
                                val downloaded = onnxObjectDetectionSample.downloadModel(object : AiliaOnnxObjectDetectionSample.DownloadListener {
                                    override fun onProgress(fileName: String, bytesDownloaded: Long, totalBytes: Long) {
                                        val percent = if (totalBytes > 0) (bytesDownloaded * 100 / totalBytes) else 0
                                        runOnUiThread {
                                            processingTimeTextView.text = "Downloading $fileName... $percent%"
                                        }
                                    }
                                    override fun onComplete() {}
                                    override fun onError(error: String) {
                                        runOnUiThread {
                                            processingTimeTextView.text = "Download error: $error"
                                        }
                                    }
                                })
                                Log.i("AILIA_Main", "ONNX Tracking: download result=$downloaded")
                                if (downloaded) {
                                    Log.i("AILIA_Main", "ONNX Tracking: initializing with envId=$selectedEnvId")
                                    val detectorSuccess = onnxObjectDetectionSample.initializeObjectDetection(selectedEnvId)
                                    val trackerSuccess = if (detectorSuccess) trackerSample.initializeTracker() else false
                                    Log.i("AILIA_Main", "ONNX Tracking: detector=$detectorSuccess, tracker=$trackerSuccess")
                                    isInitialized = trackerSuccess
                                    isDownloadingModel.set(false)
                                    runOnUiThread {
                                        if (trackerSuccess) {
                                            processingTimeTextView.text = "ONNX model ready"
                                            if (modeRadioGroup.checkedRadioButtonId == R.id.imageRadioButton) {
                                                processImageMode()
                                            }
                                        } else {
                                            processingTimeTextView.text = "Failed to initialize ONNX tracking"
                                        }
                                    }
                                } else {
                                    isDownloadingModel.set(false)
                                }
                            } catch (e: Exception) {
                                Log.e("AILIA_Main", "ONNX Tracking: exception in cameraExecutor", e)
                                isDownloadingModel.set(false)
                                runOnUiThread {
                                    processingTimeTextView.text = "Error: ${e.message}"
                                }
                            }
                        }
                        return
                    } else {
                        //val yoloxModel: ByteArray? = loadRawFile(R.raw.yolox_tiny)
                        val yoloxModel: ByteArray? = loadRawFile(R.raw.yolox_s)
                        if (objectDetectionSample.initializeObjectDetection(
                                yoloxModel,
                                env = selectedEnvId
                            )
                        ) {
                            isInitialized = trackerSample.initializeTracker()
                        }
                    }
                }

                AlgorithmType.SPEECH_TO_TEXT -> {
                    // モデルダウンロードはRun/Record押下時(ensureSpeechReady)まで遅延する
                    return
                }

                AlgorithmType.TEXT_TO_SPEECH -> {
                    // モデルダウンロードはGenerate押下時(setupVoiceGenerateButton)まで遅延する
                    return
                }

                AlgorithmType.LLM, AlgorithmType.MULTIMODAL_LLM -> {
                    // モデルダウンロードはSend押下時(initialize*Async)まで遅延する
                    return
                }
            }

            if (isInitialized) {
                Log.i("AILIA_Main", "Algorithm ${currentAlgorithm.name} initialized successfully")
            } else {
                Log.e("AILIA_Error", "Failed to initialize algorithm ${currentAlgorithm.name}")
            }
        } catch (e: Exception) {
            Log.e(
                "AILIA_Error",
                "Error initializing algorithm ${currentAlgorithm.name}: ${e.message}"
            )
        }
    }

    /** チャット吹き出しを追加する(ユーザー右寄せ/AI左寄せ) */
    private fun addChatBubble(message: String, isUser: Boolean): TextView {
        val bubble = TextView(this)
        bubble.text = message
        bubble.setTextColor(Color.BLACK)
        bubble.textSize = 14f
        bubble.setBackgroundResource(if (isUser) R.drawable.chat_bubble_user else R.drawable.chat_bubble_assistant)
        val pad = (10 * resources.displayMetrics.density).toInt()
        bubble.setPadding(pad, pad, pad, pad)
        bubble.maxWidth = (resources.displayMetrics.widthPixels * 0.75f).toInt()
        val params = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.WRAP_CONTENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        )
        val margin = (4 * resources.displayMetrics.density).toInt()
        params.setMargins(margin, margin, margin, margin)
        params.gravity = if (isUser) android.view.Gravity.END else android.view.Gravity.START
        llmChatContainer.addView(bubble, params)
        scrollResultToBottom()
        return bubble
    }

    private fun scrollResultToBottom() {
        rootScrollView.post { rootScrollView.fullScroll(View.FOCUS_DOWN) }
    }

    /** LLMモデルをダウンロード+初期化し、成功時にonReadyをUIスレッドで呼ぶ */
    private fun initializeLLMAsync(onReady: (() -> Unit)? = null) {
        llmStatusTextView.text = "Status: Downloading model..."
        llmSendButton.isEnabled = false
        algorithmSpinner.isEnabled = false
        llmSample.modelType = selectedLLMModelType
        cameraExecutor.execute {
            val success = llmSample.initialize(this@MainActivity, object : ModelDownloader.DownloadListener {
                override fun onProgress(bytesDownloaded: Long, totalBytes: Long) {
                    val percent = if (totalBytes > 0) (bytesDownloaded * 100 / totalBytes) else 0
                    runOnUiThread {
                        llmStatusTextView.text = "Status: Downloading model... $percent%"
                    }
                }
                override fun onComplete(file: java.io.File) {
                    Log.i("AILIA_Main", "Model download complete")
                }
                override fun onError(error: String) {
                    runOnUiThread {
                        llmStatusTextView.text = "Status: Download error - $error"
                    }
                }
            })
            runOnUiThread {
                isInitialized = success
                algorithmSpinner.isEnabled = true
                if (success) {
                    llmStatusTextView.text = "Status: Ready"
                    llmSendButton.isEnabled = true
                    onReady?.invoke()
                } else {
                    llmStatusTextView.text = "Status: Initialization failed"
                }
            }
        }
    }

    private fun setupLLMSendButton() {
        llmSendButton.setOnClickListener {
            val userInput = llmInputEditText.text.toString().trim()
            if (userInput.isEmpty()) {
                llmStatusTextView.text = "Status: Please enter a message"
                return@setOnClickListener
            }
            // モデルダウンロードはSend押下時に行う
            if (!isInitialized) {
                initializeLLMAsync {
                    performLLMChat(userInput)
                }
            } else {
                performLLMChat(userInput)
            }
        }
    }

    private fun performLLMChat(userInput: String) {
        llmSendButton.isEnabled = false
        algorithmSpinner.isEnabled = false
        llmStatusTextView.text = "Status: Generating..."
        // チャット風表示: 履歴は消さず、ユーザー発言とAI応答の吹き出しを追加する
        addChatBubble(userInput, isUser = true)
        val assistantBubble = addChatBubble("", isUser = false)
        llmInputEditText.setText("")

        cameraExecutor.execute {
            val processingTime = llmSample.chat(userInput, object : AiliaLLMSample.LLMListener {
                override fun onToken(token: String) {
                    runOnUiThread {
                        assistantBubble.append(token)
                        scrollResultToBottom()
                    }
                }
                override fun onComplete(fullResponse: String) {
                    runOnUiThread {
                        llmStatusTextView.text = "Status: Complete"
                    }
                }
                override fun onError(error: String) {
                    runOnUiThread {
                        llmStatusTextView.text = "Status: Error - $error"
                    }
                }
            })
            runOnUiThread {
                llmSendButton.isEnabled = true
                algorithmSpinner.isEnabled = true
                if (processingTime > 0) {
                    processingTimeTextView.text = "Processing Time: ${processingTime}ms (LLM)"
                }
            }
        }
    }

    /** MultimodalLLMモデルをダウンロード+初期化し、成功時にonReadyをUIスレッドで呼ぶ */
    private fun initializeMultimodalAsync(onReady: (() -> Unit)? = null) {
        llmStatusTextView.text = "Status: Downloading model..."
        llmSendButton.isEnabled = false
        algorithmSpinner.isEnabled = false
        cameraExecutor.execute {
            val success = multimodalLLMSample.initialize(this@MainActivity, object : AiliaMultimodalLLMSample.MultimodalLLMListener {
                override fun onDownloadProgress(fileName: String, bytesDownloaded: Long, totalBytes: Long) {
                    val percent = if (totalBytes > 0) (bytesDownloaded * 100 / totalBytes) else 0
                    runOnUiThread {
                        llmStatusTextView.text = "Status: Downloading $fileName... $percent%"
                    }
                }
                override fun onStatus(status: String) {}
                override fun onToken(token: String) {}
                override fun onComplete(fullResponse: String) {}
                override fun onError(error: String) {
                    runOnUiThread {
                        llmStatusTextView.text = "Status: Error - $error"
                    }
                }
            })
            Log.i("AILIA_Main", "MultimodalLLM: init done on cameraExecutor, success=$success")
            runOnUiThread {
                isInitialized = success
                algorithmSpinner.isEnabled = true
                if (success) {
                    llmStatusTextView.text = "Status: Ready"
                    llmSendButton.isEnabled = true
                    // Load the sample image (image mode)
                    loadSampleImageForMultimodal()
                    onReady?.invoke()
                } else {
                    llmStatusTextView.text = "Status: Initialization failed"
                }
            }
        }
    }

    private fun setupMultimodalLLMSendButton() {
        llmSendButton.setOnClickListener {
            val userInput = llmInputEditText.text.toString().trim()
            if (userInput.isEmpty()) {
                llmStatusTextView.text = "Status: Please enter a question about the image"
                return@setOnClickListener
            }
            // モデルダウンロードはSend押下時に行う
            if (!isInitialized) {
                initializeMultimodalAsync {
                    performMultimodalChat(userInput)
                }
            } else {
                performMultimodalChat(userInput)
            }
        }
    }

    private fun performMultimodalChat(userInput: String) {
            llmSendButton.isEnabled = false
            algorithmSpinner.isEnabled = false
            modeRadioGroup.isEnabled = false
            for (i in 0 until modeRadioGroup.childCount) {
                modeRadioGroup.getChildAt(i).isEnabled = false
            }
            llmStatusTextView.text = "Status: Generating..."
            // チャット風表示: 履歴は消さず、ユーザー発言とAI応答の吹き出しを追加する
            addChatBubble(userInput, isUser = true)
            val assistantBubble = addChatBubble("", isUser = false)
            llmInputEditText.setText("")

            val isCameraMode = modeRadioGroup.checkedRadioButtonId == R.id.cameraRadioButton
            val imagePath = if (isCameraMode && latestCameraBitmap != null) {
                val tmpFile = File(cacheDir, "camera_frame.png")
                FileOutputStream(tmpFile).use { latestCameraBitmap!!.compress(Bitmap.CompressFormat.PNG, 100, it) }
                tmpFile.absolutePath
            } else {
                null
            }

            Log.i("AILIA_Main", "MultimodalLLM Send: submitting chatWithImage to cameraExecutor, imagePath=$imagePath, userInput='$userInput'")
            cameraExecutor.execute {
                Log.i("AILIA_Main", "MultimodalLLM Send: cameraExecutor task started, calling chatWithImage...")
                val processingTime = multimodalLLMSample.chatWithImage(imagePath, userInput, object : AiliaMultimodalLLMSample.MultimodalLLMListener {
                    override fun onDownloadProgress(fileName: String, bytesDownloaded: Long, totalBytes: Long) {}
                    override fun onStatus(status: String) {
                        runOnUiThread {
                            llmStatusTextView.text = "Status: $status"
                        }
                    }
                    override fun onToken(token: String) {
                        runOnUiThread {
                            assistantBubble.append(token)
                            scrollResultToBottom()
                        }
                    }
                    override fun onComplete(fullResponse: String) {
                        runOnUiThread {
                            llmStatusTextView.text = "Status: Complete"
                        }
                    }
                    override fun onError(error: String) {
                        runOnUiThread {
                            llmStatusTextView.text = "Status: Error - $error"
                        }
                    }
                })
                runOnUiThread {
                    llmSendButton.isEnabled = true
                    algorithmSpinner.isEnabled = true
                    modeRadioGroup.isEnabled = true
                    for (i in 0 until modeRadioGroup.childCount) {
                        modeRadioGroup.getChildAt(i).isEnabled = true
                    }
                    if (processingTime > 0) {
                        processingTimeTextView.text = "Processing Time: ${processingTime}ms (MultimodalLLM)"
                    }
                }
            }
    }

    private fun loadSampleImageForMultimodal() {
        val isImageMode = modeRadioGroup.checkedRadioButtonId == R.id.imageRadioButton
        if (!isImageMode) return  // Camera mode: don't load sample image
        val imagePath = multimodalLLMSample.getSampleImagePath()
        if (imagePath != null) {
            val bitmap = android.graphics.BitmapFactory.decodeFile(imagePath)
            if (bitmap != null) {
                multimodalImageView.setImageBitmap(bitmap)
            }
        }
    }

    /** 各Voiceモデルのデフォルト入力テキスト(V1は英語モデルのため英文) */
    private fun defaultVoiceText(type: VoiceModelType): String {
        return if (type == VoiceModelType.GPT_SOVITS_V1) {
            "Hello world. We will introduce ailia AI voice."
        } else {
            "こんにちは。今日はいい天気ですね。"
        }
    }

    /**
     * 入力テキストが日本語か英語かを判定してG2Pの言語を返す。
     * ひらがな/カタカナ/漢字が含まれていれば"ja"、それ以外は"en"。
     */
    private fun detectVoiceTextLanguage(text: String): String {
        for (ch in text) {
            val block = Character.UnicodeBlock.of(ch)
            if (block == Character.UnicodeBlock.HIRAGANA ||
                block == Character.UnicodeBlock.KATAKANA ||
                block == Character.UnicodeBlock.CJK_UNIFIED_IDEOGRAPHS) {
                return "ja"
            }
        }
        return "en"
    }

    private fun setupVoiceReplayButton() {
        voiceReplayButton.setOnClickListener {
            val audio = voiceSample.lastAudioData ?: return@setOnClickListener
            voiceSample.playAudio(audio, voiceSample.lastAudioChannels, voiceSample.lastAudioSampleRate)
            voiceWaveformView.startPlayback(audio, voiceSample.lastAudioChannels, voiceSample.lastAudioSampleRate)
        }
    }

    private fun setupVoiceGenerateButton() {
        voiceGenerateButton.setOnClickListener {
            val inputText = voiceInputEditText.text.toString().trim()
            if (inputText.isEmpty()) {
                voiceStatusTextView.text = "Status: Please enter text to speak"
                return@setOnClickListener
            }
            voiceGenerateButton.isEnabled = false
            voiceResultTextView.text = ""
            voiceWaveformView.clear()
            voiceStatusTextView.text =
                if (isInitialized) "Status: Generating..." else "Status: Downloading model..."

            cameraExecutor.execute {
                // モデルダウンロード+初期化はGenerate押下時に行う
                if (!isInitialized) {
                    voiceSample.modelType = selectedVoiceModelType
                    val success = voiceSample.initializeVoice(envId = selectedEnvId, listener = object : AiliaVoiceSample.DownloadListener {
                        override fun onProgress(fileName: String, bytesDownloaded: Long, totalBytes: Long) {
                            val percent = if (totalBytes > 0) (bytesDownloaded * 100 / totalBytes) else 0
                            runOnUiThread {
                                voiceStatusTextView.text = "Status: Downloading $fileName... $percent%"
                            }
                        }
                        override fun onComplete() {
                            Log.i("AILIA_Main", "Voice model download complete")
                        }
                        override fun onError(error: String) {
                            runOnUiThread {
                                voiceStatusTextView.text = "Status: Error - $error"
                            }
                        }
                    })
                    isInitialized = success
                    if (!success) {
                        runOnUiThread {
                            voiceGenerateButton.isEnabled = true
                            voiceStatusTextView.text = "Status: Initialization failed"
                        }
                        return@execute
                    }
                    runOnUiThread {
                        voiceStatusTextView.text = "Status: Generating..."
                    }
                }

                val refAudio: AudioUtil.WavFileData = AudioUtil().loadRawAudio(this.resources.openRawResource(R.raw.reference_audio_girl))
                // 入力テキストの日英を判定してG2Pの言語を切り替える
                val textLang = detectVoiceTextLanguage(inputText)
                val inferenceTime = voiceSample.textToSpeech(
                    refAudio.audioData,
                    refAudio.channels,
                    refAudio.sampleRate,
                    "水をマレーシアから買わなくてはならない。",
                    "ja",
                    inputText,
                    textLang,
                )
                runOnUiThread {
                    voiceGenerateButton.isEnabled = true
                    voiceStatusTextView.text = "Status: Complete"
                    voiceResultTextView.text = "${selectedVoiceModelType.displayName} Generated"
                    if (inferenceTime > 0) {
                        processingTimeTextView.text = "Processing Time: ${inferenceTime}ms (Voice)"
                    }
                    // 合成音声の波形を再生位置に追従して表示する
                    voiceSample.lastAudioData?.let { audio ->
                        voiceWaveformView.startPlayback(audio, voiceSample.lastAudioChannels, voiceSample.lastAudioSampleRate)
                        // 生成後はリプレイボタンを表示する
                        voiceReplayButton.visibility = View.VISIBLE
                    }
                }
            }
        }
    }

    private fun setupSpeechLanguageSpinner() {
        val languages = arrayOf("ja", "en", "auto")
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, languages)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        speechLanguageSpinner.adapter = adapter

        val currentIndex = languages.indexOf(selectedSpeechLanguage)
        if (currentIndex >= 0) {
            speechLanguageSpinner.setSelection(currentIndex)
        }

        speechLanguageSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                val newLanguage = languages[position]
                if (newLanguage != selectedSpeechLanguage) {
                    selectedSpeechLanguage = newLanguage
                    speechSample.language = newLanguage
                    // 言語変更のため要再初期化(ダウンロードはRun/Record押下時)
                    if (currentAlgorithm == AlgorithmType.SPEECH_TO_TEXT) {
                        stopMicRecording()
                        speechSample.releaseSpeech()
                        isInitialized = false
                        isDownloadingModel.set(false)
                        clearTranscript()
                        classificationResultTextView.text = "Speech Result: --"
                    }
                }
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
    }

    private fun setupSpeechModeRadioGroup() {
        speechModeRadioGroup.setOnCheckedChangeListener { _, checkedId ->
            when (checkedId) {
                R.id.wavRadioButton -> {
                    speechRunButton.visibility = View.VISIBLE
                    micRecordButton.visibility = View.GONE
                    waveformView.visibility = View.GONE
                    waveformInfoTextView.visibility = View.GONE
                    liveModeCheckBox.visibility = View.GONE
                    speechLanguageLabel.visibility = View.GONE
                    speechLanguageSpinner.visibility = View.GONE
                    stopMicRecording()
                    // liveモード設定が変わるため要再初期化(ダウンロードはRun押下時)
                    speechSample.releaseSpeech()
                    isInitialized = false
                    isDownloadingModel.set(false)
                    clearTranscript()
                    classificationResultTextView.text = "Speech Result: --"
                }
                R.id.micRadioButton -> {
                    speechRunButton.visibility = View.GONE
                    micRecordButton.visibility = View.VISIBLE
                    waveformView.visibility = View.VISIBLE
                    waveformView.clear()
                    waveformInfoTextView.visibility = View.VISIBLE
                    waveformInfoTextView.text = ""
                    liveModeCheckBox.visibility = View.VISIBLE
                    speechLanguageLabel.visibility = View.VISIBLE
                    speechLanguageSpinner.visibility = View.VISIBLE
                    // liveモード設定が変わるため要再初期化(ダウンロードはRecord押下時)
                    speechSample.releaseSpeech()
                    isInitialized = false
                    isDownloadingModel.set(false)
                    clearTranscript()
                    classificationResultTextView.text = "Speech Result: (tap Record)"
                }
            }
        }
    }

    private fun setupDiarizationCheckBox() {
        diarizationCheckBox.setOnCheckedChangeListener { _, isChecked ->
            speechSample.diarizationEnabled = isChecked
            // 設定変更のため要再初期化(ダウンロードはRun/Record押下時)
            stopMicRecording()
            speechSample.releaseSpeech()
            isInitialized = false
            isDownloadingModel.set(false)
            clearTranscript()
            classificationResultTextView.text = "Speech Result: --"
        }
    }

    private fun setupLiveModeCheckBox() {
        liveModeCheckBox.setOnCheckedChangeListener { _, _ ->
            // LIVEフラグ変更のため要再初期化(ダウンロードはRun/Record押下時)
            stopMicRecording()
            speechSample.releaseSpeech()
            isInitialized = false
            isDownloadingModel.set(false)
            clearTranscript()
            classificationResultTextView.text = "Speech Result: --"
        }
    }

    /** 議事録風トランスクリプトに行を追記して表示を更新する(UIスレッドで呼ぶこと) */
    private fun appendTranscriptLines(lines: List<String>) {
        if (lines.isEmpty()) return
        speechTranscript.addAll(lines)
        transcriptTextView.text = speechTranscript.joinToString("\n")
        scrollResultToBottom()
    }

    private fun clearTranscript() {
        speechTranscript.clear()
        transcriptTextView.text = ""
    }

    /**
     * Speechモデルを必要ならダウンロード+初期化してからonReadyをUIスレッドで呼ぶ。
     * モデルダウンロードはRun/Record押下時に初めて行う。
     */
    private fun ensureSpeechReady(onReady: () -> Unit) {
        if (isInitialized) {
            onReady()
            return
        }
        if (isDownloadingModel.get()) return
        isDownloadingModel.set(true)
        val isMicMode = speechModeRadioGroup.checkedRadioButtonId == R.id.micRadioButton
        // LIVEフラグはLive Modeチェックボックスで有効化(マイクモード時のみ)
        val liveMode = isMicMode && liveModeCheckBox.isChecked
        processingTimeTextView.text = "Downloading speech model (${selectedSpeechModelType.displayName})..."
        speechRunButton.isEnabled = false
        micRecordButton.isEnabled = false
        speechExecutor.execute {
            try {
                val downloaded = speechSample.downloadModel(selectedSpeechModelType, object : AiliaSpeechSample.DownloadListener {
                    override fun onProgress(fileName: String, bytesDownloaded: Long, totalBytes: Long) {
                        val percent = if (totalBytes > 0) (bytesDownloaded * 100 / totalBytes) else 0
                        runOnUiThread {
                            processingTimeTextView.text = "Downloading $fileName... $percent%"
                        }
                    }
                    override fun onComplete() {}
                    override fun onError(error: String) {
                        runOnUiThread {
                            processingTimeTextView.text = "Download error: $error"
                        }
                    }
                })
                var success = false
                if (downloaded) {
                    // Wavモードは常にauto、Micモードのみ言語選択を適用する
                    val language = if (isMicMode) selectedSpeechLanguage else "auto"
                    Log.i("AILIA_Main", "Speech: initializing with envId=$selectedEnvId, liveMode=$liveMode, language=$language")
                    speechSample.language = language
                    success = speechSample.initializeSpeech(selectedEnvId, liveMode = liveMode)
                    isInitialized = success
                }
                isDownloadingModel.set(false)
                runOnUiThread {
                    speechRunButton.isEnabled = true
                    micRecordButton.isEnabled = true
                    if (success) {
                        processingTimeTextView.text = "${selectedSpeechModelType.displayName} ready"
                        onReady()
                    } else {
                        processingTimeTextView.text = "Failed to initialize speech model"
                    }
                }
            } catch (e: Exception) {
                Log.e("AILIA_Main", "Speech: download/init error", e)
                isDownloadingModel.set(false)
                runOnUiThread {
                    speechRunButton.isEnabled = true
                    micRecordButton.isEnabled = true
                    processingTimeTextView.text = "Error: ${e.message}"
                }
            }
        }
    }

    private fun setupSpeechRunButton() {
        speechRunButton.setOnClickListener {
            if (isProcessing.get() || isDownloadingModel.get()) {
                return@setOnClickListener
            }
            speechRunButton.isEnabled = false
            ensureSpeechReady {
                // 認識結果が表示されるまでRunをグレーアウトする
                speechRunButton.isEnabled = false
                clearTranscript()
                classificationResultTextView.text = "Speech Result: Processing..."
                speechExecutor.execute {
                    try {
                        val audio: AudioUtil.WavFileData = AudioUtil().loadRawAudio(this.resources.openRawResource(R.raw.demo))
                        val startTime = System.nanoTime()
                        val lines = speechSample.process(audio.audioData, audio.channels, audio.sampleRate)
                        val endTime = System.nanoTime()
                        val timeMs = (endTime - startTime) / 1000000
                        runOnUiThread {
                            appendTranscriptLines(lines)
                            classificationResultTextView.text =
                                if (lines.isEmpty()) "Speech Result: (no speech detected)" else "Speech Result:"
                            processingTimeTextView.text = "Processing Time: $timeMs ms"
                        }
                    } catch (e: Exception) {
                        Log.e("AILIA_Main", "Speech run error", e)
                        runOnUiThread {
                            classificationResultTextView.text = "Speech Result: Error - ${e.message}"
                        }
                    } finally {
                        runOnUiThread {
                            speechRunButton.isEnabled = true
                        }
                    }
                }
            }
        }
    }

    private fun setupMicRecordButton() {
        micRecordButton.setOnClickListener {
            if (isRecording.get()) {
                stopMicRecording()
            } else {
                if (isDownloadingModel.get()) {
                    return@setOnClickListener
                }
                ensureSpeechReady {
                    startMicRecording()
                }
            }
        }
    }

    private fun startMicRecording() {
        if (!isInitialized) {
            classificationResultTextView.text = "Speech model not ready"
            return
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), REQUEST_CODE_PERMISSIONS)
            return
        }

        val sampleRate = 16000
        val channelConfig = AudioFormat.CHANNEL_IN_MONO
        val audioFormat = AudioFormat.ENCODING_PCM_FLOAT
        // 波形表示を滑らかにするため100msごとに読み出し、認識には従来通り1秒分をまとめて渡す
        val readChunkSize = sampleRate / 10
        val recognitionChunkSize = sampleRate
        // AudioRecord internal buffer: at least 2 seconds
        val audioRecordBufferBytes = recognitionChunkSize * 4 * 2

        try {
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                channelConfig,
                audioFormat,
                audioRecordBufferBytes
            )

            if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                classificationResultTextView.text = "Failed to initialize AudioRecord"
                audioRecord = null
                return
            }

            audioRecord?.startRecording()
            isRecording.set(true)
            micRecordButton.text = "Stop"
            classificationResultTextView.text = "Recording..."
            clearTranscript()

            // 波形表示とREC経過時間の初期化
            waveformView.clear()
            recStartMs = android.os.SystemClock.elapsedRealtime()
            waveformInfoTextView.text = "● REC 00:00"
            recTimerHandler.post(recTimerRunnable)

            // マイク読み出しループ。認識(transcribe)は専用のspeechExecutorへ
            // 非同期に投げることで、認識中もマイク読み出しと波形表示を止めない。
            cameraExecutor.execute {
                val floatBuffer = FloatArray(readChunkSize)
                val recognitionBuffer = FloatArray(recognitionChunkSize)
                var recognitionFill = 0

                while (isRecording.get()) {
                    val readResult = audioRecord?.read(floatBuffer, 0, floatBuffer.size, AudioRecord.READ_BLOCKING) ?: -1
                    if (readResult > 0) {
                        // 波形表示(約10msごとのピーク振幅ブロック、PCM_FLOATは[-1.0, 1.0])
                        val blocks = WaveformView.peakBlocks(floatBuffer, readResult, sampleRate)
                        runOnUiThread {
                            waveformView.push(blocks)
                        }

                        // 認識には1秒分をまとめて渡す
                        val toCopy = minOf(readResult, recognitionChunkSize - recognitionFill)
                        System.arraycopy(floatBuffer, 0, recognitionBuffer, recognitionFill, toCopy)
                        recognitionFill += toCopy
                        if (recognitionFill < recognitionChunkSize) {
                            continue
                        }
                        val chunk = recognitionBuffer.copyOf(recognitionFill)
                        recognitionFill = 0
                        speechExecutor.execute {
                            if (!isInitialized) return@execute
                            try {
                                val lines = speechSample.pushLiveAudio(chunk, 1, sampleRate)
                                if (lines.isNotEmpty()) {
                                    runOnUiThread {
                                        appendTranscriptLines(lines)
                                    }
                                }
                            } catch (e: Exception) {
                                Log.e("AILIA_Main", "pushLiveAudio error (speech may have been released): ${e.message}")
                            }
                        }
                    }
                }

                // 録音停止後、1秒に満たない残りの音声を送ってから確定処理
                // (speechExecutorは単一スレッドなので、キュー済みのチャンク処理後に実行される)
                val tail = recognitionBuffer.copyOf(recognitionFill)
                speechExecutor.execute {
                    if (!isInitialized) return@execute
                    try {
                        if (tail.isNotEmpty()) {
                            val lines = speechSample.pushLiveAudio(tail, 1, sampleRate)
                            runOnUiThread {
                                appendTranscriptLines(lines)
                            }
                        }
                        val finalLines = speechSample.finalizeLiveAudio()
                        runOnUiThread {
                            appendTranscriptLines(finalLines)
                            classificationResultTextView.text =
                                if (speechTranscript.isEmpty()) "Speech Result: (no speech detected)" else "Speech Result:"
                        }
                    } catch (e: Exception) {
                        Log.e("AILIA_Main", "finalizeLiveAudio error (speech may have been released): ${e.message}")
                    }
                }
            }
        } catch (e: SecurityException) {
            Log.e("AILIA_Main", "SecurityException starting mic recording: ${e.message}")
            classificationResultTextView.text = "Microphone permission denied"
        } catch (e: Exception) {
            Log.e("AILIA_Main", "Error starting mic recording: ${e.message}")
            classificationResultTextView.text = "Error starting recording: ${e.message}"
        }
    }

    private fun stopMicRecording() {
        if (isRecording.get()) {
            isRecording.set(false)
            try {
                audioRecord?.stop()
            } catch (e: Exception) {
                Log.e("AILIA_Main", "Error stopping AudioRecord: ${e.message}")
            }
            try {
                audioRecord?.release()
            } catch (e: Exception) {
                Log.e("AILIA_Main", "Error releasing AudioRecord: ${e.message}")
            }
            audioRecord = null
            recTimerHandler.removeCallbacks(recTimerRunnable)
            runOnUiThread {
                micRecordButton.text = "Record"
                waveformInfoTextView.text = ""
            }
        }
    }

    private fun processImageMode() {
        // Speech to Text uses speech model spinner and Wav/Mic mode
        // モデルダウンロードはRun/Record押下時まで遅延する
        if (currentAlgorithm == AlgorithmType.SPEECH_TO_TEXT) {
            if (!isInitialized) {
                setupSpeechLanguageSpinner()
                setupSpeechModeRadioGroup()
                setupDiarizationCheckBox()
                setupLiveModeCheckBox()
                setupSpeechRunButton()
                setupMicRecordButton()
            }
            return
        }

        // 画像系/TokenizeはRunボタンが押されるまでダウンロード・実行しない
        if (needsVisionRunButton() && !runRequested) {
            // Run押下前でも入力画像は表示しておく(Imageモードのみ。カメラモードはプレビューが見える)
            if (currentAlgorithm != AlgorithmType.TOKENIZE &&
                modeRadioGroup.checkedRadioButtonId == R.id.imageRadioButton) {
                val options = Options()
                options.inScaled = false
                val personBmp = BitmapFactory.decodeResource(this.resources, R.raw.person, options)
                imageView.setImageBitmap(personBmp)
            }
            return
        }

        // TOKENIZE (MiniLMv2) is async download + init
        if (currentAlgorithm == AlgorithmType.TOKENIZE) {
            if (!isInitialized) {
                initializeAilia()
                return
            }
        }

        // 非同期モデルダウンロードが必要なモード
        if (selectedRuntime == "ONNX" && (currentAlgorithm == AlgorithmType.OBJECT_DETECTION ||
                    currentAlgorithm == AlgorithmType.CLASSIFICATION ||
                    currentAlgorithm == AlgorithmType.TRACKING)) {
            if (!isInitialized) {
                initializeAilia()
                return
            }
        }

        // TEXT_TO_SPEECHはGenerate押下時にダウンロード+初期化する
        if (currentAlgorithm == AlgorithmType.TEXT_TO_SPEECH) {
            if (!isInitialized) {
                setupVoiceGenerateButton()
            }
            return
        }

        // LLMはSend押下時にダウンロード+初期化する
        if (currentAlgorithm == AlgorithmType.LLM) {
            return
        }

        // MultimodalLLMはperson画像の表示のみ(ダウンロードはSend押下時)
        if (currentAlgorithm == AlgorithmType.MULTIMODAL_LLM) {
            val isImageMode = modeRadioGroup.checkedRadioButtonId == R.id.imageRadioButton
            if (isImageMode) {
                val options = BitmapFactory.Options()
                options.inScaled = false
                val personBmp = BitmapFactory.decodeResource(this.resources, R.raw.person, options)
                multimodalImageView.setImageBitmap(personBmp)
            }
            return
        }

        if (isProcessing.get()) {
            return
        }

        isProcessing.set(true)

        try {
            if (!isInitialized) {
                initializeAilia()
            }

            if (!isInitialized) {
                runOnUiThread {
                    processingTimeTextView.text = "Failed to initialize ${currentAlgorithm.name}"
                }
                return
            }

            val options = Options()
            options.inScaled = false
            val personBmp = BitmapFactory.decodeResource(this.resources, R.raw.person, options)

            val img = ImageUtil().loadRawImage(personBmp)
            val w = personBmp.width
            val h = personBmp.height

            val bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            bitmap.copyPixelsFromBuffer(ByteBuffer.wrap(img))

            val canvas = Canvas(bitmap)
            val paint = Paint().apply {
                color = Color.WHITE
            }

            val paint2 = Paint().apply {
                style = Paint.Style.STROKE
                color = Color.RED
                strokeWidth = 5f
            }

            val textPaint = Paint().apply {
                color = Color.BLACK
                textSize = 50f
                isAntiAlias = true
            }

            val processingTime = processAlgorithm(img, personBmp, canvas, w, h)

            runOnUiThread {
                if (currentAlgorithm != AlgorithmType.TOKENIZE) {
                    imageView.setImageBitmap(bitmap)
                }
                var timeText = "Processing Time: ${processingTime}ms (${currentAlgorithm.name})"
                when (currentAlgorithm) {
                    AlgorithmType.CLASSIFICATION -> {
                        val result = if (selectedRuntime == "ONNX") onnxClassificationSample.getLastClassificationResult() else classificationSample.getLastClassificationResult()
                        timeText += "\n$result"
                    }
                    AlgorithmType.TRACKING -> {
                        timeText += "\n${trackerSample.getLastTrackingResult()}"
                    }
                    else -> {}
                }
                processingTimeTextView.text = timeText
            }

        } catch (e: Exception) {
            Log.e("AILIA_Error", "Error in image mode: ${e.message}")
            runOnUiThread {
                processingTimeTextView.text = "Processing Error: ${e.message}"
            }
        } finally {
            isProcessing.set(false)

            pendingAlgorithmSwitch?.let { pendingAlgorithm ->
                pendingAlgorithmSwitch = null
                executeAlgorithmSwitch(pendingAlgorithm)
            }

            pendingModeSwitch?.let { pendingMode ->
                pendingModeSwitch = null
                executeModeSwitch(pendingMode)
            }
        }
    }

    private fun startCamera() {
        isStopCamera.set(false)

        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(cameraPreviewView.surfaceProvider)
            }

            imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, CameraFrameAnalyzer())
                }

            val cameraSelector = cameraChoices[selectedCameraIndex].selector

            try {
                cameraProvider.unbindAll()
                camera = cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer
                )
            } catch (exc: Exception) {
                Log.e("AILIA_Error", "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun stopCamera() {
        isStopCamera.set(true)
        try {
            val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
            cameraProviderFuture.addListener({
                val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
                cameraProvider.unbindAll()
                camera = null
                imageAnalyzer = null
            }, ContextCompat.getMainExecutor(this))
        } catch (e: Exception) {
            Log.e("AILIA_Error", "Error stopping camera: ${e.message}")
        }
    }

    fun cropToSquare(bitmap: Bitmap): Bitmap {
        val width = bitmap.width
        val height = bitmap.height

        // 正方形のサイズは、元のBitmapの幅と高さのうち小さい方に合わせます
        val newSize = if (width < height) width else height

        // 中央を基準にクロップするための開始XとYを計算します
        val startX = (width - newSize) / 2
        val startY = (height - newSize) / 2

        // Bitmapをクロップして正方形の新しいBitmapを作成します
        return Bitmap.createBitmap(bitmap, startX, startY, newSize, newSize)
    }

    private inner class CameraFrameAnalyzer : ImageAnalysis.Analyzer {
        override fun analyze(image: ImageProxy) {
            // MultimodalLLMはプレビュー表示とフレーム保持のみ行う(推論はSend押下時)
            if (currentAlgorithm == AlgorithmType.MULTIMODAL_LLM) {
                processCameraFrame(image)
                image.close()
                return
            }
            // 画像系はRunボタンが押されるまでダウンロード・実行しない
            if (needsVisionRunButton() && !runRequested) {
                image.close()
                return
            }
            if (!isInitialized) {
                initializeAilia()
            }
            if (isInitialized) {
                processCameraFrame(image)
            }
            image.close()
        }

        private fun processCameraFrame(image: ImageProxy) {
            if (isProcessing.get()) {
                return
            }
            if (isWaitAlgorithmSwitch.get()) {
                return
            }
            if (isWaitModeSwitch.get()) {
                return
            }
            if (isStopCamera.get()) {
                return
            }

            isProcessing.set(true)

            try {
                // フロントカメラはPreviewViewと同じ鏡像表示になるよう左右反転する
                val isFrontCamera = cameraChoices[selectedCameraIndex].selector == CameraSelector.DEFAULT_FRONT_CAMERA
                var camera_bitmap = ImageUtil().imageProxyToBitmap(image, mirror = isFrontCamera)
                camera_bitmap = cropToSquare(camera_bitmap)

                val img = ImageUtil().loadRawImage(camera_bitmap)
                val w = camera_bitmap.width
                val h = camera_bitmap.height

                Log.i("AILIA_Main", "${w} ${h}")

                val bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
                val canvas = Canvas(bitmap)
                canvas.drawBitmap(camera_bitmap, 0f, 0f, null)

                val processingTime = processAlgorithm(img, bitmap, canvas, w, h)

                runOnUiThread {
                    // Stop押下後に処理中だったフレームの結果で表示を上書きしない
                    if (needsVisionRunButton() && !runRequested) {
                        return@runOnUiThread
                    }
                    if (currentAlgorithm == AlgorithmType.MULTIMODAL_LLM) {
                        latestCameraBitmap = camera_bitmap
                        multimodalImageView.setImageBitmap(camera_bitmap)
                    } else if (currentAlgorithm != AlgorithmType.TOKENIZE) {
                        imageView.setImageBitmap(bitmap)
                    }

                    val fps = if (processingTime > 0) 1000 / processingTime else 0
                    var timeText = "Processing Time: ${processingTime}ms (${currentAlgorithm.name}) - FPS: $fps"
                    when (currentAlgorithm) {
                        AlgorithmType.CLASSIFICATION -> {
                            val result = if (selectedRuntime == "ONNX") onnxClassificationSample.getLastClassificationResult() else classificationSample.getLastClassificationResult()
                            timeText += "\n$result"
                        }
                        AlgorithmType.TRACKING -> {
                            timeText += "\n${trackerSample.getLastTrackingResult()}"
                        }
                        else -> {}
                    }
                    processingTimeTextView.text = timeText
                }

            } catch (e: Exception) {
                Log.e("AILIA_Error", "Error processing camera frame: ${e.message}")
            } finally {
                isProcessing.set(false)

                pendingAlgorithmSwitch?.let { pendingAlgorithm ->
                    pendingAlgorithmSwitch = null
                    isWaitAlgorithmSwitch.set(true)
                    runOnUiThread {
                        executeAlgorithmSwitch(pendingAlgorithm)
                        isWaitAlgorithmSwitch.set(false)
                    }
                }

                pendingModeSwitch?.let { pendingMode ->
                    pendingModeSwitch = null
                    isWaitModeSwitch.set(true)
                    runOnUiThread {
                        executeModeSwitch(pendingMode)
                        isWaitModeSwitch.set(false)
                    }
                }
            }
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                initializeAilia()
            } else {
                Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT)
                    .show()
                finish()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        stopMicRecording()
        releaseCurrentAlgorithm()
        cameraExecutor.shutdown()
        speechExecutor.shutdown()
    }

    @Throws(IOException::class)
    fun inputStreamToByteArray(`in`: InputStream): ByteArray? {
        val bout = ByteArrayOutputStream()
        BufferedOutputStream(bout).use { out ->
            val buf = ByteArray(128)
            var n = 0
            while (`in`.read(buf).also { n = it } > 0) {
                out.write(buf, 0, n)
            }
        }
        return bout.toByteArray()
    }

    @Throws(IOException::class)
    fun loadRawFile(resourceId: Int): ByteArray? {
        val resources = this.resources
        resources.openRawResource(resourceId).use { `in` -> return inputStreamToByteArray(`in`) }
    }
}
