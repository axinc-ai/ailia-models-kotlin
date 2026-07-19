package jp.axinc.ailia_kotlin

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.BitmapFactory.Options
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
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
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

class MainActivity : AppCompatActivity() {
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var imageView: ImageView
    private lateinit var cameraPreviewView: PreviewView
    private lateinit var modeRadioGroup: RadioGroup
    private lateinit var algorithmSpinner: Spinner
    private lateinit var envSpinner: Spinner
    private lateinit var processingTimeTextView: TextView
    private lateinit var modelDownloadProgressBar: ProgressBar
    private lateinit var resultScrollView: FrameLayout
    private lateinit var classificationResultTextView: TextView
    private lateinit var tokenizerInputLabel: TextView
    private lateinit var tokenizerInputEditText: EditText
    private lateinit var tokenizerOutputLabel: TextView
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
    private var speechIntermediateText: String? = null

    private val modelDirectory by lazy { ModelDownloader.modelDirectory(this) }
    private val poseEstimatorSample = AiliaPoseEstimatorSample()
    private val objectDetectionSample = AiliaTFLiteObjectDetectionSample()
    private val classificationSample by lazy { AiliaTFLiteClassificationSample(modelDirectory) }
    private val miniLMv2Sample by lazy { AiliaMiniLMv2Sample(modelDirectory) }
    private val trackerSample = AiliaTrackerSample()
    private val speechSample by lazy { AiliaSpeechSample(modelDirectory) }
    private val voiceSample by lazy { AiliaVoiceSample(modelDirectory) }
    private val llmSample = AiliaLLMSample()
    private val multimodalLLMSample = AiliaMultimodalLLMSample()
    private val onnxObjectDetectionSample by lazy { AiliaOnnxObjectDetectionSample(modelDirectory) }
    private val onnxClassificationSample by lazy { AiliaOnnxClassificationSample(modelDirectory) }
    private val u2netSample by lazy { AiliaU2NetSample(modelDirectory) }
    private val detrSample by lazy { AiliaDetrSample(modelDirectory) }

    // ObjectDetectionのONNXモデルとしてDETRを使うかどうか(falseならYOLOX)
    private var useDetr = false

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
    private val operationGeneration = AtomicLong(0)
    private val visionDisplayGeneration = AtomicLong(0)
    private val activityDestroyed = AtomicBoolean(false)

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
    private var selectedSpeechModelType: SpeechModelType = SpeechModelType.SENSEVOICE_SMALL
    private var selectedSpeechLanguage: String = "ja"
    private var selectedLLMModelType: LLMModelType = LLMModelType.GEMMA_4_E2B
    private var isUpdatingSpeechOptionChecks = false

    // マイク録音のREC経過時間表示用
    private val recTimerHandler = android.os.Handler(android.os.Looper.getMainLooper())
    private var recStartMs: Long = 0
    private val recTimerRunnable = object : Runnable {
        override fun run() {
            if (speechSample.isMicRecording) {
                val elapsedSec = (android.os.SystemClock.elapsedRealtime() - recStartMs) / 1000
                waveformInfoTextView.text = String.format(Locale.ROOT, "● REC %02d:%02d", elapsedSec / 60, elapsedSec % 60)
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
        BACKGROUND_REMOVAL,
        SPEECH_TO_TEXT,
        TEXT_TO_SPEECH,
        LLM,
        MULTIMODAL_LLM,
    }

    companion object {
        private const val REQUEST_CODE_CAMERA_PERMISSION = 10
        private const val REQUEST_CODE_AUDIO_PERMISSION = 11

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

        initializeViews()
        adjustContentSizeForScreen()
        setupModeSelection()
        updateUIVisibility()

        // Camera and microphone permissions are requested only when their feature is selected.
    }

    private fun initializeViews() {
        imageView = findViewById(R.id.imageView)
        cameraPreviewView = findViewById(R.id.cameraPreviewView)
        modeRadioGroup = findViewById(R.id.modeRadioGroup)
        algorithmSpinner = findViewById(R.id.algorithmSpinner)
        envSpinner = findViewById(R.id.envSpinner)
        processingTimeTextView = findViewById(R.id.processingTimeTextView)
        modelDownloadProgressBar = findViewById(R.id.modelDownloadProgressBar)
        resultScrollView = findViewById(R.id.resultScrollView)
        classificationResultTextView = findViewById(R.id.classificationResultTextView)
        tokenizerInputLabel = findViewById(R.id.tokenizerInputLabel)
        tokenizerInputEditText = findViewById(R.id.tokenizerInputEditText)
        tokenizerOutputLabel = findViewById(R.id.tokenizerOutputLabel)
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
            "BackgroundRemoval",
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
            AlgorithmType.BACKGROUND_REMOVAL,
            AlgorithmType.TOKENIZE -> true
            else -> false
        }
    }

    /** Run状態をリセットしてボタン表記をRunに戻す */
    private fun resetRunState() {
        runRequested = false
        visionRunButton.text = "Run"
    }

    /** モデル変更前の推論結果を消し、現在の入力表示へ戻す。 */
    private fun resetVisionDisplayForModelChange() {
        // 切り替え前に開始したカメラ推論のUI反映も無効化する。
        visionDisplayGeneration.incrementAndGet()
        latestCameraBitmap = null
        processingTimeTextView.text = "Processing Time: -- ms"
        classificationResultTextView.text = "Classification Result: --"
        trackingResultTextView.text = "Tracking Results: --"

        if (modeRadioGroup.checkedRadioButtonId == R.id.cameraRadioButton) {
            // ImageViewはPreviewViewより手前にあるため、前モデルの最終フレームを消す。
            imageView.setImageBitmap(null)
            imageView.invalidate()
        } else if (currentAlgorithm != AlgorithmType.TOKENIZE) {
            val options = Options().apply { inScaled = false }
            imageView.setImageBitmap(
                BitmapFactory.decodeResource(resources, R.raw.person, options)
            )
        }
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
                        resetVisionDisplayForModelChange()
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
            AlgorithmType.POSE_ESTIMATION, AlgorithmType.BACKGROUND_REMOVAL -> {
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
                                resetVisionDisplayForModelChange()
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
                selectedIndex = when {
                    selectedRuntime == "ONNX" && useDetr -> 2
                    selectedRuntime == "ONNX" -> 1
                    else -> 0
                }
                arrayOf("YOLOX-S (TFLite)", "YOLOX-S (ONNX)", "DETR ResNet50 (ONNX)")
            }
            AlgorithmType.TRACKING -> {
                selectedIndex = if (selectedRuntime == "ONNX") 1 else 0
                arrayOf("YOLOX-S + ByteTrack (TFLite)", "YOLOX-S + ByteTrack (ONNX)")
            }
            AlgorithmType.CLASSIFICATION -> {
                selectedIndex = when {
                    selectedRuntime == "TFLite" && classificationSample.modelType == TFLiteClassificationModelType.RESNET50 -> 1
                    selectedRuntime == "TFLite" -> 0
                    onnxClassificationSample.modelType == OnnxClassificationModelType.RESNET50 -> 3
                    else -> 2
                }
                arrayOf("MobileNetV2 (TFLite)", "ResNet50 (TFLite)", "MobileNetV2 (ONNX)", "ResNet50 (ONNX)")
            }
            AlgorithmType.TOKENIZE -> arrayOf("Multilingual MiniLMv2 (L12)")
            AlgorithmType.BACKGROUND_REMOVAL -> {
                selectedRuntime = "ONNX"
                arrayOf("U-2-Net")
            }
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
            AlgorithmType.OBJECT_DETECTION -> {
                val newRuntime = if (position >= 1) "ONNX" else "TFLite"
                val newUseDetr = position == 2
                if (newRuntime != selectedRuntime || newUseDetr != useDetr) {
                    selectedRuntime = newRuntime
                    useDetr = newUseDetr
                    updateEnvSpinner(algorithm)
                    isInitialized = false
                    isDownloadingModel.set(false)
                    resetRunState()
                    resetVisionDisplayForModelChange()
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
            AlgorithmType.TRACKING -> {
                val newRuntime = if (position == 1) "ONNX" else "TFLite"
                if (newRuntime != selectedRuntime) {
                    selectedRuntime = newRuntime
                    updateEnvSpinner(algorithm)
                    isInitialized = false
                    isDownloadingModel.set(false)
                    resetRunState()
                    resetVisionDisplayForModelChange()
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
            AlgorithmType.CLASSIFICATION -> {
                val newRuntime = if (position <= 1) "TFLite" else "ONNX"
                val newTfliteModel = if (position == 1) TFLiteClassificationModelType.RESNET50 else TFLiteClassificationModelType.MOBILENETV2
                val newOnnxModel = if (position == 3) OnnxClassificationModelType.RESNET50 else OnnxClassificationModelType.MOBILENETV2
                if (newRuntime != selectedRuntime ||
                    newTfliteModel != classificationSample.modelType ||
                    newOnnxModel != onnxClassificationSample.modelType) {
                    selectedRuntime = newRuntime
                    classificationSample.modelType = newTfliteModel
                    onnxClassificationSample.modelType = newOnnxModel
                    updateEnvSpinner(algorithm)
                    isInitialized = false
                    isDownloadingModel.set(false)
                    resetRunState()
                    resetVisionDisplayForModelChange()
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
                    stopMicRecording(finalize = false)
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
                    voiceInputEditText.setText(newType.defaultText)
                    voiceStatusTextView.text = "Status: Press Generate to synthesize"
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
                    llmStatusTextView.text = "Status: Press Send to run"
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
        h: Int,
        tokenizerInput: String? = null,
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
                if (selectedRuntime == "ONNX" && useDetr) {
                    detrSample.processObjectDetection(bitmap, canvas, paint2, textPaint, w, h)
                } else if (selectedRuntime == "ONNX") {
                    onnxObjectDetectionSample.processObjectDetection(
                        img, canvas, paint2, textPaint, w, h
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
                    runOnUiThreadIfActive {
                        classificationResultTextView.text = "Classification Result: $result"
                    }
                    time
                } else {
                    val time = classificationSample.processClassification(bitmap)
                    val result = classificationSample.getLastClassificationResult()
                    runOnUiThreadIfActive {
                        classificationResultTextView.text = "Classification Result: $result"
                    }
                    time
                }
            }

            AlgorithmType.BACKGROUND_REMOVAL -> {
                u2netSample.process(bitmap, canvas, w, h)
            }

            AlgorithmType.TOKENIZE -> {
                val inputText = tokenizerInput.orEmpty().ifEmpty { "今日、新しいiPhoneが発売されました" }
                val labels = listOf("スマートフォン", "エンタメ", "スポーツ", "政治", "科学")
                val time = miniLMv2Sample.predict(inputText, labels)
                val result = miniLMv2Sample.getLastResult()
                runOnUiThreadIfActive {
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
                    runOnUiThreadIfActive {
                        trackingResultTextView.text = "Tracking Results: $trackingInfo"
                    }
                    detectionTime + trackingTime
                } else {
                    // First run object detection to get detection results without drawing
                    val detectionTime = objectDetectionSample.processObjectDetectionWithoutDrawing(
                        bitmap, threshold = 0.1f, iou = 1.0f
                    )
                    val detectionResults = objectDetectionSample.getDetectionResults()
                    // Then run tracking with the detection results and draw the tracking results
                    val trackingTime = trackerSample.processTrackingWithDetections(
                        canvas, paint2, w, h, detectionResults
                    )
                    val trackingInfo = trackerSample.getLastTrackingResult()
                    runOnUiThreadIfActive {
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

    /** updateUIVisibilityで管理するView。可視性の初期値はすべてGONEにする。 */
    private fun managedAlgorithmViews(): List<View> = listOf(
        modeRadioGroup,
        imageView,
        cameraPreviewView,
        resultScrollView,
        classificationResultTextView,
        tokenizerInputLabel,
        tokenizerInputEditText,
        tokenizerOutputLabel,
        tokenizerOutputTextView,
        trackingResultTextView,
        transcriptTextView,
        multimodalImageView,
        llmInputLabel,
        llmInputEditText,
        llmSendButton,
        llmOutputLabel,
        llmChatContainer,
        llmStatusTextView,
        voiceInputEditText,
        voiceStatusTextView,
        voiceGenerateButton,
        voiceEnvSpinner,
        voiceResultTextView,
        speechLanguageLabel,
        speechLanguageSpinner,
        speechModeRadioGroup,
        diarizationCheckBox,
        liveModeCheckBox,
        speechRunButton,
        micRecordButton,
        waveformView,
        waveformInfoTextView,
        voiceWaveformView,
        voiceReplayButton,
    )

    /** アルゴリズムごとに表示するView集合。共通レイアウトは同じ集合を共有する。 */
    private fun visibleViewsByAlgorithm(isCameraMode: Boolean): Map<AlgorithmType, Set<View>> {
        val visionViews = mutableSetOf<View>(modeRadioGroup, imageView)
        if (isCameraMode) {
            visionViews.add(cameraPreviewView)
        }

        val tokenizeViews = setOf<View>(
            resultScrollView,
            tokenizerInputLabel,
            tokenizerInputEditText,
            tokenizerOutputLabel,
            tokenizerOutputTextView,
        )

        val isMicMode = speechModeRadioGroup.checkedRadioButtonId == R.id.micRadioButton
        val speechViews = mutableSetOf<View>(
            resultScrollView,
            classificationResultTextView,
            transcriptTextView,
            speechModeRadioGroup,
            diarizationCheckBox,
        )
        if (isMicMode) {
            speechViews.addAll(
                listOf(
                    speechLanguageLabel,
                    speechLanguageSpinner,
                    liveModeCheckBox,
                    micRecordButton,
                    waveformView,
                    waveformInfoTextView,
                )
            )
        } else {
            speechViews.add(speechRunButton)
        }

        val llmViews = setOf<View>(
            resultScrollView,
            llmInputLabel,
            llmInputEditText,
            llmSendButton,
            llmOutputLabel,
            llmChatContainer,
            llmStatusTextView,
        )

        val multimodalViews = llmViews.toMutableSet().apply {
            add(modeRadioGroup)
            add(multimodalImageView)
            if (isCameraMode) {
                add(cameraPreviewView)
            }
        }

        val voiceViews = setOf<View>(
            resultScrollView,
            voiceInputEditText,
            voiceStatusTextView,
            voiceGenerateButton,
            voiceEnvSpinner,
            voiceResultTextView,
            voiceWaveformView,
        )

        return mapOf(
            AlgorithmType.POSE_ESTIMATION to visionViews,
            AlgorithmType.OBJECT_DETECTION to visionViews,
            AlgorithmType.TRACKING to visionViews,
            AlgorithmType.TOKENIZE to tokenizeViews,
            AlgorithmType.CLASSIFICATION to visionViews,
            AlgorithmType.BACKGROUND_REMOVAL to visionViews,
            AlgorithmType.SPEECH_TO_TEXT to speechViews,
            AlgorithmType.TEXT_TO_SPEECH to voiceViews,
            AlgorithmType.LLM to llmViews,
            AlgorithmType.MULTIMODAL_LLM to multimodalViews,
        )
    }

    private fun updateUIVisibility() {
        val isCameraMode = modeRadioGroup.checkedRadioButtonId == R.id.cameraRadioButton
        val visibleViews = visibleViewsByAlgorithm(isCameraMode).getValue(currentAlgorithm)
        managedAlgorithmViews().forEach { view ->
            view.visibility = if (view in visibleViews) View.VISIBLE else View.GONE
        }

        // 表示切り替え時に必要なアルゴリズム固有の初期状態を設定する。
        when (currentAlgorithm) {
            AlgorithmType.SPEECH_TO_TEXT -> {
                classificationResultTextView.text = "Speech Result: --"
            }

            AlgorithmType.LLM -> {
                llmInputEditText.setText("Hello!")
                llmChatContainer.removeAllViews()
                llmStatusTextView.text = "Status: Press Send to run"
                llmSendButton.isEnabled = true
            }

            AlgorithmType.MULTIMODAL_LLM -> {
                llmInputEditText.setText("What is in this image?")
                llmChatContainer.removeAllViews()
                llmStatusTextView.text = "Status: Press Send to run"
                llmSendButton.isEnabled = true
            }

            AlgorithmType.TEXT_TO_SPEECH -> {
                voiceWaveformView.clear()
                voiceGenerateButton.isEnabled = true
                voiceResultTextView.text = ""
                voiceStatusTextView.text = "Status: Press Generate to synthesize"
            }

            else -> Unit
        }

        imageView.scaleType = if (isCameraMode) {
            ImageView.ScaleType.CENTER_CROP
        } else {
            ImageView.ScaleType.FIT_CENTER
        }

        cameraSpinner.visibility =
            if (modeRadioGroup.visibility == View.VISIBLE && isCameraMode) {
                View.VISIBLE
            } else {
                View.GONE
            }
        visionRunButton.visibility =
            if (needsVisionRunButton()) View.VISIBLE else View.GONE
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
        // アルゴリズム切り替え時も、前モデルが描画した画像を残さない。
        resetVisionDisplayForModelChange()
        // アルゴリズム切り替え時に波形表示をリセット
        hideModelDownloadProgress()
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
                voiceInputEditText.setText(selectedVoiceModelType.defaultText)
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

    private fun releaseCurrentAlgorithm(includeSpeech: Boolean = true) {
        try {
            stopMicRecording(finalize = false)
            poseEstimatorSample.releasePoseEstimator()
            objectDetectionSample.releaseObjectDetection()
            classificationSample.releaseClassification()
            onnxObjectDetectionSample.releaseObjectDetection()
            onnxClassificationSample.releaseClassification()
            u2netSample.release()
            detrSample.release()
            miniLMv2Sample.release()
            trackerSample.releaseTracker()
            if (includeSpeech) speechSample.releaseSpeech()
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
                if (hasPermission(Manifest.permission.CAMERA)) {
                    updateUIVisibility()
                    imageView.setImageBitmap(null)
                    startCamera()
                } else {
                    Toast.makeText(this, "Camera permission required", Toast.LENGTH_SHORT).show()
                    ActivityCompat.requestPermissions(
                        this,
                        arrayOf(Manifest.permission.CAMERA),
                        REQUEST_CODE_CAMERA_PERMISSION,
                    )
                    modeRadioGroup.check(R.id.imageRadioButton)
                }
            }
        }
    }

    /**
     * モデルのダウンロード、初期化、進捗・結果表示を共通の非同期フローで実行する。
     * ダウンロード済みの場合も同じ経路で初期化し、成功後にonReadyをUIスレッドで呼ぶ。
     */
    private fun initializeDownloadedModelAsync(
        logName: String,
        download: (ModelDownloadListener) -> Boolean,
        initialize: () -> Boolean,
        onReady: () -> Unit = {
            if (modeRadioGroup.checkedRadioButtonId == R.id.imageRadioButton) {
                processImageMode()
            }
        }
    ) {
        val operationId = beginModelOperation() ?: return
        runOnUiThreadIfActive {
            if (isCurrentOperation(operationId)) {
                processingTimeTextView.text = "Processing Time: -- ms"
            }
        }
        Log.i("AILIA_Main", "$logName: submitting download/init task")

        cameraExecutor.execute {
            var initializationSucceeded = false
            try {
                val downloaded = download(object : ModelDownloadListener {
                    override fun onProgress(
                        fileName: String,
                        bytesDownloaded: Long,
                        totalBytes: Long
                    ) {
                        Log.d("AILIA_Main", "$logName: downloading $fileName")
                        if (!isCurrentOperation(operationId)) return
                        runOnUiThreadIfActive {
                            if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                            showModelDownloadProgress(bytesDownloaded, totalBytes)
                        }
                    }

                    override fun onComplete() = Unit

                    override fun onError(error: String) {
                        runOnUiThreadIfActive {
                            if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                            processingTimeTextView.text = "Download error: $error"
                            hideModelDownloadProgress()
                        }
                    }
                })
                Log.i("AILIA_Main", "$logName: download result=$downloaded")
                if (!downloaded) return@execute

                initializationSucceeded = initialize()
                Log.i("AILIA_Main", "$logName: initialization result=$initializationSucceeded")
                runOnUiThreadIfActive {
                    if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                    isInitialized = initializationSucceeded
                    if (initializationSucceeded) {
                        processingTimeTextView.text = "Processing Time: -- ms"
                    } else {
                        processingTimeTextView.text = "Initialization failed"
                    }
                }
            } catch (e: Exception) {
                Log.e("AILIA_Main", "$logName: download/init error", e)
                runOnUiThreadIfActive {
                    if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                    processingTimeTextView.text = "Error: ${e.message}"
                }
            } finally {
                runOnUiThreadIfActive {
                    if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                    hideModelDownloadProgress()
                    finishModelOperation(operationId)
                    if (initializationSucceeded) onReady()
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
                    if (selectedRuntime == "ONNX" && useDetr) {
                        initializeDownloadedModelAsync(
                            logName = "DETR",
                            download = { detrSample.downloadModel(it) },
                            initialize = { detrSample.initialize(selectedEnvId) }
                        )
                        return
                    } else if (selectedRuntime == "ONNX") {
                        initializeDownloadedModelAsync(
                            logName = "ONNX ObjDet",
                            download = { onnxObjectDetectionSample.downloadModel(it) },
                            initialize = {
                                onnxObjectDetectionSample.initializeObjectDetection(selectedEnvId)
                            }
                        )
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
                        initializeDownloadedModelAsync(
                            logName = "ONNX Classification",
                            download = { onnxClassificationSample.downloadModel(it) },
                            initialize = {
                                onnxClassificationSample.initializeClassification(selectedEnvId)
                            }
                        )
                        return
                    } else if (classificationSample.modelType == TFLiteClassificationModelType.RESNET50) {
                        initializeDownloadedModelAsync(
                            logName = "TFLite ResNet50",
                            download = { classificationSample.downloadModel(it) },
                            initialize = {
                                classificationSample.initializeFromFile(env = selectedEnvId)
                            }
                        )
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
                    initializeDownloadedModelAsync(
                        logName = "MiniLMv2",
                        download = { miniLMv2Sample.downloadModel(it) },
                        initialize = { miniLMv2Sample.initialize(selectedEnvId) },
                        onReady = { processImageMode() }
                    )
                    return
                }

                AlgorithmType.TRACKING -> {
                    if (selectedRuntime == "ONNX") {
                        initializeDownloadedModelAsync(
                            logName = "ONNX Tracking",
                            download = { onnxObjectDetectionSample.downloadModel(it) },
                            initialize = {
                                val detectorSuccess =
                                    onnxObjectDetectionSample.initializeObjectDetection(selectedEnvId)
                                detectorSuccess && trackerSample.initializeTracker()
                            }
                        )
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

                AlgorithmType.BACKGROUND_REMOVAL -> {
                    initializeDownloadedModelAsync(
                        logName = "U2Net",
                        download = { u2netSample.downloadModel(it) },
                        initialize = { u2netSample.initialize(selectedEnvId) }
                    )
                    return
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

    /** モデルダウンロードの進捗を画面下部の共通ProgressBarへ表示する。 */
    private fun showModelDownloadProgress(bytesDownloaded: Long = 0, totalBytes: Long = 0) {
        modelDownloadProgressBar.visibility = View.VISIBLE
        if (totalBytes > 0) {
            modelDownloadProgressBar.isIndeterminate = false
            modelDownloadProgressBar.progress =
                ((bytesDownloaded * 100 / totalBytes).coerceIn(0, 100)).toInt()
        } else {
            modelDownloadProgressBar.isIndeterminate = true
        }
    }

    private fun hideModelDownloadProgress() {
        modelDownloadProgressBar.visibility = View.GONE
        modelDownloadProgressBar.isIndeterminate = false
        modelDownloadProgressBar.progress = 0
    }

    /** Prevents model/environment changes while a native model is being initialized or used. */
    private fun beginModelOperation(): Long? {
        if (!isDownloadingModel.compareAndSet(false, true)) return null
        val operationId = operationGeneration.incrementAndGet()
        if (android.os.Looper.myLooper() == android.os.Looper.getMainLooper()) {
            setModelOperationControlsEnabled(false)
        } else {
            runOnUiThreadIfActive {
                if (isCurrentOperation(operationId)) setModelOperationControlsEnabled(false)
            }
        }
        return operationId
    }

    private fun isCurrentOperation(operationId: Long): Boolean =
        !activityDestroyed.get() && operationGeneration.get() == operationId

    private fun finishModelOperation(operationId: Long) {
        if (!isCurrentOperation(operationId)) return
        isDownloadingModel.set(false)
        setModelOperationControlsEnabled(true)
    }

    private fun setModelOperationControlsEnabled(enabled: Boolean) {
        algorithmSpinner.isEnabled = enabled
        modelSpinner.isEnabled = enabled
        envSpinner.isEnabled = enabled
        voiceEnvSpinner.isEnabled = enabled
        modeRadioGroup.isEnabled = enabled
        for (index in 0 until modeRadioGroup.childCount) {
            modeRadioGroup.getChildAt(index).isEnabled = enabled
        }
        speechModeRadioGroup.isEnabled = enabled
        speechLanguageSpinner.isEnabled = enabled
        diarizationCheckBox.isEnabled = enabled
        liveModeCheckBox.isEnabled = enabled && !diarizationCheckBox.isChecked
        visionRunButton.isEnabled = enabled
    }

    private fun runOnUiThreadIfActive(action: () -> Unit) {
        if (activityDestroyed.get()) return
        runOnUiThread {
            if (!activityDestroyed.get()) action()
        }
    }

    private fun setupLLMSendButton() {
        llmSendButton.setOnClickListener {
            val userInput = llmInputEditText.text.toString().trim()
            if (userInput.isEmpty()) {
                llmStatusTextView.text = "Status: Please enter a message"
                return@setOnClickListener
            }
            performLLMChat(userInput)
        }
    }

    private fun performLLMChat(userInput: String) {
        val operationId = beginModelOperation() ?: return
        val needsInitialization = !isInitialized
        val modelType = selectedLLMModelType
        llmSendButton.isEnabled = false
        processingTimeTextView.text = "Processing Time: -- ms"
        llmStatusTextView.text = if (needsInitialization) "Status: Initializing..." else "Status: Generating..."
        // チャット風表示: 履歴は消さず、ユーザー発言とAI応答の吹き出しを追加する
        addChatBubble(userInput, isUser = true)
        val assistantBubble = addChatBubble("", isUser = false)
        llmInputEditText.setText("")

        cameraExecutor.execute {
            try {
                val initialized = if (needsInitialization) {
                    llmSample.modelType = modelType
                    llmSample.initialize(this@MainActivity, object : ModelDownloader.DownloadListener {
                        override fun onProgress(bytesDownloaded: Long, totalBytes: Long) {
                            if (!isCurrentOperation(operationId)) return
                            val percent = if (totalBytes > 0) bytesDownloaded * 100 / totalBytes else 0
                            runOnUiThreadIfActive {
                                if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                                llmStatusTextView.text = "Status: Downloading model... $percent%"
                                showModelDownloadProgress(bytesDownloaded, totalBytes)
                            }
                        }

                        override fun onComplete(file: File) = Unit

                        override fun onError(error: String) {
                            runOnUiThreadIfActive {
                                if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                                llmStatusTextView.text = "Status: Download error - $error"
                            }
                        }
                    })
                } else {
                    true
                }

                if (!initialized || !isCurrentOperation(operationId)) {
                    runOnUiThreadIfActive {
                        if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                        isInitialized = false
                        llmStatusTextView.text = "Status: Initialization failed"
                        hideModelDownloadProgress()
                        llmSendButton.isEnabled = true
                        finishModelOperation(operationId)
                    }
                    return@execute
                }

                runOnUiThreadIfActive {
                    if (isCurrentOperation(operationId)) llmStatusTextView.text = "Status: Generating..."
                }
                val processingTime = llmSample.chat(userInput, object : AiliaLLMSample.LLMListener {
                    override fun onToken(token: String) {
                        runOnUiThreadIfActive {
                            if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                            assistantBubble.append(token)
                            scrollResultToBottom()
                        }
                    }

                    override fun onComplete(fullResponse: String) = Unit

                    override fun onError(error: String) {
                        runOnUiThreadIfActive {
                            if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                            llmStatusTextView.text = "Status: Error - $error"
                        }
                    }
                })
                runOnUiThreadIfActive {
                    if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                    isInitialized = true
                    llmSendButton.isEnabled = true
                    hideModelDownloadProgress()
                    if (processingTime >= 0) {
                        llmStatusTextView.text = "Status: Complete"
                        processingTimeTextView.text = "Processing Time: ${processingTime}ms"
                    }
                    finishModelOperation(operationId)
                }
            } catch (e: Exception) {
                Log.e("AILIA_Main", "LLM request failed", e)
                runOnUiThreadIfActive {
                    if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                    llmSendButton.isEnabled = true
                    hideModelDownloadProgress()
                    llmStatusTextView.text = "Status: Error - ${e.message}"
                    finishModelOperation(operationId)
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
            performMultimodalChat(userInput)
        }
    }

    private fun performMultimodalChat(userInput: String) {
            val operationId = beginModelOperation() ?: return
            val needsInitialization = !isInitialized
            llmSendButton.isEnabled = false
            processingTimeTextView.text = "Processing Time: -- ms"
            llmStatusTextView.text = if (needsInitialization) "Status: Initializing..." else "Status: Generating..."
            // チャット風表示: 履歴は消さず、ユーザー発言とAI応答の吹き出しを追加する
            addChatBubble(userInput, isUser = true)
            val assistantBubble = addChatBubble("", isUser = false)
            llmInputEditText.setText("")

            val isCameraMode = modeRadioGroup.checkedRadioButtonId == R.id.cameraRadioButton
            val cameraFrame = if (isCameraMode) latestCameraBitmap?.copy(Bitmap.Config.ARGB_8888, false) else null

            cameraExecutor.execute {
                try {
                    val imagePath = cameraFrame?.let { bitmap ->
                        try {
                            val file = File(cacheDir, "camera_frame.png")
                            FileOutputStream(file).use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
                            file.absolutePath
                        } finally {
                            bitmap.recycle()
                        }
                    }
                    val listener = object : AiliaMultimodalLLMSample.MultimodalLLMListener {
                        override fun onDownloadProgress(fileName: String, bytesDownloaded: Long, totalBytes: Long) {
                            if (!isCurrentOperation(operationId)) return
                            val percent = if (totalBytes > 0) bytesDownloaded * 100 / totalBytes else 0
                            runOnUiThreadIfActive {
                                if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                                llmStatusTextView.text = "Status: Downloading $fileName... $percent%"
                                showModelDownloadProgress(bytesDownloaded, totalBytes)
                            }
                        }

                        override fun onStatus(status: String) {
                            runOnUiThreadIfActive {
                                if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                                llmStatusTextView.text = "Status: $status"
                            }
                        }

                        override fun onToken(token: String) {
                            runOnUiThreadIfActive {
                                if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                                assistantBubble.append(token)
                                scrollResultToBottom()
                            }
                        }

                        override fun onComplete(fullResponse: String) = Unit

                        override fun onError(error: String) {
                            runOnUiThreadIfActive {
                                if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                                llmStatusTextView.text = "Status: Error - $error"
                            }
                        }
                    }

                    val initialized = if (needsInitialization) {
                        multimodalLLMSample.initialize(this@MainActivity, listener)
                    } else {
                        true
                    }
                    if (!initialized || !isCurrentOperation(operationId)) {
                        runOnUiThreadIfActive {
                            if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                            isInitialized = false
                            llmStatusTextView.text = "Status: Initialization failed"
                            llmSendButton.isEnabled = true
                            hideModelDownloadProgress()
                            finishModelOperation(operationId)
                        }
                        return@execute
                    }

                    val processingTime = multimodalLLMSample.chatWithImage(imagePath, userInput, listener)
                    runOnUiThreadIfActive {
                        if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                        isInitialized = true
                        llmSendButton.isEnabled = true
                        hideModelDownloadProgress()
                        if (processingTime >= 0) {
                            llmStatusTextView.text = "Status: Complete"
                            processingTimeTextView.text = "Processing Time: ${processingTime}ms"
                        }
                        if (needsInitialization) loadSampleImageForMultimodal()
                        finishModelOperation(operationId)
                    }
                } catch (e: Exception) {
                    Log.e("AILIA_Main", "Multimodal LLM request failed", e)
                    cameraFrame?.takeUnless(Bitmap::isRecycled)?.recycle()
                    runOnUiThreadIfActive {
                        if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                        llmSendButton.isEnabled = true
                        hideModelDownloadProgress()
                        llmStatusTextView.text = "Status: Error - ${e.message}"
                        finishModelOperation(operationId)
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
            val operationId = beginModelOperation() ?: return@setOnClickListener
            val needsInitialization = !isInitialized
            val modelType = selectedVoiceModelType
            val envId = selectedEnvId
            voiceGenerateButton.isEnabled = false
            voiceResultTextView.text = ""
            voiceWaveformView.clear()
            voiceStatusTextView.text =
                if (needsInitialization) "Status: Initializing..." else "Status: Generating..."
            if (needsInitialization) {
                processingTimeTextView.text = "Processing Time: -- ms"
            }

            cameraExecutor.execute {
                try {
                    // モデルダウンロード+初期化はGenerate押下時に行う
                    val initialized = if (needsInitialization) {
                        voiceSample.modelType = modelType
                        voiceSample.initializeVoice(envId = envId, listener = object : ModelDownloadListener {
                            override fun onProgress(fileName: String, bytesDownloaded: Long, totalBytes: Long) {
                                if (!isCurrentOperation(operationId)) return
                                val percent = if (totalBytes > 0) bytesDownloaded * 100 / totalBytes else 0
                                runOnUiThreadIfActive {
                                    if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                                    voiceStatusTextView.text = "Status: Downloading $fileName... $percent%"
                                    showModelDownloadProgress(bytesDownloaded, totalBytes)
                                }
                            }

                            override fun onComplete() = Unit

                            override fun onError(error: String) {
                                runOnUiThreadIfActive {
                                    if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                                    voiceStatusTextView.text = "Status: Error - $error"
                                }
                            }
                        })
                    } else {
                        true
                    }
                    if (!initialized || !isCurrentOperation(operationId)) {
                        runOnUiThreadIfActive {
                            if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                            isInitialized = false
                            voiceGenerateButton.isEnabled = true
                            voiceStatusTextView.text = "Status: Initialization failed"
                            hideModelDownloadProgress()
                            finishModelOperation(operationId)
                        }
                        return@execute
                    }
                    runOnUiThreadIfActive {
                        if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                        voiceStatusTextView.text = "Status: Generating..."
                        hideModelDownloadProgress()
                    }

                    val refAudio = AudioUtil().loadRawAudio(resources.openRawResource(R.raw.reference_audio_girl))
                    // 入力テキストの日英を判定してG2Pの言語を切り替える
                    val textLang = AiliaVoiceSample.detectLanguage(inputText)
                    val inferenceTime = voiceSample.textToSpeech(
                        refAudio.audioData,
                        refAudio.channels,
                        refAudio.sampleRate,
                        "水をマレーシアから買わなくてはならない。",
                        "ja",
                        inputText,
                        textLang,
                    )
                    runOnUiThreadIfActive {
                        if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                        isInitialized = true
                        voiceGenerateButton.isEnabled = true
                        if (inferenceTime >= 0) {
                            voiceStatusTextView.text = "Status: Complete"
                            voiceResultTextView.text = "${modelType.displayName} Generated"
                            processingTimeTextView.text = "Processing Time: ${inferenceTime}ms"
                        } else {
                            voiceStatusTextView.text = "Status: Generation failed"
                        }
                        // 合成音声の波形を再生位置に追従して表示する
                        voiceSample.lastAudioData?.let { audio ->
                            voiceWaveformView.startPlayback(audio, voiceSample.lastAudioChannels, voiceSample.lastAudioSampleRate)
                            voiceReplayButton.visibility = View.VISIBLE
                        }
                        finishModelOperation(operationId)
                    }
                } catch (e: Exception) {
                    Log.e("AILIA_Main", "Voice request failed", e)
                    runOnUiThreadIfActive {
                        if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                        voiceGenerateButton.isEnabled = true
                        hideModelDownloadProgress()
                        voiceStatusTextView.text = "Status: Error - ${e.message}"
                        finishModelOperation(operationId)
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
                        stopMicRecording(finalize = false)
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
                    stopMicRecording(finalize = false)
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
        speechSample.diarizationEnabled = diarizationCheckBox.isChecked
        updateLiveModeAvailability(diarizationCheckBox.isChecked)
        diarizationCheckBox.setOnCheckedChangeListener { _, isChecked ->
            speechSample.diarizationEnabled = isChecked
            updateLiveModeAvailability(isChecked)
            resetSpeechForOptionChange()
        }
    }

    private fun setupLiveModeCheckBox() {
        liveModeCheckBox.setOnCheckedChangeListener { _, _ ->
            if (!isUpdatingSpeechOptionChecks) {
                resetSpeechForOptionChange()
            }
        }
    }

    /** Speaker Diarizationと併用できないLive ModeをOFFにして選択不可にする。 */
    private fun updateLiveModeAvailability(diarizationEnabled: Boolean) {
        isUpdatingSpeechOptionChecks = true
        try {
            if (diarizationEnabled) {
                liveModeCheckBox.isChecked = false
            }
            liveModeCheckBox.isEnabled = !diarizationEnabled
        } finally {
            isUpdatingSpeechOptionChecks = false
        }
    }

    /** 音声認識オプション変更後、次回のRun/Recordで設定を反映する。 */
    private fun resetSpeechForOptionChange() {
        stopMicRecording(finalize = false)
        speechSample.releaseSpeech()
        isInitialized = false
        isDownloadingModel.set(false)
        clearTranscript()
        classificationResultTextView.text = "Speech Result: --"
    }

    /** 議事録風トランスクリプトに行を追記して表示を更新する(UIスレッドで呼ぶこと) */
    private fun appendTranscriptLines(lines: List<String>) {
        if (lines.isEmpty()) return
        speechIntermediateText = null
        speechTranscript.addAll(lines)
        updateTranscriptDisplay()
    }

    /** 確定済み行の末尾にIntermediate結果を一時表示する。次の結果で同じ行を置き換える。 */
    private fun showIntermediateTranscript(text: String) {
        speechIntermediateText = text.trim().takeIf { it.isNotEmpty() }
        updateTranscriptDisplay()
    }

    private fun updateTranscriptDisplay() {
        val displayLines = speechTranscript.toMutableList()
        speechIntermediateText?.let { displayLines.add("$it...") }
        transcriptTextView.text = displayLines.joinToString("\n")
        scrollResultToBottom()
    }

    private fun clearTranscript() {
        speechTranscript.clear()
        speechIntermediateText = null
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
        val operationId = beginModelOperation() ?: return
        val isMicMode = speechModeRadioGroup.checkedRadioButtonId == R.id.micRadioButton
        // LIVEフラグはLive Modeチェックボックスで有効化(マイクモード時のみ)
        val liveMode = isMicMode && liveModeCheckBox.isChecked
        val modelType = selectedSpeechModelType
        val envId = selectedEnvId
        val language = if (isMicMode) selectedSpeechLanguage else "auto"
        processingTimeTextView.text = "Processing Time: -- ms"
        speechRunButton.isEnabled = false
        micRecordButton.isEnabled = false
        speechExecutor.execute {
            try {
                val downloaded = speechSample.downloadModel(modelType, object : ModelDownloadListener {
                    override fun onProgress(fileName: String, bytesDownloaded: Long, totalBytes: Long) {
                        Log.d("AILIA_Main", "Speech: downloading $fileName")
                        runOnUiThreadIfActive {
                            if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                            showModelDownloadProgress(bytesDownloaded, totalBytes)
                        }
                    }
                    override fun onComplete() {}
                    override fun onError(error: String) {
                        runOnUiThreadIfActive {
                            if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                            processingTimeTextView.text = "Download error: $error"
                            hideModelDownloadProgress()
                        }
                    }
                })
                var success = false
                if (downloaded) {
                    // Wavモードは常にauto、Micモードのみ言語選択を適用する
                    Log.i("AILIA_Main", "Speech: initializing with envId=$envId, liveMode=$liveMode, language=$language")
                    speechSample.language = language
                    success = speechSample.initializeSpeech(envId, liveMode = liveMode)
                }
                runOnUiThreadIfActive {
                    if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                    isInitialized = success
                    speechRunButton.isEnabled = true
                    micRecordButton.isEnabled = true
                    hideModelDownloadProgress()
                    finishModelOperation(operationId)
                    if (success) {
                        processingTimeTextView.text = "Processing Time: -- ms"
                        onReady()
                    } else {
                        processingTimeTextView.text = "Failed to initialize speech model"
                    }
                }
            } catch (e: Exception) {
                Log.e("AILIA_Main", "Speech: download/init error", e)
                runOnUiThreadIfActive {
                    if (!isCurrentOperation(operationId)) return@runOnUiThreadIfActive
                    speechRunButton.isEnabled = true
                    micRecordButton.isEnabled = true
                    hideModelDownloadProgress()
                    finishModelOperation(operationId)
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
                        runOnUiThreadIfActive {
                            appendTranscriptLines(lines)
                            classificationResultTextView.text =
                                if (lines.isEmpty()) "Speech Result: (no speech detected)" else "Speech Result:"
                            processingTimeTextView.text = "Processing Time: $timeMs ms"
                        }
                    } catch (e: Exception) {
                        Log.e("AILIA_Main", "Speech run error", e)
                        runOnUiThreadIfActive {
                            classificationResultTextView.text = "Speech Result: Error - ${e.message}"
                        }
                    } finally {
                        runOnUiThreadIfActive {
                            speechRunButton.isEnabled = true
                        }
                    }
                }
            }
        }
    }

    private fun setupMicRecordButton() {
        micRecordButton.setOnClickListener {
            if (speechSample.isMicRecording) {
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
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), REQUEST_CODE_AUDIO_PERMISSION)
            return
        }

        val started = speechSample.startMicRecording(object : AiliaSpeechSample.MicRecordingListener {
            override fun onWaveform(samples: FloatArray, sampleRate: Int) {
                val blocks = WaveformView.peakBlocks(samples, samples.size, sampleRate)
                runOnUiThreadIfActive {
                    waveformView.push(blocks)
                }
            }

            override fun onIntermediateResult(text: String) {
                runOnUiThreadIfActive {
                    showIntermediateTranscript(text)
                    classificationResultTextView.text = "Recording..."
                }
            }

            override fun onResult(lines: List<String>, isFinal: Boolean) {
                runOnUiThreadIfActive {
                    appendTranscriptLines(lines)
                    if (isFinal) {
                        speechIntermediateText = null
                        updateTranscriptDisplay()
                        classificationResultTextView.text =
                            if (speechTranscript.isEmpty()) "Speech Result: (no speech detected)" else "Speech Result:"
                        recTimerHandler.removeCallbacks(recTimerRunnable)
                        micRecordButton.text = "Record"
                        micRecordButton.isEnabled = true
                        waveformInfoTextView.text = ""
                    }
                }
            }

            override fun onError(error: String) {
                runOnUiThreadIfActive {
                    Log.e("AILIA_Main", "Microphone recording error: $error")
                    classificationResultTextView.text = "Speech Result: Error - $error"
                    if (!speechSample.isMicRecording) {
                        recTimerHandler.removeCallbacks(recTimerRunnable)
                        micRecordButton.text = "Record"
                        waveformInfoTextView.text = ""
                    }
                    micRecordButton.isEnabled = true
                }
            }
        })

        if (started) {
            micRecordButton.text = "Stop"
            classificationResultTextView.text = "Recording..."
            clearTranscript()

            // 波形表示とREC経過時間の初期化
            waveformView.clear()
            recStartMs = android.os.SystemClock.elapsedRealtime()
            waveformInfoTextView.text = "● REC 00:00"
            recTimerHandler.post(recTimerRunnable)
        }
    }

    private fun stopMicRecording(finalize: Boolean = true) {
        if (speechSample.isMicRecording) {
            speechSample.stopMicRecording(finalize)
            recTimerHandler.removeCallbacks(recTimerRunnable)
            runOnUiThreadIfActive {
                micRecordButton.text = "Record"
                micRecordButton.isEnabled = !finalize
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
                    currentAlgorithm == AlgorithmType.BACKGROUND_REMOVAL ||
                    currentAlgorithm == AlgorithmType.TRACKING)) {
            if (!isInitialized) {
                initializeAilia()
                return
            }
        }

        // TFLiteのResNet50もダウンロードが必要なため非同期で初期化する
        if (currentAlgorithm == AlgorithmType.CLASSIFICATION && selectedRuntime == "TFLite" &&
            classificationSample.modelType == TFLiteClassificationModelType.RESNET50) {
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

        if (!isProcessing.compareAndSet(false, true)) return
        val tokenizerInput = tokenizerInputEditText.text.toString()
        setModelOperationControlsEnabled(false)
        cameraExecutor.execute { runImageModeInference(tokenizerInput) }
    }

    /** Runs bundled-model initialization and still-image inference away from the UI thread. */
    private fun runImageModeInference(tokenizerInput: String) {
        try {
            if (!isInitialized) {
                initializeAilia()
            }

            if (!isInitialized) {
                runOnUiThreadIfActive {
                    processingTimeTextView.text = "Initialization failed"
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
            val processingTime = processAlgorithm(img, personBmp, canvas, w, h, tokenizerInput)

            runOnUiThreadIfActive {
                if (currentAlgorithm != AlgorithmType.TOKENIZE) {
                    imageView.setImageBitmap(bitmap)
                }
                var timeText = "Processing Time: ${processingTime}ms"
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
            runOnUiThreadIfActive {
                processingTimeTextView.text = "Processing Error: ${e.message}"
            }
        } finally {
            isProcessing.set(false)
            runOnUiThreadIfActive {
                if (!isDownloadingModel.get()) setModelOperationControlsEnabled(true)
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
            val displayGeneration = visionDisplayGeneration.get()

            try {
                // フロントカメラはPreviewViewと同じ鏡像表示になるよう左右反転する
                val isFrontCamera = cameraChoices[selectedCameraIndex].selector == CameraSelector.DEFAULT_FRONT_CAMERA
                val imageUtil = ImageUtil()
                var camera_bitmap = imageUtil.imageProxyToBitmap(image, mirror = isFrontCamera)
                camera_bitmap = imageUtil.cropToSquare(camera_bitmap)

                val img = ImageUtil().loadRawImage(camera_bitmap)
                val w = camera_bitmap.width
                val h = camera_bitmap.height

                Log.i("AILIA_Main", "${w} ${h}")

                val bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
                val canvas = Canvas(bitmap)
                canvas.drawBitmap(camera_bitmap, 0f, 0f, null)

                val processingTime = processAlgorithm(img, bitmap, canvas, w, h)

                runOnUiThreadIfActive {
                    // モデル/アルゴリズム切り替え前に開始したフレームは表示しない。
                    if (displayGeneration != visionDisplayGeneration.get()) {
                        return@runOnUiThreadIfActive
                    }
                    // Stop押下後に処理中だったフレームの結果で表示を上書きしない
                    if (needsVisionRunButton() && !runRequested) {
                        return@runOnUiThreadIfActive
                    }
                    if (currentAlgorithm == AlgorithmType.MULTIMODAL_LLM) {
                        latestCameraBitmap = camera_bitmap
                        multimodalImageView.setImageBitmap(camera_bitmap)
                    } else if (currentAlgorithm != AlgorithmType.TOKENIZE) {
                        imageView.setImageBitmap(bitmap)
                    }

                    val fps = if (processingTime > 0) 1000 / processingTime else 0
                    var timeText = "Processing Time: ${processingTime}ms - FPS: $fps"
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
                    runOnUiThreadIfActive {
                        executeAlgorithmSwitch(pendingAlgorithm)
                        isWaitAlgorithmSwitch.set(false)
                    }
                }

                pendingModeSwitch?.let { pendingMode ->
                    pendingModeSwitch = null
                    isWaitModeSwitch.set(true)
                    runOnUiThreadIfActive {
                        executeModeSwitch(pendingMode)
                        isWaitModeSwitch.set(false)
                    }
                }
            }
        }
    }

    private fun hasPermission(permission: String): Boolean =
        ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        val granted = grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED
        when (requestCode) {
            REQUEST_CODE_CAMERA_PERMISSION -> if (granted) {
                modeRadioGroup.check(R.id.cameraRadioButton)
            } else {
                Toast.makeText(this, "Camera permission was not granted.", Toast.LENGTH_SHORT).show()
            }
            REQUEST_CODE_AUDIO_PERMISSION -> if (granted) {
                startMicRecording()
            } else {
                Toast.makeText(this, "Microphone permission was not granted.", Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onDestroy() {
        activityDestroyed.set(true)
        operationGeneration.incrementAndGet()
        isDownloadingModel.set(false)
        llmSample.cancelGeneration()
        multimodalLLMSample.cancelGeneration()
        recTimerHandler.removeCallbacksAndMessages(null)
        stopMicRecording(finalize = false)
        stopCamera()

        // Native objects are released after already queued work on the same executor.
        // This avoids destroying a model while its inference call is still running.
        speechExecutor.execute { speechSample.releaseSpeech() }
        speechExecutor.shutdown()
        cameraExecutor.execute { releaseCurrentAlgorithm(includeSpeech = false) }
        cameraExecutor.shutdown()
        super.onDestroy()
    }

    @Throws(IOException::class)
    fun inputStreamToByteArray(`in`: InputStream): ByteArray? {
        val bout = ByteArrayOutputStream()
        BufferedOutputStream(bout).use { out ->
            val buf = ByteArray(128)
            while (true) {
                val count = `in`.read(buf)
                if (count <= 0) break
                out.write(buf, 0, count)
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
