package jp.axinc.ailia_kotlin

/** モデルファイルのダウンロード進捗をサンプル実装からUIへ通知する共通リスナー。 */
interface ModelDownloadListener {
    fun onProgress(fileName: String, bytesDownloaded: Long, totalBytes: Long)
    fun onComplete()
    fun onError(error: String)
}
