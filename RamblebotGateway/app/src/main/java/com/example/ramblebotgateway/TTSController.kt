package com.example.ramblebotgateway

import android.content.Context
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import java.util.Locale
import java.util.concurrent.ConcurrentLinkedQueue

/**
 * Text-to-Speech controller for robot commentary.
 *
 * Supports Finnish (fi-FI) and other languages.
 * Provides HTTP endpoint for remote TTS control.
 */
class TTSController(context: Context) : TextToSpeech.OnInitListener {

    private val tag = "TTSController"
    private var tts: TextToSpeech? = null
    private var isInitialized = false
    private var currentLocale: Locale = Locale("fi", "FI")

    // Queue for utterances before TTS is ready
    private val pendingQueue = ConcurrentLinkedQueue<PendingUtterance>()

    // Stats
    @Volatile var utteranceCount = 0
        private set
    @Volatile var lastSpokenText = ""
        private set
    @Volatile var isSpeaking = false
        private set

    private data class PendingUtterance(val text: String, val lang: String, val queue: Boolean)

    init {
        tts = TextToSpeech(context, this)
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            // Set Finnish as default
            val result = tts?.setLanguage(currentLocale)
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.w(tag, "Finnish not available, trying English")
                tts?.setLanguage(Locale.US)
            }

            // Set up progress listener
            tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                override fun onStart(utteranceId: String?) {
                    isSpeaking = true
                }

                override fun onDone(utteranceId: String?) {
                    isSpeaking = false
                }

                @Deprecated("Deprecated in Java")
                override fun onError(utteranceId: String?) {
                    isSpeaking = false
                    Log.e(tag, "TTS error for utterance: $utteranceId")
                }

                override fun onError(utteranceId: String?, errorCode: Int) {
                    isSpeaking = false
                    Log.e(tag, "TTS error $errorCode for utterance: $utteranceId")
                }
            })

            // Set speech rate (slightly faster than default)
            tts?.setSpeechRate(1.1f)

            // Set pitch
            tts?.setPitch(1.0f)

            isInitialized = true
            Log.i(tag, "TTS initialized with locale: $currentLocale")

            // Process pending queue
            while (pendingQueue.isNotEmpty()) {
                val pending = pendingQueue.poll()
                if (pending != null) {
                    speak(pending.text, pending.lang, pending.queue)
                }
            }
        } else {
            Log.e(tag, "TTS initialization failed with status: $status")
        }
    }

    /**
     * Speak text with specified language.
     *
     * @param text Text to speak
     * @param lang Language code (e.g., "fi-FI", "en-US")
     * @param queue If true, add to queue; if false, interrupt current speech
     */
    fun speak(text: String, lang: String = "fi-FI", queue: Boolean = false): Boolean {
        if (!isInitialized) {
            pendingQueue.offer(PendingUtterance(text, lang, queue))
            return false
        }

        if (text.isBlank()) return false

        // Parse language code
        val locale = parseLocale(lang)
        if (locale != currentLocale) {
            val result = tts?.setLanguage(locale)
            if (result != TextToSpeech.LANG_MISSING_DATA && result != TextToSpeech.LANG_NOT_SUPPORTED) {
                currentLocale = locale
            }
        }

        val queueMode = if (queue) TextToSpeech.QUEUE_ADD else TextToSpeech.QUEUE_FLUSH
        val utteranceId = "utt_${System.currentTimeMillis()}"

        tts?.speak(text, queueMode, null, utteranceId)

        utteranceCount++
        lastSpokenText = text
        Log.d(tag, "Speaking: $text (lang: $lang, queue: $queue)")

        return true
    }

    /**
     * Stop current speech and clear queue.
     */
    fun stop() {
        tts?.stop()
        isSpeaking = false
    }

    /**
     * Check if TTS is currently speaking.
     */
    fun isBusy(): Boolean {
        return tts?.isSpeaking == true || isSpeaking
    }

    /**
     * Get available languages.
     */
    fun getAvailableLanguages(): List<String> {
        return tts?.availableLanguages?.map { it.toLanguageTag() } ?: emptyList()
    }

    /**
     * Set speech rate.
     *
     * @param rate 0.5 (slow) to 2.0 (fast), 1.0 is normal
     */
    fun setSpeechRate(rate: Float) {
        tts?.setSpeechRate(rate.coerceIn(0.5f, 2.0f))
    }

    /**
     * Get TTS status as JSON.
     */
    fun toJson(): String {
        return buildString {
            append("{")
            append("\"initialized\":$isInitialized,")
            append("\"speaking\":$isSpeaking,")
            append("\"locale\":\"${currentLocale.toLanguageTag()}\",")
            append("\"utteranceCount\":$utteranceCount,")
            append("\"lastText\":\"${lastSpokenText.replace("\"", "\\\"").take(50)}\"")
            append("}")
        }
    }

    /**
     * Clean up TTS resources.
     */
    fun shutdown() {
        tts?.stop()
        tts?.shutdown()
        tts = null
        isInitialized = false
    }

    private fun parseLocale(langCode: String): Locale {
        val parts = langCode.split("-", "_")
        return when (parts.size) {
            1 -> Locale(parts[0])
            else -> Locale(parts[0], parts[1])
        }
    }
}
