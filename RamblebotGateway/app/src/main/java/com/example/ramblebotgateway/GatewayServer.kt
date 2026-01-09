package com.example.ramblebotgateway

import fi.iki.elonen.NanoHTTPD
import java.io.PipedInputStream
import java.io.PipedOutputStream
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

class GatewayServer(
    private val bt: BluetoothController,
    private val sensors: SensorHub,
    private val arCore: ArCoreTracker,
    private val tts: TTSController,
    private val webHtml: String
) : NanoHTTPD(8765) {

    override fun serve(session: IHTTPSession): Response {
        val uri = session.uri ?: "/"

        return try {
            when {
                uri == "/" -> {
                    newFixedLengthResponse(Response.Status.OK, "text/html; charset=utf-8", webHtml)
                }

                uri == "/stream.mjpeg" -> {
                    val pis = PipedInputStream(64 * 1024)
                    val pos = PipedOutputStream(pis)
                    MjpegHub.startWriterThread(pos)

                    val resp = newChunkedResponse(
                        Response.Status.OK,
                        "multipart/x-mixed-replace; boundary=frame",
                        pis
                    )
                    resp.addHeader("Cache-Control", "no-cache, no-store, must-revalidate")
                    resp.addHeader("Pragma", "no-cache")
                    resp.addHeader("Connection", "close")
                    resp
                }

                uri == "/sensors.json" -> {
                    val json = sensors.toJson()
                    val resp = newFixedLengthResponse(Response.Status.OK, "application/json", json)
                    resp.addHeader("Cache-Control", "no-cache, no-store, must-revalidate")
                    resp.addHeader("Pragma", "no-cache")
                    resp
                }

                uri == "/arcore.json" -> {
                    val pose = arCore.latest()
                    val payload = if (pose == null) {
                        """{"available":false,"running":${arCore.isRunning()}}"""
                    } else {
                        """{"available":true,"running":${arCore.isRunning()},"trackingState":"${pose.trackingState}","timestampNs":${pose.timestampNs},"position":${pose.position},"rotation":${pose.rotation}}"""
                    }
                    val resp = newFixedLengthResponse(Response.Status.OK, "application/json", payload)
                    resp.addHeader("Cache-Control", "no-cache, no-store, must-revalidate")
                    resp.addHeader("Pragma", "no-cache")
                    resp
                }

                uri.startsWith("/cmd") -> handleCmd(session)

                uri.startsWith("/speak") -> handleSpeak(session)

                uri == "/tts.json" -> {
                    val json = tts.toJson()
                    val resp = newFixedLengthResponse(Response.Status.OK, "application/json", json)
                    resp.addHeader("Cache-Control", "no-cache")
                    resp
                }

                else -> newFixedLengthResponse(Response.Status.NOT_FOUND, "text/plain", "Not found")
            }
        } catch (e: Exception) {
            newFixedLengthResponse(
                Response.Status.INTERNAL_ERROR,
                "text/plain",
                "ERR: ${e.javaClass.simpleName}: ${e.message}"
            )
        }
    }

    private fun handleCmd(session: IHTTPSession): Response {
        val cmd = session.parameters["do"]?.firstOrNull()?.lowercase()

        return when (cmd) {
            "drive" -> {
                val l = session.parameters["l"]?.firstOrNull()?.toIntOrNull()
                val r = session.parameters["r"]?.firstOrNull()?.toIntOrNull()
                if (l == null || r == null) {
                    newFixedLengthResponse(
                        Response.Status.BAD_REQUEST,
                        "text/plain",
                        "Missing l/r. Example: /cmd?do=drive&l=200&r=180"
                    )
                } else {
                    bt.sendLine(buildDriveCommand(l, r))
                    newFixedLengthResponse("OK")
                }
            }

            "stop" -> {
                bt.sendLine("0")
                newFixedLengthResponse("OK")
            }

            // ✅ UUSI: Head tilt (Servo Head) based on .ino case '3'
            // /cmd?do=head&pos=90&speed=0
            "head" -> {
                val pos = session.parameters["pos"]?.firstOrNull()?.toIntOrNull()
                val speed = session.parameters["speed"]?.firstOrNull()?.toIntOrNull() ?: 0

                if (pos == null) {
                    return newFixedLengthResponse(
                        Response.Status.BAD_REQUEST,
                        "text/plain",
                        "Missing pos. Example: /cmd?do=head&pos=90&speed=0"
                    )
                }

                val p = pos.coerceIn(0, 180)
                val s = speed.coerceIn(0, 9)

                // Format required by Arduino:
                // command '3' then parseInt = [servocommand][pos(3 digits)][speedDigit]
                // Use servocommand=1 to preserve leading zeros in pos.
                val payload = "1" + p.toString().padStart(3, '0') + s.toString()
                bt.sendLine("3 $payload")

                newFixedLengthResponse("OK head pos=$p speed=$s")
            }

            // legacy shortcuts
            "forward" -> { bt.sendLine(buildDriveCommand(200, 200)); newFixedLengthResponse("OK") }
            "back"    -> { bt.sendLine(buildDriveCommand(-200, -200)); newFixedLengthResponse("OK") }
            "left"    -> { bt.sendLine(buildDriveCommand(-140, 140)); newFixedLengthResponse("OK") }
            "right"   -> { bt.sendLine(buildDriveCommand(140, -140)); newFixedLengthResponse("OK") }

            else -> newFixedLengthResponse(
                Response.Status.BAD_REQUEST,
                "text/plain",
                "Unknown. Use do=drive&l=..&r=.., do=stop, do=head&pos=..&speed=.."
            )
        }
    }

    private fun buildDriveCommand(left: Int, right: Int): String {
        val (dl, sl) = encodeWheel(left)
        val (dr, sr) = encodeWheel(right)
        return "1 ${dl}${sl}${dr}${sr}"
    }

    private fun encodeWheel(v: Int): Pair<Int, String> {
        val clamped = max(-255, min(255, v))
        val dir = if (clamped >= 0) 1 else 2
        val speed = abs(clamped)
        val sss = speed.toString().padStart(3, '0')
        return dir to sss
    }

    /**
     * Handle TTS speak requests.
     * /speak?text=Hello&lang=fi-FI&queue=0
     * /speak?stop=1
     */
    private fun handleSpeak(session: IHTTPSession): Response {
        // Stop command
        val stopParam = session.parameters["stop"]?.firstOrNull()
        if (stopParam == "1") {
            tts.stop()
            return newFixedLengthResponse("OK stopped")
        }

        // Get text to speak
        val text = session.parameters["text"]?.firstOrNull()
        if (text.isNullOrBlank()) {
            return newFixedLengthResponse(
                Response.Status.BAD_REQUEST,
                "text/plain",
                "Missing text. Example: /speak?text=Moi!&lang=fi-FI"
            )
        }

        // Language (default Finnish)
        val lang = session.parameters["lang"]?.firstOrNull() ?: "fi-FI"

        // Queue mode (0 = interrupt, 1 = add to queue)
        val queue = session.parameters["queue"]?.firstOrNull() == "1"

        // Speak
        val success = tts.speak(text, lang, queue)

        return if (success) {
            newFixedLengthResponse("OK speaking: $text")
        } else {
            newFixedLengthResponse(
                Response.Status.SERVICE_UNAVAILABLE,
                "text/plain",
                "TTS not ready or failed"
            )
        }
    }
}
