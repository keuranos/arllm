package com.example.ramblebotgateway

import android.content.Context
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import android.os.Build
import android.util.Log
import com.google.ar.core.Config
import com.google.ar.core.Frame
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.NotYetAvailableException
import java.io.ByteArrayOutputStream
import java.util.concurrent.atomic.AtomicBoolean

data class ArCorePose(
    val trackingState: String,
    val timestampNs: Long,
    val position: List<Float>,
    val rotation: List<Float>
)

class ArCoreTracker(private val context: Context) {
    private var session: Session? = null
    private val running = AtomicBoolean(false)
    private var thread: Thread? = null
    private var lastImageTimestamp = 0L

    @Volatile
    private var latestPose: ArCorePose? = null

    fun start(): Boolean {
        if (running.get()) return true
        return try {
            session = Session(context).apply {
                val config = Config(this).apply {
                    updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
                    focusMode = Config.FocusMode.AUTO
                }
                configure(config)
            }
            session?.setCameraTextureName(0)
            session?.resume()
            running.set(true)
            thread = Thread { loop() }.also { it.start() }
            true
        } catch (e: Exception) {
            Log.w("ArCoreTracker", "Failed to start ARCore: ${e.message}")
            stop()
            false
        }
    }

    fun stop() {
        running.set(false)
        thread?.join(200)
        thread = null
        try { session?.pause() } catch (_: Exception) {}
        try { session?.close() } catch (_: Exception) {}
        session = null
    }

    fun isRunning(): Boolean = running.get()

    fun latest(): ArCorePose? = latestPose

    private fun loop() {
        while (running.get()) {
            try {
                val frame = session?.update()
                if (frame != null) {
                    updatePose(frame)
                }
                Thread.sleep(50)
            } catch (e: Exception) {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                    Log.w("ArCoreTracker", "ARCore loop error: ${e.message}")
                }
                Thread.sleep(200)
            }
        }
    }

    private fun updatePose(frame: Frame) {
        val camera = frame.camera
        val pose = camera.pose
        val tracking = when (camera.trackingState) {
            TrackingState.TRACKING -> "tracking"
            TrackingState.PAUSED -> "paused"
            TrackingState.STOPPED -> "stopped"
        }
        val position = listOf(pose.tx(), pose.ty(), pose.tz())
        val rotation = floatArrayOf(0f, 0f, 0f, 0f).also { pose.getRotationQuaternion(it, 0) }
        latestPose = ArCorePose(
            trackingState = tracking,
            timestampNs = frame.timestamp,
            position = position,
            rotation = rotation.toList()
        )
        updateCameraFrame(frame)
    }

    private fun updateCameraFrame(frame: Frame) {
        if (frame.timestamp <= 0) return
        if (frame.timestamp - lastImageTimestamp < 100_000_000L) return
        try {
            val image = frame.acquireCameraImage()
            try {
                val jpeg = imageToJpeg(image, 60)
                if (jpeg != null) {
                    MjpegHub.updateFrame(jpeg)
                    lastImageTimestamp = frame.timestamp
                }
            } finally {
                image.close()
            }
        } catch (_: NotYetAvailableException) {
        } catch (_: Exception) {
        }
    }

    private fun imageToJpeg(image: Image, quality: Int): ByteArray? {
        if (image.format != ImageFormat.YUV_420_888) return null
        val nv21 = imageToNv21(image)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), quality, out)
        return out.toByteArray()
    }

    private fun imageToNv21(image: Image): ByteArray {
        val w = image.width
        val h = image.height
        val ySize = w * h
        val uvSize = w * h / 2
        val nv21 = ByteArray(ySize + uvSize)

        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer

        val yRowStride = yPlane.rowStride
        val yPixelStride = yPlane.pixelStride

        var yIndex = 0
        for (row in 0 until h) {
            val yRowStart = row * yRowStride
            for (col in 0 until w) {
                nv21[yIndex++] = yBuffer.get(yRowStart + col * yPixelStride)
            }
        }

        val uvRowStride = uPlane.rowStride
        val uvPixelStride = uPlane.pixelStride
        var uvIndex = ySize
        val halfH = h / 2
        val halfW = w / 2
        for (row in 0 until halfH) {
            val uvRowStart = row * uvRowStride
            for (col in 0 until halfW) {
                val uvOffset = uvRowStart + col * uvPixelStride
                nv21[uvIndex++] = vBuffer.get(uvOffset)
                nv21[uvIndex++] = uBuffer.get(uvOffset)
            }
        }
        return nv21
    }
}
