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
                    captureFrameForMjpeg(frame)
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
            TrackingState.TRACKING -> "TRACKING"
            TrackingState.PAUSED -> "PAUSED"
            TrackingState.STOPPED -> "STOPPED"
        }
        val position = listOf(pose.tx(), pose.ty(), pose.tz())
        val rotation = floatArrayOf(0f, 0f, 0f, 0f).also { pose.getRotationQuaternion(it, 0) }
        latestPose = ArCorePose(
            trackingState = tracking,
            timestampNs = frame.timestamp,
            position = position,
            rotation = rotation.toList()
        )
    }

    /**
     * Capture camera frame from ARCore and feed to MjpegHub for streaming.
     */
    private fun captureFrameForMjpeg(frame: Frame) {
        var image: Image? = null
        try {
            image = frame.acquireCameraImage()
            val jpeg = imageToJpeg(image, 60)
            if (jpeg != null) {
                MjpegHub.updateFrame(jpeg)
            }
        } catch (e: NotYetAvailableException) {
            // Frame not ready yet, skip
        } catch (e: Exception) {
            Log.w("ArCoreTracker", "Failed to capture frame: ${e.message}")
        } finally {
            image?.close()
        }
    }

    /**
     * Convert YUV_420_888 Image to JPEG bytes.
     */
    private fun imageToJpeg(image: Image, quality: Int): ByteArray? {
        if (image.format != ImageFormat.YUV_420_888) {
            return null
        }

        val width = image.width
        val height = image.height

        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        // NV21 format: Y plane followed by interleaved VU
        val nv21 = ByteArray(ySize + width * height / 2)

        // Copy Y plane
        yBuffer.get(nv21, 0, ySize)

        // Interleave V and U (NV21 expects VUVU...)
        val uvPixelStride = uPlane.pixelStride
        val uvRowStride = uPlane.rowStride

        var uvIndex = ySize
        for (row in 0 until height / 2) {
            for (col in 0 until width / 2) {
                val vIndex = row * uvRowStride + col * uvPixelStride
                val uIndex = row * uvRowStride + col * uvPixelStride

                if (vIndex < vSize && uIndex < uSize && uvIndex + 1 < nv21.size) {
                    vBuffer.position(vIndex)
                    uBuffer.position(uIndex)
                    nv21[uvIndex++] = vBuffer.get()
                    nv21[uvIndex++] = uBuffer.get()
                }
            }
        }

        // Reset buffer positions
        vBuffer.rewind()
        uBuffer.rewind()

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), quality, out)
        return out.toByteArray()
    }
}
