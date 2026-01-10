package com.example.ramblebotgateway

import android.content.Context
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraManager
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

    // Flashlight control
    private val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    private var flashlightOn = false
    private var flashCameraId: String? = null

    init {
        // Find camera with flash
        try {
            for (id in cameraManager.cameraIdList) {
                val chars = cameraManager.getCameraCharacteristics(id)
                val hasFlash = chars.get(android.hardware.camera2.CameraCharacteristics.FLASH_INFO_AVAILABLE) ?: false
                if (hasFlash) {
                    flashCameraId = id
                    break
                }
            }
            Log.i("ArCoreTracker", "Flash camera ID: $flashCameraId")
        } catch (e: Exception) {
            Log.w("ArCoreTracker", "Failed to find flash camera: ${e.message}")
        }
    }

    fun setFlashlight(on: Boolean): Boolean {
        if (flashCameraId == null) {
            Log.w("ArCoreTracker", "No flash available")
            return false
        }
        return try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                cameraManager.setTorchMode(flashCameraId!!, on)
                flashlightOn = on
                Log.i("ArCoreTracker", "Flashlight: ${if (on) "ON" else "OFF"}")
                true
            } else {
                false
            }
        } catch (e: CameraAccessException) {
            Log.w("ArCoreTracker", "Failed to set flashlight: ${e.message}")
            false
        }
    }

    fun isFlashlightOn(): Boolean = flashlightOn

    fun toggleFlashlight(): Boolean {
        return setFlashlight(!flashlightOn)
    }

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
        var frameCount = 0
        var lastLog = System.currentTimeMillis()

        while (running.get()) {
            try {
                val frame = session?.update()
                if (frame != null) {
                    updatePose(frame)
                    captureFrameForMjpeg(frame)
                    frameCount++

                    // Log status every 5 seconds
                    val now = System.currentTimeMillis()
                    if (now - lastLog > 5000) {
                        val state = latestPose?.trackingState ?: "UNKNOWN"
                        Log.i("ArCoreTracker", "ARCore running: state=$state, frames=$frameCount")
                        lastLog = now
                    }
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
    private var lastCaptureLog = 0L
    private var captureSuccessCount = 0
    private var captureFailCount = 0

    private fun captureFrameForMjpeg(frame: Frame) {
        var image: Image? = null
        try {
            image = frame.acquireCameraImage()
            val jpeg = imageToJpeg(image, 60)
            if (jpeg != null) {
                MjpegHub.updateFrame(jpeg)
                captureSuccessCount++
            } else {
                captureFailCount++
            }
        } catch (e: NotYetAvailableException) {
            // Frame not ready yet, skip - this is normal during startup
            captureFailCount++
        } catch (e: Exception) {
            Log.w("ArCoreTracker", "Failed to capture frame: ${e.message}")
            captureFailCount++
        } finally {
            image?.close()
        }

        // Log capture stats every 10 seconds
        val now = System.currentTimeMillis()
        if (now - lastCaptureLog > 10000) {
            Log.i("ArCoreTracker", "MJPEG capture: success=$captureSuccessCount, fail=$captureFailCount")
            lastCaptureLog = now
        }
    }

    /**
     * Convert YUV_420_888 Image to JPEG bytes.
     * Handles various YUV formats from different devices.
     */
    private fun imageToJpeg(image: Image, quality: Int): ByteArray? {
        try {
            if (image.format != ImageFormat.YUV_420_888) {
                Log.w("ArCoreTracker", "Unexpected image format: ${image.format}")
                return null
            }

            val width = image.width
            val height = image.height

            if (width <= 0 || height <= 0) {
                return null
            }

            val yPlane = image.planes[0]
            val uPlane = image.planes[1]
            val vPlane = image.planes[2]

            val yBuffer = yPlane.buffer.duplicate()
            val uBuffer = uPlane.buffer.duplicate()
            val vBuffer = vPlane.buffer.duplicate()

            val yRowStride = yPlane.rowStride
            val uvRowStride = uPlane.rowStride
            val uvPixelStride = uPlane.pixelStride

            // Allocate NV21 buffer
            val nv21 = ByteArray(width * height + width * height / 2)

            // Copy Y plane row by row (handles rowStride != width)
            var yOffset = 0
            for (row in 0 until height) {
                yBuffer.position(row * yRowStride)
                yBuffer.get(nv21, yOffset, width)
                yOffset += width
            }

            // Interleave V and U for NV21 format
            var uvOffset = width * height
            for (row in 0 until height / 2) {
                for (col in 0 until width / 2) {
                    val uvIndex = row * uvRowStride + col * uvPixelStride
                    if (uvOffset + 1 < nv21.size) {
                        vBuffer.position(uvIndex)
                        uBuffer.position(uvIndex)
                        nv21[uvOffset++] = vBuffer.get()
                        nv21[uvOffset++] = uBuffer.get()
                    }
                }
            }

            val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
            val out = ByteArrayOutputStream()
            yuvImage.compressToJpeg(Rect(0, 0, width, height), quality, out)
            return out.toByteArray()
        } catch (e: Exception) {
            Log.w("ArCoreTracker", "YUV to JPEG conversion failed: ${e.message}")
            return null
        }
    }
}
