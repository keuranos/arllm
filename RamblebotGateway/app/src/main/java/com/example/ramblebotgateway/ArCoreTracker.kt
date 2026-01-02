package com.example.ramblebotgateway

import android.content.Context
import android.graphics.SurfaceTexture
import android.os.Build
import android.util.Log
import com.google.ar.core.Config
import com.google.ar.core.Frame
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import java.util.concurrent.atomic.AtomicBoolean

data class ArCorePose(
    val trackingState: String,
    val timestampNs: Long,
    val position: List<Float>,
    val rotation: List<Float>
)

class ArCoreTracker(private val context: Context) {
    private var session: Session? = null
    private var surfaceTexture: SurfaceTexture? = null
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
            surfaceTexture = SurfaceTexture(0).also {
                session?.setCameraTextureName(it)
            }
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
        try { surfaceTexture?.release() } catch (_: Exception) {}
        surfaceTexture = null
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
    }
}
