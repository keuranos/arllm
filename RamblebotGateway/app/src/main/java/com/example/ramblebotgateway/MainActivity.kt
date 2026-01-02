package com.example.ramblebotgateway

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Build
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import fi.iki.elonen.NanoHTTPD
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {

    private var bt: BluetoothController? = null
    private var server: GatewayServer? = null
    private lateinit var sensors: SensorHub
    private lateinit var arCore: ArCoreTracker

    private lateinit var status: TextView
    private lateinit var macEdit: EditText
    private lateinit var connectBtn: Button

    private val REQ_BT = 1001
    private val REQ_CAM = 1002
    private val REQ_ACTIVITY = 1003

    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private val useArCore = true
    private var arCoreEnabled = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        status = findViewById(R.id.statusText)
        macEdit = findViewById(R.id.macEdit)
        connectBtn = findViewById(R.id.connectBtn)

        sensors = SensorHub(this).also { it.start() }
        arCore = ArCoreTracker(this)
        arCoreEnabled = useArCore && arCore.start()

        ensureActivityRecognition()

        // Start camera if permitted (MJPEG feed)
        if (!arCoreEnabled) {
            ensureCamera()
        }

        connectBtn.setOnClickListener {
            if (!hasBtPermissions()) {
                requestBtPermissions()
                status.text = "Requesting Bluetooth permissions…"
                return@setOnClickListener
            }

            val mac = macEdit.text.toString().trim()
            if (!isMacLike(mac)) {
                status.text = "Invalid MAC. Example: AA:BB:CC:DD:EE:FF"
                return@setOnClickListener
            }

            status.text = "Connecting…"

            Thread {
                try {
                    bt?.close()
                    bt = BluetoothController(mac)
                    bt!!.connect()

                    if (server == null) {
                        server = GatewayServer(bt!!, sensors, arCore, WebUi.html())
                        server!!.start(NanoHTTPD.SOCKET_READ_TIMEOUT, false)
                    }

                    runOnUiThread {
                        status.text = "Connected ✅  Web: http://PHONE_IP:8765/"
                    }
                } catch (e: Exception) {
                    runOnUiThread {
                        status.text = "Error: ${e.javaClass.simpleName}: ${e.message}"
                    }
                }
            }.start()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        try { server?.stop() } catch (_: Exception) {}
        try { bt?.close() } catch (_: Exception) {}
        sensors.stop()
        arCore.stop()
        cameraExecutor.shutdown()
    }

    private fun isMacLike(s: String): Boolean =
        Regex("^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$").matches(s)

    private fun hasBtPermissions(): Boolean {
        if (Build.VERSION.SDK_INT < 31) return true
        return checkSelfPermission(Manifest.permission.BLUETOOTH_CONNECT) == PackageManager.PERMISSION_GRANTED &&
                checkSelfPermission(Manifest.permission.BLUETOOTH_SCAN) == PackageManager.PERMISSION_GRANTED
    }

    private fun requestBtPermissions() {
        if (Build.VERSION.SDK_INT >= 31) {
            requestPermissions(
                arrayOf(
                    Manifest.permission.BLUETOOTH_CONNECT,
                    Manifest.permission.BLUETOOTH_SCAN
                ),
                REQ_BT
            )
        }
    }

    private fun ensureActivityRecognition() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return
        val granted = checkSelfPermission(Manifest.permission.ACTIVITY_RECOGNITION) ==
            PackageManager.PERMISSION_GRANTED
        if (!granted) {
            requestPermissions(arrayOf(Manifest.permission.ACTIVITY_RECOGNITION), REQ_ACTIVITY)
        }
    }

    private fun ensureCamera() {
        val granted = checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        if (!granted) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), REQ_CAM)
            return
        }
        startCameraMjpeg()
    }

    private fun startCameraMjpeg() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val provider = cameraProviderFuture.get()

            val selector = CameraSelector.DEFAULT_BACK_CAMERA

            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            analysis.setAnalyzer(cameraExecutor) { image ->
                try {
                    val jpeg = imageProxyToJpeg(image, 60)
                    if (jpeg != null) MjpegHub.updateFrame(jpeg)
                } catch (_: Exception) {
                } finally {
                    image.close()
                }
            }

            val preview = Preview.Builder().build()
            // We do not need to display preview on screen for MJPEG streaming.

            provider.unbindAll()
            provider.bindToLifecycle(this, selector, preview, analysis)

        }, ContextCompat.getMainExecutor(this))
    }

    // YUV_420_888 -> JPEG
    private fun imageProxyToJpeg(image: ImageProxy, quality: Int): ByteArray? {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        // Y
        yBuffer.get(nv21, 0, ySize)

        // VU (NV21 expects V then U)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), quality, out)
        return out.toByteArray()
    }
}
