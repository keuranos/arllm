package com.example.ramblebotgateway

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap

data class SensorSample(
    val values: FloatArray,
    val accuracy: Int,
    val eventTimestampNs: Long,
    val updatedAtMs: Long
)

class SensorHub(context: Context) : SensorEventListener {
    private val sensorManager =
        context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val sensors: List<Sensor> = sensorManager.getSensorList(Sensor.TYPE_ALL)
    private val samples = ConcurrentHashMap<String, SensorSample>()

    fun start() {
        sensors.forEach { sensor ->
            sensorManager.registerListener(this, sensor, SensorManager.SENSOR_DELAY_UI)
        }
    }

    fun stop() {
        sensorManager.unregisterListener(this)
    }

    fun toJson(): String {
        val now = System.currentTimeMillis()
        val sb = StringBuilder()
        sb.append("{")
        sb.append("\"deviceTimeMs\":").append(now).append(',')
        sb.append("\"sensorCount\":").append(sensors.size).append(',')
        sb.append("\"sensors\":[")
        sensors.forEachIndexed { idx, sensor ->
            if (idx > 0) sb.append(',')
            sb.append(sensorToJson(sensor))
        }
        sb.append("]}")
        return sb.toString()
    }

    override fun onSensorChanged(event: SensorEvent) {
        val key = sensorKey(event.sensor)
        samples[key] = SensorSample(
            event.values.clone(),
            event.accuracy,
            event.timestamp,
            System.currentTimeMillis()
        )
    }

    override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {
        val key = sensorKey(sensor)
        val existing = samples[key]
        if (existing != null) {
            samples[key] = existing.copy(accuracy = accuracy)
        }
    }

    private fun sensorToJson(sensor: Sensor): String {
        val key = sensorKey(sensor)
        val sample = samples[key]
        val sb = StringBuilder()
        sb.append("{")
        sb.append("\"key\":\"").append(jsonEscape(key)).append("\",")
        sb.append("\"name\":\"").append(jsonEscape(sensor.name)).append("\",")
        sb.append("\"type\":").append(sensor.type).append(',')
        sb.append("\"stringType\":\"").append(jsonEscape(sensor.stringType)).append("\",")
        sb.append("\"vendor\":\"").append(jsonEscape(sensor.vendor)).append("\",")
        sb.append("\"version\":").append(sensor.version).append(',')
        sb.append("\"maxRange\":").append(sensor.maximumRange).append(',')
        sb.append("\"resolution\":").append(sensor.resolution).append(',')
        sb.append("\"power\":").append(sensor.power).append(',')
        sb.append("\"minDelay\":").append(sensor.minDelay).append(',')
        sb.append("\"maxDelay\":").append(sensor.maxDelay).append(',')
        sb.append("\"reportingMode\":\"").append(reportingMode(sensor.reportingMode)).append("\",")
        sb.append("\"isWakeUp\":").append(sensor.isWakeUpSensor)
        if (sample != null) {
            sb.append(',')
            sb.append("\"hasReading\":true,")
            sb.append("\"accuracy\":").append(sample.accuracy).append(',')
            sb.append("\"eventTimestampNs\":").append(sample.eventTimestampNs).append(',')
            sb.append("\"updatedAtMs\":").append(sample.updatedAtMs).append(',')
            sb.append("\"values\":[")
            sample.values.forEachIndexed { idx, v ->
                if (idx > 0) sb.append(',')
                sb.append(String.format(Locale.US, "%.5f", v))
            }
            sb.append("]")
        } else {
            sb.append(",\"hasReading\":false")
        }
        sb.append("}")
        return sb.toString()
    }

    private fun reportingMode(mode: Int): String {
        return when (mode) {
            Sensor.REPORTING_MODE_CONTINUOUS -> "continuous"
            Sensor.REPORTING_MODE_ON_CHANGE -> "on_change"
            Sensor.REPORTING_MODE_ONE_SHOT -> "one_shot"
            Sensor.REPORTING_MODE_SPECIAL_TRIGGER -> "special_trigger"
            else -> "unknown"
        }
    }

    private fun sensorKey(sensor: Sensor): String {
        return "${sensor.type}:${sensor.name}"
    }

    private fun jsonEscape(value: String): String {
        return value
            .replace("\\", "\\\\")
            .replace("\"", "\\\"")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
    }
}
