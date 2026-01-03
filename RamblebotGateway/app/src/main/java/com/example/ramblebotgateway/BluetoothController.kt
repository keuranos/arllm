package com.example.ramblebotgateway

import android.annotation.SuppressLint
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothSocket
import java.io.OutputStream
import java.util.UUID

class BluetoothController(private val mac: String) {

    private val adapter: BluetoothAdapter = BluetoothAdapter.getDefaultAdapter()
        ?: throw IllegalStateException("Bluetooth not supported")

    private val sppUuid: UUID =
        UUID.fromString("00001101-0000-1000-8000-00805F9B34FB")

    private var socket: BluetoothSocket? = null
    private var out: OutputStream? = null

    @SuppressLint("MissingPermission")
    fun connect() {
        if (!adapter.isEnabled) throw IllegalStateException("Bluetooth OFF")
        val device: BluetoothDevice = adapter.getRemoteDevice(mac)

        try { adapter.cancelDiscovery() } catch (_: Exception) {}
        try { socket?.close() } catch (_: Exception) {}

        try {
            socket = device.createRfcommSocketToServiceRecord(sppUuid)
            socket!!.connect()
        } catch (_: Exception) {
            val m = device.javaClass.getMethod("createRfcommSocket", Int::class.javaPrimitiveType)
            socket = m.invoke(device, 1) as BluetoothSocket
            socket!!.connect()
        }

        out = socket!!.outputStream
    }

    @Synchronized
    fun sendLine(line: String) {
        val o = out ?: throw IllegalStateException("Not connected")
        o.write((line + "\n").toByteArray(Charsets.UTF_8))
        o.flush()
    }

    fun close() {
        try { out?.close() } catch (_: Exception) {}
        try { socket?.close() } catch (_: Exception) {}
        out = null
        socket = null
    }
}
