package com.example.ramblebotgateway

import java.io.OutputStream
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.atomic.AtomicReference

object MjpegHub {
    private val latestJpeg = AtomicReference<ByteArray?>(null)
    private val clients = CopyOnWriteArrayList<OutputStream>()

    fun updateFrame(jpeg: ByteArray) {
        latestJpeg.set(jpeg)
    }

    fun addClient(os: OutputStream) {
        clients.add(os)
    }

    fun removeClient(os: OutputStream) {
        clients.remove(os)
        try { os.close() } catch (_: Exception) {}
    }

    fun startWriterThread(os: OutputStream) {
        addClient(os)

        Thread {
            val boundary = "--frame"
            try {
                // HTTP multipart stream header is handled by NanoHTTPD Response header;
                // here we just write multipart parts repeatedly.
                while (true) {
                    val frame = latestJpeg.get()
                    if (frame == null) {
                        Thread.sleep(50)
                        continue
                    }

                    val header =
                        "$boundary\r\n" +
                                "Content-Type: image/jpeg\r\n" +
                                "Content-Length: ${frame.size}\r\n\r\n"

                    os.write(header.toByteArray())
                    os.write(frame)
                    os.write("\r\n".toByteArray())
                    os.flush()

                    // ~10 fps default
                    Thread.sleep(100)
                }
            } catch (_: Exception) {
                // client disconnected
            } finally {
                removeClient(os)
            }
        }.start()
    }
}
