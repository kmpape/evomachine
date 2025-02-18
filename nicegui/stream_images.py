import cv2
import numpy as np
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/")
async def get():
    return HTMLResponse("""
   <!DOCTYPE html>
<html>
<head>
    <title>Image Stream</title>
</head>
<body>
    <h1>Image Stream</h1>
    <img id="image" width="800" />
    <script>
        const ws = new WebSocket("ws://localhost:8000/ws");
        ws.binaryType = 'arraybuffer';  // Set WebSocket to handle binary data

        ws.onmessage = async event => {
            const message = new Uint8Array(event.data);
            if (message[0] === 123) {  // Check if the message starts with '{' (ASCII 123), indicating JSON
                const jsonMessage = JSON.parse(new TextDecoder().decode(message));
                // Handle WebRTC signaling messages
            } else {
                // Handle binary image data
                const blob = new Blob([message], { type: 'image/jpeg' });
                const url = URL.createObjectURL(blob);
                const img = document.getElementById('image');
                img.src = url;
            }
        };
    </script>
</body>
</html>
    """)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        frame = (255 * np.random.rand(3200, 3200)).astype('uint8')
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 5])
        jpg_as_text = buffer.tobytes()
        await websocket.send_bytes(jpg_as_text)
        await asyncio.sleep(0.05)

# uvicorn stream_images:app --reload