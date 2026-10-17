const localVideo = document.getElementById('localVideo');
const remoteVideo = document.getElementById('remoteVideo');
const captureCanvas = document.getElementById('captureCanvas');
const captureCtx = captureCanvas.getContext('2d');

const snrSlider = document.getElementById('snrSlider');
const snrValue = document.getElementById('snrValue');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const wsStatusDot = document.getElementById('wsStatusDot');
const wsStatusText = document.getElementById('wsStatusText');
const noiseOverlay = document.getElementById('noiseOverlay');

let stream;
let ws;
let isStreaming = false;

// We match the PyTorch JSCC input requirement
captureCanvas.width = 128;
captureCanvas.height = 128;

snrSlider.addEventListener('input', (e) => {
    snrValue.textContent = parseFloat(e.target.value).toFixed(1);
    
    // Add a slight static overlay if SNR drops extremely low
    const snr = parseFloat(e.target.value);
    if (snr < 3) {
        noiseOverlay.style.opacity = '0.4';
    } else if (snr < 8) {
        noiseOverlay.style.opacity = '0.15';
    } else {
        noiseOverlay.style.opacity = '0';
    }
});

async function startWebcam() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                aspectRatio: 1,
                width: { ideal: 480 },
                height: { ideal: 480 }
            } 
        });
        localVideo.srcObject = stream;
    } catch (err) {
        console.error("Error accessing webcam: ", err);
        alert("Please allow webcam access in your browser to test the telemetry feed.");
    }
}

function stopWebcam() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    localVideo.srcObject = null;
}

function connectWebSocket() {
    // Connect to the FastAPI JSCC Server dynamically based on the current host IP
    const protocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
    ws = new WebSocket(protocol + window.location.hostname + ':8000/ws/stream');
    
    ws.onopen = () => {
        wsStatusDot.classList.add('connected');
        wsStatusText.textContent = 'Connected to JSCC Channel';
        isStreaming = true;
        sendFrame(); // Kick off the asynchronous streaming loop
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.image_reconstructed) {
            remoteVideo.src = data.image_reconstructed;
        }
        
        // Loop continuously to send the next frame as soon as the last one arrives
        if (isStreaming) {
            requestAnimationFrame(sendFrame);
        }
    };
    
    ws.onclose = () => {
        wsStatusDot.classList.remove('connected');
        wsStatusText.textContent = 'Disconnected';
        isStreaming = false;
    };
}

const channelSelect = document.getElementById('channelSelect');

function sendFrame() {
    if (!isStreaming || ws.readyState !== WebSocket.OPEN) return;
    
    // Draw current video frame to canvas at 128x128
    captureCtx.drawImage(localVideo, 0, 0, captureCanvas.width, captureCanvas.height);
    
    // Convert to JPEG base64 string
    const base64Image = captureCanvas.toDataURL('image/jpeg', 0.8);
    
    // Send to backend with selected channel model and SNR
    ws.send(JSON.stringify({
        image: base64Image,
        snr: snrSlider.value,
        channel: channelSelect ? channelSelect.value : 'rician'
    }));
}

startBtn.addEventListener('click', async () => {
    startBtn.disabled = true;
    await startWebcam();
    connectWebSocket();
    stopBtn.disabled = false;
});

stopBtn.addEventListener('click', () => {
    stopBtn.disabled = true;
    isStreaming = false;
    if (ws) ws.close();
    stopWebcam();
    startBtn.disabled = false;
    remoteVideo.src = "";
    noiseOverlay.style.opacity = '0';
});
