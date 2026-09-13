const input = document.getElementById("images");
const fileCount = document.getElementById("file-count");
const video = document.getElementById("camera-preview");
const snapshot = document.getElementById("camera-snapshot");
const placeholder = document.getElementById("camera-placeholder");
const canvas = document.getElementById("camera-canvas");
const hiddenCameraInput = document.getElementById("camera_image");
const startButton = document.getElementById("start-camera");
const captureButton = document.getElementById("capture-frame");
const retakeButton = document.getElementById("retake-frame");
const cameraStatus = document.getElementById("camera-status");

let cameraStream = null;

if (input && fileCount) {
    input.addEventListener("change", () => {
        const count = input.files ? input.files.length : 0;
        if (count === 0) {
            fileCount.textContent = "No files selected";
            return;
        }
        if (count === 1) {
            fileCount.textContent = input.files[0].name;
            return;
        }
        fileCount.textContent = `${count} files selected`;
    });
}

async function startCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        cameraStatus.textContent = "Live camera is not supported in this browser.";
        return;
    }

    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: { ideal: "environment" } },
            audio: false,
        });
        video.srcObject = cameraStream;
        video.style.display = "block";
        snapshot.style.display = "none";
        placeholder.style.display = "none";
        hiddenCameraInput.value = "";
        captureButton.disabled = false;
        retakeButton.disabled = true;
        cameraStatus.textContent = "Camera is live. Capture a frame when the sign is clear.";
    } catch (error) {
        cameraStatus.textContent = "Could not access the camera. Please allow camera permission or use image upload.";
    }
}

function stopCameraTracks() {
    if (!cameraStream) {
        return;
    }
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
}

function captureFrame() {
    if (!video || !canvas || !hiddenCameraInput) {
        return;
    }
    const width = video.videoWidth;
    const height = video.videoHeight;
    if (!width || !height) {
        cameraStatus.textContent = "Camera frame is not ready yet. Please wait a moment and try again.";
        return;
    }

    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, width, height);

    const dataUrl = canvas.toDataURL("image/png");
    hiddenCameraInput.value = dataUrl;
    snapshot.src = dataUrl;
    snapshot.style.display = "block";
    video.style.display = "none";
    captureButton.disabled = true;
    retakeButton.disabled = false;
    cameraStatus.textContent = "Frame captured. You can analyze now or retake a clearer image.";
    stopCameraTracks();
}

async function retakeFrame() {
    hiddenCameraInput.value = "";
    snapshot.removeAttribute("src");
    snapshot.style.display = "none";
    placeholder.style.display = "none";
    await startCamera();
}

if (startButton) {
    startButton.addEventListener("click", startCamera);
}

if (captureButton) {
    captureButton.addEventListener("click", captureFrame);
}

if (retakeButton) {
    retakeButton.addEventListener("click", retakeFrame);
}

function formatPredictionList(predictions) {
    return predictions.map((item, index) => `${index + 1}. ${item.localized_label} — ${item.score}%`).join("\n");
}

function downloadResultPdf(event) {
    const button = event.currentTarget;
    const image = button.dataset.image;
    const filename = button.dataset.filename || "prediction";
    const label = button.dataset.label || "Prediction";
    const confidence = button.dataset.confidence || "0";
    const time = button.dataset.time || "";
    const topPredictions = JSON.parse(button.dataset.topPredictions || "[]");

    const doc = new window.jspdf.jsPDF({ orientation: "portrait", unit: "pt", format: "a4" });
    const margin = 40;
    let y = margin;

    doc.setFontSize(18);
    doc.text("Traffic Sign Prediction", margin, y);
    y += 28;

    doc.setFontSize(12);
    doc.text(`File: ${filename}`, margin, y);
    y += 18;
    doc.text(`Prediction: ${label}`, margin, y);
    y += 18;
    doc.text(`Confidence: ${confidence}%`, margin, y);
    y += 18;
    doc.text(`Time: ${time}`, margin, y);
    y += 24;

    if (image) {
        try {
            const maxWidth = doc.internal.pageSize.getWidth() - margin * 2;
            const imageProps = doc.getImageProperties(image);
            const imageHeight = (imageProps.height * maxWidth) / imageProps.width;
            doc.addImage(image, "PNG", margin, y, maxWidth, imageHeight);
            y += imageHeight + 20;
        } catch (error) {
            console.error("PDF image error:", error);
        }
    }

    doc.setFontSize(14);
    doc.text("Top 5 Predictions:", margin, y);
    y += 18;
    doc.setFontSize(12);

    formatPredictionList(topPredictions).split("\n").forEach((line) => {
        const splitText = doc.splitTextToSize(line, doc.internal.pageSize.getWidth() - margin * 2);
        doc.text(splitText, margin, y);
        y += splitText.length * 14;
        if (y > doc.internal.pageSize.getHeight() - margin) {
            doc.addPage();
            y = margin;
        }
    });

    const safeName = filename.replace(/[^a-zA-Z0-9-_\.]/g, "_");
    const sanitizedTime = time.replace(/[: ]/g, "_");
    doc.save(`${safeName}_${sanitizedTime || Date.now()}.pdf`);
}

document.body.addEventListener("click", (event) => {
    const button = event.target.closest(".download-pdf-btn");
    if (!button) {
        return;
    }
    event.preventDefault();
    downloadResultPdf({ currentTarget: button });
});

window.addEventListener("beforeunload", stopCameraTracks);
