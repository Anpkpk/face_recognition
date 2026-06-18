window.addEventListener('DOMContentLoaded', () => {
  const video = document.getElementById('video');
  const croppedImage = document.getElementById('croppedImage');
  const labelResult = document.getElementById('labelResult');
  const labelRegister = document.getElementById('labelRegister');
  const registerBtn = document.getElementById('registerButton');
  const registerDialog = document.getElementById('registerDialog');
  const nameInput = document.getElementById('nameInput');
  const submitRegister = document.getElementById('okBtn');
  const passwordInput = document.getElementById('passwordInput');
  const passwordSection = document.getElementById('passwordSection');
  const nameSection = document.getElementById('nameSection');
  const cancelBtn = document.getElementById('cancelBtn');

  const overlay = document.getElementById('overlay');
  const overlayCtx = overlay.getContext('2d');

  let currentName = "Đang nhận diện...";
  let lastRecognizeTime = 0;
  let lastBBox = null; 
  let currentFaceImage = null; 

  // --- CÁC BIẾN LƯU TRỮ THÔNG SỐ HIỆU SUẤT ---
  // 1. Client FPS (Camera)
  let frameCount = 0;
  let lastFpsTime = performance.now();
  let clientFps = 0;
  
  // 2. Server Stats (Từ Flask API)
  let serverRam = 0;
  let serverFps = 0;
  let serverLatency = 0;
  // -------------------------------------------

  // Khởi tạo MediaPipe Face Detection
  const faceDetection = new FaceDetection({
    locateFile: (file) => {
      return `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`;
    }
  });

  faceDetection.setOptions({
    model: 'short',
    minDetectionConfidence: 0.5
  });

  // Hàm này tự động chạy mỗi khi Camera có khung hình mới (~30FPS)
  faceDetection.onResults((results) => {
    // --- TÍNH TOÁN CLIENT FPS ---
    frameCount++;
    const nowTime = performance.now();
    if (nowTime - lastFpsTime >= 1000) { 
      clientFps = frameCount;
      frameCount = 0;
      lastFpsTime = nowTime;
    }

    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

    // --- VẼ THÔNG SỐ LÊN MÀN HÌNH (GÓC TRÊN TRÁI) ---
    // Vẽ nền mờ để dễ đọc chữ hơn
    overlayCtx.fillStyle = "rgba(0, 0, 0, 0.5)";
    overlayCtx.fillRect(5, 5, 200, 110);

    // In Client FPS
    overlayCtx.fillStyle = "yellow"; 
    overlayCtx.font = "bold 16px Arial";
    overlayCtx.fillText(`Client FPS: ${clientFps}`, 15, 25);
    
    // In Server Stats (Nếu đã nhận được dữ liệu)
    if (serverLatency > 0) {
      overlayCtx.fillStyle = "#00FFFF"; // Màu Cyan
      overlayCtx.fillText(`Server FPS: ${serverFps}`, 15, 50);
      overlayCtx.fillStyle = "#00FF00"; // Màu Xanh lá
      overlayCtx.fillText(`Server RAM: ${serverRam} MB`, 15, 75);
      overlayCtx.fillStyle = "#FF9900"; // Màu Cam
      overlayCtx.fillText(`Độ trễ: ${serverLatency} ms`, 15, 100);
    }
    // -----------------------------------------------

    if (results.detections.length > 0) {
      const detection = results.detections[0];
      const bbox = detection.boundingBox;

      // Tính tọa độ pixel thực tế
      const x = bbox.xCenter * overlay.width - (bbox.width * overlay.width) / 2;
      const y = bbox.yCenter * overlay.height - (bbox.height * overlay.height) / 2;
      const w = bbox.width * overlay.width;
      const h = bbox.height * overlay.height;

      // Cập nhật lastBBox để dùng cho chức năng Register
      lastBBox = { width: w, height: h, x: x, y: y };

      // Vẽ khung xanh cực mượt
      overlayCtx.strokeStyle = "lime";
      overlayCtx.lineWidth = 3;
      overlayCtx.strokeRect(x, y, w, h);

      // Cắt mặt và lưu vào currentFaceImage
      const cropCanvas = document.createElement('canvas');
      cropCanvas.width = w;
      cropCanvas.height = h;
      const cropCtx = cropCanvas.getContext('2d');
      cropCtx.drawImage(results.image, x, y, w, h, 0, 0, w, h);
      currentFaceImage = cropCanvas.toDataURL("image/jpeg", 0.8); // Nén ảnh crop

      // Gọi server để nhận diện tên (1 giây/lần để tránh spam server)
      const now = Date.now();
      if (now - lastRecognizeTime > 1000) {
        lastRecognizeTime = now;
        
        // Hiển thị ảnh crop sang thẻ img
        if (croppedImage) croppedImage.src = currentFaceImage;

        doPredict(currentFaceImage);
      }
    } else {
      // Không thấy mặt
      lastBBox = null;
      currentFaceImage = null;
      currentName = "No face";
      labelResult.textContent = currentName;
    }
  });

  // Hàm gọi API /predict nhận diện
  function doPredict(base64Face) {
    const formData = new FormData();
    formData.append("image", base64Face);

    fetch('/predict', {
      method: 'POST',
      body: formData
    })
      .then(res => res.json())
      .then(data => {
        if (data.success) {
          // 1. Cập nhật tên và độ chính xác
          if (data.label === "Unknown") {
            currentName = "Unknown";
            labelResult.textContent = "Name: Unknown";
          } else {
            currentName = `${data.label} (${data.distance.toFixed(4)})`;
            labelResult.textContent = `Name: ${data.label} (Khoảng cách: ${data.distance.toFixed(4)})`;
          }

          // 2. Cập nhật thông số Server để hàm onResults vẽ lên màn hình
          if (data.stats) {
            serverRam = data.stats.backend_ram_mb;
            serverFps = data.stats.backend_fps;
            serverLatency = data.stats.inference_time_ms;
          }
        }
      })
      .catch(err => console.error("Lỗi khi gọi /predict:", err));
  }

  // Khởi động Camera qua MediaPipe
  const camera = new Camera(video, {
    onFrame: async () => {
      // Mỗi khi có frame mới, resize canvas overlay cho khớp và đưa frame vào MediaPipe
      if (overlay.width !== video.videoWidth) {
        overlay.width = video.videoWidth;
        overlay.height = video.videoHeight;
      }
      await faceDetection.send({ image: video });
    },
    width: 640,
    height: 480
  });
  camera.start().catch(err => {
    console.error("Lỗi khi mở camera:", err);
    alert(`Không thể mở camera. Vui lòng kiểm tra quyền truy cập và thử lại.\nChi tiết: ${err.message}`);
  });

  // ================= PHẦN REGISTER GIỮ NGUYÊN LOGIC GIAO DIỆN =================

  // Nút Register
  registerBtn.addEventListener('click', () => {
    passwordInput.value = "";
    nameInput.value = "";
    passwordSection.style.display = "block";
    nameSection.classList.add("hidden");

    registerDialog.style.display = "block";
    passwordInput.focus();
  });

  // Nút Cancel
  cancelBtn.addEventListener('click', () => {
    registerDialog.style.display = "none";
    passwordInput.value = "";
    nameInput.value = "";
  });

  // Nút OK (hai bước)
  submitRegister.addEventListener('click', () => {
    // Nếu đang ở bước mật khẩu
    if (nameSection.classList.contains("hidden")) {
      const password = passwordInput.value.trim();
      if (password === "1") {
        passwordSection.style.display = "none";
        nameSection.classList.remove("hidden");
        passwordSection.style.display = "none";
        nameInput.focus();
      } else {
        alert("Sai mật khẩu!");
      }
      return;
    }

    // Nếu đang ở bước nhập tên
    const name = nameInput.value.trim();
    if (!name) {
      alert("Nhập tên trước khi đăng ký!");
      return;
    }
    
    registerDialog.style.display = "none";

    // Reset fields
    passwordInput.value = "";
    nameInput.value = "";

    let countdown = 3;
    let timer;

    function startCountdown() {
      clearInterval(timer);
      countdown = 3;
      labelRegister.textContent = `Chụp ảnh sau ${countdown}...`;

      timer = setInterval(() => {
        const minSize = 136;

        if (!lastBBox || lastBBox.width < minSize || lastBBox.height < minSize) {
          // Nếu bbox không đủ lớn thì reset countdown
          countdown = 3;
          labelRegister.textContent = `Kích thước mặt chưa đủ. Chờ ${countdown}...`;
          return;
        }

        countdown--;
        if (countdown > 0) {
          labelRegister.textContent = `Chụp ảnh sau ${countdown}...`;
        } else {
          clearInterval(timer);

          // Chụp ảnh khi đủ điều kiện
          const canvas = document.createElement('canvas');
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          const ctx = canvas.getContext('2d');
          ctx.drawImage(video, 0, 0);

          fetch('/register', {
            method: 'POST',
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              name: name,
              image: canvas.toDataURL('image/jpeg')
            })
          })
            .then(res => res.json())
            .then(data => {
              labelRegister.textContent = "Đã đăng ký: " + data.name;
              registerDialog.style.display = "none";
              
              nameInput.value = "";
              passwordInput.value = "";

              setTimeout(() => {
                labelRegister.textContent = "";
                labelRegister.style.display = "none";
              }, 2000);
            })
            .catch(err => console.error(err));
        }
      }, 1000);
    }

    startCountdown();
  });
});
