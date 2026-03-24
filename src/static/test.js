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

  // 🎯 Canvas dùng cố định cho capture
  const captureCanvas = document.createElement('canvas');
  const captureCtx = captureCanvas.getContext('2d');

  // Khi video load metadata → đặt size cho canvas
  video.addEventListener('loadedmetadata', () => {
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;
  });

  // Mở camera
  navigator.mediaDevices.getUserMedia({ video: { width: 400, height: 300 } })
    .then(stream => {
      video.srcObject = stream;
      video.play();
    })
    .catch(err => {
      console.error("Lỗi khi mở camera:", err);
      alert(`Không thể mở camera. Vui lòng kiểm tra quyền truy cập và thử lại.\nChi tiết: ${err.message}`);
    });

  let lastBBox = null;
  let lastCropTime = 0;

  // Chống chồng request → tăng FPS
  let isBusy = false;

  async function doPredict() {
    if (isBusy) {
      requestAnimationFrame(doPredict);
      return;
    }
    isBusy = true;

    // Chụp frame vào canvas cố định
    captureCtx.drawImage(video, 0, 0);

    // Chuyển canvas sang Blob → nhỏ + nhanh
    captureCanvas.toBlob(async blob => {
      const formData = new FormData();
      formData.append("image", blob, "frame.jpg");

      try {
        const res = await fetch('/predict', {
          method: "POST",
          body: formData
        });

        const data = await res.json();

        if (data.success) {
          if (data.video) {
            const x = data.x;
            const y = data.y;
            const w = data.width;
            const h = data.height;

            // Chỉ vẽ lại khi bbox thay đổi → giảm load
            if (
              !lastBBox ||
              lastBBox.x !== x ||
              lastBBox.y !== y ||
              lastBBox.w !== w ||
              lastBBox.h !== h
            ) {
              overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
              overlayCtx.strokeStyle = "lime";
              overlayCtx.lineWidth = 2;
              overlayCtx.strokeRect(x, y, w, h);
            }

            lastBBox = { x, y, w, h };

            if (data.label === "Unknown") {
              labelResult.textContent = `Name: ${data.label}`;
            } else {
              labelResult.textContent = `Name: ${data.label} (Khoảng cách: ${data.distance.toFixed(4)})`;
            }
          }

          // Crop ảnh mỗi 2s
          const now = Date.now();
          if (now - lastCropTime > 2000) {
            croppedImage.src = data.crop;
            lastCropTime = now;
          }
        }
        else if (data.label === "No face") {
          labelResult.textContent = `${data.label}`;
          lastBBox = null;
        }
      } catch (err) {
        console.error("Lỗi khi gọi /predict:", err);
      }

      isBusy = false;
      requestAnimationFrame(doPredict);  // gọi predict tiếp
    }, "image/jpeg", 0.7); // giảm chất lượng nhẹ để tăng tốc
  }

  requestAnimationFrame(doPredict);

  // ==== Register giữ nguyên logic CỦA ANH ====
  registerBtn.addEventListener('click', () => {
    passwordInput.value = "";
    nameInput.value = "";
    passwordSection.style.display = "block";
    nameSection.classList.add("hidden");
    registerDialog.style.display = "block";
    passwordInput.focus();
  });

  cancelBtn.addEventListener('click', () => {
    registerDialog.style.display = "none";
    passwordInput.value = "";
    nameInput.value = "";
  });

  submitRegister.addEventListener('click', () => {
    if (nameSection.classList.contains("hidden")) {
      const password = passwordInput.value.trim();
      if (password === "1") {
        passwordSection.style.display = "none";
        nameSection.classList.remove("hidden");
        nameInput.focus();
      } else {
        alert("Sai mật khẩu!");
      }
      return;
    }

    const name = nameInput.value.trim();
    if (!name) {
      alert("Nhập tên trước khi đăng ký!");
      return;
    }

    registerDialog.style.display = "none";

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

        if (!lastBBox || lastBBox.w < minSize || lastBBox.h < minSize) {
          countdown = 3;
          labelRegister.textContent = `Kích thước mặt chưa đủ. Chờ ${countdown}...`;
          return;
        }

        countdown--;
        if (countdown > 0) {
          labelRegister.textContent = `Chụp ảnh sau ${countdown}...`;
        } else {
          clearInterval(timer);

          const regCanvas = document.createElement('canvas');
          regCanvas.width = video.videoWidth;
          regCanvas.height = video.videoHeight;
          const ctx = regCanvas.getContext('2d');
          ctx.drawImage(video, 0, 0);

          fetch('/register', {
            method: 'POST',
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              name: name,
              image: regCanvas.toDataURL('image/jpeg')
            })
          })
            .then(res => res.json())
            .then(data => {
              labelRegister.textContent = "Đã đăng ký: " + data.name;
              setTimeout(() => labelRegister.textContent = "", 2000);
            })
            .catch(err => console.error(err));
        }
      }, 1000);
    }

    startCountdown();
  });
});

