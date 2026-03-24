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

  video.addEventListener('loadedmetadata', () => {
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
  });

  // Bật camera
  navigator.mediaDevices.getUserMedia({ video: { width: 400, height: 300 } })
    .then(stream => {
      video.srcObject = stream;
      video.play();
    })
    .catch(err => {
      console.error("Lỗi khi mở camera:", err);
      alert(`Không thể mở camera. Vui lòng kiểm tra quyền truy cập và thử lại.\nChi tiết: ${err.message}`);
    });

  // Biến lưu bbox mới nhất từ predict
  // ... (Phần setup camera giữ nguyên)

  let lastBBox = null;
  let lastCropTime = 0;
  let isPredicting = false; // Cờ kiểm tra xem có đang gửi request không

  function doPredict() {
    if (isPredicting) return;
    isPredicting = true;

    const now = Date.now();
    const canvas = document.createElement('canvas');
    // Mẹo: Bạn có thể scale nhỏ canvas width/height ở đây (ví dụ 320x240) 
    // để model AI chạy nhanh hơn nếu AI của bạn hỗ trợ ảnh nhỏ.
    canvas.width = video.videoWidth; 
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);

    const formData = new FormData();
    // Nén JPEG xuống 70% chất lượng để gửi qua HTTP siêu nhanh
    formData.append("image", canvas.toDataURL("image/jpeg", 0.7)); 

    // Kiểm tra xem đã đến lúc cần lấy ảnh crop chưa (2s một lần)
    const needCrop = (now - lastCropTime > 2000);
    formData.append("need_crop", needCrop);

    fetch('/predict', {
      method: 'POST',
      body: formData
    })
      .then(res => res.json())
      .then(data => {
        if (data.success) {
          // 1. Chỉ vẽ khung (không cần load lại ảnh từ server)
          const x = data.x;
          const y = data.y;
          const w = data.width;
          const h = data.height;

          lastBBox = { width: w, height: h };
          overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

          overlayCtx.lineWidth = 2; // Tăng độ dày cho dễ nhìn
          overlayCtx.strokeStyle = "lime";
          overlayCtx.strokeRect(x, y, w, h);

          // 2. Cập nhật label
          if (data.label === "Unknown") {
            labelResult.textContent = `Name: ${data.label}`;
          } else {
            labelResult.textContent = `Name: ${data.label} (Khoảng cách: ${data.distance.toFixed(4)})`;
          }

          // 3. Cập nhật ảnh crop nếu server có trả về
          if (needCrop && data.crop) {
            croppedImage.src = data.crop;
            lastCropTime = Date.now(); // Reset lại thời gian
          }
        } 
        else if (data.label === "No face") {
          overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
          labelResult.textContent = `${data.label}`;
        }
      })
      .catch(err => console.error("Lỗi khi gọi /predict:", err))
      .finally(() => {
        isPredicting = false;
        // Chỉ gửi request tiếp theo sau khi request này đã hoàn thành
        // Đợi thêm khoảng 100ms - 200ms để giảm tải cho CPU
        setTimeout(doPredict, 150); 
      });
  }

  // Bắt đầu vòng lặp thay vì dùng setInterval
  doPredict();

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
