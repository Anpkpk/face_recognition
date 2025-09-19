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

  // Bật camera
  navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
    .then(stream => {
      video.srcObject = stream;
      video.play();
    })
    .catch(err => {
      console.error("Lỗi khi mở camera:", err);
      alert(`Không thể mở camera. Vui lòng kiểm tra quyền truy cập và thử lại.\nChi tiết: ${err.message}`);
    });

  // Hàm gửi ảnh lên server (nhận diện)
  let lastCropTime = 0;   // thời gian lần cuối crop

  function doPredict() {
    const now = Date.now();
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    const ctx = canvas.getContext('2d');
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    const formData = new FormData();
    formData.append("image", canvas.toDataURL("image/jpeg"));

    fetch('/predict', {
      method: 'POST',
      body: formData
    })
      .then(res => res.json())
      .then(data => {
        if (data.success) {
          if (data.video) {
            const img = new Image();
            img.onload = () => {
              // vẽ lại frame đầy đủ
              ctx.clearRect(0, 0, canvas.width, canvas.height);
              ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

              // vẽ bounding box
              ctx.strokeStyle = "lime";
              ctx.lineWidth = 2;
              ctx.strokeRect(data.x, data.y, data.width, data.height);
            };
            img.src = data.video;
            labelResult.textContent = `Name: ${data.label} (Khoảng cách: ${data.distance.toFixed(4)})`;
          }

          // crop chỉ update mỗi 3 giây
          if (now - lastCropTime > 3000) {
            croppedImage.src = data.crop;
            lastCropTime = now;
          }
        } else {
          console.warn("Detect thất bại:", data.message);
        }
      })
      .catch(err => console.error("Lỗi khi gọi /predict:", err))
      .finally(() => {
        requestAnimationFrame(doPredict);
      });
  }





  // Auto predict mỗi 2s
  setInterval(doPredict, 2000);

  // Nút Register
  registerBtn.addEventListener('click', () => {
    // reset dialog
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
      })
      .catch(err => console.error(err));
  });
});
