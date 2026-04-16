# Age & Gender Prediction using Multi-task Learning

Dự án Computer Vision ứng dụng học sâu (Deep Learning) để dự đoán đồng thời Tuổi và Giới tính của người trong ảnh. Mô hình được xây dựng dựa trên kiến trúc Multi-task Learning với PyTorch, giúp tối ưu hóa tài nguyên và tăng độ chính xác nhờ việc chia sẻ đặc trưng khuôn mặt.

## 🧠 Kiến trúc Mô hình (Multi-task CNN)
Thay vì huấn luyện hai mô hình riêng biệt, dự án sử dụng cấu trúc rẽ nhánh (Y-shape):
* **Backbone:** ResNet-18 (được loại bỏ lớp Fully Connected cuối cùng) đóng vai trò trích xuất đặc trưng khuôn mặt (Shared Features).
* **Gender Head (Phân loại):** Nhánh dự đoán giới tính sử dụng hàm loss `BCEWithLogitsLoss`. Đầu ra 1 node giải mã bằng hàm Sigmoid với quy ước nhãn thực tế: `1 = Nam`, `0 = Nữ`.
* **Age Head (Hồi quy):** Nhánh dự đoán tuổi sử dụng hàm loss `MSELoss` để dự đoán trực tiếp một con số tuổi liên tục. Đầu ra 1 node.

## 📊 Dữ liệu (Dataset)
Sử dụng bộ dữ liệu [Krishnancool/age-gender-prediction](https://huggingface.co/datasets/Krishnancool/age-gender-prediction) từ Hugging Face.
* **Tiền xử lý:** Tự động hóa Pipeline với `Custom Dataset` và `DataLoader`.
* **Augmentation:** Xử lý ảnh về chuẩn `224x224`, áp dụng RandomHorizontalFlip, ColorJitter và chuẩn hóa (Normalize) theo ImageNet để tăng tính robust cho mô hình.
* **Debug Kỹ thuật:** Đã xử lý triệt để lỗi "Flipped Label" (ngược nhãn) thường gặp trong các bộ dữ liệu do cộng đồng đóng góp.

## 🚀 Cài đặt và Sử dụng

**1. Cài đặt thư viện**
```bash
pip install torch torchvision datasets pillow tqdm
```
**2. Huấn luyện mô hình (Training) **

Mô hình được tối ưu hóa để huấn luyện trên GPU CUDA.

```bash
python train.py
```

**3. Dự đoán trên ảnh thực tế (Evaluation)**

Script test được tích hợp tính năng tự động so sánh Ground Truth (nhãn thực tế người dùng nhập) và tính toán hàm Loss trực tiếp.

Chuẩn bị một bức ảnh chân dung rõ mặt, đổi tên thành test.jpg và đặt cùng thư mục.

```bash
python evaluate.py
```
##
## 📈 Kết quả Huấn luyện (Training & Validation Results)

Thay vì huấn luyện cố định toàn bộ 15 Epochs, quá trình Train đã được tích hợp thuật toán **Early Stopping** (với `patience=3`) và tự động dừng ở **Epoch 10** nhằm ngăn chặn triệt để hiện tượng Overfitting. 

Các kết quả nổi bật trong quá trình huấn luyện bao gồm:

* **Điểm hội tụ tối ưu (Best Weights):** Mô hình đạt trạng thái tổng quát hóa tốt nhất tại **Epoch 7**, với **Train Loss đạt 0.7182** và **Validation Loss chạm đáy cực tiểu ở mức 0.8247**. Trọng số tại Epoch này đã được lưu lại để đánh giá.
* **Nhánh Giới tính (Gender Classification):** Tích hợp hàm `BCEWithLogitsLoss` giúp mô hình hội tụ nhanh, độ tự tin (Confidence Score) trên các đặc trưng giới tính rõ ràng phân bổ ổn định và chính xác.
* **Nhánh Tuổi (Age Regression):** Việc áp dụng mạnh tay **Data Augmentation** (Random Rotation, Horizontal Flip, Color Jitter) đã ép nhánh này không được "học vẹt" màu nền hay góc chụp. Thay vào đó, nó phản ứng và nhạy cảm rất tốt với các đặc điểm lão hóa cốt lõi như cấu trúc xương hàm và sự thay đổi của nếp nhăn/cấu trúc da.
## 📊 Đánh giá Mô hình & Phân tích Lỗi (Tập dữ liệu UTKFace)

Sau khi áp dụng **Tăng cường dữ liệu (Data Augmentation)**, **L2 Regularization (Weight Decay: 1e-4)**, và **Dừng sớm (Early Stopping)**, mô hình đã được đánh giá trên một tập dữ liệu hoàn toàn mới là UTKFace.

- **Độ chính xác Giới tính (Accuracy):** 74.41%
- **Sai số Tuyệt đối Trung bình của Tuổi (MAE):** 11.51 năm
- **Tổng Loss:** 3.0590 (Giới tính: 0.6474 | Tuổi: 2.4115)

### 🔍 Phân tích Chuyên sâu: Vấn đề Mất cân bằng Dữ liệu (Data Imbalance)
Phân tích chi tiết chỉ số MAE theo từng nhóm tuổi đã lật tẩy một vấn đề kinh điển về **Mất cân bằng dữ liệu** bên trong tập huấn luyện ban đầu:

| Nhóm tuổi | Sai số (MAE) | Số lượng ảnh Test | Phân tích Hành vi Mô hình |
|:---:|:---:|:---:|---|
| **20-39** | `~5.10 năm` | ~11,800 | **Xuất sắc.** Mô hình hoạt động tốt nhất ở nhóm này do lượng dữ liệu huấn luyện (người trưởng thành trẻ tuổi) chiếm số lượng áp đảo. |
| **40-69** | `10 - 21 năm`| ~5,800 | **Trung bình.** Độ chính xác giảm dần khi số lượng mẫu ít đi và các đặc điểm lão hóa trên khuôn mặt trở nên phức tạp, khó đoán hơn. |
| **0-9 & 70+** | `22 - 37 năm`| ~4,400 | **Kém.** Mô hình dự đoán rất tệ đối với trẻ em và người già. Do tập huấn luyện thiếu vắng các nhóm này, kết hợp với hàm `MSELoss` thường phạt rất nặng các dự đoán cực đoan, mô hình đã bị ép phải "chơi an toàn" bằng cách luôn thiên vị và kéo dự đoán về khoảng tuổi trung bình (25-35 tuổi). |

**🚀 Hướng phát triển tiếp theo:** Các phiên bản nâng cấp sau sẽ thử nghiệm việc chuyển đổi bài toán hồi quy (đoán 1 số tuổi) thành bài toán phân loại bằng kỹ thuật **DEX (Deep EXpectation)**, hoặc áp dụng **Hàm mất mát có trọng số (Weighted Loss)** để phạt nặng hơn khi mô hình đoán sai trên các nhóm tuổi thiểu số.
##
👨‍💻 Tác giả

**Nguyễn Quang Minh**

Sinh viên chuyên ngành Trí tuệ Nhân tạo (AI).
