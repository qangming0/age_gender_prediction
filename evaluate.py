import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from models.multitask_cnn import AgeGenderModel
import os

def evaluate_with_loss(image_path, model_path="best_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AgeGenderModel(pretrained=False).to(device)
    
    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy file trọng số: {model_path}")
        return
        
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception as e:
        print(f"❌ Lỗi khi tải mô hình: {e}")
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"❌ Không tìm thấy ảnh tại: {image_path}")
        return

    img_tensor = transform(img).unsqueeze(0).to(device)

    print(f"\n--- PHÂN TÍCH ẢNH: {image_path} ---")
    try:
        true_age_input = float(input("👉 Nhập tuổi thật của người trong ảnh: "))
        
        true_gender_input = int(input("👉 Nhập giới tính thật (1 cho Nam, 0 cho Nữ): "))
        if true_gender_input not in [0, 1]:
            raise ValueError("Giới tính chỉ được là 0 hoặc 1.")
            
    except ValueError:
        print("❌ Nhập sai định dạng. Vui lòng chạy lại và nhập số.")
        return

    criterion_age = nn.MSELoss()
    criterion_gender = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        gen_logits, age_pred = model(img_tensor)

        gen_prob = torch.sigmoid(gen_logits).item()
        pred_gender_label = "Nam" if gen_prob >= 0.5 else "Nữ"
        pred_gender_conf = gen_prob if pred_gender_label == "Nam" else (1 - gen_prob)
        pred_gender_conf *= 100

        pred_age = age_pred.item()

        true_age_tensor = torch.tensor([[true_age_input]], dtype=torch.float32).to(device)
        true_gender_tensor = torch.tensor([[true_gender_input]], dtype=torch.float32).to(device)

        loss_age = criterion_age(age_pred, true_age_tensor)
        loss_gen = criterion_gender(gen_logits, true_gender_tensor)

    true_gender_label = "Nam" if true_gender_input == 1 else "Nữ"

    print("\n" + "="*50)
    print("🎯 BÁO CÁO DỰ ĐOÁN & ĐÁNH GIÁ LOSS")
    print("="*50)
    print("👤 GIỚI TÍNH:")
    print(f"   - Nhãn thực tế : {true_gender_label}")
    print(f"   - Mô hình đoán : {pred_gender_label} (Tự tin: {pred_gender_conf:.2f}%)")
    print(f"   -> Loss (BCE)  : {loss_gen.item():.4f}")
    if true_gender_label != pred_gender_label:
        print("   ⚠️ CẢNH BÁO: Đoán sai giới tính!")
        
    print("-" * 50)
    print("🎂 ĐỘ TUỔI:")
    print(f"   - Nhãn thực tế : {true_age_input} tuổi")
    print(f"   - Mô hình đoán : {pred_age:.1f} tuổi")
    print(f"   -> Loss (MSE)  : {loss_age.item():.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    TEST_IMAGE_PATH = "Test/TestVTH2.jpg"
    evaluate_with_loss(TEST_IMAGE_PATH)