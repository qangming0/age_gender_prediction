import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, mean_absolute_error

from models.multitask_cnn import AgeGenderModel

class UTKFaceEvalDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.file_list = [f for f in os.listdir(folder_path) if f.endswith('.jpg') and len(f.split('_')) >= 3]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        img_path = os.path.join(self.folder_path, file_name)
        
        parts = file_name.split('_')
        age = float(parts[0])
        utk_gender = int(parts[1]) 
        gender = 1 - utk_gender # Đảo nhãn giới tính
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(age, dtype=torch.float32), torch.tensor(gender, dtype=torch.long)

DATA_PATH = 'data/UTKFace' 
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

eval_dataset = UTKFaceEvalDataset(DATA_PATH, transform=test_transform)
eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

def evaluate(model, loader):
    model.eval()
    
    # -----------------------------------------------------
    # MANG HÀM LOSS TỪ TRAIN.PY SANG ĐỂ TÍNH TOÁN
    # -----------------------------------------------------
    criterion_gender = nn.BCEWithLogitsLoss()
    criterion_age = nn.MSELoss() 
    AGE_LOSS_WEIGHT = 0.01 
    
    total_eval_loss = 0.0
    total_gen_loss = 0.0
    total_age_loss = 0.0
    
    all_true_ages, all_pred_ages = [], []
    all_true_genders, all_pred_genders = [], []
    
    print(f"Đang bắt đầu đánh giá trên {len(eval_dataset)} ảnh...")
    
    with torch.no_grad():
        for images, ages, genders in loader:
            images = images.to(DEVICE)
            
            # Định dạng lại kích thước Tensor giống y hệt lúc Train
            genders_tensor = genders.to(DEVICE).view(-1, 1).float()
            ages_tensor = ages.to(DEVICE).view(-1, 1).float()
            
            outputs_gender, outputs_age = model(images)
            
            # --- 1. TÍNH TOÁN CHỈ SỐ LOSS (0.x - 1.x) ---
            loss_gen = criterion_gender(outputs_gender, genders_tensor)
            loss_age = criterion_age(outputs_age, ages_tensor)
            
            batch_total_loss = loss_gen + (loss_age * AGE_LOSS_WEIGHT)
            
            total_eval_loss += batch_total_loss.item()
            total_gen_loss += loss_gen.item()
            total_age_loss += (loss_age.item() * AGE_LOSS_WEIGHT)
            
            # --- 2. TÍNH TOÁN MAE / ACCURACY ---
            pred_gender_probs = torch.sigmoid(outputs_gender)
            pred_gender = (pred_gender_probs > 0.5).long().squeeze()
            pred_age = outputs_age.squeeze()

            all_true_ages.extend(ages.numpy())
            all_pred_ages.extend(pred_age.cpu().numpy())
            all_true_genders.extend(genders.numpy())
            all_pred_genders.extend(pred_gender.cpu().numpy())

    # Tính trung bình các loại Loss trên toàn bộ tập test
    avg_total_loss = total_eval_loss / len(loader)
    avg_gen_loss = total_gen_loss / len(loader)
    avg_age_loss = total_age_loss / len(loader)

    mae = mean_absolute_error(all_true_ages, all_pred_ages)
    acc = accuracy_score(all_true_genders, all_pred_genders)
    
    print("\n" + "=" * 40)
    print(f"🏆 KẾT QUẢ TỔNG QUAN TRÊN UTKFACE:")
    print(f"🔹 TỔNG LOSS (Thang đo Train): {avg_total_loss:.4f}")
    print(f"   - Loss Giới tính:           {avg_gen_loss:.4f}")
    print(f"   - Loss Tuổi (đã x 0.01):    {avg_age_loss:.4f}")
    print("-" * 40)
    print(f"1. Giới tính (Accuracy): {acc * 100:.2f}%") 
    print(f"2. Tuổi (MAE tổng):      {mae:.2f} năm")
    print("=" * 40)

    # ==========================================
    # PHÂN TÍCH SAI SỐ THEO NHÓM TUỔI
    # ==========================================
    print("\n📊 PHÂN TÍCH SAI SỐ TUỔI CHI TIẾT:")
    true_ages_np = np.array(all_true_ages)
    pred_ages_np = np.array(all_pred_ages)
    errors = np.abs(true_ages_np - pred_ages_np) 

    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 120]
    labels = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
    
    group_stats = []

    for i in range(len(labels)):
        mask = (true_ages_np >= bins[i]) & (true_ages_np < bins[i+1])
        count = np.sum(mask) 
        
        if count > 0:
            group_mae = np.mean(errors[mask])
            group_stats.append({
                "label": labels[i],
                "mae": group_mae,
                "count": count
            })

    group_stats.sort(key=lambda x: x["mae"], reverse=True)

    for stat in group_stats:
        print(f" ⚠️ Nhóm {stat['label']:>5} tuổi | Sai số: {stat['mae']:>5.2f} năm | Dữ liệu: {stat['count']} ảnh")
    print("-" * 40)

if __name__ == "__main__":
    print(f"🚀 Đang nạp mô hình lên thiết bị: {DEVICE}")
    
    my_model = AgeGenderModel(pretrained=False).to(DEVICE)
    my_model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
    
    evaluate(my_model, eval_loader)