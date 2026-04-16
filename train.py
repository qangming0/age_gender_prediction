import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm 
from data.dataset import get_dataloaders
from models.multitask_cnn import AgeGenderModel

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Đang sử dụng thiết bị: {device}")

    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 0.0001
    AGE_LOSS_WEIGHT = 0.01 
    WEIGHT_DECAY = 1e-4 

    # Lấy thêm test_loader để làm tập Validation
    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE, num_workers=0)

    model = AgeGenderModel(pretrained=True).to(device)

    criterion_gender = nn.BCEWithLogitsLoss() 
    criterion_age = nn.MSELoss()              
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # --- THÔNG SỐ EARLY STOPPING ---
    best_val_loss = float('inf') 
    patience = 3                 
    trigger_times = 0           

    for epoch in range(EPOCHS):
        model.train()
        running_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for images, genders, ages in pbar:
            images = images.to(device)
            genders = genders.to(device).view(-1, 1)
            ages = ages.to(device).view(-1, 1)

            optimizer.zero_grad()
            gen_out, age_out = model(images)
            loss_gen = criterion_gender(gen_out, genders)
            loss_age = criterion_age(age_out, ages)
            
            total_loss = loss_gen + (loss_age * AGE_LOSS_WEIGHT)
            total_loss.backward()
            optimizer.step()

            running_train_loss += total_loss.item()
            pbar.set_postfix({'loss': total_loss.item()})

        train_loss = running_train_loss / len(train_loader)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, genders, ages in test_loader:
                images = images.to(device)
                genders = genders.to(device).view(-1, 1)
                ages = ages.to(device).view(-1, 1)

                gen_out, age_out = model(images)
                loss_gen = criterion_gender(gen_out, genders)
                loss_age = criterion_age(age_out, ages)
                
                total_loss = loss_gen + (loss_age * AGE_LOSS_WEIGHT)
                running_val_loss += total_loss.item()
                
        val_loss = running_val_loss / len(test_loader)
        
        print(f"✅ End Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            print(f"🌟 Val loss giảm từ {best_val_loss:.4f} xuống {val_loss:.4f}. Đang lưu mô hình mới!")
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            trigger_times = 0 # Reset bộ đếm
        else:
            trigger_times += 1
            print(f"⚠️ Val loss không giảm. Cảnh báo Early Stopping: {trigger_times}/{patience}")
            if trigger_times >= patience:
                print("🛑 Early Stopping kích hoạt! Đã dừng việc huấn luyện để chống Overfitting.")
                break # Dừng luôn vòng lặp Epoch

if __name__ == "__main__":
    train_model()