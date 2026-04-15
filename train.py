import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm # Thư viện tạo thanh tiến trình
from data.dataset import get_dataloaders
from models.multitask_cnn import AgeGenderModel

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Đang sử dụng thiết bị: {device}")
    if torch.cuda.is_available():
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")

    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 0.0001
    AGE_LOSS_WEIGHT = 0.01 

    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE, num_workers=0)

    model = AgeGenderModel(pretrained=True).to(device)

    criterion_gender = nn.BCEWithLogitsLoss() # Cho SECH
    criterion_age = nn.MSELoss()              # Cho tuổi
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
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

            running_loss += total_loss.item()
            pbar.set_postfix({'loss': total_loss.item()})

        epoch_loss = running_loss / len(train_loader)
        print(f"✅ End Epoch {epoch+1}, Loss trung bình: {epoch_loss:.4f}")

        torch.save(model.state_dict(), "best_model.pth")

if __name__ == "__main__":
    train_model()