import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

#Carregar dataset
digits = load_digits()
X = digits.images   # (1797, 8, 8)
y = digits.target   # (1797,)

#Normalizar e converter para tensores
X = X / 16.0
X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, 8, 8)
y = torch.tensor(y, dtype=torch.long)

#Dividir treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test, y_test)

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=32)

#Definir modelo CNN
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)   # (1 canal → 16 filtros)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*3*3, 32)  # ajustar tamanho depois do pooling
        self.fc2 = nn.Linear(32, 10)      # 10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # convolução + ReLU + pooling
        x = x.view(-1, 16*3*3)               # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN()

#Otimizador e função de perda
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Treinamento
loss_arr = []
for epoch in range(20):
    model.train()
    epoch_loss = 0.0
    for xb, yb in train_dl:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss_arr.append(loss.item())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_dl)
    loss_arr.append(avg_loss)
    print(f"Época {epoch+1}, loss: {loss.item():.4f}")

# Avaliação
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for xb, yb in test_dl:
        out = model(xb)
        _, preds = torch.max(out, 1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

print("Acurácia no teste:", correct / total)
#Gráfico da perda
plt.plot(loss_arr, marker="o")
plt.xlabel("Época")
plt.ylabel("Loss médio")
plt.title("Evolução da perda durante o treino")
plt.grid()
plt.savefig("loss_curve.png")
plt.show()


