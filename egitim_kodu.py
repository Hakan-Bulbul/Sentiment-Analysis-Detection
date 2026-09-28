import torch #Pytorch ana kütüphane
import torch.nn as nn #Katmanlar(Linear/Loss)
import torch.optim as optim #Optimizasyon Algoritmları(AdamW)
from torch.utils.data import DataLoader #Dataseti batch'lere böler
from torchvision import transforms, models, datasets #transforms->augmentation, models->pretrained models(efficientnet b-3), datasets->hazır datasets loader
import matplotlib.pyplot as plt #Eğitim grafikleri
import seaborn as sns  #Confusion matrix çizimi
from sklearn.metrics import confusion_matrix, classification_report, f1_score #Confusion matrix hesaplama, Precision/Recall/F1 raporu, Weighted F1 skoru
from tqdm import tqdm #Progress bar
from collections import Counter #Class sayımı (class imbalance için)

# ------------------ GÖRSELLEŞTİRME ------------------
def plot_final_results(history, y_true, y_pred, class_names): #Eğitim sürecini Final confusion matrix’i tek ekranda gösterir.
    epochs = range(1, len(history['train_acc']) + 1) #Kaç epoch olduysa eğitim ona göre oluşur
    plt.figure(figsize=(18, 5)) #Yatay geniş grafik

    plt.subplot(1, 3, 1) #Grafik 1 - Accuracy
    plt.plot(epochs, history['train_acc'], label='Train Acc') #Train doğruluğu
    plt.plot(epochs, history['val_acc'], label='Val Acc') #Validation doğruluğu
    plt.legend()
    plt.title("Accuracy (%)")

    plt.subplot(1, 3, 2) #Grafik 2 - F1 Score
    plt.plot(epochs, history['val_f1'], 'orange', label='Val F1') #f1 score
    plt.legend()
    plt.title("F1-Score")

    plt.subplot(1, 3, 3) #Grafik 3 - Confusion matrix (gerçek vs tahmin karşılaştırması)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=class_names, yticklabels=class_names) #Hangi duygu hangisiyle karışıyor gördüğümüz kısım.
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    plt.tight_layout()
    plt.show()

# ------------------ ANA PROGRAM ------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #GPU varsa kullanır yoksa CPU kullanır.

    # -------- PARAMETRELER --------
    EPOCHS = 30 #Eğitim turu
    BATCH_SIZE = 128 #Her adımda görüntü sayısı
    LR = 8e-5 #Öğrenme oranı
    PATIENCE = 5 #Kaç epoch boyunca öğrenme olmazsa durma sayısı
    LABEL_SMOOTHING = 0.1 #Modelin aşırı emin olmasını engeller

    #Early Stopping kontrol değişkenleri
    best_val_f1 = 0.0
    early_stop_counter = 0

    SAVE_PATH = "model_dosyasi.pth"

    train_dir = r"C:\Users\hakan\Downloads\Karma_Veri_Seti\train"
    val_dir   = r"C:\Users\hakan\Downloads\Karma_Veri_Seti\val"
    test_dir  = r"C:\Users\hakan\Downloads\Karma_Veri_Seti\test"

    # -------- TRAIN AUGMENTATION --------
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(300, scale=(0.8, 1.0)), #Rastgele crop
        transforms.RandomHorizontalFlip(p=0.5), #Yüz sağ, sol simetrisi
        transforms.RandomRotation(10), #Kamera açısı değişimi
        transforms.ColorJitter(0.2, 0.2, 0.2), #Işık değişimi simülasyonu
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)) #Yüzün bir kısmı kapanmış gibi
    ])

    eval_transform = transforms.Compose([ #Resize ve normalize
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # -------- DATASET --------
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform) #Klasör isimlerini sınıf etiketi yapar
    val_dataset   = datasets.ImageFolder(val_dir, transform=eval_transform)
    test_dataset  = datasets.ImageFolder(test_dir, transform=eval_transform)

    class_names = train_dataset.classes #Sınıf isimlerini alır
    num_classes = len(class_names)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,shuffle=True, num_workers=4, pin_memory=True) #Train set suffle(karıştırmak) edilir
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,shuffle=False, num_workers=4, pin_memory=True) #Val set suffle(karıştırmak) edilir
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,shuffle=False, num_workers=4, pin_memory=True) #Test set suffle(karıştırmak) edilir

    # -------- CLASS WEIGHTS --------
    targets = [label for _, label in train_dataset.samples] #Tüm etiketler çıkarılır
    counts = Counter(targets) #Her sınıfın kaç görüntüsü var
    max_count = max(counts.values())
    class_weights = torch.tensor([
        (max_count / counts[i]) ** 0.5  #Karekök ile yumuşatılmış
        for i in range(num_classes)
    ], dtype=torch.float).to(device)

    # -------- MODEL --------
    model = models.efficientnet_b3(
        weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1 #IMAGENET pretrained model (EfficientNet B-3)
    )

    # Tüm katmanları dondur
    for p in model.parameters():
        p.requires_grad = False

    # Son 3 blok + classifier açık
    for p in model.features[-2:].parameters(): #Sadece üst seviye katmanlar öğrenir
        p.requires_grad = True

    #Dropout
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),  # %40 oranında nöronu rastgele kapatarak ezberlemeyi önler
        nn.Linear(num_ftrs, num_classes)
    )
    model.to(device)

    # -------- LOSS --------
    criterion = nn.CrossEntropyLoss( #Class imbalance çözüldü, Overconfidence azaltıldı
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING
    )

    optimizer = optim.AdamW( # Modern standart optimizer
        #filter(lambda p: p.requires_grad, model.parameters()),
        model.parameters(),lr=LR, weight_decay=5e-4 #kontrol
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau( #f1 artmazsa LR düşer
        optimizer,
        mode='max',
        factor=0.5,
        patience=4,
        #verbose=True
    )

    history = {'train_acc': [], 'val_acc': [], 'val_f1': []}

    # -------- TRAIN LOOP --------
    for epoch in range(EPOCHS):

        # ---- GRADUAL UNFREEZE (Kademeli Katman Açma) ----
        if epoch == 7: #Daha fazla katman açılır
            for p in model.features[-3:].parameters():
                p.requires_grad = True
            print("features[-3:] açıldı")

        model.train() #Dropout/BatchNorm aktif
        correct, total = 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"): #Batch döngüsü
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    #Standart backpropagation

            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
    #Accuracy hesaplama

        train_acc = 100 * correct / total

        # -------- VALIDATION --------
        model.eval() #Dropout kapanır
        v_correct, v_total = 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad(): #Gradient hesaplanmaz
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = outputs.max(1)

                v_total += labels.size(0)
                v_correct += preds.eq(labels).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = 100 * v_correct / v_total
        val_f1 = f1_score(all_labels, all_preds, average='weighted') #F1 hesaplama

        scheduler.step(val_f1) #Plateau varsa LR düşer.

        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        print(f"Train {train_acc:.2f}% | Val {val_acc:.2f}% | F1 {val_f1:.4f}") #kontrol

        # -------- EARLY STOPPING --------
        if val_f1 > best_val_f1: #F1 artarsa model kaydeder ve sayaç sıfırlanır
            best_val_f1 = val_f1
            early_stop_counter = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print("⭐ En iyi model kaydedildi")
        else:
            early_stop_counter += 1 #F1 artmazsa sayaç devam eder ve sınıra(PATIENCE) gelirse eğitim durur
            if early_stop_counter >= PATIENCE:
                print("Early stopping")
                break

    # -------- TEST --------
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device)) #En iyi model yüklenir ve test set üzerinde tahmin yapılır
    model.eval()

    final_preds, final_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            final_preds.extend(preds.cpu().numpy())
            final_labels.extend(labels.cpu().numpy())

    print("\n--- TEST SONUÇLARI ---")
    print(classification_report(final_labels, final_preds, target_names=class_names))

    plot_final_results(history, final_labels, final_preds, class_names) #Eğitim eğrileri ve confusion matrix gösterilir

if __name__ == "__main__":
    main()