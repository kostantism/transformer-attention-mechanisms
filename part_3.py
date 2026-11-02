import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import random_split
import torchvision.transforms as transforms
from torchvision.models import (
    resnet50, ResNet50_Weights,
    vit_b_16, ViT_B_16_Weights,
)
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import matplotlib.pyplot as plt
import time

#question 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

#load oxford pet train dataset split
train_dataset = datasets.OxfordIIITPet(
    root='./data',
    split='trainval',  
    target_types='category',
    download=True,
)

#load oxford pet test dataset split
test_dataset = datasets.OxfordIIITPet(
    root='./data',
    split='test',   
    target_types='category',
    download=True,
)

#calculate subset sizes for train and validation split
train_size = int(0.8 * len(train_dataset))
validation_size = len(train_dataset) - train_size 

torch.manual_seed(42)

#split train dataset into train and validation subsets
train_dataset_new, validation_dataset = random_split(train_dataset, [train_size, validation_size])

print(f"train samples: {len(train_dataset_new)}")
print(f"validation samples: {len(validation_dataset)}")
print(f"test samples: {len(test_dataset)}")

#get pretrained weights configs
resnet_weights = ResNet50_Weights.IMAGENET1K_V1
vit_weights = ViT_B_16_Weights.IMAGENET1K_V1

#retrieve built-in weight transforms
resnet_transforms = (ResNet50_Weights.IMAGENET1K_V1).transforms()
vit_transforms = (ViT_B_16_Weights.IMAGENET1K_V1).transforms()

#align input distribution with pretrained imagenet stats for transfer learning
imagenet_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


#define training augmentations for resnet
train_transforms_resnet = transforms.Compose([
    transforms.RandomResizedCrop(size=224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    imagenet_normalize
])

#define validation and test transforms for resnet without augmentation
test_transforms_resnet = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    imagenet_normalize
])

#define training augmentations for vit
train_transforms_vit = transforms.Compose([
    transforms.RandomResizedCrop(size=224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    imagenet_normalize
])

#define validation and test transforms for vit without augmentation
test_transforms_vit = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    imagenet_normalize
])



#load pretrained resnet50 and modify fc head for 37 classes
cnn = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
num_features_cnn = cnn.fc.in_features
cnn.fc = nn.Linear(num_features_cnn, 37)
cnn = cnn.to(device) 

#load pretrained vit and modify classification head for 37 classes
vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
num_features_vit = vit.heads.head.in_features
vit.heads.head = nn.Linear(num_features_vit, 37)
vit = vit.to(device) 




#collate function for resnet to handle dynamic phase transforms
def collate_cnn(batch):
    images, labels = [], []
   #select transform based on model state
    transform_to_apply = train_transforms_resnet if cnn.training else test_transforms_resnet
    
    for img, label in batch:
        labels.append(label)
        images.append(transform_to_apply(img))
    labels  = torch.tensor(labels, dtype=torch.long)
    images = torch.stack(images)
        
    return images.to(device), labels.to(device)


#collate function for vit to handle dynamic phase transforms
def collate_vit(batch):
    images, labels = [], []
    transform_to_apply = train_transforms_vit if vit.training else test_transforms_vit
    
    for img, label in batch:
        labels.append(label)
        images.append(transform_to_apply(img))    
    labels  = torch.tensor(labels, dtype=torch.long)
    images = torch.stack(images)
        
    return images.to(device), labels.to(device)



BATCH_SIZE = 32

#create dataloaders for resnet
train_loader_cnn = DataLoader(train_dataset_new, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_cnn)
val_loader_cnn   = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_cnn)
test_loader_cnn  = DataLoader(test_dataset,       batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_cnn)

#create dataloaders for vit
train_loader_vit = DataLoader(train_dataset_new, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_vit)
val_loader_vit   = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_vit)
test_loader_vit  = DataLoader(test_dataset,       batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_vit)




#freeze all resnet parameters
for param in cnn.parameters():
    param.requires_grad = False

#unfreeze only the fc layer parameters for linear probing
for param in cnn.fc.parameters():
    param.requires_grad = True

#freeze all vit parameters
for param in vit.parameters():
    param.requires_grad = False

#unfreeze only the head layer parameters for linear probing
for param in vit.heads.head.parameters():
    param.requires_grad = True

#define loss function
criterion = nn.CrossEntropyLoss()

#pass all parameters to optimizers but only heads will update due to requires_grad=false
optimizer_cnn = optim.Adam(cnn.parameters(), lr=1e-3)
optimizer_vit = optim.Adam(vit.parameters(), lr=1e-3)


# optimizer_cnn = optim.Adam(cnn.parameters(), lr=1)



#standard training epoch execution loop
def train(model, dataloader, optimizer, criterion):
    model.train()
    train_loss, total_acc, total_count = 0.0, 0, 0

    for images, label in dataloader:
        images, label = images.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        total_acc += (output.argmax(1) == label).sum().item()
        total_count += label.size(0)
        
    epoch_loss = train_loss / total_count
    epoch_acc = total_acc / total_count
    return epoch_loss, epoch_acc

#standard evaluation phase loop
def evaluate(model, dataloader, criterion):
    model.eval()
    val_loss, total_acc, total_count = 0.0, 0, 0
    with torch.no_grad():
        for image, label in dataloader:
            image, label = image.to(device), label.to(device)
            output = model(image)
            loss = criterion(output, label)
            val_loss += loss.item() * image.size(0)
            total_acc += (output.argmax(1) == label).sum().item()
            total_count += label.size(0)
            
    epoch_loss = val_loss / total_count
    epoch_acc = total_acc / total_count
    return epoch_loss, epoch_acc


print("\n")


#initialize storage lists for resnet stats
EPOCHS = 10
train_loss_resnet = []
train_acc_resnet  = []
val_loss_resnet    = []
val_acc_resnet     = []

print("ResNet50 Training")
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    
    #execute training and validation steps
    train_loss, train_acc = train(cnn, train_loader_cnn, optimizer_cnn, criterion)
    val_loss, val_acc = evaluate(cnn, val_loader_cnn, criterion)
    
    #save records
    train_loss_resnet.append(train_loss)
    train_acc_resnet.append(train_acc)
    val_loss_resnet.append(val_loss)
    val_acc_resnet.append(val_acc)
    
    print(f"Epoch {epoch:2d}/{EPOCHS} | time {time.time()-t0:5.1f}s "
          f"train acc {train_acc:.3f} | train loss {train_loss:.3f}"
          f" val acc {val_acc:.3f} | val loss {val_loss:.3f}")
    

print("\n")

#initialize storage lists for vit stats
EPOCHS = 10
train_loss_vit = []
train_acc_vit  = []
val_loss_vit    = []
val_acc_vit     = []

print("ViT-B/16 Training")
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    
    train_loss, train_acc = train(vit, train_loader_vit, optimizer_vit, criterion)
    val_loss, val_acc = evaluate(vit, val_loader_vit, criterion)
    
    train_loss_vit.append(train_loss)
    train_acc_vit.append(train_acc)
    val_loss_vit.append(val_loss)
    val_acc_vit.append(val_acc)
    
    print(f"Epoch {epoch:2d}/{EPOCHS} | time {time.time()-t0:5.1f}s "
          f"train acc {train_acc:.3f} | train loss {train_loss:.3f}"
          f" val acc {val_acc:.3f} | val loss {val_loss:.3f}")
    
print("\n")

#evaluate final performance of resnet50 on test dataset
_, test_acc_cnn = evaluate(cnn, test_loader_cnn, criterion)
print(f"ResNet50 Accuracy: {test_acc_cnn:.3f}")

#evaluate final performance of vit on test dataset
_, test_acc_vit = evaluate(vit, test_loader_vit, criterion)
print(f"ViT-B/16 Accuracy: {test_acc_vit:.3f}")
    
print("\n")

epochs_range = range(1, EPOCHS + 1)

plt.close('all')

plt.figure()

#plot accuracy curves for linear probing
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_acc_resnet, 'b--', label='ResNet50 Train Acc')
plt.plot(epochs_range, val_acc_resnet, 'b-', label='ResNet50 Val Acc')
plt.plot(epochs_range, train_acc_vit, 'g--', label='ViT-B/16 Train Acc')
plt.plot(epochs_range, val_acc_vit, 'g-', label='ViT-B/16 Val Acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

#plot loss curves for linear probing
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_loss_resnet, 'b--', label='ResNet50 Train Loss')
plt.plot(epochs_range, val_loss_resnet, 'b-', label='ResNet50 Val Loss')
plt.plot(epochs_range, train_loss_vit, 'g--', label='ViT-B/16 Train Loss')
plt.plot(epochs_range, val_loss_vit, 'g-', label='ViT-B/16 Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()





#question 2




#reload resnet from imagenet to perform full fine-tuning
cnn = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
num_features_cnn = cnn.fc.in_features
cnn.fc = nn.Linear(num_features_cnn, 37)
cnn = cnn.to(device)

#reload vit from imagenet to perform full fine-tuning
vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
num_features_vit = vit.heads.head.in_features
vit.heads.head = nn.Linear(num_features_vit, 37)
vit = vit.to(device)

#unfreeze all parameters for full backpropagation
for param in cnn.parameters():
    param.requires_grad = True

for param in vit.parameters():
    param.requires_grad = True

#initialize sub-lists to separate parameters for differential learning rates
backbone_params_cnn = []
head_params_cnn = []
backbone_params_vit = []
head_params_vit = []

#group resnet parameters into backbone and head components
for name, param in cnn.named_parameters():
    if "fc" in name:  
        head_params_cnn.append(param)
    else:
        backbone_params_cnn.append(param)

#group vit parameters into backbone and head components
for name, param in vit.named_parameters():
    if "heads.head" in name:  
        head_params_vit.append(param)
    else:
        backbone_params_vit.append(param)

#configure optimizers with lower learning rate for pre-trained weights
optimizer_cnn_ft = optim.Adam([
    {'params': backbone_params_cnn, 'lr': 1e-5}, 
    {'params': head_params_cnn, 'lr': 1e-3}      
])

optimizer_vit_ft = optim.Adam([
    {'params': backbone_params_vit, 'lr': 1e-5}, 
    {'params': head_params_vit, 'lr': 1e-3}      
])

#initialize history storage containers for full fine-tuning
train_loss_cnn_ft = []
train_acc_cnn_ft = []
val_loss_cnn_ft = []
val_acc_cnn_ft = []

train_loss_vit_ft = []
train_acc_vit_ft = []
val_loss_vit_ft = []
val_acc_vit_ft = []

best_val_acc_cnn_ft = 0.0
best_val_acc_vit_ft = 0.0

print("\n")


#full fine tuning training loop for resnet
print("ResNet50 Full Fine-Tuning Training")
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    
    train_loss, train_acc = train(cnn, train_loader_cnn, optimizer_cnn_ft, criterion)
    val_loss, val_acc = evaluate(cnn, val_loader_cnn, criterion)
    
    train_loss_cnn_ft.append(train_loss)
    train_acc_cnn_ft.append(train_acc)
    val_loss_cnn_ft.append(val_loss)
    val_acc_cnn_ft.append(val_acc)
    
    if val_acc > best_val_acc_cnn_ft:
        best_val_acc_cnn_ft = val_acc
        
    print(f"Epoch {epoch:2d}/{EPOCHS} | time {time.time()-t0:5.1f}s "
          f"train acc {train_acc:.3f} | train loss {train_loss:.3f} | "
          f"val acc {val_acc:.3f} | val loss {val_loss:.3f}")
    
print("\n")

    
#full fine tuning training loop for vit
print("ViT-B/16 Full Fine-Tuning Training")
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    
    train_loss, train_acc = train(vit, train_loader_vit, optimizer_vit_ft, criterion)
    val_loss, val_acc = evaluate(vit, val_loader_vit, criterion)
    
    train_loss_vit_ft.append(train_loss)
    train_acc_vit_ft.append(train_acc)
    val_loss_vit_ft.append(val_loss)
    val_acc_vit_ft.append(val_acc)
    
    if val_acc > best_val_acc_vit_ft:
        best_val_acc_vit_ft = val_acc
        
    print(f"Epoch {epoch:2d}/{EPOCHS} | time {time.time()-t0:5.1f}s "
          f"train acc {train_acc:.3f} | train loss {train_loss:.3f} | "
          f"val acc {val_acc:.3f} | val loss {val_loss:.3f}")

print("\n")

#perform final evaluation of finetuned resnet model on test split
_, test_acc_cnn_ft = evaluate(cnn, test_loader_cnn, criterion)
print(f"ResNet50 Fine-Tuned Accuracy: {test_acc_cnn_ft:.3f}")

#perform final evaluation of finetuned vit model on test split
_, test_acc_vit_ft = evaluate(vit, test_loader_vit, criterion)
print(f"ViT-B/16 Fine-Tuned Accuracy: {test_acc_vit_ft:.3f}")

print("\n")

#generate plots for fine tuning performance tracking
plt.close('all')
plt.figure()

#plot accuracy curves for fine tuning
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_acc_cnn_ft, 'b--', label='ResNet50 FT Train Acc')
plt.plot(epochs_range, val_acc_cnn_ft, 'b-', label='ResNet50 FT Val Acc')
plt.plot(epochs_range, train_acc_vit_ft, 'g--', label='ViT-B/16 FT Train Acc')
plt.plot(epochs_range, val_acc_vit_ft, 'g-', label='ViT-B/16 FT Val Acc')
plt.title('Fine-Tuning Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

#plot loss curves for fine tuning
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_loss_cnn_ft, 'b--', label='ResNet50 FT Train Loss')
plt.plot(epochs_range, val_loss_cnn_ft, 'b-', label='ResNet50 FT Val Loss')
plt.plot(epochs_range, train_loss_vit_ft, 'g--', label='ViT-B/16 FT Train Loss')
plt.plot(epochs_range, val_loss_vit_ft, 'g-', label='ViT-B/16 FT Val Loss')
plt.title('Fine-Tuning Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\n")



#question 3


#define visualization function to capture hits and misse
def visualize(model, dataset, transform_to_apply, model_name):
    model.eval()
    correct, wrong = [], []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            img, label = dataset[i]
            
            #apply corresponding transform configuration passed as parameter
            img_tensor = transform_to_apply(img).unsqueeze(0).to(device)
            
            #compute outputs and map raw logits to percentages using softmax
            output = model(img_tensor)
            probabilities = torch.softmax(output, 1)[0] 
            pred = output.argmax(1).item()
            
            #extract prediction probability margins
            true_prob = probabilities[label].item() * 100
            pred_prob = probabilities[pred].item() * 100
            
            if pred == label and len(correct) < 4:
                #store correct classifications
                correct.append((img, label, pred, true_prob))
                
            elif pred != label and len(wrong) < 4:
                #store wrong predictions along with correct and incorrect scores
                wrong.append((img, label, pred, true_prob, pred_prob))
                
            if len(correct) == 4 and len(wrong) == 4:
                break

    #initialize grid setup for image grid displaying samples
    fig, axes = plt.subplots(2, 4, figsize=(12, 10)) 
    fig.suptitle(f'{model_name}') 
    class_names = dataset.classes

    #render first row with successful predictions
    for idx, (img, label, pred, true_prob) in enumerate(correct):
        axes[0, idx].imshow(img)
        axes[0, idx].axis('off')
        axes[0, idx].set_title(f"True: {class_names[label]}\nprob: {true_prob:.1f}%", color='green') 

    #render second row with error classifications
    for idx, (img, label, pred, true_prob, pred_prob) in enumerate(wrong):
        axes[1, idx].imshow(img)
        axes[1, idx].axis('off')
        axes[1, idx].set_title(f"False\nTrue: {class_names[label]} ({true_prob:.1f}%)\npred: {class_names[pred]} ({pred_prob:.1f}%)", color='red') 

    plt.tight_layout()
    plt.show()

#visualization calls 
visualize(cnn, test_dataset, test_transforms_resnet, "ResNet50")
visualize(vit, test_dataset, test_transforms_vit, "ViT-B/16")

