import matplotlib.pyplot as plt
import torch
def display_training_curves(training, validation, title, subplot):

    plt.subplots(figsize=(10,10), facecolor='#F0F0F0')
    plt.tight_layout()

    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training, color="indianred", marker='o')
    ax.plot(validation, color="darkorange", marker='o')
    
    ax.set_ylabel(title)
    ax.set_xlabel('epoch')
    ax.set_title('Model '+ title)
    ax.legend(['train', 'valid'])
    plt.savefig(f'{title}.png')

    
val_f1s = [0] + [torch.load('../models/val_f1_{}.pt'.format(i)) for i in range(3)]
train_f1s = [0] + [torch.load('../models/train_f1_{}.pt'.format(i)) for i in range(3)]
val_losses = [0.25] + [torch.load('../models/val_loss_{}.pt'.format(i)) for i in range(3)]
train_losses = [0.25] + [torch.load('../models/train_loss_{}.pt'.format(i)) for i in range(3)]

display_training_curves(train_f1s, val_f1s, "F1 Score vs. Epochs", 212)
display_training_curves(train_losses, val_losses, "BCE Loss vs. Epochs", 211)
