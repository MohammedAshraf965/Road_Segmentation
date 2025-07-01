import torch

from tqdm import tqdm
from utils import draw_translucent_seg_maps
from metrics import IOUEval

def train(model, train_dataloader, device, optimizer, criterion, classes_to_train):
    print('Training started...')
    model.train()
    train_running_loss = 0.0
    num_batches = len(train_dataloader)
    prog_bar = tqdm(train_dataloader, total=num_batches, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    batch_counter = 0
    num_classes = len(classes_to_train)
    iou_eval = IOUEval(num_classes)

    for i, data in enumerate(prog_bar):
        batch_counter += 1
        data, target = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(data)['out']

        loss = criterion(outputs, target)
        train_running_loss += loss.item()

        loss.backward()
        optimizer.step()

        iou_eval.addBatch(outputs.max(1)[1].data, target.data)
        
    # Per epoch loss
    train_loss = train_running_loss / batch_counter
    
    overall_acc, per_class_acc, per_class_iu, mIOU = iou_eval.getMetric()
    return train_loss, overall_acc, mIOU

def validate(model, valid_dataloader, device, criterion, classes_to_train, label_colors_list, epoch, save_dir):
    print('Validation started...')
    model.eval()
    valid_running_loss = 0.0
    num_batches = len(valid_dataloader)
    num_classes = len(classes_to_train)
    iou_eval = IOUEval(num_classes)

    with torch.no_grad():
        prog_bar = tqdm(valid_dataloader, total=num_batches, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        batch_counter = 0 # To keep track of batch counter.
        for i, data in enumerate(prog_bar):
            batch_counter += 1
            data, target = data[0].to(device), data[1].to(device)
            outputs = model(data)['out']
            
            # Save the validation segmentation maps every last batch of each epoch
            if i == num_batches - 1:
                draw_translucent_seg_maps(data, outputs, epoch, i, save_dir, label_colors_list)

            loss = criterion(outputs, target)
            valid_running_loss += loss.item()

            iou_eval.addBatch(outputs.max(1)[1].data, target.data)
        
    # Per epoch loss
    valid_loss = valid_running_loss / batch_counter

    overall_acc, per_class_acc, per_class_iu, mIOU = iou_eval.getMetric()
    return valid_loss, overall_acc, mIOU