import torch

from metric import compute_micro_stats
import torch.nn.functional as F


def validate(generator, classifier, annotation_loader, device):
    num_true_pos = 0.
    num_predicted_pos = 0.
    num_real_pos = 0.
    num_words = 0

    TP = 0
    TN = 0
    FN = 0
    FP = 0

    for (batch, (inputs, masks, labels,
                 annotations)) in enumerate(annotation_loader):
        inputs, masks, labels, annotations = inputs.to(device), masks.to(device), labels.to(device), annotations.to(
            device)

        # rationales -- (batch_size, seq_length, 2)
        rationales = generator(inputs, masks)

        num_true_pos_, num_predicted_pos_, num_real_pos_ = compute_micro_stats(
            annotations, rationales[:, :, 1])

        # cls
        logits = classifier(inputs, masks, z=rationales[:, :, -1])

        soft_pred = F.softmax(logits, -1)
        _, pred = torch.max(soft_pred, dim=-1)

        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        FP += ((pred == 1) & (labels == 0)).cpu().sum()

        num_true_pos += num_true_pos_
        num_predicted_pos += num_predicted_pos_
        num_real_pos += num_real_pos_
        num_words += torch.sum(masks)

    micro_precision = num_true_pos / num_predicted_pos
    micro_recall = num_true_pos / num_real_pos
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision +
                                                       micro_recall)
    sparsity = num_predicted_pos / num_words

    # cls
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print("annotation dataset : recall:{:.4f} precision:{:.4f} f1-score:{:.4f}".format(recall, precision,
                                                                                       f1_score))
    return sparsity, micro_precision, micro_recall, micro_f1


def validate_att(model, annotation_loader, device):
    num_true_pos = 0.
    num_predicted_pos = 0.
    num_real_pos = 0.
    num_words = 0

    TP = 0
    TN = 0
    FN = 0
    FP = 0

    for (batch, (inputs, masks, labels,
                 annotations)) in enumerate(annotation_loader):
        inputs, masks, labels, annotations = inputs.to(device), masks.to(device), labels.to(device), annotations.to(
            device)

        # rationales -- (batch_size, seq_length, 2)
        att_score, rationales, cls_logits = model(inputs, masks)

        num_true_pos_, num_predicted_pos_, num_real_pos_ = compute_micro_stats(
            annotations, rationales)

        soft_pred = F.softmax(cls_logits, -1)
        _, pred = torch.max(soft_pred, dim=-1)

        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        FP += ((pred == 1) & (labels == 0)).cpu().sum()

        num_true_pos += num_true_pos_
        num_predicted_pos += num_predicted_pos_
        num_real_pos += num_real_pos_
        num_words += torch.sum(masks)

    micro_precision = num_true_pos / num_predicted_pos
    micro_recall = num_true_pos / num_real_pos
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision +
                                                       micro_recall)
    sparsity = num_predicted_pos / num_words

    # cls
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)

    print("annotation dataset : recall:{:.4f} precision:{:.4f} f1-score:{:.4f}".format(recall, precision,
                                                                                       f1_score))
    return sparsity, micro_precision, micro_recall, micro_f1


def validate_share(model, annotation_loader, device):
    num_true_pos = 0.
    num_predicted_pos = 0.
    num_real_pos = 0.
    num_words = 0

    TP = 0
    TN = 0
    FN = 0
    FP = 0

    for (batch, (inputs, masks, labels, annotations)) in enumerate(annotation_loader):
        inputs, masks, labels, annotations = inputs.to(device), masks.to(device), labels.to(device), annotations.to(
            device)

        # rationales -- (batch_size, seq_length, 2)
        _, rationales, cls_logits = model(inputs, masks)

        num_true_pos_, num_predicted_pos_, num_real_pos_ = compute_micro_stats(
            annotations, rationales[:, :, 1])

        soft_pred = F.softmax(cls_logits, -1)
        _, pred = torch.max(soft_pred, dim=-1)

        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        FP += ((pred == 1) & (labels == 0)).cpu().sum()

        num_true_pos += num_true_pos_
        num_predicted_pos += num_predicted_pos_
        num_real_pos += num_real_pos_
        num_words += torch.sum(masks)

    micro_precision = num_true_pos / num_predicted_pos
    micro_recall = num_true_pos / num_real_pos
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision +
                                                       micro_recall)
    sparsity = num_predicted_pos / num_words

    # cls
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print(
        "annotation dataset : recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(recall, precision,
                                                                                                     f1_score,
                                                                                                     accuracy))
    return sparsity, micro_precision, micro_recall, micro_f1

def validate_share_rnp(model, annotation_loader, device):
    num_true_pos = 0.
    num_predicted_pos = 0.
    num_real_pos = 0.
    num_words = 0

    TP = 0
    TN = 0
    FN = 0
    FP = 0

    for (batch, (inputs, masks, labels,annotations)) in enumerate(annotation_loader):
        inputs, masks, labels, annotations = inputs.to(device), masks.to(device), labels.to(device), annotations.to(
            device)

        # rationales -- (batch_size, seq_length, 2)
        rationales, cls_logits = model(inputs, masks)

        num_true_pos_, num_predicted_pos_, num_real_pos_ = compute_micro_stats(
            annotations, rationales[:, :, 1])

        soft_pred = F.softmax(cls_logits, -1)
        _, pred = torch.max(soft_pred, dim=-1)

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()

        num_true_pos += num_true_pos_
        num_predicted_pos += num_predicted_pos_
        num_real_pos += num_real_pos_
        num_words += torch.sum(masks)

    micro_precision = num_true_pos / num_predicted_pos
    micro_recall = num_true_pos / num_real_pos
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision +
                                                       micro_recall)
    sparsity = num_predicted_pos / num_words

    # cls
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print(
        "annotation dataset : recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(recall, precision,
                                                                                                     f1_score,
                                                                                                     accuracy))
    return sparsity, micro_precision, micro_recall, micro_f1


def validate_share_write_best_prediction(model, annotation_loader, device,args_root_save_dir):
    args,root_save_dir = args_root_save_dir
    num_true_pos = 0.
    num_predicted_pos = 0.
    num_real_pos = 0.
    num_words = 0

    TP = 0
    TN = 0
    FN = 0
    FP = 0

    gold_rationales = []
    pred_rationales = []
    gold_labels =[]
    pred_labels = []


    with torch.no_grad():
        for (batch, (inputs, masks, labels,annotations)) in enumerate(annotation_loader):
            inputs, masks, labels, annotations = inputs.to(device), masks.to(device), labels.to(device), annotations.to(
                device)

            # rationales -- (batch_size, seq_length, 2)
            _, rationales, cls_logits = model(inputs, masks)

            num_true_pos_, num_predicted_pos_, num_real_pos_ = compute_micro_stats(
                annotations, rationales[:, :, 1])

            soft_pred = F.softmax(cls_logits, -1)
            _, pred = torch.max(soft_pred, dim=-1)

            # TP predict 和 label 同时为1
            TP += ((pred == 1) & (labels == 1)).cpu().sum()
            # TN predict 和 label 同时为0
            TN += ((pred == 0) & (labels == 0)).cpu().sum()
            # FN predict 0 label 1
            FN += ((pred == 0) & (labels == 1)).cpu().sum()
            # FP predict 1 label 0
            FP += ((pred == 1) & (labels == 0)).cpu().sum()

            num_true_pos += num_true_pos_
            num_predicted_pos += num_predicted_pos_
            num_real_pos += num_real_pos_
            num_words += torch.sum(masks)

            gold_rationales.extend(annotations.detach().cpu().numpy().tolist())
            pred_rationales.extend(rationales[:, :, 1].detach().cpu().numpy().tolist())
            gold_labels.extend(labels.detach().cpu().numpy().tolist())
            pred_labels.extend([int(item) for item in pred.detach().cpu().numpy().tolist()])

    assert len(gold_rationales)==len(pred_rationales),"rationale labels len not equal"
    assert len(gold_labels)==len(pred_labels),"cls labels len not equal"

    import os
    prediction_file = os.path.join(root_save_dir,"{}_{}_best_pred.json".format(args.data_type,args.aspect))
    print(prediction_file)
    with open(prediction_file,encoding='utf-8',mode='w') as file:
        for i in range(len(gold_rationales)):
            gold_rationale = gold_rationales[i]
            pred_rationale = pred_rationales[i]
            gold_label, pred_label = gold_labels[i],pred_labels[i]
            file.write("{}\t{}\t{}\t{}\n".format(str(gold_label),str(pred_label),str(gold_rationale),str(pred_rationale)))
    

    micro_precision = num_true_pos / num_predicted_pos
    micro_recall = num_true_pos / num_real_pos
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision +
                                                       micro_recall)
    sparsity = num_predicted_pos / num_words

    # cls
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print("annotation dataset : recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(recall, precision,
                                                                                                     f1_score,
                                                                                                     accuracy))
    
    hyper_result_file = os.path.join(root_save_dir,"{}_{}_result.json".format(args.data_type,args.aspect))
    with open(hyper_result_file,encoding='utf-8',mode='w') as file:
        file.write("sparsity: {}\n precision: {}\n recall: {}\n f1: {}\n ".
                   format(sparsity,micro_precision,micro_recall,micro_f1))
            

    return sparsity, micro_precision, micro_recall, micro_f1




def validate_annotation_sentence(model, annotation_loader, device):
    TP = 0
    TN = 0
    FN = 0
    FP = 0

    for (batch, (inputs, masks, labels,
                 annotations)) in enumerate(annotation_loader):
        inputs, masks, labels, annotations = inputs.to(device), masks.to(device), labels.to(device), annotations.to(
            device)

        # rationales -- (batch_size, seq_length, 2)
        cls_logits = model.train_one_step(inputs, masks)

        soft_pred = F.softmax(cls_logits, -1)
        _, pred = torch.max(soft_pred, dim=-1)

        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        FP += ((pred == 1) & (labels == 0)).cpu().sum()

    # cls
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print("annotation dataset : recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(recall, precision,
                                                                                                     f1_score,
                                                                                                     accuracy))


def validate_dev_sentence(model, dev_loader, device,writer_epoch):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    writer,epoch=writer_epoch
    for (batch, (inputs, masks, labels)) in enumerate(dev_loader):
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        # rationales -- (batch_size, seq_length, 2)
        cls_logits = model.train_one_step(inputs, masks)

        soft_pred = F.softmax(cls_logits, -1)
        _, pred = torch.max(soft_pred, dim=-1)

        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        FP += ((pred == 1) & (labels == 0)).cpu().sum()

    # cls
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    print("dev dataset : recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(recall, precision,
                                                                                                f1_score, accuracy))
    writer.add_scalar('./sent_acc',accuracy,epoch)
    return f1_score,accuracy


def validate_rationales(model, annotation_loader, device,writer_epoch):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    writer,epoch=writer_epoch
    for (batch, (inputs, masks, labels,
                 annotations)) in enumerate(annotation_loader):
        inputs, masks, labels, annotations = inputs.to(device), masks.to(device), labels.to(
            device), annotations.to(
            device)

        masks = annotations
        logits = model.train_one_step(inputs, masks)

        soft_pred = F.softmax(logits, -1)
        _, pred = torch.max(soft_pred, dim=-1)

        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        FP += ((pred == 1) & (labels == 0)).cpu().sum()

    # cls
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    writer.add_scalar('rat_acc',accuracy,epoch)
    print("rationale dataset : recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(recall, precision,
                                                                                                      f1_score,
                                                                                                      accuracy))


def validate_annotation_sentence_toynet(model, annotation_loader, device):
    TP = 0
    TN = 0
    FN = 0
    FP = 0

    for (batch, (inputs, masks, labels, annotations)) in enumerate(annotation_loader):
        inputs, masks, labels, annotations = inputs.to(device), masks.to(device), labels.to(device), annotations.to(device)

        # rationales -- (batch_size, seq_length, 2)
        cls_logits = model(inputs, masks)

        soft_pred = F.softmax(cls_logits, -1)
        _, pred = torch.max(soft_pred, dim=-1)

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()

    # cls
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print("annotation dataset : recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(recall, precision,
                                                                                                     f1_score,
                                                                                                     accuracy))
    return recall, precision,f1_score,accuracy

def validate_rationales_toynet(model, annotation_loader, device,writer_epoch_args):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    writer,epoch,args=writer_epoch_args
    loss,neg_loss = 0.0,0.0
    for (batch, (inputs, masks, labels,annotations)) in enumerate(annotation_loader):
        inputs, masks, labels, annotations = inputs.to(device), masks.to(device), labels.to(device), annotations.to(device)
        masks = annotations


        logits = model(inputs, masks)
        soft_pred = F.softmax(logits, -1)
        _, pred = torch.max(soft_pred, dim=-1)
        # compute sample loss
        cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)
        loss = loss + cls_loss


        neg_logits = model(inputs, ~masks)
        neg_cls_loss = args.cls_lambda * F.cross_entropy(neg_logits, labels)
        neg_loss = neg_loss + neg_cls_loss 


        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()

    print("Test rationale Loss:%f" % loss)
    print("Test non-rationale Loss:%f" % neg_loss)

    # cls
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    writer.add_scalar('rat_acc',accuracy,epoch)
    print("rationale dataset : recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(recall, precision,
                                                                                                      f1_score,
                                                                                                      accuracy))
    return recall, precision, f1_score, accuracy



# def validate_dev_sentence_toy(model, dev_loader, device,writer_epoch):
#     TP = 0
#     TN = 0
#     FN = 0
#     FP = 0
#     writer,epoch=writer_epoch
#     for (batch, (inputs, masks, labels,gold_rationales)) in enumerate(dev_loader):
#         inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

#         # rationales -- (batch_size, seq_length, 2)
#         cls_logits = model.train_one_step(inputs, masks)

#         soft_pred = F.softmax(cls_logits, -1)
#         _, pred = torch.max(soft_pred, dim=-1)

#         TP += ((pred == 1) & (labels == 1)).cpu().sum()
#         TN += ((pred == 0) & (labels == 0)).cpu().sum()
#         FN += ((pred == 0) & (labels == 1)).cpu().sum()
#         FP += ((pred == 1) & (labels == 0)).cpu().sum()

#     # cls
#     precision = TP / (TP + FP)
#     recall = TP / (TP + FN)
#     f1_score = 2 * recall * precision / (recall + precision)
#     accuracy = (TP + TN) / (TP + TN + FP + FN)

#     print("dev dataset : recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(recall, precision,
#                                                                                                 f1_score, accuracy))
#     writer.add_scalar('./sent_acc',accuracy,epoch)
#     return f1_score,accuracy


def validate_rationales_toy(model, annotation_loader, device,writer_epoch):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    writer,epoch=writer_epoch
    for (batch, (inputs, masks, labels,annotations)) in enumerate(annotation_loader):
        inputs, masks, labels, annotations = inputs.to(device), masks.to(device), labels.to(
            device), annotations.to(device)

        masks = annotations
        logits = model.train_one_step(inputs, masks)

        soft_pred = F.softmax(logits, -1)
        _, pred = torch.max(soft_pred, dim=-1)

        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        FP += ((pred == 1) & (labels == 0)).cpu().sum()

    # cls
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    writer.add_scalar('rat_acc',accuracy,epoch)
    print("rationale dataset : recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(recall, precision,
                                                                                                      f1_score,
                                                                                                      accuracy))



'''PolicyNet '''
def validate_annotation_sentence_plolicy_net(model, annotation_loader, device):
    TP = 0
    TN = 0
    FN = 0
    FP = 0

    for (batch, (inputs, masks, labels, annotations)) in enumerate(annotation_loader):
        inputs, masks, labels, annotations = inputs.to(device), masks.to(device), labels.to(device), annotations.to(
            device)

        # rationales -- (batch_size, seq_length, 2)
        cls_logits = model(inputs, masks)

        soft_pred = F.softmax(cls_logits, -1)
        _, pred = torch.max(soft_pred, dim=-1)

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()

    # cls
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print("annotation dataset : recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(recall, precision,
                                                                                                     f1_score,
                                                                                                     accuracy))
    return recall, precision,f1_score,accuracy

def validate_rationales_plolicy_net(model, annotation_loader, device,writer_epoch):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    writer,epoch=writer_epoch
    for (batch, (inputs, masks, labels,
                 annotations)) in enumerate(annotation_loader):
        inputs, masks, labels, annotations = inputs.to(device), masks.to(device), labels.to(
            device), annotations.to(
            device)

        masks = annotations
        logits = model(inputs, masks)

        soft_pred = F.softmax(logits, -1)
        _, pred = torch.max(soft_pred, dim=-1)

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()

    # cls
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    writer.add_scalar('rat_acc',accuracy,epoch)
    print("rationale dataset : recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(recall, precision,
                                                                                                      f1_score,
                                                                                                      accuracy))
    return recall, precision, f1_score, accuracy

