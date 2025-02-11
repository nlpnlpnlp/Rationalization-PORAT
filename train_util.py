import torch
import torch.nn.functional as F
from metric import get_sparsity_loss, get_continuity_loss
import numpy as np
EPS = 1e-15


def train_porat(model, model_pt, optimizer, dataset, device, args,writer_epoch,grad,grad_loss,freezing=0):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    train_sp = []
    batch_len=len(dataset)

    previous_baseline_list = []
    current_baseline_list = []
    avg_reward = []

    reward_mode = args.reward_mode

    epoch_loss,epoch_cls_loss,epoch_spa_loss,epoch_con_loss,epoch_rl_loss = 0.0,0.0,0.0,0.0,0.0

    for (batch, (inputs, masks, labels)) in enumerate(dataset):
        ##
        optimizer.zero_grad()
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        batch_loss = 0.
        topK_ratio = args.topK_ratio
        if topK_ratio == 0:
            valid_budget = 0
        elif topK_ratio < 1:
            valid_budget = max(int(topK_ratio * torch.sum(masks==1).item() / torch.numel(masks)), 1)
        else:
            valid_budget = topK_ratio


        gen_logits, rationales, logits = model(inputs, masks, train_flag=False)
        init_masks = torch.argmax(gen_logits, dim=2)  # (batch_size, sequence_length)

        full_logits=0.
        new_logits=0.
        if(args.pretrain_agent):
            with torch.no_grad():
                full_sub_input_pred = F.softmax(model_pt(inputs, init_masks), dim=-1)  # shape -- (batch_size, num_classes)
        else:   
            if(args.embedding_agent==0):
                zeros_like_logits = torch.zeros_like(logits)
                full_logits = zeros_like_logits
                with torch.no_grad():
                    full_sub_input_pred = F.softmax(zeros_like_logits, dim=-1)
            elif args.embedding_agent==1:
                rand_logits = torch.rand_like(logits)
                full_logits = rand_logits
                with torch.no_grad():
                    full_sub_input_pred = F.softmax(rand_logits, dim=-1)


        masks_state = ~(masks == 0).to(torch.bool)  
        init_state = torch.zeros_like(init_masks, dtype=torch.bool)  
        # current_state = torch.tensor(init_masks, dtype=torch.bool)   
        current_state = init_masks.clone().detach().to(torch.bool)

        pre_reward = torch.zeros(labels.size()).to(device)
        num_beam = args.num_beam  # 8

        for budget in range(valid_budget):
            available_action = ~current_state.clone() 
            new_state = init_state.clone()

            beam_reward_list = []
            beam_action_list = []
            beam_action_probs_list = []

            trained_masks = masks
            for beam in range(num_beam):
                beam_available_action = ~current_state.clone() #& masks_state  # (batch_size, sequence_length)
                beam_new_state = current_state.clone()

                if beam == 0:
                    gen_logits, rationales, logits, ava_action_probs, added_action_probs, added_actions \
                        = model(inputs, masks, state=current_state, labels=labels,k=args.action_K,
                                                                 available_action=beam_available_action, train_flag=False)
                else:
                    gen_logits, rationales, logits, ava_action_probs, added_action_probs, added_actions \
                        = model(inputs, masks, state=current_state, labels=labels,k=args.action_K,
                                                                 available_action=beam_available_action, train_flag=True)

                beam_inavailable_action = ~beam_available_action 
                batch_indices = torch.arange(beam_inavailable_action.size(0)).view(-1, 1)
                beam_inavailable_action[batch_indices, added_actions] = 1
                beam_available_action = ~beam_inavailable_action # mini update

                beam_new_state = beam_inavailable_action  
                beam_new_state = beam_new_state  & masks_state

                beam_new_masks = torch.where(beam_new_state, torch.tensor(1).to(device), torch.tensor(0).to(device))

                if (args.pretrain_agent):
                    with torch.no_grad():
                        new_sub_input_pred = model_pt(inputs, beam_new_masks)
                        new_sub_input_pred = F.softmax(new_sub_input_pred, dim=-1)
                else:
                    if(args.embedding_agent==0):
                        zeros_like_logits = torch.zeros_like(logits)
                        new_logits = zeros_like_logits
                        with torch.no_grad():
                            new_sub_input_pred = F.softmax(zeros_like_logits, dim=-1)
                    elif args.embedding_agent==1:
                        rand_logits = torch.rand_like(logits)
                        new_logits = rand_logits
                        with torch.no_grad():
                            new_sub_input_pred = F.softmax(rand_logits, dim=-1)

                reward = get_reward(full_sub_input_pred, new_sub_input_pred, labels, pre_reward=pre_reward,mode=reward_mode,full_logits=full_logits,new_logits=new_logits)

                if len(previous_baseline_list) - 1 < budget:
                    baseline_reward = 0.
                else:
                    baseline_reward = previous_baseline_list[budget]

                if len(current_baseline_list) - 1 < budget:
                    current_baseline_list.append([torch.mean(reward)])   
                else:
                    current_baseline_list[budget].append(torch.mean(reward))

                if(args.pre_ward):
                    reward -= baseline_reward


                avg_reward += reward.tolist()
                beam_reward_list.append(reward)
                beam_action_list.append(added_actions)
                beam_action_probs_list.append(added_action_probs)

            beam_reward_list = torch.stack(beam_reward_list).T
            beam_action_list = torch.stack(beam_action_list).T
            beam_action_probs_list1 = torch.stack(beam_action_probs_list).T  
            beam_action_probs_list = F.softmax(beam_action_probs_list1, dim=1)  

            batch_loss += torch.mean(-torch.log(beam_action_probs_list + EPS) * beam_reward_list)

            max_reward, max_reward_idx = torch.max(beam_reward_list, dim=1)  
            max_actions = torch.gather(beam_action_list, 2,max_reward_idx.unsqueeze(0).unsqueeze(-1).expand(beam_action_list.size(0), -1, -1))

            inavailable_action = ~available_action  
            batch_indices_ = torch.arange(inavailable_action.size(0)).view(-1, 1)
            max_actions = torch.transpose(torch.squeeze(max_actions, dim=-1), 0, 1)
            inavailable_action[batch_indices_, max_actions] = 1
            available_action = ~inavailable_action 

            new_state = ~available_action
            current_state = new_state.clone()

        previous_baseline_list = [torch.mean(torch.stack(cur_baseline)) for cur_baseline in current_baseline_list]
        current_baseline_list = []
        cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)
        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(rationales[:, :, 1], masks, args.sparsity_percentage)
        sparsity = (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item()
        train_sp.append((torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item())
        continuity_loss = args.continuity_lambda * get_continuity_loss(rationales[:, :, 1])
        loss = cls_loss + sparsity_loss + continuity_loss + batch_loss

        epoch_loss = loss + epoch_loss
        epoch_cls_loss = cls_loss + epoch_cls_loss
        epoch_spa_loss = sparsity_loss + epoch_spa_loss
        epoch_con_loss = continuity_loss + epoch_con_loss
        epoch_rl_loss = batch_loss + epoch_rl_loss

        if topK_ratio == 0:
            lr_alphi = 1
        else:
            lr_alphi = 1 if sparsity==0 else sparsity
            lr_alphi = 0.05 if lr_alphi < 0.05 else lr_alphi
        
        optimizer.param_groups[1]['lr'] = optimizer.param_groups[0]['lr'] * lr_alphi

        loss.backward()
        optimizer.step()

        ###
        if(freezing==0):
            gen_list=[]
            for idx,p in model.generator.named_parameters():
                gen_list.append(idx) if p.requires_grad==True else None
                p.requires_grad = False if p.requires_grad==True else False                    
        elif(freezing==1): 
            cls_list=[]
            cls_fc_list=[]
            for idx,p in model.cls.named_parameters():
                cls_list.append(idx) if p.requires_grad==True else None
                p.requires_grad = False if p.requires_grad==True else False
            for idx,p in model.cls_fc.named_parameters():
                cls_fc_list.append(idx) if p.requires_grad==True else None
                p.requires_grad = False if p.requires_grad==True else False
        elif(freezing==2):
            pass
        else:
            break

        optimizer.zero_grad()
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        batch_loss = 0.
        topK_ratio = args.topK_ratio
        if topK_ratio == 0:
            valid_budget = 0
        elif topK_ratio < 1:
            valid_budget = max(int(topK_ratio * torch.sum(masks==1).item() / torch.numel(masks)), 1)
        else:
            valid_budget = topK_ratio


        gen_logits, rationales, logits = model(inputs, masks, train_flag=False)
        init_masks = torch.argmax(gen_logits, dim=2)  # (batch_size, sequence_length)

        full_logits=0.
        new_logits=0.
        if(args.pretrain_agent):
            with torch.no_grad():
                full_sub_input_pred = F.softmax(model_pt(inputs, init_masks), dim=-1)  # shape -- (batch_size, num_classes)
        else:   
            if(args.embedding_agent==0):
                zeros_like_logits = torch.zeros_like(logits)
                full_logits = zeros_like_logits
                with torch.no_grad():
                    full_sub_input_pred = F.softmax(zeros_like_logits, dim=-1)
            elif args.embedding_agent==1:
                rand_logits = torch.rand_like(logits)
                full_logits = rand_logits
                with torch.no_grad():
                    full_sub_input_pred = F.softmax(rand_logits, dim=-1)


        masks_state = ~(masks == 0).to(torch.bool)  
        init_state = torch.zeros_like(init_masks, dtype=torch.bool)  
        # current_state = torch.tensor(init_masks, dtype=torch.bool)   
        current_state = init_masks.clone().detach().to(torch.bool)

        pre_reward = torch.zeros(labels.size()).to(device)
        num_beam = args.num_beam  # 8

        for budget in range(valid_budget):
            available_action = ~current_state.clone() 
            new_state = init_state.clone()

            beam_reward_list = []
            beam_action_list = []
            beam_action_probs_list = []

            trained_masks = masks
            for beam in range(num_beam):
                beam_available_action = ~current_state.clone() #& masks_state  # (batch_size, sequence_length)
                beam_new_state = current_state.clone()

                if beam == 0:
                    gen_logits, rationales, logits, ava_action_probs, added_action_probs, added_actions \
                        = model(inputs, masks, state=current_state, labels=labels,k=args.action_K,
                                                                 available_action=beam_available_action, train_flag=False)
                else:
                    gen_logits, rationales, logits, ava_action_probs, added_action_probs, added_actions \
                        = model(inputs, masks, state=current_state, labels=labels,k=args.action_K,
                                                                 available_action=beam_available_action, train_flag=True)

                beam_inavailable_action = ~beam_available_action 
                batch_indices = torch.arange(beam_inavailable_action.size(0)).view(-1, 1)
                beam_inavailable_action[batch_indices, added_actions] = 1
                beam_available_action = ~beam_inavailable_action # mini update

                beam_new_state = beam_inavailable_action  
                beam_new_state = beam_new_state  & masks_state

                beam_new_masks = torch.where(beam_new_state, torch.tensor(1).to(device), torch.tensor(0).to(device))

                if (args.pretrain_agent):
                    with torch.no_grad():
                        new_sub_input_pred = model_pt(inputs, beam_new_masks)
                        new_sub_input_pred = F.softmax(new_sub_input_pred, dim=-1)
                else:
                    if(args.embedding_agent==0):
                        zeros_like_logits = torch.zeros_like(logits)
                        new_logits = zeros_like_logits
                        with torch.no_grad():
                            new_sub_input_pred = F.softmax(zeros_like_logits, dim=-1)
                    elif args.embedding_agent==1:
                        rand_logits = torch.rand_like(logits)
                        new_logits = rand_logits
                        with torch.no_grad():
                            new_sub_input_pred = F.softmax(rand_logits, dim=-1)

                reward = get_reward(full_sub_input_pred, new_sub_input_pred, labels, pre_reward=pre_reward,mode=reward_mode,full_logits=full_logits,new_logits=new_logits)

                if len(previous_baseline_list) - 1 < budget:
                    baseline_reward = 0.
                else:
                    baseline_reward = previous_baseline_list[budget]

                if len(current_baseline_list) - 1 < budget:
                    current_baseline_list.append([torch.mean(reward)])   
                else:
                    current_baseline_list[budget].append(torch.mean(reward))

                if(args.pre_ward):
                    reward -= baseline_reward


                avg_reward += reward.tolist()
                beam_reward_list.append(reward)
                beam_action_list.append(added_actions)
                beam_action_probs_list.append(added_action_probs)

            beam_reward_list = torch.stack(beam_reward_list).T
            beam_action_list = torch.stack(beam_action_list).T
            beam_action_probs_list1 = torch.stack(beam_action_probs_list).T  
            beam_action_probs_list = F.softmax(beam_action_probs_list1, dim=1)  

            batch_loss += torch.mean(-torch.log(beam_action_probs_list + EPS) * beam_reward_list)

            max_reward, max_reward_idx = torch.max(beam_reward_list, dim=1)  
            max_actions = torch.gather(beam_action_list, 2,max_reward_idx.unsqueeze(0).unsqueeze(-1).expand(beam_action_list.size(0), -1, -1))

            inavailable_action = ~available_action  
            batch_indices_ = torch.arange(inavailable_action.size(0)).view(-1, 1)
            max_actions = torch.transpose(torch.squeeze(max_actions, dim=-1), 0, 1)
            inavailable_action[batch_indices_, max_actions] = 1
            available_action = ~inavailable_action 

            new_state = ~available_action
            current_state = new_state.clone()

        previous_baseline_list = [torch.mean(torch.stack(cur_baseline)) for cur_baseline in current_baseline_list]
        current_baseline_list = []
        cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)
        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(rationales[:, :, 1], masks, args.sparsity_percentage)
        sparsity = (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item()
        train_sp.append((torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item())
        continuity_loss = args.continuity_lambda * get_continuity_loss(rationales[:, :, 1])
        loss = cls_loss + sparsity_loss + continuity_loss + batch_loss

        epoch_loss = loss + epoch_loss
        epoch_cls_loss = cls_loss + epoch_cls_loss
        epoch_spa_loss = sparsity_loss + epoch_spa_loss
        epoch_con_loss = continuity_loss + epoch_con_loss
        epoch_rl_loss = batch_loss + epoch_rl_loss

        if topK_ratio == 0:
            lr_alphi = 1
        else:
            lr_alphi = 1 if sparsity==0 else sparsity
            lr_alphi = 0.05 if lr_alphi < 0.05 else lr_alphi
        
        optimizer.param_groups[1]['lr'] = optimizer.param_groups[0]['lr'] * lr_alphi

        loss.backward()
        optimizer.step()

        if(freezing==0):
            for idx,p in model.generator.named_parameters():
                if idx in gen_list:
                    p.requires_grad = True
        elif(freezing==1):
            for idx,p in model.cls.named_parameters():
                if idx in cls_list:
                    p.requires_grad = True
            for idx,p in model.cls_fc.named_parameters():
                if idx in cls_fc_list:
                    p.requires_grad = True
        elif(freezing==2):
            pass
        else:
            break

        ###
        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        FP += ((pred == 1) & (labels == 0)).cpu().sum()
        cls_l += cls_loss.cpu().item()
        spar_l += sparsity_loss.cpu().item()
        cont_l += continuity_loss.cpu().item()

    avg_reward = torch.mean(torch.FloatTensor(avg_reward))
    print('Episode: %d, loss: %.4f, cls loss: %.4f, spa loss: %.4f, con loss: %.4f, rl loss: %.4f, avg_reward: %.4f' % (writer_epoch[1], epoch_loss.detach(), epoch_cls_loss.detach(), epoch_spa_loss.detach(), epoch_con_loss.detach(), epoch_rl_loss,avg_reward.detach()))

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return precision, recall, f1_score, accuracy

def relabel_rationale(input, selection):

    import copy
    rationale = input.clone()
    padding_length = rationale.size(1) - selection.size(1)

    rationale_logits = torch.softmax(rationale, dim=-1)
    _, rationale_pred = torch.max(rationale_logits, axis=-1)
    boolean_pred = rationale_pred == 1  

    padded_selection = F.pad(selection, (0, padding_length))
    result = torch.logical_or(boolean_pred, padded_selection)
    return result.int()

def relabel_rationale_masks(beam_new_state):
    return torch.where(beam_new_state, torch.tensor(1), torch.tensor(0))

def get_reward(full_sub_input_pred, new_sub_input_pred, target_y, pre_reward, mode='mutual_info',full_logits=None,new_logits=None):
    if mode in ['mutual_info']:
        reward = torch.sum(full_sub_input_pred * torch.log(new_sub_input_pred + EPS), dim=1)
        reward += 2 * (target_y == new_sub_input_pred.argmax(dim=1)).float() - 1.

    elif mode in ['causal_effect']:
        full_sub_input_pred_clamped = torch.clamp(full_sub_input_pred, EPS, 1.0)
        new_sub_input_pred_clamped = torch.clamp(new_sub_input_pred, EPS, 1.0)
        reward = -torch.sum(full_sub_input_pred_clamped * torch.log1p(full_sub_input_pred_clamped / new_sub_input_pred_clamped), dim=1)
        reward += 2 * (target_y == new_sub_input_pred.argmax(dim=1)).float() - 1

    reward += 0.9 * pre_reward
    return reward

def train_skew(model, optimizer, dataset, device, args):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    for (batch, (inputs, masks, labels)) in enumerate(dataset):
        optimizer.zero_grad()
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
        logits=model.train_skew(inputs,masks,labels)
        cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)
        cls_loss.backward()
        optimizer.step()
        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return precision, recall, f1_score, accuracy

def train_base_rnp(model, optimizer, dataset, device, args,writer_epoch):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    train_sp = []
    batch_len=len(dataset)
    for (batch, (inputs, masks, labels)) in enumerate(dataset):
        # return self.input_ids[i], self.masks[i], self.labels[i], self.rationales[i]
        optimizer.zero_grad()
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        # rationales, cls_logits
        rationales, logits = model(inputs, masks)

        # computer loss
        cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)
        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(rationales[:, :, 1], masks, args.sparsity_percentage)

        sparsity = (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item()
        train_sp.append((torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item())
        continuity_loss = args.continuity_lambda * get_continuity_loss(rationales[:, :, 1])

        loss = cls_loss + sparsity_loss + continuity_loss
        # update gradient

        loss.backward()
        optimizer.step()


        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()
        cls_l += cls_loss.cpu().item()
        spar_l += sparsity_loss.cpu().item()
        cont_l += continuity_loss.cpu().item()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    writer_epoch[0].add_scalar('cls', cls_l, writer_epoch[1])
    writer_epoch[0].add_scalar('spar_l', spar_l, writer_epoch[1])
    writer_epoch[0].add_scalar('cont_l', cont_l, writer_epoch[1])
    writer_epoch[0].add_scalar('train_sp', np.mean(train_sp), writer_epoch[1])


    return precision, recall, f1_score, accuracy
