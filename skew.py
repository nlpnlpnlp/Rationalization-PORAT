import argparse
import os
import time

import torch

from beer import Beer, BeerAnnotation,BeerData_ReadSkew,Beer_decorrelated
from embedding import get_glove_embedding
from torch.utils.data import DataLoader

from model import Base_RNP, ReAGR, ReDR
from train_util import train_agr,train_pagr,train_skew
from validate_util import validate_share, validate_dev_sentence, validate_annotation_sentence, validate_rationales
from tensorboardX import SummaryWriter


def parse():
    parser = argparse.ArgumentParser(description="PORAT")

    # Skew Parameter
    parser.add_argument('--skew_lr',
                        type=float,
                        default=0.001,
                        help='compliment skew learning rate [default: 0.001] (smilarlly A2R)')
    parser.add_argument('--skew',
                        type=int,
                        default=10,
                        help='Number of pre-training epoch')
    parser.add_argument('--skew_batch_size',
                        type=int,
                        default=500,
                        help='Batch size [default: 500] (smilarlly A2R)')
    
    # dataset parameters
    parser.add_argument('--data_dir',
                        type=str,
                        default='./data/beer',
                        help='Path of the dataset')
    parser.add_argument('--data_type',
                        type=str,
                        default='beer',
                        help='0:beer,1:hotel')
    parser.add_argument('--aspect',
                        type=int,
                        default=0,
                        help='The aspect number of beer review [0, 1, 2]')
    parser.add_argument('--seed',
                        type=int,
                        default=12252018,
                        help='The aspect number of beer review [20226666,12252018]')
    parser.add_argument('--annotation_path',
                        type=str,
                        default='./data/beer/annotations.json',
                        help='Path to the annotation')
    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help='Batch size [default: 100]')
    parser.add_argument('--correlated',
                        type=int,
                        default=1,
                        help='correlated dataset of decorrelated dataset of beer; hotel decorrelated')


    # model parameters
    parser.add_argument('--dis_lr',
                        type=float,
                        default=1,
                        help='0:rnp,1:dis')
    parser.add_argument('--save',
                        type=int,
                        default=0,
                        help='save model, 0:do not save, 1:save')
    parser.add_argument('--cell_type',
                        type=str,
                        default="GRU",
                        help='Cell type: LSTM, GRU [default: GRU]')
    parser.add_argument('--num_layers',
                        type=int,
                        default=1,
                        help='RNN cell layers')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.2,
                        help='Network Dropout')
    parser.add_argument('--embedding_dim',
                        type=int,
                        default=100,
                        help='Embedding dims [default: 100]')
    parser.add_argument('--hidden_dim',
                        type=int,
                        default=200,
                        help='RNN hidden dims [default: 100]')
    parser.add_argument('--num_class',
                        type=int,
                        default=2,
                        help='Number of predicted classes [default: 2]')


    # learning parameters
    parser.add_argument('--epochs',
                        type=int,
                        default=37,
                        help='Number of training epoch')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0001,
                        help='compliment learning rate [default: 1e-3]')
    parser.add_argument('--sparsity_lambda',
                        type=float,
                        default=12.,
                        help='Sparsity trade-off [default: 1.]')
    parser.add_argument('--continuity_lambda',
                        type=float,
                        default=10.,
                        help='Continuity trade-off [default: 4.]')
    parser.add_argument('--sparsity_percentage',
                        type=float,
                        default=0.1,
                        help='Regularizer to control highlight percentage [default: .2]')
    parser.add_argument('--cls_lambda',
                        type=float,
                        default=0.9,
                        help='lambda for classification loss')
    parser.add_argument('--gpu',
                        type=str,
                        default='0',
                        help='id(s) for CUDA_VISIBLE_DEVICES [default: None]')
    parser.add_argument('--share',
                        type=int,
                        default=0,
                        help='')
    
    # Agent parameters
    parser.add_argument('--num_beam',
                        type=int,
                        default=8,
                        help='num_beam')
    parser.add_argument('--topK_ratio',
                        type=float,
                        default=0.1,
                        help='topK_ratio')
    parser.add_argument('--freezing',
                        type=int,
                        default=0,
                        help='freezing')

    
    parser.add_argument('--action_K',
                        type=int,
                        default=3,
                        help='action_K')
    parser.add_argument('--pretrain_agent',
                        type=bool,
                        default=False,
                        help='pretrain_agent')
    parser.add_argument('--reward_mode',
                        type=str,
                        default="causal_effect",
                        help='reward_mode')
    parser.add_argument('--embedding_agent',
                        type=int,
                        default=1,
                        help='[0:zero,1:rand]')
    parser.add_argument('--pre_ward',
                        type=bool,
                        default=False,
                        help='pre_ward')
    
    # pretrained embeddings
    parser.add_argument('--embedding_dir',
                        type=str,
                        default='./data/hotel/embeddings',
                        help='Dir. of pretrained embeddings [default: None]')
    parser.add_argument('--embedding_name',
                        type=str,
                        default='glove.6B.100d.txt',
                        help='File name of pretrained embeddings [default: None]')
    parser.add_argument('--max_length',
                        type=int,
                        default=256,
                        help='Max sequence length [default: 256]')

    parser.add_argument('--log_date',
                        type=str,
                        default="",
                        help='log_date')
    parser.add_argument('--writer',
                        type=str,
                        default='./noname',
                        help='Regularizer to control highlight percentage [default: .2]')
    args = parser.parse_args()
    return args

args = parse()
#####################
# set seed
#####################
torch.manual_seed(args.seed)

#####################
# parse arguments
#####################
print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))
root_save_dir = '{}/trained_model/{}/'.format(args.writer, args.data_type)
if not os.path.exists(root_save_dir):
    os.makedirs(root_save_dir)

######################
# device
######################
torch.cuda.set_device(int(args.gpu))
device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
torch.cuda.manual_seed(args.seed)

######################
# load embedding
######################
pretrained_embedding, word2idx = get_glove_embedding(os.path.join(args.embedding_dir, args.embedding_name))
args.vocab_size = len(word2idx)
args.pretrained_embedding = pretrained_embedding

######################
# load dataset dataset
######################
if args.data_type=='beer':       #beer
    if args.correlated==1:
        train_data = Beer(args.data_dir, args.aspect, 'train', word2idx, balance=True)
        dev_data = Beer(args.data_dir, args.aspect, 'dev', word2idx,balance=True)
        annotation_data = BeerAnnotation(args.annotation_path, args.aspect, word2idx)
    else:
        print('decorrelated')
        train_data = Beer_decorrelated(args.data_dir, args.aspect, 'train', word2idx, balance=True)
        dev_data = Beer_decorrelated(args.data_dir, args.aspect, 'dev', word2idx,balance=True)
        annotation_data = BeerAnnotation(args.annotation_path, args.aspect, word2idx)
        
    sentence_data = BeerData_ReadSkew(args.data_dir, args.aspect, 'train', word2idx, balance=True)

# shuffle and batch the dataset
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
sentence_loader = DataLoader(sentence_data, batch_size=500, shuffle=True) # args.skew_batch_size=500

dev_loader = DataLoader(dev_data, batch_size=args.batch_size)
annotation_loader = DataLoader(annotation_data, batch_size=args.batch_size)



######################
# load model
######################
writer=SummaryWriter(args.writer)
model=ReAGR(args)
model.to(device)


if args.data_type=='beer':   
    if args.correlated==1:
        '''correlated beer'''
        if args.aspect==0:
            model_pt = torch.load("./trained_embedding/beer/pretrain_aspect0.pkl",map_location=torch.device("cuda:{}".format(args.gpu)))
            model_pt.to(device)
        elif args.aspect==1:
            model_pt = torch.load("./trained_embedding/beer/pretrain_aspect0.pkl",map_location=torch.device("cuda:{}".format(args.gpu)))
            model_pt.to(device)
        elif args.aspect==2:
            model_pt = torch.load("./trained_embedding/beer/pretrain_aspect0.pkl",map_location=torch.device("cuda:{}".format(args.gpu)))
            model_pt.to(device)
    else:
        '''decorrelated beer'''
        print("decorrelated embedding")
        if args.aspect==0:
            model_pt = torch.load("./trained_embedding/beer_decorr/pretrain_aspect0_beer_decorr.pkl",map_location=torch.device("cuda:{}".format(args.gpu)))
            model_pt.to(device)
        elif args.aspect==1:
            model_pt = torch.load("./trained_embedding/beer_decorr/pretrain_aspect1_beer_decorr.pkl",map_location=torch.device("cuda:{}".format(args.gpu)))
            model_pt.to(device)
        elif args.aspect==2:
            model_pt = torch.load("./trained_embedding/beer_decorr/pretrain_aspect2_beer_decorr.pkl",map_location=torch.device("cuda:{}".format(args.gpu)))
            model_pt.to(device)

######################
# Training
######################
g_para=list(map(id, model.generator.parameters()))
p_para=filter(lambda p: id(p) not in g_para, model.parameters())
lr2=args.lr
lr1=args.lr
para=[
    {'params': model.generator.parameters(), 'lr':lr1},
    {'params':p_para,'lr':lr2}
]
optimizer = torch.optim.Adam(para)
skew_opt = torch.optim.Adam(model.get_cls_param(), lr=0.001)  

######################
# Training
######################
strat_time=time.time()
best_all = 0
f1_best_dev = [0]
best_dev_epoch = [0]
acc_best_dev = [0]
grad=[]
grad_loss=[]

# To skew predictor
for e in range(args.skew):
    model.train()
    precision, recall, f1_score, accuracy = train_skew(model, skew_opt, sentence_loader, device, args)
    print('skew={},p={:.4f},r={:.4f},f1={:.4f},acc={:.4f}'.format(e,precision,recall,f1_score, accuracy))

for epoch in range(args.epochs):

    start = time.time()
    model.train()
    precision, recall, f1_score, accuracy = train_pagr(model, model_pt,optimizer, train_loader, device, args,(writer,epoch),grad,grad_loss,freezing=args.freezing)
    end = time.time()
    print('\nTrain time for epoch #%d : %f second' % (epoch, end - start))
    print('gen_lr={}, pred_lr={}'.format(optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']))
    print("traning epoch:{} recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(epoch, recall,precision, f1_score,accuracy))
    writer.add_scalar('train_acc',accuracy,epoch)
    writer.add_scalar('time',time.time()-strat_time,epoch)
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    model.eval()
    print("Validate")
    with torch.no_grad():
        for (batch, (inputs, masks, labels)) in enumerate(dev_loader):
            inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
            _, __, logits = model(inputs, masks)
            # pdb.set_trace()
            logits = torch.softmax(logits, dim=-1)
            _, pred = torch.max(logits, axis=-1)
            # compute accuarcy
            TP += ((pred == 1) & (labels == 1)).cpu().sum()
            TN += ((pred == 0) & (labels == 0)).cpu().sum()
            FN += ((pred == 0) & (labels == 1)).cpu().sum()
            FP += ((pred == 1) & (labels == 0)).cpu().sum()
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * precision * recall / (recall + precision)
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        print("dev epoch:{} recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(epoch, recall,
                                                                                                   precision,
                                                                                                   f1_score, accuracy))

        writer.add_scalar('dev_acc',accuracy,epoch)
        print("Validate Sentence")
        validate_dev_sentence(model, dev_loader, device,(writer,epoch))
        print("Annotation")
        annotation_results = validate_share(model, annotation_loader, device)
        print(
            "The annotation performance: sparsity: %.4f, precision: %.4f, recall: %.4f, f1: %.4f"
            % (100 * annotation_results[0], 100 * annotation_results[1],
               100 * annotation_results[2], 100 * annotation_results[3]))
        writer.add_scalar('f1',100 * annotation_results[3],epoch)
        writer.add_scalar('sparsity',100 * annotation_results[0],epoch)
        writer.add_scalar('p', 100 * annotation_results[1], epoch)
        writer.add_scalar('r', 100 * annotation_results[2], epoch)
        print("Annotation Sentence")
        validate_annotation_sentence(model, annotation_loader, device)
        print("Rationale")
        validate_rationales(model, annotation_loader, device,(writer,epoch))
        if accuracy>acc_best_dev[-1]:
            acc_best_dev.append(accuracy)
            best_dev_epoch.append(epoch)
            f1_best_dev.append(annotation_results[3])
        if best_all<annotation_results[3]:
            best_all=annotation_results[3]
print(best_all)
print(acc_best_dev)
print(best_dev_epoch)
print(f1_best_dev)

if args.save==1:
    save_path = str(root_save_dir) + '//aspect{}_dis{}.pkl'.format(args.aspect,args.dis_lr)
    if args.data_type=='beer':
        torch.save(model.state_dict(),save_path)
        print('save the model')
    elif args.data_type=='hotel':
        torch.save(model.state_dict(),save_path)
        print('save the model')
else:
    print('not save')

