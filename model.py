import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear,  ELU
from torch.nn.utils import spectral_norm


class Rnn(nn.Module):
    def __init__(self, cell_type, embedding_dim, hidden_dim, num_layers):
        super(Rnn, self).__init__()
        if cell_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=embedding_dim,
                               hidden_size=hidden_dim // 2,
                               num_layers=num_layers,
                               batch_first=True,
                               bidirectional=True)
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(input_size=embedding_dim,
                              hidden_size=hidden_dim // 2,
                              num_layers=num_layers,
                              batch_first=True,
                              bidirectional=True)
        else:
            raise NotImplementedError('cell_type {} is not implemented'.format(cell_type))

    def forward(self, x):
        """
        Inputs:
        x - - (batch_size, seq_length, input_dim)
        Outputs:
        h - - bidirectional(batch_size, seq_length, hidden_dim)
        """
        h = self.rnn(x)
        return h

class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]

class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pretrained_embedding=None):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embedding is not None:
            self.embedding.weight.data = torch.from_numpy(pretrained_embedding)
        self.embedding.weight.requires_grad = False

    def forward(self, x):
        """
        Inputs:
        x -- (batch_size, seq_length)
        Outputs
        shape -- (batch_size, seq_length, embedding_dim)
        """
        return self.embedding(x)

class Base_RNP(nn.Module):
    def __init__(self, args):
        super(Base_RNP, self).__init__()
        self.args = args
        self.embedding_layer = Embedding(args.vocab_size,
                                         args.embedding_dim,
                                         args.pretrained_embedding)
        self.gen = nn.GRU(input_size=args.embedding_dim,
                                  hidden_size=args.hidden_dim // 2,
                                  num_layers=args.num_layers,
                                  batch_first=True,
                                  bidirectional=True)
        self.cls = nn.GRU(input_size=args.embedding_dim,
                          hidden_size=args.hidden_dim // 2,
                          num_layers=args.num_layers,
                          batch_first=True,
                          bidirectional=True)


        self.cls_fc = nn.Linear(args.hidden_dim, args.num_class)

        self.z_dim = 2
        self.gen_fc = nn.Linear(args.hidden_dim, self.z_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.generator=nn.Sequential(self.gen,
                                     SelectItem(0),
                                     self.dropout,
                                     self.gen_fc)

    def _independent_soft_sampling(self, rationale_logits):
        """
        Use the hidden states at all time to sample whether each word is a rationale or not.
        No dependency between actions. Return the sampled (soft) rationale mask.
        Outputs:
                z -- (batch_size, sequence_length, 2)
        """
        z = torch.softmax(rationale_logits, dim=-1)

        return z

    def independent_straight_through_sampling(self, rationale_logits):
        """
        Straight through sampling.
        Outputs:
            z -- shape (batch_size, sequence_length, 2)
        """
        z = F.gumbel_softmax(rationale_logits, tau=1, hard=True)
        return z

    def forward(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)

        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  
        gen_output, _ = self.gen(embedding)  
        gen_logits = self.gen_fc(self.dropout(gen_output))  
        ########## Sample ##########
        z = self.independent_straight_through_sampling(gen_logits)  
        cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  
        cls_outputs, _ = self.cls(cls_embedding)  
        cls_outputs = cls_outputs * masks_ + (1. - masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return z, cls_logits

    def train_one_step(self, inputs, masks):    
        masks_ = masks.unsqueeze(-1)
        # (batch_size, seq_length, embedding_dim)
        embedding = masks_ * self.embedding_layer(inputs)
        outputs, _ = self.cls(embedding)  
        outputs = outputs * masks_ + (1. - masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        outputs = torch.transpose(outputs, 1, 2)
        outputs, _ = torch.max(outputs, axis=2)
        logits = self.cls_fc(self.dropout(outputs))
        return logits


    def get_rationale(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)
        embedding = masks_ * self.embedding_layer(inputs)  
        gen_logits = self.generator(embedding)
        z = self.independent_straight_through_sampling(gen_logits)  
        return z


    def g_skew(self,inputs, masks):
        masks_ = masks.unsqueeze(-1)
        embedding = masks_ * self.embedding_layer(inputs) 
        gen_output, _ = self.gen(embedding)  
        gen_output = self.layernorm1(gen_output)
        gen_logits = self.gen_fc(self.dropout(gen_output))  
        soft_log=self._independent_soft_sampling(gen_logits)
        return soft_log

class ReAGR(nn.Module):       
    def __init__(self, args):
        super(ReAGR, self).__init__()
        self.lay=True
        self.args = args
        self.embedding_layer = Embedding(args.vocab_size,
                                         args.embedding_dim,
                                         args.pretrained_embedding)
        self.gen = nn.GRU(input_size=args.embedding_dim,
                                  hidden_size=args.hidden_dim // 2,
                                  num_layers=args.num_layers,
                                  batch_first=True,
                                  bidirectional=True)
        self.cls = nn.GRU(input_size=args.embedding_dim,
                          hidden_size=args.hidden_dim // 2,
                          num_layers=args.num_layers,
                          batch_first=True,
                          bidirectional=True)
        self.cls_fc = nn.Linear(args.hidden_dim, args.num_class)
        self.z_dim = 2
        self.gen_fc = nn.Linear(args.hidden_dim, self.z_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.layernorm1 = nn.LayerNorm(args.hidden_dim)
        self.layernorm2 = nn.LayerNorm(args.hidden_dim)
        self.generator=nn.Sequential(self.gen,
                                     SelectItem(0),
                                     self.layernorm1,
                                     self.dropout,
                                     self.gen_fc)
        

        self.hidden_size = args.hidden_dim
        self.num_labels = args.num_class
        self.edge_action_prob_generator = self.build_action_prob_generator()
        self.enc = Rnn(args.cell_type,
                       args.embedding_dim,
                       args.hidden_dim,
                       args.num_layers)


    def forward(self, inputs, masks, state=None, available_action=None, labels=None, train_flag=False,k=3):
        masks_ = masks.unsqueeze(-1)
        embedding = masks_ * self.embedding_layer(inputs)  
        gen_logits=self.generator(embedding)
        z = self.independent_straight_through_sampling(gen_logits)  
        cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  
        cls_outputs, _ = self.cls(cls_embedding)  
        cls_outputs = cls_outputs * masks_ + (1. - masks_) * (-1e6)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))

        # ********** find next action **************
        if state is not None or available_action:
            zk = torch.argmax(z, dim=2)  # (batch_size, seq_length)  
            Zk_1 = state.clone()  # (batch_size, seq_length)  current_state
            beam_available_action = available_action.clone()  # (batch_size, seq_length)
            Z = zk | Zk_1

            reshaped_Z = Z.unsqueeze(2)   
            reshaped_zk = zk.unsqueeze(2)  
            reshaped_beam_available_action = beam_available_action.unsqueeze(2) 

            Z_embedding, _ = self.enc(reshaped_Z * self.embedding_layer(inputs))  
            Zk_embedding, _ = self.enc(reshaped_zk * self.embedding_layer(inputs)) 
            ava_action_embedding, _ = self.enc(reshaped_beam_available_action * self.embedding_layer(inputs))  

            ava_action_probs = self.predict_star(Z_embedding, Zk_embedding, ava_action_embedding, labels)  
            added_action_probs, added_actions = torch.topk(torch.softmax(ava_action_probs, dim=1), k)  
            if train_flag:
                device = torch.device("cuda:{}".format(self.args.gpu) if torch.cuda.is_available() else "cpu")
                rand_action_probs = torch.rand(ava_action_probs.size()).to(device)
                rand_action_probs, rand_actions = torch.topk(torch.softmax(rand_action_probs, dim=1),k) 
                return gen_logits, z, cls_logits, ava_action_probs, rand_action_probs, rand_actions
            return gen_logits, z, cls_logits, ava_action_probs, added_action_probs, added_actions

        return gen_logits, z, cls_logits
    

    def predict_star(self, z_rep, sub_z_rep, ava_action_reps, target_y):
        action_z_reps = z_rep - sub_z_rep  
        action_z_reps = torch.cat([ava_action_reps, action_z_reps], dim=2) 

        action_probs = []
        for i_explainer in self.edge_action_prob_generator:
            i_action_probs = i_explainer(action_z_reps)   
            action_probs.append(i_action_probs)
        action_probs = torch.cat(action_probs, dim=2)  

        ''' aggregation '''
        batch_size, seq_length, class_num = action_probs.size()
        indices = target_y.view(batch_size, 1, 1).expand(batch_size, seq_length, class_num)
        new_action_probs = action_probs.gather(2, indices)
        new_action_probs = new_action_probs[:, :, 0]   # (batch_size, seq_length,)

        return new_action_probs

    def build_action_prob_generator(self):
        device = torch.device("cuda:{}".format(self.args.gpu) if torch.cuda.is_available() else "cpu")
        edge_action_prob_generator = nn.ModuleList()
        for i in range(self.num_labels):
            i_explainer = Sequential(
                Linear(self.hidden_size * 2, self.hidden_size * 2),
                ELU(),
                Linear(self.hidden_size * 2, self.hidden_size),
                ELU(),
                Linear(self.hidden_size, 1)
            ).to(device)
            edge_action_prob_generator.append(i_explainer)

        return edge_action_prob_generator
    
    def _independent_soft_sampling(self, rationale_logits):
        """
        Use the hidden states at all time to sample whether each word is a rationale or not.
        No dependency between actions. Return the sampled (soft) rationale mask.
        Outputs:
                z -- (batch_size, sequence_length, 2)
        """
        z = torch.softmax(rationale_logits, dim=-1)

        return z

    def independent_straight_through_sampling(self, rationale_logits):
        """
        Straight through sampling.
        Outputs:
            z -- shape (batch_size, sequence_length, 2)
        """
        z = self._independent_soft_sampling(rationale_logits)
        z = F.gumbel_softmax(rationale_logits, tau=1, hard=True)
        return z

    def train_one_step(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)
        embedding = masks_ * self.embedding_layer(inputs)
        outputs, _ = self.cls(embedding)  
        outputs = outputs * masks_ + (1. - masks_) * (-1e6)
        outputs = torch.transpose(outputs, 1, 2)
        outputs, _ = torch.max(outputs, axis=2)
        logits = self.cls_fc(self.dropout(outputs))
        return logits

    def grad(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)

        embedding = masks_ * self.embedding_layer(inputs) 
        gen_logits=self.generator(embedding)

        z = self.independent_straight_through_sampling(gen_logits)  
        embedding2=embedding.clone().detach()
        embedding2.requires_grad=True
        cls_embedding =embedding2  * (z[:, :, 1].unsqueeze(-1))  
        cls_outputs, _ = self.cls(cls_embedding)  
        cls_outputs = cls_outputs * masks_ + (1. - masks_) * (-1e6)

        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return z, cls_logits,embedding2,cls_embedding


    def g_skew(self,inputs, masks):
        #  masks_ (batch_size, seq_length, 1)
        masks_ = masks.unsqueeze(-1)
        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        gen_output = self.layernorm1(gen_output)
        gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        soft_log=self._independent_soft_sampling(gen_logits)
        return soft_log

    
    def train_skew(self,inputs,masks,labels):
        masks_ = masks.unsqueeze(-1)

        labels_=labels.detach().unsqueeze(-1)       #batch*1
        pos=torch.ones_like(inputs)[:,:10]*labels_
        neg=-pos+1
        skew_pad=torch.cat((pos,neg),dim=1)
        latter=torch.zeros_like(inputs)[:,20:]

        masks_=torch.cat((skew_pad,latter),dim=1).unsqueeze(-1)
        # (batch_size, seq_length, embedding_dim)

        embedding = masks_ * self.embedding_layer(inputs)
        outputs, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        outputs = self.layernorm1(outputs)
        outputs = outputs * masks_ + (1. - masks_) * (-1e6)

        # (batch_size, hidden_dim, seq_length)
        outputs = torch.transpose(outputs, 1, 2)
        outputs, _ = torch.max(outputs, axis=2)
        # shape -- (batch_size, num_classes)
        logits = self.cls_fc(self.dropout(outputs))
        return logits

    def get_cls_param(self):
        # layers = [self.gen, self.layernorm1, self.cls_fc]
        layers = [self.cls, self.cls_fc]
        params = []
        for layer in layers:
            params.extend([param for param in layer.parameters() if param.requires_grad])
        return params
    
    def get_gen_param(self):
        layers = [self.gen_fc]
        layers = [self.embedding_layer, self.gen, self.layernorm1, self.gen_fc]
        params = []
        for layer in layers:
            params.extend([param for param in layer.parameters() if param.requires_grad])
        return params

class ReDR(nn.Module):       
    def __init__(self, args):
        ''' Need to control the asymmetric learning rates between two players'''
        super(ReDR, self).__init__()
        self.lay=True
        self.args = args
        self.embedding_layer = Embedding(args.vocab_size,
                                         args.embedding_dim,
                                         args.pretrained_embedding)
        self.gen = nn.GRU(input_size=args.embedding_dim,
                                  hidden_size=args.hidden_dim // 2,
                                  num_layers=args.num_layers,
                                  batch_first=True,
                                  bidirectional=True)
        self.cls = nn.GRU(input_size=args.embedding_dim,
                          hidden_size=args.hidden_dim // 2,
                          num_layers=args.num_layers,
                          batch_first=True,
                          bidirectional=True)
        if args.sp_norm==1:
            self.cls_fc = spectral_norm(nn.Linear(args.hidden_dim, args.num_class))
        elif args.sp_norm==0:
            self.cls_fc = nn.Linear(args.hidden_dim, args.num_class)
        else:
            print('wrong norm')
        self.z_dim = 2
        self.gen_fc = nn.Linear(args.hidden_dim, self.z_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.layernorm1 = nn.LayerNorm(args.hidden_dim)
        self.layernorm2 = nn.LayerNorm(args.hidden_dim)
        self.generator=nn.Sequential(self.gen,
                                     SelectItem(0),
                                     self.layernorm1,
                                     self.dropout,
                                     self.gen_fc)

    def _independent_soft_sampling(self, rationale_logits):
        """
        Use the hidden states at all time to sample whether each word is a rationale or not.
        No dependency between actions. Return the sampled (soft) rationale mask.
        Outputs:
                z -- (batch_size, sequence_length, 2)
        """
        z = torch.softmax(rationale_logits, dim=-1)

        return z

    def independent_straight_through_sampling(self, rationale_logits):
        """
        Straight through sampling.
        Outputs:
            z -- shape (batch_size, sequence_length, 2)
        """
        z = self._independent_soft_sampling(rationale_logits)
        z = F.gumbel_softmax(rationale_logits, tau=1, hard=True)
        return z

    def forward(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)

        embedding = masks_ * self.embedding_layer(inputs)  
        gen_logits=self.generator(embedding)
        z = self.independent_straight_through_sampling(gen_logits)  
        cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1)) 
        cls_outputs, _ = self.cls(cls_embedding)  
        cls_outputs = cls_outputs * masks_ + (1. - masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return z, cls_logits

    def train_one_step(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)
        # (batch_size, seq_length, embedding_dim)
        embedding = masks_ * self.embedding_layer(inputs)
        outputs, _ = self.cls(embedding)  # (batch_size, seq_length, hidden_dim)
        outputs = outputs * masks_ + (1. - masks_) * (-1e6)
        outputs = torch.transpose(outputs, 1, 2)
        outputs, _ = torch.max(outputs, axis=2)
        # shape -- (batch_size, num_classes)
        logits = self.cls_fc(self.dropout(outputs))
        return logits

    def grad(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)

        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        gen_logits=self.generator(embedding)
        z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)
        embedding2=embedding.clone().detach()
        embedding2.requires_grad=True
        cls_embedding =embedding2  * (z[:, :, 1].unsqueeze(-1))  # (batch_size, seq_length, embedding_dim)
        cls_outputs, _ = self.cls(cls_embedding)  # (batch_size, seq_length, hidden_dim)
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return z, cls_logits,embedding2,cls_embedding

    def g_skew(self,inputs, masks):
        masks_ = masks.unsqueeze(-1)
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        gen_output = self.layernorm1(gen_output)
        gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        soft_log=self._independent_soft_sampling(gen_logits)
        return soft_log

    
    def train_skew(self,inputs,masks,labels):
        masks_ = masks.unsqueeze(-1)

        labels_=labels.detach().unsqueeze(-1)       #batch*1
        pos=torch.ones_like(inputs)[:,:10]*labels_
        neg=-pos+1
        skew_pad=torch.cat((pos,neg),dim=1)
        latter=torch.zeros_like(inputs)[:,20:]

        masks_=torch.cat((skew_pad,latter),dim=1).unsqueeze(-1)

        embedding = masks_ * self.embedding_layer(inputs)
        outputs, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        outputs = self.layernorm1(outputs)
        outputs = outputs * masks_ + (1. - masks_) * (-1e6)

        outputs = torch.transpose(outputs, 1, 2)
        outputs, _ = torch.max(outputs, axis=2)
        logits = self.cls_fc(self.dropout(outputs))
        return logits


    def get_cls_param(self):
        layers = [self.cls, self.cls_fc]
        params = []
        for layer in layers:
            params.extend([param for param in layer.parameters() if param.requires_grad])
        return params
    
    def get_gen_param(self):
        layers = [self.gen_fc]
        layers = [self.embedding_layer, self.gen, self.layernorm1, self.gen_fc]
        params = []
        for layer in layers:
            params.extend([param for param in layer.parameters() if param.requires_grad])
        return params


class PolicyNet(nn.Module):
    def __init__(self, args):
        super(PolicyNet, self).__init__()
        self.args = args
        # initialize embedding layers
        self.embedding_layer = Embedding(args.vocab_size,
                                         args.embedding_dim,
                                         args.pretrained_embedding)
        # initialize a RNN encoder and a fc layer
        self.encoder = Rnn(args.cell_type, args.embedding_dim, args.hidden_dim, args.num_layers)
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(args.hidden_dim, args.num_class)

    def forward(self, inputs, masks, z=None):
        """
        Inputs:
            inputs -- (batch_size, seq_length)
            masks -- (batch_size, seq_length)
        Outputs:
            logits -- (batch_size, num_class)
        """
        masks_ = masks.unsqueeze(-1)
        embeddings = masks_ * self.embedding_layer(inputs)
        if z is not None:
            embeddings = embeddings * (z.unsqueeze(-1))
        outputs, _ = self.encoder(embeddings)
        outputs = outputs * masks_ + (1. - masks_) * (-1e6)
        outputs = torch.transpose(outputs, 1, 2)
        outputs, _ = torch.max(outputs, axis=2)
        logits = self.fc(self.dropout(outputs))
        return logits

class ToyNet(nn.Module):
    def __init__(self, args):
        super(ToyNet, self).__init__()
        self.args = args
        self.embedding_layer = Embedding(args.vocab_size,
                                         args.embedding_dim,
                                         args.pretrained_embedding)
        # initialize a RNN encoder and  a fc layer
        self.encoder = Rnn(args.cell_type, args.embedding_dim, args.hidden_dim, args.num_layers)
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(args.hidden_dim, args.num_class)

    def forward(self, inputs, masks, z=None):
        """
        Inputs:
            inputs -- (batch_size, seq_length)
            masks -- (batch_size, seq_length)
        Outputs:
            logits -- (batch_size, num_class)
        """
        masks_ = masks.unsqueeze(-1)
        embeddings = masks_ * self.embedding_layer(inputs)
        if z is not None:
            embeddings = embeddings * (z.unsqueeze(-1))
        outputs, _ = self.encoder(embeddings)
        outputs = outputs * masks_ + (1. - masks_) * (-1e6)

        outputs = torch.transpose(outputs, 1, 2)
        outputs, _ = torch.max(outputs, axis=2)
        logits = self.fc(self.dropout(outputs))
        return logits
