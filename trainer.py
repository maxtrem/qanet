from torch import tensor



class ModelTrainer:
    def __init__(self, model, dataset, optimizer=None, eval_dataset=None, name='QA', 
                 update_interval=5, scheduler=None, ema=None, grad_clip=5.0, na_y = 400):
        
        self.model         = model
        self.optimizer     = optimizer
        self.scheduler     = scheduler
        self.train_dataset = dataset
        self.eval_dataset  = eval_dataset
        self.ema           = ema
        self.grad_clip     = grad_clip
        
        self.name          = name

        self.update_interval = update_interval
        self.epoch_dict    = {}
        
        self.print_prec    = 3
        self.na_y          = na_y
        
        self.epoch_history = []
        self.best_epoch    = (0, float('inf'))
        self.save_models   = []

        self.save_dir      = 'model_saves'
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
            print(f'Created directory: "{self.save_dir}"')
        
        self.save_keys     = ['epoch_history', 'best_epoch', 'save_models', 'name']
        
    def get_id(self):
        return str(id(self))[:-7]
        
    def call_thread(self, func, path):
        thread = threading.Thread(target=func, args=(path,))
        thread.start()
        
    def download(self, load_path, save_path=None):
        session = ftplib.FTP('85.214.200.53','ftp-user','oqu7iyiJongae6Oon5foo5mau')
        session.cwd('models')
        save_path = save_path if save_path else load_path
        with open(save_path, 'wb') as f:
            session.retrbinary(f'RETR {load_path}', f.write, blocksize=2**32)
        session.quit()
        print(f'"Server model "{load_path}" written to "{save_path}"')
        
    def get_server_models(self):
        session = ftplib.FTP('85.214.200.53','ftp-user','oqu7iyiJongae6Oon5foo5mau')
        session.cwd('models') 
        ls = []
        session.dir(ls.append)
        model_files = list(filter(lambda x: x.endswith('.history'), map(lambda line: line.split()[-1], ls)))
        models = []
        def handle_binary(fetched_data):
            models.append(eval(fetched_data.decode("utf-8")))

        for filename in model_files:
            session.retrbinary(f'RETR {filename}', callback=handle_binary, blocksize=2**32)
        session.quit()
        model_ids = list(map(lambda x: x.split('.')[0].split('_')[0], model_files))
        return dict(zip(model_ids, models))
    
    def list_server_models(self):
        models = self.get_server_models()
        for model_id in models:
            epochs_trained = len(models[model_id]['save_models'])
            best_epoch, best_loss = models[model_id]['best_epoch']
            print(f'Model: {model_id} (Name: {models[model_id].get("name")})')
            print(f'Epochs trained: {epochs_trained}')
            print(f'Best Loss {best_loss} in epoch {best_epoch+1}.')
            print()
            
    def load_server_model(self, model_id=None):
        models = self.get_server_models()
        if model_id:
            best_epoch = models[model_id]['best_epoch'][0]
            load_path  = models[model_id]['save_models'][best_epoch].split('/')[-1]
            save_path  = f'{model_trainer.save_dir}/{load_path}'
            self.download(load_path, save_path)
            
            new_state_dict = torch.load(save_path, map_location=lambda storage, loc: storage)
            self.model.load_state_dict(new_state_dict)
            update_dict = models[model_id]
            if update_dict.get('name'):
                del update_dict['name']
            self.__dict__.update(update_dict)
            print(f'Loaded model from: "{load_path}"')
        else:
            model_loss_map = list(map(lambda x: (x[0], x[1]['best_epoch'][1]), models.items()))
            best_model_id  = min(model_loss_map, key=lambda x: x[1])[0]
            print(f'Loading model: "{best_model_id}"')
            self.load_server_model(best_model_id)
        
    def update_dict(self, exact_match, f1, loss_start, loss_end, correct_start, correct_end, na_stats, mode='train'):
        assert self.epoch_dict != {}, 'epoch_dict not initialized, use init_epoch_dict() before.'

        self.epoch_dict[f'{mode}_sum_loss_start']    += loss_start
        self.epoch_dict[f'{mode}_sum_loss_end']      += loss_end
        self.epoch_dict[f'{mode}_correct_start'] += correct_start
        self.epoch_dict[f'{mode}_correct_end']   += correct_end
        self.epoch_dict[f'{mode}_exact_match']   += exact_match
        self.epoch_dict[f'{mode}_F1']            += f1
        self.epoch_dict[f'{mode}_na_stats']      += na_stats # true predictions, predictions, gold
        self.epoch_dict[f'{mode}_step']          += 1
        
    def get_metric(self, mode='train', epoch_idx=None, update_mean_loss=True):
        
        d = self.epoch_dict
        if epoch_idx:
            d = self.epoch_history[epoch_idx]

        step       = d[f'{mode}_step']
        num_steps  = d[f'{mode}_num_steps']
        batch_size = d[f'{mode}_batch_size']
        
        acc_start  = round(d[f'{mode}_correct_start'] / (step*batch_size), self.print_prec)
        acc_end    = round(d[f'{mode}_correct_end']   / (step*batch_size), self.print_prec)
        
        loss_start = round(d[f'{mode}_sum_loss_start'] / step, self.print_prec)
        loss_end   = round(d[f'{mode}_sum_loss_end']   / step, self.print_prec)
        em         = round(d[f'{mode}_exact_match']    / (step*batch_size), self.print_prec)
        f1         = round(d[f'{mode}_F1']             / (step*batch_size), self.print_prec)
        
        if update_mean_loss:
            d[f'{mode}_mean_loss_start'] = loss_start
            d[f'{mode}_mean_loss_end']   = loss_end
        

        return em, f1, acc_start, acc_end, loss_start, loss_end, step, num_steps, 

    def get_best_epoch(self, history):
        summed_mean_loss = list(map(lambda d: d[f'{mode}_mean_loss_start'] + d[f'{mode}_mean_loss_end'], history))
        best_epoch  = min(range(len(summed_mean_loss)), key=lambda i: summed_mean_loss[i]) + 1
        best_loss   = summed_mean_loss[best_epoch]
        return best_epoch, best_loss
        
    
    def print_metric(self, mode='train', epoch=None):
        em, f1, acc_start, acc_end, loss_start, loss_end, step, num_steps = self.get_metric(mode, epoch)
        
        prefix = ''
        improvement = ''

        if step == num_steps:
            best_epoch, best_loss = self.best_epoch
            curr_loss   = (loss_start + loss_end)
            difference  = round(best_loss - curr_loss, self.print_prec)
            
            improved_loss = (difference > 0)
            correct_mode  = ((mode == 'eval') or (not self.eval_dataset))
            if improved_loss and correct_mode:
                self.best_epoch = len(self.epoch_history), round(curr_loss, self.print_prec)
                    
                if self.epoch_history:
                    improvement = f'\nMean Loss (start+end) improved compared to epoch {best_epoch+1} by {difference}' 
        else:
            prefix = f'Step [{step}/{num_steps}] - '
            

        body = f'EM: {em} - F1: {f1} - acc_start: {acc_start} - acc_end: {acc_end} - loss_start: {loss_start} - loss_end: {loss_end}'
        print(prefix + body + improvement)
        

    def init_epoch_dict(self, num_steps, batch_size, mode):
        self.epoch_dict.update({f'{mode}_sum_loss_start':0,
                                f'{mode}_sum_loss_end':0,
                                f'{mode}_mean_loss_start':0,
                                f'{mode}_mean_loss_end':0,
                                f'{mode}_correct_start':0,
                                f'{mode}_correct_end':0,
                                f'{mode}_exact_match':0,
                                f'{mode}_F1':0,
                                f'{mode}_step': 0,
                                f'{mode}_num_steps':num_steps,
                                f'{mode}_batch_size':batch_size,
                                f'{mode}_duration': time.time(),
                                f'{mode}_na_stats': tensor([0, 0, 0]), # true predictions, predictions, gold
                                f'save':None,
                                f'uploaded':False,
                                f'model_name':self.name,
                                })
        
    def save_model(self, info, epoch_model=False, suffix='save'):
      
        if self.ema:
            self.ema.assign(self.model)
        model_id = self.get_id()
        
        
        path = f'{self.save_dir}/{model_id}_{self.name}_{info}.{suffix}'
        
        torch.save(self.model.state_dict(), path)
        print(f"Model saved: '{path}'")
        
        self.save_models.append(path)
        self.epoch_dict['save'] = path
        if self.ema:
            self.ema.resume(self.model)
        self.call_thread(upload, path)


        
        history = dict(map(lambda k: (k, self.__dict__[k]), self.save_keys))
        path = f'{self.save_dir}/{model_id}_{self.name}.history'
        with open(path, 'w') as f:
            f.write(repr(history))
            
        self.call_thread(upload, path)
        
    def fit(self, epochs, batch_size, shuffle=True, eval_batch_size=50, start_epoch=1):

        print(f'Start Training: {self.get_id()}\n- Epochs: {epochs} \n- Batch Size: {batch_size}')
        print('\nTrain dataset:', len(self.train_dataset))
        print('Eval dataset: ', len(self.eval_dataset) if self.eval_dataset else None)
        print('\nOptimizer:\n')
        print(self.optimizer)
        if self.epoch_history and start_epoch == 1:
            start_epoch = len(self.epoch_history) + 1
        for epoch in range(start_epoch, epochs+start_epoch):            
            self.run(epoch, batch_size, mode='train')
            if self.eval_dataset:
                if self.ema:
                    self.ema.assign(self.model)
                with torch.no_grad():
                    self.model.eval()
                    self.run(epoch, eval_batch_size, mode='eval', shuffle=False)
                    self.model.train()
                if self.ema:
                    self.ema.resume(self.model)

            self.save_model(f'epoch_{epoch}')
            self.epoch_history.append(self.epoch_dict)
            self.epoch_dict = {}

            print()
            
    def make_step(self, input_features, mode='train'):
        context_tok, context_char, \
        question_tok, question_char, \
        answer_start, answer_end, \
        no_answer, question_id = input_features
        #Cwid, Ccid, Qwid, Qcid
        logits_start, logits_end = self.model(context_tok, context_char, question_tok, question_char)

        loss_start = F.cross_entropy(logits_start, answer_start)
        loss_end   = F.cross_entropy(logits_end, answer_end)
        if mode == 'train':
            self.optimizer.zero_grad()
            
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            (loss_start + loss_end).backward()
            self.optimizer.step()
            if self.ema:
                self.ema(self.model, self.epoch_dict[f'{mode}_step'])
            if self.scheduler:
                self.scheduler.step()
        
        # correct answers vector für batch
        pred_start = logits_start.argmax(dim=1)
        pred_end   = logits_end.argmax(dim=1)
        
        ## Adding NA count
        na_pred = torch.max((pred_end == self.na_y), (pred_start == self.na_y))
        na_true = torch.max((answer_end == self.na_y), (answer_start == self.na_y))
        na_true_pred = (na_pred + na_true) == 2
        
        # true predictions, predictions, gold
        na_stats = (na_true_pred.sum(), na_pred.sum(), na_true.sum())
        na_stats = torch.stack(na_stats)
        
        # setting setting both answer and end to no answer idx even if only one is predicted
        # to make computation of metrics easier
        pred_start[na_pred] = self.na_y
        pred_end[na_pred]   = self.na_y
        ##
        
        # getting max of real and pred start and end
        intersect_start = torch.cat((answer_start.view(-1, 1), pred_start.view(-1, 1)), dim=1).max(dim=1)[0]
        intersect_end   = torch.cat((answer_end.view(-1, 1), pred_end.view(-1, 1)), dim=1).min(dim=1)[0]

        correct_pred_spans = intersect_end - intersect_start
        correct_pred_spans[correct_pred_spans < 0] = -1 
        correct_pred_spans += 1

        eps = 0.00001
        true_positive = correct_pred_spans.float() + eps

        pred_positive = (pred_end - pred_start)
        pred_positive[pred_positive < 0] = -1
        pred_positive += 1
        pred_positive = pred_positive.float() + eps

        real_positive = ((answer_end - answer_start) + 1).float()

        precsion = true_positive / pred_positive
        recall   = true_positive / real_positive
        f1 = 2 * ((precsion * recall) / (precsion + recall)).sum().item()

        correct_start = (answer_start == pred_start) #.sum().item()
        correct_end   = (answer_end == pred_end)   #.sum().item()

        # finding examples with an exact match, both start and end equal 1
        exact_match   = ((correct_start + correct_end) == 2).sum().item()

        # summing up correct
        correct_start = correct_start.sum().item()
        correct_end   = correct_end.sum().item()
        
        self.update_dict(exact_match, f1, loss_start.item(), loss_end.item(), correct_start, correct_end, na_stats, mode=mode)

    def run(self, epoch, batch_size, mode='train', shuffle=True):        
        subject = f'Training Epoch {epoch}' if mode == 'train' else 'Evaluation'
        print(f'- Start {subject}:')
        
        start = time.time()
        self._run(batch_size, mode=mode, shuffle=shuffle)
        
        duration = round(time.time()-start)
        self.epoch_dict[f'{mode}_duration'] = duration
        self.epoch_dict['learning_rate']    = self.optimizer.param_groups[0]['lr']
        self.epoch_dict['optimizer']        = self.optimizer.__class__.__name__
        
        print(f'{subject} finished after {datetime.timedelta(seconds=duration)}.')
        self.print_metric(mode=mode)

    def _run(self, batch_size, mode='train', shuffle=True):
        dataset = self.train_dataset if mode == 'train' else self.eval_dataset
        
        dataset_iterator = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        num_steps = len(dataset_iterator)
        self.init_epoch_dict(num_steps, batch_size, mode)
        
        for step, input_features in enumerate(dataset_iterator, start=1):
            self.make_step(input_features, mode=mode)
            
            if ((step % self.update_interval == 0) and (mode == 'train')):
                self.print_metric(mode=mode)

if __name__ == '__main__':


        import torch
        import torch.nn as nn
        import torch.optim as optim
        import torch.nn.functional as F

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        from torch.utils.data import Dataset, DataLoader

        import numpy as np

        import random, json, math, time, os, datetime, ftplib

        from modules.helpers import upload
        from model import QANet

        print('Loading Dataset..')
        from squad_loader import SQuADDataset

        if not 'eval_dataset' in locals():
            eval_dataset = SQuADDataset('data/dev.npz')
            train_dataset = SQuADDataset('data/train.npz')

        from model import QANet
        from bangliu_model import QANet as BangLuiQANet



        print('Loading Embeddigs..')
        char_emb_matrix = np.array(json.load(open('data/char_emb.json')), dtype=np.float32)
        word_emb_matrix = np.array(json.load(open('data/word_emb.json')), dtype=np.float32)  

        print('Create Model..')

        d_model = 64
        batch_size = 20
        q_max_len = 50
        c_max_len = 400
        char_dim = 16
        num_head = 8




        model = QANet(d_model, c_max_len, q_max_len, word_emb_matrix, char_emb_matrix, droprate=0.1).to(device)

        from modules.ema import EMA
        ema = EMA(0.9999)
        if ema:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    ema.register(name, param.data)

        #model = BangLuiQANet(torch.tensor(word_emb_matrix), torch.tensor(char_emb_matrix),
        #                     c_max_len, q_max_len, d_model, train_cemb=False, heads=num_head).to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.8, 0.999), eps=1e-08, weight_decay=3e-7, amsgrad=False)
        del char_emb_matrix, word_emb_matrix

        warm_up = 1000
        warm_up_f  = lambda x: math.log(x+1)/math.log(warm_up) if x < warm_up else 1

        #scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[warm_up_f])
         
        cr = 1.0 / math.log(warm_up)
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda ee: cr * math.log(ee + 1)
            if ee < warm_up else 1)
          
        model_trainer = ModelTrainer(model, train_dataset, optimizer=optimizer, 
                                     eval_dataset=eval_dataset, name='cq_&_emb', 
                                     update_interval=100, scheduler=scheduler,
                                     ema=ema)
        #model_trainer.list_server_models()
        #model_trainer.load_server_model('13986904')

        model_trainer.fit(epochs=15, batch_size=batch_size)

