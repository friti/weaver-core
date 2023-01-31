import numpy as np
import awkward as ak
import tqdm
import time
import torch
import gc

from collections import defaultdict, Counter
from .metrics import evaluate_metrics
from ..data.tools import _concat
from ..logger import _logger


def _flatten_label(label, mask=None):
    if label.ndim > 1:
        label = label.view(-1)
        if mask is not None:
            label = label[mask.view(-1)]
    return label

def _flatten_preds(preds, mask=None, label_axis=1):
    if preds.ndim > 2:
        # assuming axis=1 corresponds to the classes
        preds = preds.transpose(label_axis, -1).contiguous()
        preds = preds.view((-1, preds.shape[-1]))
        if mask is not None:
            preds = preds[mask.view(-1)]
    return preds


## train classification + regssion into a total loss --> best training epoch decided on the loss function
def train_classreg(model, loss_func, opt, scheduler, train_loader, dev, epoch, steps_per_epoch=None, grad_scaler=None, tb_helper=None):

    model.train()
    torch.backends.cudnn.benchmark = True;
    torch.backends.cudnn.enabled = True;
    gc.enable();

    data_config = train_loader.dataset.config

    num_batches, total_loss, total_cat_loss, total_reg_loss, total_domain_loss, count_cat, count_domain = 0, 0, 0, 0, 0, 0, 0;
    label_cat_counter = Counter()
    total_cat_correct, total_domain_correct, sum_sqr_err = 0, 0 ,0;
    inputs, target, label_cat, label_domain, model_output, model_output_cat, model_output_reg, model_output_domain = None, None, None, None, None, None, None, None;
    loss, loss_cat, loss_reg, loss_domain, pred_cat, pred_reg, pred_domain, residual_reg, correct_cat, correct_domain = None, None, None, None, None, None, None, None, None, None;

    ### number of classification labels
    num_labels = len(data_config.label_value);
    ### number of targets
    if type(data_config.target_value) == dict:
        num_targets = sum(len(dct) if type(dct) == list else 1 for dct in data_config.target_value.values())
    else:
        num_targets = len(data_config.target_value);
    ### number of domain regions
    num_domains = len(data_config.label_domain_names);
    ### total number of domain labels
    if type(data_config.label_domain_value) == dict:
        num_labels_domain = sum(len(dct) if type(dct) == list else 1 for dct in data_config.label_domain_value.values())
    else:
        num_labels_domain = len(data_config.label_domain_value);
    ### number of labels per region as a list
    if type(data_config.label_domain_value) == dict:
        ldomain = [len(dct) if type(dct) == list else 1 for dct in data_config.label_domain_value.values()]
    else:
        ldomain = [len(data_config.label_domain_value)];
    ### label domain counter
    label_domain_counter = [];
    for idx, names in enumerate(data_config.label_domain_names):
        label_domain_counter.append(Counter())

    start_time = time.time()
    with tqdm.tqdm(train_loader) as tq:
        for X, y_cat, y_reg, y_domain, _, y_cat_check, y_domain_check in tq:

            ### input features for the model
            inputs = [X[k].to(dev,non_blocking=True) for k in data_config.input_names]

            ### build classification true labels (numpy argmax)
            label_cat  = y_cat[data_config.label_names[0]].long()
            cat_check  = y_cat_check[data_config.labelcheck_names[0]].long()
            index_cat  = cat_check.nonzero();
            label_cat  = label_cat[index_cat];

            ### build regression targets
            for idx, (k, v) in enumerate(y_reg.items()):
                if idx == 0:
                    target = v.float();
                else:
                    target = torch.column_stack((target,v.float()))
            target = target[index_cat];

            ### build domain true labels (numpy argmax)
            for idx, (k, v) in enumerate(y_domain.items()):
                if idx == 0:
                    label_domain = v.long();
                else:
                    label_domain = torch.column_stack((label_domain,v.long()))
                
            ### store indexes to separate classification+regression events from DA
            for idx, (k, v) in enumerate(y_domain_check.items()):
                if idx == 0:
                    label_domain_check = v.long();
                    index_domain_all = v.long().nonzero();
                else:
                    label_domain_check = torch.column_stack((label_domain_check,v.long()))
                    index_domain_all = torch.cat((index_domain_all,v.long().nonzero()),0)

            ### edit labels
            label_domain = label_domain[index_domain_all];
            label_domain_check = label_domain_check[index_domain_all];
            label_cat = _flatten_label(label_cat,None)
            label_domain = label_domain.squeeze()
            label_domain_check = label_domain_check.squeeze()            

            ### Number of samples in the batch
            num_cat_examples = max(label_cat.shape[0],target.shape[0]);
            num_domain_examples = label_domain.shape[0];

            if label_cat.nelement():
                label_cat_counter.update(label_cat.cpu().numpy().astype(dtype=np.int32))

            index_domain = defaultdict(list)
            for idx, (k,v) in enumerate(y_domain_check.items()):
                if num_domains == 1:
                    index_domain[k] = label_domain_check.nonzero();
                    if label_domain[index_domain[k]].nelement():
                        label_domain_counter[idx].update(label_domain[index_domain[k]].squeeze().cpu().numpy().astype(dtype=np.int32))
                else:                    
                    index_domain[k] = label_domain_check[:,idx].nonzero();
                    if label_domain[index_domain[k],idx].nelement():
                        a = label_domain[index_domain[k],idx].squeeze().cpu().numpy().astype(dtype=np.int32);
                        print(type(a)," ",a.shape)
                        label_domain_counter[idx].update(a)

            ### send to GPU
            label_cat = label_cat.to(dev,non_blocking=True)
            label_domain = label_domain.to(dev,non_blocking=True)
            label_domain_check = label_domain_check.to(dev,non_blocking=True)
            target = target.to(dev,non_blocking=True)            

            ### loss minimization
            model.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                ### evaluate the model
                model_output  = model(*inputs)         
                model_output_cat = model_output[:,:num_labels]
                model_output_reg = model_output[:,num_labels:num_labels+num_targets];
                model_output_domain = model_output[:,num_labels+num_targets:num_labels+num_targets+num_labels_domain]
                model_output_cat    = _flatten_preds(model_output_cat,None);
                model_output_cat    = model_output_cat[index_cat].squeeze().float();
                model_output_reg    = model_output_reg[index_cat].squeeze().float();
                model_output_domain = model_output_domain[index_domain_all].squeeze().float();
                label_cat    = label_cat.squeeze();
                label_domain = label_domain.squeeze();
                label_domain_check = label_domain_check.squeeze();
                target       = target.squeeze();
                ### evaluate loss function            
                loss, loss_cat, loss_reg, loss_domain = loss_func(model_output_cat,label_cat,model_output_reg,target,model_output_domain,label_domain,label_domain_check);

            ### back propagation
            if grad_scaler is None:
                loss.backward()
                opt.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt)
                grad_scaler.update()

            if scheduler and getattr(scheduler, '_update_per_step', True):
                scheduler.step()

            ### evaluate loss function and counters
            num_batches += 1
            loss = loss.detach().item()
            total_loss += loss
            if loss_cat:
                loss_cat = loss_cat.detach().item()
                total_cat_loss += loss_cat;
            if loss_reg:
                loss_reg = loss_reg.detach().item()
                total_reg_loss += loss_reg;
            if loss_domain:
                loss_domain = loss_domain.detach().item()
                total_domain_loss += loss_domain;

            ## take the classification prediction and compare with the true labels            
            label_cat = label_cat.detach()
            label_domain = label_domain.detach()
            target = target.detach()

            if label_cat.nelement():
                _, pred_cat = model_output_cat.detach().max(1);
                correct_cat = (pred_cat == label_cat).sum().item()
                total_cat_correct += correct_cat
                count_cat += num_cat_examples;
                ## take the regression prediction and compare with true targets        
                pred_reg = model_output_reg.float();
                residual_reg = pred_reg - target;            
                sqr_err = residual_reg.square().sum().item()
                sum_sqr_err += sqr_err

            ## single domain region
            if num_domains == 1:
                if not label_domain.nelement(): continue;
                _, pred_domain = model_output_domain.detach().max(1);
                correct_domain = (pred_domain == label_domain).sum().item()
                total_domain_correct += correct_domain
                count_domain += num_domain_examples;
            ## multiple domain regions
            else:
                correct_domain = 0;
                for idx, (k,v) in enumerate(y_domain_check.items()):                    
                    if not label_domain[index_domain[k],idx].nelement(): continue;
                    id_dom = idx*ldomain[idx];
                    pred_domain = model_output_domain[:,id_dom:id_dom+ldomain[idx]];
                    _, pred_domain = pred_domain[index_domain[k]].squeeze().detach().max(1);
                    label = label_domain[index_domain[k],idx].squeeze()
                    correct_domain += (pred_domain == label).sum().item()
                total_domain_correct += correct_domain
                count_domain += num_domain_examples;

            ### monitor metrics
            tq.set_postfix({
                'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                'Loss': '%.5f' % loss,
                'AvgLoss': '%.5f' % (total_loss / num_batches),
                'AccCat': '%.5f' % (correct_cat / num_cat_examples if num_cat_examples else 0),
                'AvgAccCat': '%.5f' % (total_cat_correct / count_cat if count_cat else 0),
                'AccDomain': '%.5f' % (correct_domain / (num_domain_examples) if num_domain_examples else 0),
                'AvgAccDomain': '%.5f' % (total_domain_correct / (count_domain) if count_domain else 0),
                'MSE': '%.5f' % (sqr_err / num_cat_examples if num_cat_examples else 0),
                'AvgMSE': '%.5f' % (sum_sqr_err / count_cat if count_cat else 0)
            })

            if tb_helper:
                tb_helper.write_scalars([
                    ("Loss/train", loss, tb_helper.batch_train_count + num_batches),
                    ("AccCat/train", correct_cat / num_cat_examples if num_cat_examples else 0, tb_helper.batch_train_count + num_batches),
                    ("AccDomain/train", correct_domain / (num_domain_examples) if num_domain_examples else 0, tb_helper.batch_train_count + num_batches),
                    ("MSE/train", sqr_err / num_examples_cat if num_examples_cat else 0, tb_helper.batch_train_count + num_batches),
                    ])
                if tb_helper.custom_fn:
                    with torch.no_grad():
                        tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=num_batches, mode='train')

            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break

    ### training summary
    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count_cat+count_domain, (count_cat+count_domain) / time_diff))
    _logger.info('Train AvgLoss: %.5f'% (total_loss / num_batches))
    _logger.info('Train AvgLoss Cat: %.5f'% (total_cat_loss / num_batches))
    _logger.info('Train AvgLoss Domain: %.5f'% (total_domain_loss / num_batches))
    _logger.info('Train AvgLoss Reg: %.5f'% (total_reg_loss / num_batches))
    _logger.info('Train AvgAccCat: %.5f'%(total_cat_correct / count_cat if count_cat else 0))
    _logger.info('Train AvgAccDomain: %.5f'%(total_domain_correct / (count_domain) if count_domain else 0))
    _logger.info('Train AvgMSE: %.5f'%(sum_sqr_err / count_cat if count_cat else 0))
    _logger.info('Train class distribution: \n %s', str(sorted(label_cat_counter.items())))
    _logger.info('Train domain distribution: \n %s', ' '.join([str(sorted(i.items())) for i in label_domain_counter]))

    if tb_helper:
        tb_helper.write_scalars([
            ("Loss/train (epoch)", total_loss / num_batches, epoch),
            ("Loss Cat/train (epoch)", total_cat_lloss / num_batches, epoch),
            ("Loss Domain/train (epoch)", total_domain_loss / num_batches, epoch),
            ("Loss Reg/train (epoch)", total_reg_loss / num_batches, epoch),
            ("AccCat/train (epoch)", total_cat_correct / count_cat, epoch),
            ("AccDomain/train (epoch)", total_domain_correct / (count_domain), epoch),
            ("MSE/train (epoch)", sum_sqr_err / count, epoch),
            ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode='train')
        # update the batch state
        tb_helper.batch_train_count += num_batches

    torch.cuda.empty_cache()
    gc.collect();

## evaluate classification + regression task
def evaluate_classreg(model, test_loader, dev, epoch, for_training=True, loss_func=None, steps_per_epoch=None, tb_helper=None,
                      eval_cat_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix'],
                      eval_reg_metrics=['mean_squared_error', 'mean_absolute_error', 'median_absolute_error', 'mean_gamma_deviance']):

    model.eval()
    if for_training:
        torch.backends.cudnn.benchmark = True;
        torch.backends.cudnn.enabled = True;
    else:
        torch.backends.cudnn.benchmark = False;
        torch.backends.cudnn.enabled = False;
    gc.enable();

    data_config = test_loader.dataset.config
    label_cat_counter = Counter()
    total_loss, total_cat_loss, total_reg_loss, total_domain_loss, num_batches, total_cat_correct, total_domain_correct = 0, 0, 0, 0, 0, 0, 0;
    sum_sqr_err, count_cat, count_domain = 0, 0, 0;
    inputs, label_cat, label_domain, target, model_output, model_output_cat, model_output_reg, model_output_domain  = None, None, None, None, None , None, None, None;
    pred_cat, pred_domain, pred_reg, correct_cat, correct_domain = None, None, None, None, None;
    loss, loss_cat, loss_domain, loss_reg = None, None, None, None;
    scores_cat, scores_reg, indexes_cat = [], [], [];
    scores_domain  = defaultdict(list); 
    labels_cat, labels_domain, targets, observers = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list);
    indexes_domain = defaultdict(list); 
    index_offset = 0;

    ### number of classification labels
    num_labels = len(data_config.label_value);
    ### number of targets
    if type(data_config.target_value) == dict:
        num_targets = sum(len(dct) if type(dct) == list else 1 for dct in data_config.target_value.values())
    else:
        num_targets = len(data_config.target_value);
    ### total number of domain regions
    num_domains = len(data_config.label_domain_names);
    ### total number of domain labels
    if type(data_config.label_domain_value) == dict:
        num_labels_domain = sum(len(dct) if type(dct) == list else 1 for dct in data_config.label_domain_value.values())
    else:
        num_labels_domain = len(data_config.label_domain_value);

    ### number of labels per region as a list
    if type(data_config.label_domain_value) == dict:
        ldomain = [len(dct) if type(dct) == list else 1 for dct in data_config.label_domain_value.values()]
    else:
        ldomain = [len(data_config.label_domain_value)];
    ### label counter
    label_domain_counter = [];
    for idx, names in enumerate(data_config.label_domain_names):
        label_domain_counter.append(Counter())

    start_time = time.time()
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y_cat, y_reg, y_domain, Z, y_cat_check, y_domain_check in tq:

                ### input features for the model
                inputs = [X[k].to(dev,non_blocking=True) for k in data_config.input_names]

                ### build classification true labels
                label_cat = y_cat[data_config.label_names[0]].long()
                cat_check = y_cat_check[data_config.labelcheck_names[0]].long()
                index_cat = cat_check.nonzero();
                label_cat = label_cat[index_cat];

                ### build regression targets                                                                                                                                                         
                for idx, (k, v) in enumerate(y_reg.items()):
                    if idx == 0:
                        target = v.float();
                    else:
                        target = torch.column_stack((target,v.float()))
                target = target[index_cat];

                ### build domain true labels (numpy argmax)                                                                                                                                   
                for idx, (k,v) in enumerate(y_domain.items()):
                    if idx == 0:
                        label_domain = v.long();
                    else:
                        label_domain = torch.column_stack((label_domain,v.long()))

                for idx, (k, v) in enumerate(y_domain_check.items()):
                    if idx == 0:
                        label_domain_check = v.long();
                        index_domain_all = v.long().nonzero();
                    else:
                        label_domain_check = torch.column_stack((label_domain_check,v.long()))
                        index_domain_all = torch.cat((index_domain_all,v.long().nonzero()),0)

                label_domain = label_domain[index_domain_all];
                label_domain_check = label_domain_check[index_domain_all];

                ### edit labels                                                                                                                                                                      
                label_cat = _flatten_label(label_cat,None)
                label_domain = label_domain.squeeze()
                label_domain_check = label_domain_check.squeeze()

                ### counters
                if label_cat.nelement():
                    label_cat_counter.update(label_cat.cpu().numpy().astype(dtype=np.int32))

                index_domain = defaultdict(list)
                for idx, (k,v) in enumerate(y_domain_check.items()):
                    if num_domains == 1:
                        index_domain[k] = label_domain_check.nonzero();
                        if label_domain[index_domain[k]].nelement():
                            label_domain_counter[idx].update(label_domain[index_domain[k]].squeeze().cpu().numpy().astype(dtype=np.int32))
                    else:
                        index_domain[k] = label_domain_check[:,idx].nonzero();
                        if label_domain[index_domain[k],idx].nelement():
                            label_domain_counter[idx].update(label_domain[index_domain[k],idx].squeeze().cpu().numpy().astype(dtype=np.int32))

                ### update counters
                num_cat_examples = max(label_cat.shape[0],target.shape[0]);
                num_domain_examples = label_domain.shape[0]

                ### send to gpu
                label_cat = label_cat.to(dev,non_blocking=True)
                label_domain = label_domain.to(dev,non_blocking=True)
                label_domain_check = label_domain_check.to(dev,non_blocking=True)
                target = target.to(dev,non_blocking=True)            

                ### store truth labels for classification and regression as well as observers
                for k, v in y_cat.items():
                    if not for_training:
                        labels_cat[k].append(_flatten_label(v,None).cpu().numpy().astype(dtype=np.int32))
                    else:
                        labels_cat[k].append(_flatten_label(v[index_cat],None).cpu().numpy().astype(dtype=np.int32))

                for k, v in y_reg.items():
                    if not for_training:
                        targets[k].append(v.cpu().numpy().astype(dtype=np.float32))                
                    else:
                        targets[k].append(v[index_cat].cpu().numpy().astype(dtype=np.float32))
                    
                if not for_training:
                    indexes_cat.append((index_offset+index_cat).cpu().numpy().astype(dtype=np.int32));
                    for k, v in Z.items():                
                        observers[k].append(v.cpu().numpy().astype(dtype=np.float32))

                for idx, (k, v) in enumerate(y_domain.items()):
                    if not for_training:
                        labels_domain[k].append(v.squeeze().cpu().numpy().astype(dtype=np.int32))
                        indexes_domain[k].append((index_offset+index_domain[list(y_domain_check.keys())[idx]]).cpu().numpy().astype(dtype=np.int32));
                    else:
                        labels_domain[k].append(v[index_domain[list(y_domain_check.keys())[idx]]].squeeze().cpu().numpy().astype(dtype=np.int32))
                            
                ### evaluate model
                model_output  = model(*inputs)
                model_output_cat = model_output[:,:num_labels]
                model_output_reg = model_output[:,num_labels:num_labels+num_targets];
                model_output_domain = model_output[:,num_labels+num_targets:num_labels+num_targets+num_labels_domain]
                model_output_cat = _flatten_preds(model_output_cat,None);
                label_cat    = label_cat.squeeze();
                label_domain = label_domain.squeeze();
                target       = target.squeeze();

                ### in validation only filter interesting events
                if for_training:                        
                    model_output_cat = model_output_cat[index_cat];
                    model_output_reg = model_output_reg[index_cat];
                    model_output_domain = model_output_domain[index_domain_all];                
                    ### adjsut outputs
                    model_output_cat = model_output_cat.squeeze().float();
                    model_output_reg = model_output_reg.squeeze().float();
                    model_output_domain = model_output_domain.squeeze().float();

                    scores_cat.append(torch.softmax(model_output_cat,dim=1).cpu().numpy().astype(dtype=np.float32));
                    scores_reg.append(model_output_reg.cpu().numpy().astype(dtype=np.float32));
                    for idx, name in enumerate(y_domain.keys()):
                        id_dom = idx*ldomain[idx];
                        score_domain = model_output_domain[:,id_dom:id_dom+ldomain[idx]];
                        scores_domain[name].append(torch.softmax(
                            score_domain[index_domain[list(y_domain_check.keys())[idx]]].squeeze(),dim=1).cpu().numpy().astype(dtype=np.float32));
                else:

                    model_output_cat = model_output_cat.float();
                    model_output_reg = model_output_reg.float();
                    model_output_domain = model_output_domain.float();

                    scores_cat.append(torch.softmax(model_output_cat,dim=1).cpu().numpy().astype(dtype=np.float32));
                    scores_reg.append(model_output_reg.cpu().numpy().astype(dtype=np.float32));
                    for idx, name in enumerate(y_domain.keys()):
                        id_dom = idx*ldomain[idx];
                        score_domain = model_output_domain[:,id_dom:id_dom+ldomain[idx]];
                        scores_domain[name].append(torch.softmax(score_domain.squeeze(),dim=1).cpu().numpy().astype(dtype=np.float32));
                        
                    model_output_cat = model_output_cat[index_cat];
                    model_output_reg = model_output_reg[index_cat];
                    model_output_domain = model_output_domain[index_domain_all];                        
                    ### adjsut outputs
                    model_output_cat = model_output_cat.squeeze().float();
                    model_output_reg = model_output_reg.squeeze().float();
                    model_output_domain = model_output_domain.squeeze().float();

                ### evaluate loss function
                num_batches += 1
                index_offset += (num_cat_examples+num_domain_examples)

                if loss_func != None:
                    loss, loss_cat, loss_reg, loss_domain = loss_func(model_output_cat,label_cat,model_output_reg,target,model_output_domain,label_domain,label_domain_check);                
                    loss = loss.item()
                    if loss_cat:
                        loss_cat = loss_cat.item()
                    if loss_reg:                        
                        loss_reg = loss_reg.item()
                    if loss_domain:
                        loss_domain = loss_domain.item()
                else:
                    loss,loss_cat,loss_reg,loss_domain = 0,0,0,0;
                                    
                total_loss += loss
                total_cat_loss += loss_cat
                total_reg_loss += loss_reg
                total_domain_loss += loss_domain

                ## prediction + metric for classification
                if label_cat.nelement():
                    _, pred_cat = model_output_cat.max(1);
                    correct_cat = (pred_cat == label_cat).sum().item()
                    count_cat += num_cat_examples
                    total_cat_correct += correct_cat
                    ## prediction + metric for regression
                    pred_reg = model_output_reg.float();
                    residual_reg = pred_reg - target;
                    sqr_err = residual_reg.square().sum().item()
                    sum_sqr_err += sqr_err

                ## single domain region                                                                                                                                                          
                if num_domains == 1:
                    if not label_domain.nelement(): continue;
                    _, pred_domain = model_output_domain.detach().max(1);
                    correct_domain = (pred_domain == label_domain).sum().item()
                    total_domain_correct += correct_domain
                    count_domain += num_domain_examples                
                ## multiple domains
                else:
                    correct_domain = 0;
                    for idx, (k,v) in enumerate(y_domain_check.items()):
                        if not label_domain[index_domain[k],idx].nelement(): continue;
                        id_dom = idx*ldomain[idx];
                        label = label_domain[index_domain[k],idx].squeeze()
                        pred_domain = model_output_domain[:,id_dom:id_dom+ldomain[idx]]
                        _, pred_domain = pred_domain[index_domain[k]].squeeze().detach().max(1);
                        correct_domain += (pred_domain == label).sum().item()
                    total_domain_correct += correct_domain
                    count_domain += num_domain_examples                

                ### monitor results
                tq.set_postfix({
                    'Loss': '%.5f' % loss,
                    'AvgLoss': '%.5f' % (total_loss / num_batches),
                    'AccCat': '%.5f' % (correct_cat / num_cat_examples if num_cat_examples else 0),
                    'AvgAccCat': '%.5f' % (total_cat_correct / count_cat if count_cat else 0),
                    'AccDomain': '%.5f' % (correct_domain / num_domain_examples if num_domain_examples else 0),
                    'AvgAccDomain': '%.5f' % (total_domain_correct / count_domain if count_domain else 0),
                    'MSE': '%.5f' % (sqr_err / num_cat_examples if num_cat_examples else 0),
                    'AvgMSE': '%.5f' % (sum_sqr_err / count_cat if count_cat else 0),
                })
                    
                if tb_helper:
                    if tb_helper.custom_fn:
                        with torch.no_grad():
                            tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=num_batches,
                                                mode='eval' if for_training else 'test')

                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count_cat+count_domain, (count_cat+count_domain) / time_diff))
    _logger.info('Eval AvgLoss: %.5f'% (total_loss / num_batches))
    _logger.info('Eval AvgLoss Cat: %.5f'% (total_cat_loss / num_batches))
    _logger.info('Eval AvgLoss Domain: %.5f'% (total_domain_loss / num_batches))
    _logger.info('Eval AvgLoss Reg: %.5f'% (total_reg_loss / num_batches))
    _logger.info('Eval AvgAccCat: %.5f'%(total_cat_correct / count_cat if count_cat else 0))
    _logger.info('Eval AvgAccDomain: %.5f'%(total_domain_correct / (count_domain) if count_domain else 0))
    _logger.info('Eval AvgMSE: %.5f'%(sum_sqr_err / count_cat if count_cat else 0))
    _logger.info('Evaluation class distribution: \n    %s', str(sorted(label_cat_counter.items())))
    _logger.info('Train domain distribution: \n %s', ' '.join([str(sorted(i.items())) for i in label_domain_counter]))

    if tb_helper:
        tb_mode = 'eval' if for_training else 'test'
        tb_helper.write_scalars([
            ("Loss/%s (epoch)"%(tb_mode), total_loss / num_batches, epoch),
            ("Loss Cat/%s (epoch)"%(tb_mode), total_cat_loss / num_batches, epoch),
            ("Loss Reg/%s (epoch)"%(tb_mode), total_reg_loss / num_batches, epoch),
            ("Loss Domain/%s (epoch)"%(tb_mode), total_domain_loss / num_batches, epoch),
            ("AccCat/%s (epoch)"%(tb_mode), total_cat_correct / count_cat if count_cat else 0, epoch),
            ("AccDomain/%s (epoch)"%(tb_mode), total_domain_correct / count_domain if count_domain else 0, epoch),
            ("MSE/%s (epoch)"%(tb_mode), sum_sqr_err / count_cat if count_cat else 0, epoch),
            ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode=tb_mode)

    #####
    scores_cat = np.concatenate(scores_cat).squeeze()
    scores_reg = np.concatenate(scores_reg).squeeze()
    scores_domain = {k: _concat(v) for k, v in scores_domain.items()}
    if not for_training:
        indexes_cat = np.concatenate(indexes_cat).squeeze()
        indexes_domain = {k: _concat(v) for k, v in indexes_domain.items()}
    labels_cat    = {k: _concat(v) for k, v in labels_cat.items()}
    labels_domain = {k: _concat(v) for k, v in labels_domain.items()}
    targets       = {k: _concat(v) for k, v in targets.items()}
    observers     = {k: _concat(v) for k, v in observers.items()}

    if not for_training:

        metric_cat_results = evaluate_metrics(labels_cat[data_config.label_names[0]][indexes_cat].squeeze(),scores_cat[indexes_cat].squeeze(),eval_metrics=eval_cat_metrics)            
        _logger.info('Evaluation Classification metrics: \n%s', '\n'.join(
            ['    - %s: \n%s' % (k, str(v)) for k, v in metric_cat_results.items()]))

        for idx, (name,element) in enumerate(labels_domain.items()):
            metric_domain_results = evaluate_metrics(element[indexes_domain[name]].squeeze(),scores_domain[name][indexes_domain[name]].squeeze(),eval_metrics=eval_cat_metrics)
            _logger.info('Evaluation Domain metrics for '+name+' : \n%s', '\n'.join(
                ['    - %s: \n%s' % (k, str(v)) for k, v in metric_domain_results.items()]))
        
        for idx, (name,element) in enumerate(targets.items()):
            if num_targets == 1:
                metric_reg_results = evaluate_metrics(element[indexes_cat].squeeze(), scores_reg[indexes_cat].squeeze(), eval_metrics=eval_reg_metrics)
            else:
                metric_reg_results = evaluate_metrics(element[indexes_cat].squeeze(), scores_reg[indexes_cat,idx].squeeze(), eval_metrics=eval_reg_metrics)
            _logger.info('Evaluation Regression metrics for '+name+' target: \n%s', '\n'.join(
                ['    - %s: \n%s' % (k, str(v)) for k, v in metric_reg_results.items()]))        
    else:
        metric_cat_results = evaluate_metrics(labels_cat[data_config.label_names[0]],scores_cat,eval_metrics=eval_cat_metrics)    
        _logger.info('Evaluation Classification metrics: \n%s', '\n'.join(
            ['    - %s: \n%s' % (k, str(v)) for k, v in metric_cat_results.items()]))

        for idx, (name,element) in enumerate(labels_domain.items()):
            metric_domain_results = evaluate_metrics(element,scores_domain[name],eval_metrics=eval_cat_metrics)
            _logger.info('Evaluation Domain metrics for '+name+' : \n%s', '\n'.join(
                ['    - %s: \n%s' % (k, str(v)) for k, v in metric_domain_results.items()]))

        for idx, (name,element) in enumerate(targets.items()):
            if num_targets == 1:
                metric_reg_results = evaluate_metrics(element, scores_reg, eval_metrics=eval_reg_metrics)
            else:
                metric_reg_results = evaluate_metrics(element, scores_reg[:,idx], eval_metrics=eval_reg_metrics)
            _logger.info('Evaluation Regression metrics for '+name+' target: \n%s', '\n'.join(
                ['    - %s: \n%s' % (k, str(v)) for k, v in metric_reg_results.items()]))        

    torch.cuda.empty_cache()
    if for_training:
        gc.collect();
        return total_loss / num_batches;
    else:
        scores_reg = scores_reg.reshape(len(scores_reg),num_targets);
        scores_domain = np.concatenate(list(scores_domain.values()),axis=1);
        scores_domain = scores_domain.reshape(len(scores_domain),num_labels_domain);
        scores = np.concatenate((scores_cat,scores_reg,scores_domain),axis=1)
        gc.collect();
        return total_loss / num_batches, scores, labels_cat, targets, labels_domain, observers


def evaluate_onnx_classreg(model_path, test_loader,
                           eval_cat_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix'],
                           eval_reg_metrics=['mean_squared_error', 'mean_absolute_error', 'median_absolute_error', 'mean_gamma_deviance']):

    import onnxruntime
    sess = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    gc.enable();

    data_config = test_loader.dataset.config
    label_cat_counter = Counter()
    total_loss, num_batches, total_cat_correct, total_domain_correct, sum_sqr_err, count_cat, count_domain = 0, 0, 0, 0, 0, 0, 0;
    labels_cat, labels_domain, targets, observers = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    scores_cat, scores_reg, scores_domain = [], [], []
    scores_domain  = defaultdict(list); 
    pred_cat, pred_domain, pred_reg = None, None, None;
    inputs, label_cat, label_domain, target, model_output, model_output_cat, model_output_reg, model_output_domain  = None, None, None, None, None , None, None, None;
    indexes_cat = [];
    indexes_domain = defaultdict(list); 
    index_offset = 0;

    ### number of classification labels
    num_labels = len(data_config.label_value);
    ### number of targets
    if type(data_config.target_value) == dict:
        num_targets = sum(len(dct) if type(dct) == list else 1 for dct in data_config.target_value.values())
    else:
        num_targets = len(data_config.target_value);
    ### total number of domain regions
    num_domains = len(data_config.label_domain_names);
    ### total number of domain labels
    if type(data_config.label_domain_value) == dict:
        num_labels_domain = sum(len(dct) if type(dct) == list else 1 for dct in data_config.label_domain_value.values())
    else:
        num_labels_domain = len(data_config.label_domain_value);
    ### number of labels per region as a list
    if type(data_config.label_domain_value) == dict:
        ldomain = [len(dct) if type(dct) == list else 1 for dct in data_config.label_domain_value.values()]
    else:
        ldomain = [len(data_config.label_domain_value)];
    ### label counter
    label_domain_counter = [];
    for idx, names in enumerate(data_config.label_domain_names):
        label_domain_counter.append(Counter())

    start_time = time.time()

    with tqdm.tqdm(test_loader) as tq:
        for X, y_cat, y_reg, y_domain, Z, y_cat_check, y_domain_check in tq:
            ### input features for the model
            inputs = {k: v.numpy().astype(dtype=np.float32) for k, v in X.items()}

            ### build classification true labels
            label_cat = y_cat[data_config.label_names[0]].long()
            cat_check = y_cat_check[data_config.labelcheck_names[0]].long()
            index_cat = cat_check.nonzero();
            label_cat = label_cat[index_cat];

            ### build regression targets
            for idx, names in enumerate(data_config.target_names):
                if idx == 0:
                    target = y_reg[names].float();
                else:
                    target = torch.column_stack((target,y_reg[names].float()))
            target = target[index_cat];

            ### build domain true labels (numpy argmax)                                                                                                                                   
            for idx, (k,v) in enumerate(y_domain.items()):
                if idx == 0:
                    label_domain = v.long();
                else:
                    label_domain = torch.column_stack((label_domain,v.long()))

            for idx, (k, v) in enumerate(y_domain_check.items()):
                if idx == 0:
                    label_domain_check = v.long();
                    index_domain_all = v.long().nonzero();
                else:
                    label_domain_check = torch.column_stack((label_domain_check,v.long()))
                    index_domain_all = torch.cat((index_domain_all,v.long().nonzero()),0)
                    
            label_domain = label_domain[index_domain_all];
            label_domain_check = label_domain_check[index_domain_all];

            ### edit labels                                                                                                                                                                      
            label_cat = _flatten_label(label_cat,None)
            label_domain = label_domain.squeeze()
            label_domain_check = label_domain_check.squeeze()

            ### counters
            label_cat_counter.update(label_cat.cpu().numpy().astype(dtype=np.int32))
            index_domain = defaultdict(list)
            for idx, (k,v) in enumerate(y_domain_check.items()):
                if num_domains == 1:
                    index_domain[k] = label_domain_check.nonzero();
                    label_domain_counter[idx].update(label_domain[index_domain[k]].squeeze().cpu().numpy().astype(dtype=np.int32))
                else:
                    index_domain[k] = label_domain_check[:,idx].nonzero();
                    label_domain_counter[idx].update(label_domain[index_domain[k],idx].squeeze().cpu().numpy().astype(dtype=np.int32))

            ### update counters
            num_cat_examples = max(label_cat.shape[0],target.shape[0]);
            num_domain_examples = label_domain.shape[0]

            ### define truth labels for classification and regression
            indexes_cat.append((index_offset+index_cat).cpu().numpy().astype(dtype=np.int32));
            for k, v in y_cat.items():
                labels_cat[k].append(_flatten_label(v,None).cpu().numpy().astype(dtype=np.int32))
            for k, v in y_reg.items():
                targets[k].append(v.cpu().numpy().astype(dtype=np.float32))                
            for k, v in Z.items():                
                observers[k].append(v.cpu().numpy().astype(dtype=np.float32))
            for idx, (k, v) in enumerate(y_domain.items()):
                labels_domain[k].append(v.squeeze().cpu().numpy().astype(dtype=np.int32))
                indexes_domain[k].append((index_offset+index_domain[list(y_domain_check.keys())[idx]]).cpu().numpy().astype(dtype=np.int32));
  
            ### send to device
            label_cat    = label_cat.to(dev,non_blocking=True)
            label_domain = label_domain.to(dev,non_blocking=True)
            target       = target.to(dev,non_blocking=True)            

            ### output of the mode
            model_output = sess.run([], inputs)
            model_output = torch.as_tensor(np.array(model_output));
            
            model_output_cat = model_output[:,:num_labels]
            model_output_reg = model_output[:,num_labels:num_labels+num_targets].float();
            model_output_domain = model_output[:,num_labels+num_targets:num_labels+num_targets+num_labels_domain].float()
            model_output_cat = _flatten_preds(model_output_cat,None).float()
            label_cat = label_cat.squeeze();
            label_domain = label_domain.squeeze();
            target = target.squeeze();

            scores_cat.append(torch.softmax(model_output_cat,dim=1).cpu().numpy().astype(dtype=np.float32));
            scores_reg.append(model_output_reg.cpu().numpy().astype(dtype=np.float32));
            for idx, name in enumerate(y_domain.keys()):
                id_dom = idx*ldomain[idx];
                score_domain = model_output_domain[:,id_dom:id_dom+ldomain[idx]];
                scores_domain[name].append(torch.softmax(score_domain.squeeze(),dim=1).cpu().numpy().astype(dtype=np.float32));
            
            model_output_cat = model_output_cat[index_cat];
            model_output_reg = model_output_reg[index_cat];
            model_output_domain = model_output_domain[index_domain_all];                        

            ### adjsut outputs
            model_output_cat = model_output_cat.squeeze().float();
            model_output_reg = model_output_reg.squeeze().float();
            model_output_domain = model_output_domain.squeeze().float();

            num_batches += 1
            index_offset += (num_cat_examples+num_domain_examples)

            ## prediction + metric for classification
            if label_cat.nelement():
                _, pred_cat = model_output_cat.max(1);
                correct_cat = (pred_cat == label_cat).sum().item()
                count_cat += num_cat_examples
                total_cat_correct += correct_cat
                ## prediction + metric for regression
                pred_reg = model_output_reg.float();
                residual_reg = pred_reg - target;
                sqr_err = residual_reg.square().sum().item()
                sum_sqr_err += sqr_err

            ## single domain region                                                                                                                                                          
            if num_domains == 1:
                if not label_domain.nelement(): continue;
                _, pred_domain = model_output_domain.detach().max(1);
                correct_domain = (pred_domain == label_domain).sum().item()
                total_domain_correct += correct_domain
                count_domain += num_domain_examples                
                ## multiple domains
            else:
                correct_domain = 0;
                for idx, (k,v) in enumerate(y_domain_check.items()):
                    if not label_domain[index_domain[k],idx].nelement(): continue;
                    id_dom = idx*ldomain[idx];
                    label = label_domain[index_domain[k],idx].squeeze()
                    pred_domain = model_output_domain[:,id_dom:id_dom+ldomain[idx]]
                    _, pred_domain = pred_domain[index_domain[k]].squeeze().detach().max(1);
                    correct_domain += (pred_domain == label).sum().item()
                total_domain_correct += correct_domain
                count_domain += num_domain_examples                
                
            ### monitor results
            tq.set_postfix({
                'AccCat': '%.5f' % (correct_cat / num_cat_examples if num_cat_examples else 0),
                'AvgAccCat': '%.5f' % (total_cat_correct / count_cat if count_cat else 0),
                'AccDomain': '%.5f' % (correct_domain / num_domain_examples if num_domain_examples else 0),
                'AvgAccDomain': '%.5f' % (total_domain_correct / count_domain if count_domain else 0),
                'MSE': '%.5f' % (sqr_err / num_cat_examples if num_cat_examples else 0),
                'AvgMSE': '%.5f' % (sum_sqr_err / count_cat if count_cat else 0),
            })
                
    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count_cat+count_domain, (count_cat+count_domain) / time_diff))
    _logger.info('Eval AvgAccCat: %.5f'%(total_cat_correct / count_cat if count_cat else 0))
    _logger.info('Eval AvgAccDomain: %.5f'%(total_domain_correct / (count_domain) if count_domain else 0))
    _logger.info('Eval AvgMSE: %.5f'%(sum_sqr_err / count_cat if count_cat else 0))
    _logger.info('Evaluation class distribution: \n    %s', str(sorted(label_cat_counter.items())))
    _logger.info('Train domain distribution: \n %s', ' '.join([str(sorted(i.items())) for i in label_domain_counter]))
    
    scores_cat = np.concatenate(scores_cat).squeeze()
    scores_reg = np.concatenate(scores_reg).squeeze()
    scores_domain = {k: _concat(v) for k, v in scores_domain.items()}
    indexes_cat = np.concatenate(indexes_cat).squeeze()
    indexes_domain = {k: _concat(v) for k, v in indexes_domain.items()}
    labels_cat  = {k: _concat(v) for k, v in labels_cat.items()}
    labels_domain  = {k: _concat(v) for k, v in labels_domain.items()}
    targets = {k: _concat(v) for k, v in targets.items()}
    observers = {k: _concat(v) for k, v in observers.items()}

    metric_cat_results = evaluate_metrics(labels_cat[data_config.label_names[0]][indexes_cat].squeeze(),scores_cat[indexes_cat].squeeze(),eval_metrics=eval_cat_metrics)            
    _logger.info('Evaluation Classification metrics: \n%s', '\n'.join(
        ['    - %s: \n%s' % (k, str(v)) for k, v in metric_cat_results.items()]))

    for idx, (name,element) in enumerate(labels_domain.items()):
        metric_domain_results = evaluate_metrics(element[indexes_domain[name]].squeeze(),scores_domain[name][indexes_domain[name]].squeeze(),eval_metrics=eval_cat_metrics)
        _logger.info('Evaluation Domain metrics for '+name+' : \n%s', '\n'.join(
            ['    - %s: \n%s' % (k, str(v)) for k, v in metric_domain_results.items()]))
        
        
    for idx, (name,element) in enumerate(targets.items()):
        if num_targets == 1:
            metric_reg_results = evaluate_metrics(element[indexes_cat].squeeze(), scores_reg[indexes_cat].squeeze(), eval_metrics=eval_reg_metrics)
        else:
            metric_reg_results = evaluate_metrics(element[indexes_cat].squeeze(), scores_reg[indexes_cat,idx].squeeze(), eval_metrics=eval_reg_metrics)
            _logger.info('Evaluation Regression metrics for '+name+' target: \n%s', '\n'.join(
                ['    - %s: \n%s' % (k, str(v)) for k, v in metric_reg_results.items()]))        
            
    scores_reg = scores_reg.reshape(len(scores_reg),num_targets);
    scores_domain = np.concatenate(list(scores_domain.values()),axis=1);
    scores_domain = scores_domain.reshape(len(scores_domain),num_labels_domain);
    scores = np.concatenate((scores_cat,scores_reg,scores_domain),axis=1)
    gc.collect();
    return total_loss / num_batches, scores, labels_cat, targets, labels_domain, observers

class TensorboardHelper(object):

    def __init__(self, tb_comment, tb_custom_fn):
        self.tb_comment = tb_comment
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(comment=self.tb_comment)
        _logger.info('Create Tensorboard summary writer with comment %s' % self.tb_comment)

        # initiate the batch state
        self.batch_train_count = 0

        # load custom function
        self.custom_fn = tb_custom_fn
        if self.custom_fn is not None:
            from utils.import_tools import import_module
            from functools import partial
            self.custom_fn = import_module(self.custom_fn, '_custom_fn')
            self.custom_fn = partial(self.custom_fn.get_tensorboard_custom_fn, tb_writer=self.writer)

    def __del__(self):
        self.writer.close()

    def write_scalars(self, write_info):
        for tag, scalar_value, global_step in write_info:
            self.writer.add_scalar(tag, scalar_value, global_step)
