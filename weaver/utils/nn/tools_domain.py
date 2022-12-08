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
    label_domain_counter = Counter()
    total_cat_correct, total_domain_correct, sum_sqr_err = 0, 0 ,0;
    inputs, target, label_cat, label_domain, model_output, model_output_cat, model_output_reg, model_output_domain = None, None, None, None, None, None, None, None;
    loss, loss_cat, loss_reg, loss_domain, pred_cat, pred_reg, pred_domain, residual_reg, correct_cat, correct_domain = None, None, None, None, None, None, None, None, None, None;

    start_time = time.time()

    with tqdm.tqdm(train_loader) as tq:
        for X, y_cat, y_reg, y_domain, _, y_cat_check, y_domain_check in tq:
            if num_batches >= 2: break;
            ### input features for the model
            inputs = [X[k].to(dev,non_blocking=True) for k in data_config.input_names]
            ### build classification true labels (numpy argmax)
            label_cat    = y_cat[data_config.label_names[0]].long()
            label_domain = y_domain[data_config.label_domain_names[0]].long()
            cat_check    = y_cat_check[data_config.labelcheck_names[0]].long()
            domain_check = y_domain_check[data_config.labelcheck_domain_names[0]].long()
            index_cat    = cat_check.nonzero();
            index_domain = domain_check.nonzero();
            label_cat    = label_cat[index_cat];
            label_domain = label_domain[index_domain];
            if label_cat.shape[0]+label_domain.shape[0] != cat_check.shape[0]:
                _logger.warning('Wrong decomposition in cat and domain for batch number %d'%(num_batches))
                num_batches += 1
                continue;
            label_cat = _flatten_label(label_cat,None)
            label_domain = _flatten_label(label_domain,None)
            label_cat_counter.update(label_cat.cpu().numpy())
            label_domain_counter.update(label_domain.cpu().numpy())
            label_cat = label_cat.to(dev,non_blocking=True)
            label_domain = label_domain.to(dev,non_blocking=True)
            
            ### build regression targets
            for idx, names in enumerate(data_config.target_names):
                if idx == 0:
                    target = y_reg[names].float();
                else:
                    target = torch.column_stack((target,y_reg[names].float()))

            target = target[index_cat];
            target = target.to(dev,non_blocking=True)            

            ### Number of samples in the batch
            num_cat_examples = max(label_cat.shape[0],target.shape[0]);
            num_domain_examples = label_domain.shape[0];
            if label_cat.shape[0] != target.shape[0]:
                _logger.warning('Wrong decomposition in cat and target for batch number %d'%(num_batches))
                num_batches += 1;
                continue;
            num_batches += 1

            ### loss minimization
            model.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                ### evaluate the model
                model_output  = model(*inputs)         
                if(model_output.dim() == 1) : continue;
                model_output_cat = model_output[:,:len(data_config.label_value)];
                model_output_reg = model_output[:,len(data_config.label_value):len(data_config.label_value)+len(data_config.target_value)];
                model_output_domain = model_output[:,len(data_config.label_value)+len(data_config.target_value):len(data_config.label_value)+len(data_config.target_value)+len(data_config.label_domain_value)];
                model_output_cat    = _flatten_preds(model_output_cat,None);
                model_output_domain = _flatten_preds(model_output_domain,None);
                model_output_cat    = model_output_cat[index_cat];
                model_output_reg    = model_output_reg[index_cat];
                model_output_domain = model_output_domain[index_domain];

                ### evaluate loss function            
                loss, loss_cat, loss_reg, loss_domain = loss_func(model_output_cat,label_cat,model_output_reg,target,model_output_domain,label_domain);

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
            loss = loss.detach().item()
            loss_cat = loss_cat.detach().item()
            loss_reg = loss_reg.detach().item()
            loss_domain = loss_domain.detach().item()
            total_loss += loss
            total_cat_loss += loss_cat;
            total_reg_loss += loss_reg;
            total_domain_loss += loss_domain;
            count_cat += num_cat_examples;
            count_domain += num_domain_examples;

            ## take the classification prediction and compare with the true labels            
            label_cat    = label_cat.detach()
            label_domain = label_domain.detach()
            target       = target.detach()
            _, pred_cat  = model_output_cat.squeeze().max(1);
            _, pred_domain = model_output_domain.squeeze().max(1);
            correct_cat = (pred_cat.detach() == label_cat.detach()).sum().item()
            correct_domain = (pred_domain.detach() == label_domain.detach()).sum().item()
            total_cat_correct += correct_cat
            total_domain_correct += correct_domain

            ## take the regression prediction and compare with true targets
            pred_reg = model_output_reg.squeeze().float();
            residual_reg = pred_reg.detach() - target.detach();            
            sqr_err = residual_reg.square().sum().item()
            sum_sqr_err += sqr_err

            ### monitor metrics
            tq.set_postfix({
                'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                'Loss': '%.5f' % loss,
                'AvgLoss': '%.5f' % (total_loss / num_batches),
                'AccCat': '%.5f' % (correct_cat / num_cat_examples if num_cat_examples else 0),
                'AvgAccCat': '%.5f' % (total_cat_correct / count_cat if count_cat else 0),
                'AccDomain': '%.5f' % (correct_domain / num_domain_examples if num_domain_examples else 0),
                'AvgAccDomain': '%.5f' % (total_domain_correct / count_domain if count_domain else 0),
                'MSE': '%.5f' % (sqr_err / num_cat_examples if num_cat_examples else 0),
                'AvgMSE': '%.5f' % (sum_sqr_err / count_cat if count_cat else 0)
            })

            if tb_helper:
                tb_helper.write_scalars([
                    ("Loss/train", loss, tb_helper.batch_train_count + num_batches),
                    ("AccCat/train", correct_cat / num_cat_examples if num_cat_examples else 0, tb_helper.batch_train_count + num_batches),
                    ("AccDomain/train", correct_domain / num_domain_examples if num_domain_examples else 0, tb_helper.batch_train_count + num_batches),
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
    _logger.info('Train AvgAccDomain: %.5f'%(total_domain_correct / count_domain if count_domain else 0))
    _logger.info('Train AvgMSE: %.5f'%(sum_sqr_err / count_cat if count_cat else 0))
    _logger.info('Train class distribution: \n %s', str(sorted(label_cat_counter.items())))
    _logger.info('Train domain distribution: \n %s', str(sorted(label_domain_counter.items())))

    if tb_helper:
        tb_helper.write_scalars([
            ("Loss/train (epoch)", total_loss / num_batches, epoch),
            ("Loss Cat/train (epoch)", total_cat_loss / num_batches, epoch),
            ("Loss Domain/train (epoch)", total_domain_loss / num_batches, epoch),
            ("Loss Reg/train (epoch)", total_reg_loss / num_batches, epoch),
            ("AccCat/train (epoch)", total_cat_correct / count_cat, epoch),
            ("AccDomain/train (epoch)", total_domain_correct / count_domain, epoch),
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
    label_domain_counter = Counter()
    total_loss, total_cat_loss, total_reg_loss, total_domain_loss, num_batches, total_cat_correct, total_domain_correct = 0, 0, 0, 0, 0, 0, 0;
    sum_sqr_err, count_cat, count_domain = 0, 0, 0;
    inputs, label_cat, label_domain, target, model_output, model_output_cat, model_output_reg, model_output_domain  = None, None, None, None, None , None, None, None;
    pred_cat, pred_domain, pred_reg, correct_cat, correct_domain = None, None, None, None, None;
    loss, loss_cat, loss_domain, loss_reg = None, None, None, None;
    scores_cat, scores_domain, scores_reg = [], [], [];
    indexes_cat = [];
    indexes_domain = []; 
    labels_cat, labels_domain, targets, observers = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list);
    start_time = time.time()

    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y_cat, y_reg, y_domain, Z, y_cat_check, y_domain_check in tq:
                if num_batches >= 2: break;
                ### input features for the model
                inputs = [X[k].to(dev,non_blocking=True) for k in data_config.input_names]
                ### build classification true labels
                label_cat    = y_cat[data_config.label_names[0]].long()
                label_domain = y_domain[data_config.label_domain_names[0]].long()
                cat_check    = y_cat_check[data_config.labelcheck_names[0]].long()
                domain_check = y_domain_check[data_config.labelcheck_domain_names[0]].long()
                index_cat    = cat_check.nonzero();
                index_domain = domain_check.nonzero();
                if not for_training:
                    indexes_cat.append(index_cat.detach().cpu().numpy());
                    indexes_domain.append(index_domain.detach().cpu().numpy());  
                ## filter labels between classification and domain 
                label_cat    = label_cat[index_cat];
                label_domain = label_domain[index_domain];

                if label_cat.shape[0]+label_domain.shape[0] != cat_check.shape[0]:
                    _logger.warning('Wrong decomposition in cat and domain for batch number %d'%(num_batches))
                    num_batches += 1
                    continue;
                    
                label_cat    = _flatten_label(label_cat,None)
                label_domain = _flatten_label(label_domain,None)                    
                label_cat_counter.update(label_cat.cpu().numpy())
                label_domain_counter.update(label_domain.cpu().numpy())
                label_cat    = label_cat.to(dev,non_blocking=True)
                label_domain = label_domain.to(dev,non_blocking=True)

                ### build regression targets
                for idx, names in enumerate(data_config.target_names):
                    if idx == 0:
                        target = y_reg[names].float();
                    else:
                        target = torch.column_stack((target,y_reg[names].float()))
                target = target[index_cat];
                target = target.to(dev,non_blocking=True)            

                ### update counters
                num_cat_examples = max(label_cat.shape[0],target.shape[0]);
                num_domain_examples = label_domain.shape[0]
                if label_cat.shape[0] != target.shape[0]:
                    _logger.warning('Wrong decomposition in cat and target for batch number %d'%(num_batches))
                    num_batches += 1;
                    continue;

                ### define truth labels for classification and regression
                if for_training:
                    for k, v in y_cat.items():
                        labels_cat[k].append(_flatten_label(v[index_cat],None).cpu().numpy())
                    for k, v in y_domain.items():
                        labels_domain[k].append(_flatten_label(v[index_domain],None).cpu().numpy())
                    for k, v in y_reg.items():
                        targets[k].append(v[index_cat].cpu().numpy())                
                else:
                    for k, v in y_cat.items():
                        labels_cat[k].append(_flatten_label(v,None).cpu().numpy())
                    for k, v in y_domain.items():
                        labels_domain[k].append(_flatten_label(v,None).cpu().numpy())
                    for k, v in y_reg.items():
                        targets[k].append(v.cpu().numpy())                
                    for k, v in Z.items():                
                        observers[k].append(v.cpu().numpy())

                ### evaluate model
                model_output = model(*inputs)
                model_output_cat = model_output[:,:len(data_config.label_value)].squeeze().float();
                model_output_reg = model_output[:,len(data_config.label_value):len(data_config.label_value)+len(data_config.target_value)].squeeze().float();
                model_output_domain = model_output[:,len(data_config.label_value)+len(data_config.target_value):len(data_config.label_value)+len(data_config.target_value)+len(data_config.label_domain_value)].squeeze().float();
                model_output_cat    = _flatten_preds(model_output_cat,None)
                model_output_domain = _flatten_preds(model_output_domain,None)

                if (model_output_cat.shape[0] == model_output_reg.shape[0] and 
                    model_output_cat.shape[0] == model_output_domain.shape[0] and 
                    model_output_cat.shape[0] == num_cat_examples+num_domain_examples):
                    if for_training:                        
                        model_output_cat = model_output_cat[index_cat];
                        model_output_reg = model_output_reg[index_cat];
                        model_output_domain = model_output_domain[index_domain];                
                    scores_cat.append(torch.softmax(model_output_cat,dim=1).detach().cpu().numpy());
                    scores_domain.append(torch.softmax(model_output_domain,dim=1).detach().cpu().numpy());
                    scores_reg.append(model_output_reg.detach().cpu().numpy())                        
                    _, pred_cat    = model_output_cat.squeeze().max(1);
                    _, pred_domain = model_output_domain.squeeze().max(1);
                    pred_reg       = model_output_reg.squeeze().float();
                else:                    
                    _logger.warning('Wrong dimension of model output (cat vs reg vs domain vs samples in batch) for batch %d'%(num_batches))
                    pred_cat = torch.zeros(num_cat_examples).cpu().numpy();
                    pred_domain = torch.zeros(num_domain_examples).cpu().numpy();
                    pred_reg = torch.zeros(num_cat_examples).cpu().numpy();
                    if for_training:
                        scores_cat.append(torch.zeros(num_cat_examples,len(data_config.label_value)).detach().cpu().numpy());
                        scores_domain.append(torch.zeros(num_domain_examples,len(data_config.label_domain_value)).detach().cpu().numpy());
                        if len(data_config.target_value) > 1:
                            scores_reg.append(torch.zeros(num_cat_examples,len(data_config.target_value)).detach().cpu().numpy());
                        else:
                            scores_reg.append(torch.zeros(num_cat_examples).detach().cpu().numpy());
                    else:
                        scores_cat.append(torch.zeros(num_cat_examples+num_domain_examples,len(data_config.label_value)).detach().cpu().numpy());
                        scores_domain.append(torch.zeros(num_cat_examples+num_domain_examples,len(data_config.label_domain_value)).detach().cpu().numpy());
                        if len(data_config.target_value) > 1:
                            scores_reg.append(torch.zeros(num_cat_examples+num_domain_examples,len(data_config.target_value)).detach().cpu().numpy());
                        else:
                            scores_reg.append(torch.zeros(num_cat_examples+num_domain_examples).detach().cpu().numpy());

                if not for_training:
                    model_output_cat = model_output_cat[index_cat];
                    model_output_reg = model_output_reg[index_cat];
                    model_output_domain = model_output_domain[index_domain];                        
                    pred_cat    = pred_cat[index_cat];
                    pred_reg    = pred_reg[index_cat];
                    pred_domain = pred_domain[index_domain];

                ### evaluate loss function
                if loss_func != None:
                    loss, loss_cat, loss_reg, loss_domain = loss_func(model_output_cat,label_cat,model_output_reg,target,model_output_domain,label_domain)
                    loss = loss.detach().item()
                    loss_cat = loss_cat.detach().item()
                    loss_reg = loss_reg.detach().item()
                    loss_domain = loss_domain.detach().item()
                    ### erase useless dimensions
                    total_loss += loss
                    total_cat_loss += loss_cat
                    total_reg_loss += loss_reg
                    total_domain_loss += loss_domain
                else:
                    loss,loss_cat,loss_reg,loss_domain = 0,0,0,0;
                    total_loss += loss
                    total_cat_loss += loss_cat
                    total_reg_loss += loss_reg
                    total_domain_loss += loss_domain
                
                num_batches += 1
                count_cat += num_cat_examples
                count_domain += num_domain_examples

                ### classification accuracy
                if (pred_cat.shape[0] == num_cat_examples and 
                    pred_reg.shape[0] == num_cat_examples and 
                    pred_domain.shape[0] == num_domain_examples):

                    correct_cat = (pred_cat == label_cat).sum().item()
                    total_cat_correct += correct_cat
                    correct_domain = (pred_domain == label_domain).sum().item()
                    total_domain_correct += correct_domain
                    residual_reg = pred_reg - target;
                    sqr_err = residual_reg.square().sum().item()
                    sum_sqr_err += sqr_err
                    
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
                else:
                    _logger.warning('Wrong dimension of pred_cat or pred_reg or pred_domain for batch %d'%(num_batches))
                    
                if tb_helper:
                    if tb_helper.custom_fn:
                        with torch.no_grad():
                            tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=num_batches,
                                                mode='eval' if for_training else 'test')

                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count_cat+count_domain, (count_cat+count_domain) / time_diff))
    _logger.info('Evaluation class distribution: \n    %s', str(sorted(label_cat_counter.items())))
    _logger.info('Evaluation domain distribution: \n    %s', str(sorted(label_domain_counter.items())))

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

    scores_cat = np.concatenate(scores_cat).squeeze()
    scores_reg = np.concatenate(scores_reg).squeeze()
    scores_domain = np.concatenate(scores_domain).squeeze()
    if not for_training:
        indexes_cat = np.concatenate(indexes_cat).squeeze()
        indexes_domain = np.concatenate(indexes_domain).squeeze()        
    labels_cat = {k: _concat(v) for k, v in labels_cat.items()}
    labels_domain = {k: _concat(v) for k, v in labels_domain.items()}
    targets    = {k: _concat(v) for k, v in targets.items()}
    observers  = {k: _concat(v) for k, v in observers.items()}

    if not for_training:

        _logger.info('Evaluation of metrics')

        metric_cat_results = evaluate_metrics(labels_cat[data_config.label_names[0]][indexes_cat],scores_cat[indexes_cat],eval_metrics=eval_cat_metrics)            
        _logger.info('Evaluation Classification metrics: \n%s', '\n'.join(
            ['    - %s: \n%s' % (k, str(v)) for k, v in metric_cat_results.items()]))

        metric_domain_results = evaluate_metrics(labels_domain[data_config.label_domain_names[0]][indexes_domain],scores_domain[indexes_domain], eval_metrics=eval_cat_metrics)    
        _logger.info('Evaluation Domain metrics: \n%s', '\n'.join(
            ['    - %s: \n%s' % (k, str(v)) for k, v in metric_domain_results.items()]))

        _logger.info('Evaluation of regression metrics')

        for idx, (name,element) in enumerate(targets.items()):
            if len(data_config.target_names) == 1:
                metric_reg_results = evaluate_metrics(element[indexes_cat], scores_reg[indexes_cat], eval_metrics=eval_reg_metrics)
            else:
                metric_reg_results = evaluate_metrics(element[indexes_cat], scores_reg[indexes_cat,idx], eval_metrics=eval_reg_metrics)
            _logger.info('Evaluation Regression metrics for '+name+' target: \n%s', '\n'.join(
                ['    - %s: \n%s' % (k, str(v)) for k, v in metric_reg_results.items()]))        
    else:

        _logger.info('Evaluation of metrics')

        metric_cat_results = evaluate_metrics(labels_cat[data_config.label_names[0]],scores_cat,eval_metrics=eval_cat_metrics)    
        _logger.info('Evaluation Classification metrics: \n%s', '\n'.join(
            ['    - %s: \n%s' % (k, str(v)) for k, v in metric_cat_results.items()]))

        metric_domain_results = evaluate_metrics(labels_domain[data_config.label_domain_names[0]],scores_domain, eval_metrics=eval_cat_metrics)    
        _logger.info('Evaluation Domain metrics: \n%s', '\n'.join(
            ['    - %s: \n%s' % (k, str(v)) for k, v in metric_domain_results.items()]))

        _logger.info('Evaluation of regression metrics')

        for idx, (name,element) in enumerate(targets.items()):
            if len(data_config.target_names) == 1:
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
        scores_reg = scores_reg.reshape(len(scores_reg),len(data_config.target_names))
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
    label_domain_counter = Counter()
    total_loss, num_batches, total_cat_correct, total_domain_correct, sum_sqr_err, count_cat, count_domain = 0, 0, 0, 0, 0, 0, 0;
    labels_cat, labels_domain, targets, observers = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    scores_cat, scores_reg, scores_domain = [], [], []
    pred_cat, pred_domain, pred_reg = None, None, None;
    inputs, label_cat, label_domain, target, model_output, model_output_cat, model_output_reg, model_output_domain  = None, None, None, None, None , None, None, None;
    indexes_cat = [];
    indexes_domain = [];
 
    start_time = time.time()

    with tqdm.tqdm(test_loader) as tq:
        for X, y_cat, y_reg, y_domain, Z, y_cat_check, y_domain_check in tq:
            ### input features for the model
            inputs = {k: v.numpy() for k, v in X.items()}
            ### build classification true labels
            label_cat    = y_cat[data_config.label_names[0]].long()
            label_domain = y_domain[data_config.label_domain_names[0]].long()
            cat_check    = y_cat_check[data_config.labelcheck_names[0]].long()
            domain_check = y_domain_check[data_config.labelcheck_domain_names[0]].long()
            index_cat    = cat_check.nonzero();
            index_domain = domain_check.nonzero();
            indexes_cat.append(index_cat.detach().cpu().numpy());
            indexes_domain.append(index_domain.detach().cpu().numpy());  
            label_cat    = label_cat[index_cat];
            label_domain = label_domain[index_domain];
            if label_cat.shape[0]+label_domain.shape[0] != cat_check.shape[0]:
                _logger.warning('Wrong decomposition in cat and domain for batch number %d'%(num_batches))
                num_batches += 1
                continue;

            label_cat    = _flatten_label(label_cat,None)
            label_domain = _flatten_label(label_domain,None)                    
            label_cat_counter.update(label_cat.cpu().numpy())
            label_domain_counter.update(label_domain.cpu().numpy())
            label_cat    = label_cat.to(dev,non_blocking=True)
            label_domain = label_domain.to(dev,non_blocking=True)

            ### build regression targets
            for idx, names in enumerate(data_config.target_names):
                if idx == 0:
                    target = y_reg[names].float();
                else:
                    target = torch.column_stack((target,y_reg[names].float()))
            target = target[index_cat];
            target = target.to(dev,non_blocking=True)            

            ### update counters
            num_cat_examples = max(label_cat.shape[0],target.shape[0]);
            num_domain_examples = label_domain.shape[0]
            if label_cat.shape[0] != target.shape[0]:
                _logger.warning('Wrong decomposition in cat and target for batch number %d'%(num_batches))
                num_batches += 1;
                continue;

            ### define truth labels for classification and regression
            for k, v in y_cat.items():
                labels_cat[k].append(_flatten_label(v,None).cpu().numpy())
            for k, v in y_domain.items():
                labels_domain[k].append(_flatten_label(v,None).cpu().numpy())
            for k, v in y_reg.items():
                targets[k].append(v.cpu().numpy())                
            for k, v in Z.items():                
                observers[k].append(v.cpu().numpy())

            ### output of the mode
            model_output = sess.run([], inputs)
            model_output = torch.as_tensor(np.array(model_output)).squeeze();
            
            model_output_reg = model_output[:,len(data_config.label_value):len(data_config.label_value)+len(data_config.target_value)];
            model_output_domain = model_output[:,len(data_config.label_value)+len(data_config.target_value):len(data_config.label_value)+len(data_config.target_value)+len(data_config.label_domain_value)];
            model_output_cat    = _flatten_preds(model_output_cat,None).squeeze().float()
            model_output_domain = _flatten_preds(model_output_domain,None).squeeze().float()
            
            if (model_output_cat.shape[0] == model_output_reg.shape[0] and 
                model_output_cat.shape[0] == model_output_domain.shape[0] and 
                model_output_cat.shape[0] == num_cat_examples+num_domain_examples):
                scores_cat.append(model_output_cat);
                scores_domain.append(model_output_domain);
                scores_reg.append(model_output_reg)                        
                _, pred_cat    = model_output_cat.squeeze().max(1);
                _, pred_domain = model_output_domain.squeeze().max(1);
                pred_reg       = model_output_reg.squeeze().float();
            else:
                _logger.warning('Wrong dimension of model output (cat vs reg vs domain vs samples in batch) for batch %d'%(num_batches))
                pred_cat = torch.zeros(num_cat_examples).cpu().numpy();
                pred_reg = torch.zeros(num_cat_examples).cpu().numpy();
                pred_domain = torch.zeros(num_domain_examples).cpu().numpy();
                scores_cat.append(torch.zeros(num_cat_examples+num_domain_examples,len(data_config.label_value)).detach().cpu().numpy());
                scores_domain.append(torch.zeros(num_cat_examples+num_domain_examples,len(data_config.label_domain_value)).detach().cpu().numpy());
                if len(data_config.target_value) > 1:
                    scores_reg.append(torch.zeros(num_cat_examples+num_domain_examples,len(data_config.target_value)).detach().cpu().numpy());
                else:
                    scores_reg.append(torch.zeros(num_cat_examples+num_domain_examples).detach().cpu().numpy());
                    
            ### check dimension of labels and target. If dimension is 1 extend them
            if label_cat.dim() == 1:
                label_cat = label_cat[:,None]
            if label_domain.dim() == 1:
                label_domain = label_domain[:,None]
            if target.dim() == 1:
                target = target[:,None]
                    
            pred_cat    = pred_cat[index_cat];
            pred_reg    = pred_reg[index_cat];
            pred_domain = pred_domain[index_domain];

            num_batches += 1
            count_cat += num_cat_examples
            count_domain += num_domain_examples
             
            if (pred_cat.shape[0] == num_cat_examples and 
                pred_reg.shape[0] == num_cat_examples and 
                pred_domain.shape[0] == num_domain_examples):

                correct_cat = (pred_cat == label_cat).sum().item()
                total_cat_correct += correct_cat
                correct_domain = (pred_domain == label_domain).sum().item()
                total_domain_correct += correct_domain
                residual_reg = pred_reg - target;
                sqr_err = residual_reg.square().sum().item()
                sum_sqr_err += sqr_err
            
                ### monitor results
                tq.set_postfix({
                    'AccCat': '%.5f' % (correct_cat / num_cat_examples if num_cat_examples else 0),
                    'AvgAccCat': '%.5f' % (total_cat_correct / count_cat if count_cat else 0),
                    'AccDomain': '%.5f' % (correct_domain / num_domain_examples if num_domain_examples else 0),
                    'AvgAccDomain': '%.5f' % (total_domain_correct / count_domain if count_domain else 0),
                    'MSE': '%.5f' % (sqr_err / num_cat_examples if num_cat_examples else 0),
                    'AvgMSE': '%.5f' % (sum_sqr_err / count_cat if count_cat else 0),
                })
            else:
                _logger.warning('Wrong dimension of pred_cat or pred_reg or pred_domain for batch %d'%(num_batches))
                
    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count_cat+count_domain, (count_cat+count_domain) / time_diff))
    _logger.info('Evaluation class distribution: \n    %s', str(sorted(label_cat_counter.items())))
    _logger.info('Evaluation domain distribution: \n    %s', str(sorted(label_domain_counter.items())))
    
    scores_cat = np.concatenate(scores_cat).squeeze()
    scores_reg = np.concatenate(scores_reg).squeeze()
    scores_domain = np.concatenate(scores_domain).squeeze()

    indexes_cat = np.concatenate(indexes_cat).squeeze()
    indexes_domain = np.concatenate(indexes_domain).squeeze()

    labels_cat  = {k: _concat(v) for k, v in labels_cat.items()}
    labels_domain  = {k: _concat(v) for k, v in labels_domain.items()}
    targets = {k: _concat(v) for k, v in targets.items()}
    observers = {k: _concat(v) for k, v in observers.items()}

    metric_cat_results = evaluate_metrics(labels_cat[data_config.label_names[0]][indexes_cat],scores_cat[indexes_cat],eval_metrics=eval_cat_metrics)        
    _logger.info('Evaluation Classification metrics: \n%s', '\n'.join(
        ['    - %s: \n%s' % (k, str(v)) for k, v in metric_cat_results.items()]))

    metric_domain_results = evaluate_metrics(labels_domain[data_config.label_domain_names[0]][indexes_domain],scores_domain[indexes_domain], eval_metrics=eval_cat_metrics)    
    _logger.info('Evaluation Domain metrics: \n%s', '\n'.join(
        ['    - %s: \n%s' % (k, str(v)) for k, v in metric_domain_results.items()]))

    _logger.info('Evaluation of regression metrics\n')
    for idx, (name,element) in enumerate(targets.items()):
        if len(data_config.target_names) == 1:
            metric_reg_results = evaluate_metrics(element[indexes_cat], scores_reg[indexes_cat], eval_metrics=eval_reg_metrics)
        else:
            metric_reg_results = evaluate_metrics(element[indexes_cat], scores_reg[indexes_cat,idx], eval_metrics=eval_reg_metrics)
        _logger.info('Evaluation Regression metrics for '+name+' target: \n%s', '\n'.join(
            ['    - %s: \n%s' % (k, str(v)) for k, v in metric_reg_results.items()]))        
    
    scores_reg = scores_reg.reshape(len(scores_reg),len(data_config.target_names))
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
