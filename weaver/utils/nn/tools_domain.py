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


## train a classifier for which classes are condensed into a single label_name --> argmax of numpy
def train_classification(model, loss_func, opt, scheduler, train_loader, dev, epoch, steps_per_epoch=None, grad_scaler=None, tb_helper=None):


    model.train()
    torch.backends.cudnn.benchmark = True; 
    torch.backends.cudnn.enabled = True;
    gc.enable();

    data_config = train_loader.dataset.config
    label_cat_counter = Counter()
    label_domain_counter = Counter()
    count_cat, count_domain, num_batches, total_loss, total_loss_cat, total_loss_domain, total_correct_cat, total_correct_domain = 0, 0, 0, 0, 0, 0, 0, 0;
    loss, loss_cat, loss_domain, inputs, label_cat, label_domain = None, None, None, None, None, None; 
    model_output, logits_cat, logits_domain, preds_cat, preds_domain, correct_cat, correct_domain = None, None, None, None, None, None;

    start_time = time.time()

    with tqdm.tqdm(train_loader) as tq:
        for X, y_cat, _, y_domain, _ in tq:
            inputs = [X[k].to(dev,non_blocking=True) for k in data_config.input_names]
            label_cat = y_cat[data_config.label_names[0]].long()
            label_domain = y_domain[data_config.label_domain_names[0]].long()

            ## erase rows with all zeros to disentangle labels 
            index_cat = label_cat.any(1).nonzero()[:,0]
            index_domain = label_domain.any(1).nonzero()[:,0]
            label_cat = label_cat[index_cat];
            label_domain = label_domain[index_domain];

            if label_cat.shape[0]+label_domain.shape[0] != inputs.shape[0]:
                 _logger.warning('Wrong decomposition in cat and domain for batch number '%(num_batches))
                 continue;

            label_cat    = _flatten_label(label_cat, None)            
            label_domain = _flatten_label(label_domain, None)
            num_examples_cat = label_cat.shape[0]
            num_examples_domain = label_domain.shape[0]

            label_cat_counter.update(label_cat.cpu().numpy())
            label_domain_counter.update(label_domain.cpu().numpy())
            label_cat = label_cat.to(dev,non_blocking=True)
            label_domain = label_domain.to(dev,non_blocking=True)

            model.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                model_output = model(*inputs)
                model_output_cat = model_output[:,:len(data_config.label_value)];
                model_output_domain = model_output[:,len(data_config.label_value):len(data_config.label_value)+len(data_config.label_domain_value)];
                model_output_cat = model_output_cat[index_cat];
                model_output_domain = model_output_domain[index_domain];
                logits_cat = _flatten_preds(model_output_cat,None); 
                logits_domain = _flatten_preds(model_output_domain,None); 
                loss, loss_cat, loss_domain = loss_func(logits_cat, label_cat, logits_domain, label_domain)

            if grad_scaler is None:
                loss.backward()
                opt.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt)
                grad_scaler.update()

            if scheduler and getattr(scheduler, '_update_per_step', True):
                scheduler.step()

            loss = loss.detach().item()
            loss_cat = loss_cat.detach().item()
            loss_domain = loss_domain.detach().item()
            label_cat = label_cat.detach();
            label_domain = label_domain.detach();

            _, preds_cat = logits_cat.detach().max(1)
            _, preds_domain = logits_domain.detach().max(1)

            num_batches += 1
            count_cat += num_examples_cat
            count_domain += num_examples_domain

            correct_cat = (preds_cat == label_cat).sum().item()
            correct_domain = (preds_domain == label_domain).sum().item()

            total_loss += loss
            total_loss_cat += loss_cat
            total_loss_domain += loss_domain
            total_correct_cat += correct_cat
            total_correct_domain += correct_domain

            tq.set_postfix({
                'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                'Loss': '%.5f' % loss,
                'AvgLoss': '%.5f' % (total_loss / num_batches),
                'AccCat': '%.5f' % (correct_cat / num_examples_cat),
                'AvgAccCat': '%.5f' % (total_correct_cat / count_cat),
                'AccDomain': '%.5f' % (correct_domain / num_examples_domain),
                'AvgAccDomain': '%.5f' % (total_correct_domain / count_domain)
            });

            if tb_helper:
                tb_helper.write_scalars([
                    ("Loss/train", loss, tb_helper.batch_train_count + num_batches),
                    ("AccCat/train", correct_cat / num_examples_cat, tb_helper.batch_train_count + num_batches),
                    ("AccDomain/train", correct_domain / num_examples_domain, tb_helper.batch_train_count + num_batches),
                    ])
                if tb_helper.custom_fn:
                    with torch.no_grad():
                        tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=num_batches, mode='train')

            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))
    _logger.info('Train AvgLoss: %.5f'% (total_loss / num_batches))
    _logger.info('Train AvgLossCat: %.5f'% (total_loss_cat / num_batches))
    _logger.info('Train AvgLossDomain: %.5f'% (total_loss_domain / num_batches))
    _logger.info('Train AvgAccCat: %.5f'%(total_correct_cat / count_cat))
    _logger.info('Train AvgAccDomain: %.5f'%(total_correct_domain / count_domain))
    _logger.info('Train class distribution: \n    %s', str(sorted(label_cat_counter.items())))
    _logger.info('Train domain distribution: \n    %s', str(sorted(label_domain_counter.items())))

    if tb_helper:
        tb_helper.write_scalars([
            ("Loss/train (epoch)", total_loss / num_batches, epoch),
            ("Loss Cat/train (epoch)", total_loss_cat / num_batches, epoch),
            ("Loss domain/train (epoch)", total_loss_domain / num_batches, epoch),
            ("AccCat/train (epoch)", total_correct_cat / count_cat, epoch),
            ("AccDomain/train (epoch)", total_correct_domain / count_domain, epoch),
            ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode='train')
        # update the batch state
        tb_helper.batch_train_count += num_batches

    torch.cuda.empty_cache()
    gc.collect();

## evaluate a classifier for which classes are condensed into a single label_name --> argmax of numpy
def evaluate_classification(model, test_loader, dev, epoch, for_training=True, loss_func=None, steps_per_epoch=None, tb_helper=None,
                            eval_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix']):
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
    num_batches, count_cat, count_domain, entry_count, total_correct_cat, total_correct_domain, total_loss, total_loss_cat, total_loss_domain = 0, 0, 0, 0, 0, 0, 0, 0, 0
    inputs, label_cat, label_domain, model_output, model_output_cat, model_output_domain = None, None, None, None, None, None, None;
    logits_cat, logits_domain, preds_cat, preds_domain, loss, loss_cat, loss_domain, correct_cat, correct_domain = None, None, None, None, None, None, None, None, None
    scores_cat = []
    scores_domain = []
    labels = defaultdict(list)
    labels_domain = defaultdict(list)
    targets = defaultdict(list)
    observers = defaultdict(list)

    start_time = time.time()

    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y_cat, _, y_domain, Z in tq:

                inputs = [X[k].to(dev,non_blocking=True) for k in data_config.input_names]
                label_cat = y_cat[data_config.label_names[0]].long()
                label_domain = y_domain[data_config.label_domain_names[0]].long()                

                ## erase rows with all zeros to disentangle labels 
                index_cat = label_cat.any(1).nonzero()[:,0]
                index_domain = label_domain.any(1).nonzero()[:,0]
                label_cat = label_cat[index_cat];
                label_domain = label_domain[index_domain];

                if label_cat.shape[0]+label_domain.shape[0] != inputs.shape[0]:
                    _logger.warning('Wrong decomposition in cat and domain for batch number '%(num_batches))
                    continue;

                label_cat    = _flatten_label(label_cat,None)
                label_domain = _flatten_label(label_domain,None)
                label_cat_counter.update(label_cat.cpu().numpy())
                label_domain_counter.update(label_domain.cpu().numpy())

                entry_count_cat += label_cat.shape[0]
                entry_count_domain += label_domain.shape[0]
                num_examples_cat = label_cat.shape[0]
                num_examples_domain = label_domain.shape[0]

                label_cat = label_cat.to(dev,non_blocking=True)
                label_domain = label_domain.to(dev,non_blocking=True)

                model_output = model(*inputs)
                model_output_cat = model_output[:,:len(data_config.label_value)];
                model_output_domain = model_output[:,len(data_config.label_value):len(data_config.label_value)+len(data_config.label_domain_value)];
                model_output_cat = model_output_cat[index_cat];
                model_output_domain = model_output_domain[index_domain];

                logits_cat = _flatten_preds(model_output_cat,None).float()
                logits_domain = _flatten_preds(model_output_domain,None).float()
                logits = flatten_preds(model_output,None).float()
                scores_cat.append(torch.softmax(logits,dim=1).detach().cpu().numpy())
                scores_domain.append(torch.softmax(logits,dim=1).detach().cpu().numpy())

                for k, v in y_cat.items():
                    labels_cat[k].append(_flatten_label(v,None).cpu().numpy())
                for k, v in y_domain.items():
                    labels_domain[k].append(_flatten_label(v,None).cpu().numpy())
                if not for_training:
                    for k, v in Z.items():
                        observers[k].append(v.cpu().numpy())

                _, preds_cat    = logits_cat.max(1)
                _, preds_domain = logits_domain.max(1)
                
                loss, loss_cat, loss_domain = 0, 0, 0
                if loss_func not None:
                    loss, loss_cat, loss_domain  = loss_func(logits_cat, label_cat, logits_domain, label_domain).detach().item();

                num_batches += 1
                count_cat += num_examples_cat
                count_domain += num_examples_domain

                correct_cat = (preds_cat == label_cat).sum().item()
                correct_domain = (preds_domain == label_domain).sum().item()

                total_loss += loss * (num_examples_cat+num_examples_domain)
                total_loss_cat += loss_cat * (num_examples_cat)
                total_loss_domain += loss_domain * (num_examples_domain)
                total_correct_cat += correct_cat
                total_correct_domain += correct_domain

                tq.set_postfix({
                    'Loss': '%.5f' % loss,
                    'AvgLoss': '%.5f' % (total_loss / num_batches),
                    'AccCat': '%.5f' % (correct_cat / num_examples_cat),
                    'AvgAccCat': '%.5f' % (total_correct_cat / count_cat),
                    'AccDomain': '%.5f' % (correct_domain / num_examples_domain),
                    'AvgAccDomain': '%.5f' % (total_correct_domain / count_domain)})

                if tb_helper:
                    if tb_helper.custom_fn:
                        with torch.no_grad():
                            tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=num_batches,
                                                mode='eval' if for_training else 'test')

                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))
    _logger.info('Evaluation class  distribution: \n    %s', str(sorted(label_cat_counter.items())))
    _logger.info('Evaluation domain distribution: \n    %s', str(sorted(label_domain_counter.items())))

    if tb_helper:
        tb_mode = 'eval' if for_training else 'test'
        tb_helper.write_scalars([
            ("Loss/%s (epoch)" % tb_mode, total_loss / num_batches, epoch),
            ("AccCat/%s (epoch)" % tb_mode, total_correct_cat / count_cat, epoch),
            ("AccDomain/%s (epoch)" % tb_mode, total_correct_domain / count_domain, epoch),
            ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode=tb_mode)

    scores_cat = np.concatenate(scores_cat)
    scores_domain = np.concatenate(scores_domain)
    labels_cat = {k: _concat(v) for k, v in labels_cat.items()}
    labels_domain = {k: _concat(v) for k, v in labels_domain.items()}
    observers = {k: _concat(v) for k, v in observers.items()}

    metric_cat_results = evaluate_metrics(labels_cat[data_config.label_names[0]], scores_cat, eval_metrics=eval_metrics)
    _logger.info('Evaluation Cat metrics: \n%s', '\n'.join(
        ['    - %s: \n%s' % (k, str(v)) for k, v in metric_cat_results.items()]))
    metric_domain_results = evaluate_metrics(labels_domain[data_config.label_domain_names[0]], scores_domain, eval_metrics=eval_metrics)
    _logger.info('Evaluation Domain metrics: \n%s', '\n'.join(
        ['    - %s: \n%s' % (k, str(v)) for k, v in metric_domain_results.items()]))

    torch.cuda.empty_cache()

    ## best epoch is the one providing the best categorization accuracy i.e. neglecting the adversarial
    if for_training:
        gc.collect();
        return total_correct_cat / count
    else:
        gc.collect();
        return total_loss / num_batches, scores_cat, labels_cat, targets, scores_domain, labels_domain, observers


## evaluate a classifier for which classes are condensed into a single label_name --> argmax of numpy --> use ONNX instead of pytorch
def evaluate_onnx_classification(model_path, test_loader, eval_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix']):

    import onnxruntime
    sess = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    data_config = test_loader.dataset.config
    gc.enable();

    label_cat_counter = Counter()
    label_domain_counter = Counter()
    total_correct_cat, total_correct_domain, count_cat, count_domain = 0, 0, 0, 0
    scores_cat = []
    scores_domain = []
    labels_cat = defaultdict(list)
    labels_domain = defaultdict(list)
    targets = defaultdict(list)
    observers = defaultdict(list)
    inputs, label_cat, label_domain, score, preds_cat, preds_domain, correct_cat, correct_domain = None, None, None, None, None, None, None, None
    start_time = time.time();

    with tqdm.tqdm(test_loader) as tq:
        for X, y_cat, _, y_domain, Z in tq:

            inputs = {k: v.cpu().numpy() for k, v in X.items()}
            label_cat = y_cat[data_config.label_names[0]].long()
            label_domain = y_domain[data_config.label_domain_names[0]].long()

            index_cat = label_cat.any(1).nonzero()[:,0]
            index_domain = label_domain.any(1).nonzero()[:,0]

            label_cat = label_cat[index_cat];
            label_domain = label_domain[index_domain];

            if label_cat.shape[0]+label_domain.shape[0] != inputs.shape[0]:
                _logger.warning('Wrong decomposition in cat and domain for batch number '%(num_batches))
                continue;

            num_examples_cat = label_cat.shape[0]
            num_examples_domain = label_domain.shape[0]
            label_cat_counter.update(label_cat.cpu.numpy())
            label_domain_counter.update(label_domain.cpu.numpy())
            score = sess.run([], inputs)
            score = torch.as_tensor(np.array(score)).squeeze();
            preds_cat = score[:,:len(data_config.label_value)].argmax(1)
            preds_cat = preds_cat[index_cat];
            preds_domain = score[:,len(data_config.label_value):len(data_config.label_value)+len(data_config.label_domain_value)].argmax(1)
            preds_domain = preds_domain[index_domain];
            
            scores_cat.append(score[:,:len(data_config.label_value)])
            scores_domain.append(score[:,len(data_config.label_value):len(data_config.label_value)+len(data_config.label_domain_value)])

            for k, v in y_cat.items():
                labels_cat[k].append(v.cpu().numpy())
            for k, v in y_domain.items():
                labels_domain[k].append(v.cpu().numpy())
            for k, v in Z.items():
                observers[k].append(v.cpu().numpy())

            correct_cat = (preds_cat == label_cat).sum()
            correct_domain = (preds_domain == label_domain).sum()
            total_correct_cat += correct_cat
            total_correct_domain += correct_domain
            count_cat += num_examples_cat
            count_domain += num_examples_domain

            tq.set_postfix({
                'AccCat': '%.5f' % (correct_cat / num_examples_cat),
                'AvgAccCat': '%.5f' % (total_correct_cat / count_cat),
                'AccDomain': '%.5f' % (correct_domain / num_examples_domain),
                'AvgAccDomain': '%.5f' % (total_correct_domain / count_domain)})

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))
    _logger.info('Evaluation class distribution: \n    %s', str(sorted(label_cat_counter.items())))
    _logger.info('Evaluation domain distribution: \n    %s', str(sorted(label_domain_counter.items())))

    scores_cat = np.concatenate(scores_cat)
    labels_cat = {k: _concat(v) for k, v in labels_cat.items()}
    scores_domain = np.concatenate(scores_domain)
    labels_domain = {k: _concat(v) for k, v in labels_domain.items()}
    observers = {k: _concat(v) for k, v in observers.items()}

    metric_cat_results = evaluate_metrics(labels_cat[data_config.label_names[0]], scores_cat, eval_metrics=eval_metrics)
    _logger.info('Evaluation Cat metrics: \n%s', '\n'.join(
        ['    - %s: \n%s' % (k, str(v)) for k, v in metric_cat_results.items()]))

    metric_domain_results = evaluate_metrics(labels_domain[data_config.label_domain_names[0]], scores_domain, eval_metrics=eval_metrics)
    _logger.info('Evaluation Domain metrics: \n%s', '\n'.join(
        ['    - %s: \n%s' % (k, str(v)) for k, v in metric_domain_results.items()]))

    gc.collect();
    return total_correct / count, scores_cat, labels_cat, targets, scores_domain, labels_domain, observers



## train classification + regssion into a total loss --> best training epoch decided on the loss function
def train_classreg(model, loss_func, opt, scheduler, train_loader, dev, epoch, steps_per_epoch=None, grad_scaler=None, tb_helper=None):

    model.train()

    torch.backends.cudnn.benchmark = True;
    torch.backends.cudnn.enabled = True;

    gc.enable();

    data_config = train_loader.dataset.config

    num_batches, total_loss, total_cat_loss, total_reg_loss, total_domain_loss, count = 0, 0, 0, 0, 0,0
    label_cat_counter = Counter()
    label_domain_counter = Counter()
    total_cat_correct, total_domain_correct, sum_abs_err, sum_sqr_err = 0, 0 ,0, 0, 0
    inputs, target, label_cat, label_domain, model_output, model_output_cat, model_output_reg, model_output_domain = None, None, None, None, None, None, None, None;
    loss, loss_cat, loss_reg, loss_domain, pred_cat, pred_reg, pred_domain, residual_reg, correct_cat, correct_loss = None, None, None, None, None, None, None, None, None, None;

    start_time = time.time()

    with tqdm.tqdm(train_loader) as tq:
        for X, y_cat, y_reg, y_domain, _ in tq:
            ### input features for the model
            inputs = [X[k].to(dev,non_blocking=True) for k in data_config.input_names]
            ### build classification true labels (numpy argmax)
            label_cat = y_cat[data_config.label_names[0]].long()
            label_domain = y_domain_cat[data_config.label_domain_names[0]].long()

            index_cat = label_cat.any(1).nonzero()[:,0]
            index_domain = label_domain.any(1).nonzero()[:,0]
            label_cat = label_cat[index_cat];
            label_domain = label_domain[index_cat];
            label_cat_counter.update(label_cat.cpu().numpy())
            label_domain_counter.update(label_domain.cpu().numpy())

            label_cat = label_cat.to(dev,non_blocking=True)
            label_dev = label_dev.to(dev,non_blocking=True)

            ### build regression targets
            for idx, names in enumerate(data_config.target_names):
                if idx == 0:
                    target = y_reg[names].float();
                else:
                    target = torch.column_stack((target,y_reg[names].float()))
            target = target[index_cat];
            target = target.to(dev,non_blocking=True)            
            ### Number of samples in the batch
            num_examples_cat = max(label_cat.shape[0],target.shape[0]);
            num_examples_domain = label_domain.shape[0];
            ### loss minimization
            model.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                ### evaluate the model
                model_output  = model(*inputs)         
                model_output_cat = model_output[:,len(
                ### check dimension of labels and target. If dimension is 1 extend them
                if label_cat.dim() == 1:
                    label_cat = label_cat[:,None]
                if label_domain.dim() == 1:
                    label_domain = label_domain[:,None]
                if target.dim() == 1:
                    target = target[:,None]
                ### erase uselss dimensions
                label_cat  = label_cat.squeeze();
                label_domain  = label_domain.squeeze();
                target = target.squeeze();
                ### evaluate loss function
                loss, loss_cat, loss_reg, loss_domain = loss_func(model_output,label_cat,target,label_domain);

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
            num_batches += 1
            count += num_examples;
            
            ## take the classification prediction and compare with the true labels            
            label_cat = label_cat.detach()
            label_domain = label_domain.detach()
            target = target.detach()
            if(model_output.dim() == 1) : continue;
            _, pred_cat = model_output[:,:len(data_config.label_value)].squeeze().max(1);
            _, pred_domain = model_output[:,len(data_config.label_value)+len(data_config.target_value:len(data_config.label_value)+len(data_config.target_value+len(data_config.label_domain_value)].squeeze().max(1);
            correct_cat = (pred_cat.detach() == label_cat.detach()).sum().item()
            correct_domain = (pred_domain.detach() == label_domain.detach()).sum().item()
            total_cat_correct += correct_cat
            total_domain_correct += correct_domain

            ## take the regression prediction and compare with true targets
            pred_reg = model_output[:,len(data_config.label_value):len(data_config.label_value)+len(data_config.target_value)].squeeze().float();
            residual_reg = pred_reg.detach() - target.detach();            
            abs_err = residual_reg.abs().sum().item();
            sum_abs_err += abs_err;
            sqr_err = residual_reg.square().sum().item()
            sum_sqr_err += sqr_err

            ### monitor metrics
            tq.set_postfix({
                'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                'Loss': '%.5f' % loss,
                'AvgLoss': '%.5f' % (total_loss / num_batches),
                'AccCat': '%.5f' % (correct_cat / num_examples),
                'AvgAccCat': '%.5f' % (total_correct_cat / count),
                'AccDomain': '%.5f' % (correct_domain / num_examples),
                'AvgAccDomain': '%.5f' % (total_correct_domain / count),
                'MSE': '%.5f' % (sqr_err / num_examples),
                'AvgMSE': '%.5f' % (sum_sqr_err / count),
                'MAE': '%.5f' % (abs_err / num_examples),
                'AvgMAE': '%.5f' % (sum_abs_err / count),
                
            })

            if tb_helper:
                tb_helper.write_scalars([
                    ("Loss/train", loss, tb_helper.batch_train_count + num_batches),
                    ("AccCat/train", correct_cat / num_examples, tb_helper.batch_train_count + num_batches),
                    ("AccDomain/train", correct_domain / num_examples, tb_helper.batch_train_count + num_batches),
                    ("MSE/train", sqr_err / num_examples, tb_helper.batch_train_count + num_batches),
                    ("MAE/train", abs_err / num_examples, tb_helper.batch_train_count + num_batches),
                    ])
                if tb_helper.custom_fn:
                    with torch.no_grad():
                        tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=num_batches, mode='train')

            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break

    ### training summary
    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))
    _logger.info('Train AvgLoss: %.5f'% (total_loss / num_batches))
    _logger.info('Train AvgLoss Cat: %.5f'% (total_cat_loss / num_batches))
    _logger.info('Train AvgLoss Domain: %.5f'% (total_domain_loss / num_batches))
    _logger.info('Train AvgLoss Reg: %.5f'% (total_reg_loss / num_batches))
    _logger.info('Train AvgAccCat: %.5f'%(total_correct_cat / count))
    _logger.info('Train AvgAccDomain: %.5f'%(total_correct_domain / count))
    _logger.info('Train AvgMSE: %.5f'%(sum_sqr_err / count))
    _logger.info('Train AvgMAE: %.5f'%(sum_abs_err / count))
    _logger.info('Train class distribution: \n %s', str(sorted(label_cat_counter.items())))
    _logger.info('Train domain distribution: \n %s', str(sorted(label_domain_counter.items())))

    if tb_helper:
        tb_helper.write_scalars([
            ("Loss/train (epoch)", total_loss / num_batches, epoch),
            ("Loss Cat/train (epoch)", total_cat_loss / num_batches, epoch),
            ("Loss Domain/train (epoch)", total_domain_loss / num_batches, epoch),
            ("Loss Reg/train (epoch)", total_reg_loss / num_batches, epoch),
            ("Acc/train (epoch)", total_correct / count, epoch),
            ("MSE/train (epoch)", sum_sqr_err / count, epoch),
            ("MAE/train (epoch)", sum_abs_err / count, epoch),
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
    sum_sqr_err, sum_abs_err, entry_count, count = 0, 0, 0;
    inputs, label_cat, label_domain target,  model_output, pred_cat_output, pred_domain_output, pred_reg = None, None, None, None, None , None, None;
    loss, loss_cat, loss_domain loss_reg = None, None, None, None;
    scores_cat, scores_domain, scores_reg = [], [], [];
    labels_cat, labels_domain, targets, observers = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list);
    start_time = time.time()

    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y_cat, y_reg, Z, y_domain in tq:
                ### input features for the model
                inputs = [X[k].to(dev,non_blocking=True) for k in data_config.input_names]
                ### build classification true labels
                label_cat  = y_cat[data_config.label_names[0]].long()
                label_domain  = y_cat[data_config.label_domain_names[0]].long()
                label_cat  = _flatten_label(label_cat,None)
                label_domain  = _flatten_label(label_domain,None)
                label_cat_counter.update(label_cat.cpu().numpy())
                label_domain_counter.update(label_domain.cpu().numpy())
                label_cat  = label_cat.to(dev,non_blocking=True)
                label_domain  = label_domain.to(dev,non_blocking=True)
                ### build regression targets
                for idx, names in enumerate(data_config.target_names):
                    if idx == 0:
                        target = y_reg[names].float();
                    else:
                        target = torch.column_stack((target,y_reg[names].float()))
                target = target.to(dev,non_blocking=True)            
                ### update counters
                num_examples = max(label_cat.shape[0],target.shape[0]);
                entry_count += num_examples
                ### evaluate model
                model_output = model(*inputs)
                ### define truth labels for classification and regression
                for k, name in enumerate(data_config.label_names):                    
                    labels[name].append(_flatten_label(y_cat[name],None).cpu().numpy())
                for k, name in enumerate(data_config.label_domain_names):                    
                    labels_domain[name].append(_flatten_label(y_domain[name],None).cpu().numpy())
                for k, name in enumerate(data_config.target_names):
                    targets[name].append(y_reg[name].cpu().numpy())                
                ### observers
                if not for_training:
                    for k, v in Z.items():
                        observers[k].append(v.cpu().numpy())
                ### build classification and regression outputs
                pred_cat_output = model_output[:,:len(data_config.label_value)].squeeze().float()
                pred_domain_output = model_output[:,len(data_config.label_value)+len(data_config.target_value):len(data_config.label_value)+len(data_config.target_value)+len(data_config.label_domain_value)].squeeze().float()
                pred_reg = model_output[:,len(data_config.label_value):len(data_config.label_value)+len(data_config.target_value)].squeeze().float();                
                if pred_cat_output.shape[0] == num_examples and pred_reg.shape[0] == num_examples:
                    _, pred_cat = pred_cat_output.max(1);
                    _, pred_domain = pred_domain_output.max(1);
                    scores_cat.append(torch.softmax(pred_cat_output,dim=1).detach().cpu().numpy());
                    scores_domain.append(torch.softmax(pred_domain_output,dim=1).detach().cpu().numpy());
                    scores_reg.append(pred_reg.detach().cpu().numpy())
                else:
                    pred_cat = torch.zeros(num_examples).cpu().numpy();
                    pred_domain = torch.zeros(num_examples).cpu().numpy();
                    scores_cat.append(torch.zeros(num_examples,len(data_config.label_value)).detach().cpu().numpy());
                    scores_domain.append(torch.zeros(num_examples,len(data_config.label_domain_value)).detach().cpu().numpy());
                    if len(data_config.target_value) > 1:
                        scores_reg.append(torch.zeros(num_examples,len(data_config.target_value)).detach().cpu().numpy());
                    else:
                        scores_reg.append(torch.zeros(num_examples).detach().cpu().numpy());
                    
                ### evaluate loss function
                if loss_func != None:
                    ### check dimension of labels and target. If dimension is 1 extend them
                    if label_cat.dim() == 1:
                        label_cat = label_cat[:,None]
                    if label_domain.dim() == 1:
                        label_domain = label_domain[:,None]
                    if target.dim() == 1:
                        target = target[:,None]
                    ### true labels and true target 
                    loss, loss_cat, loss_reg = loss_func(model_output,label_cat,target,)
                    loss = loss.detach().item()
                    loss_cat = loss_cat.detach().item()
                    loss_reg = loss_reg.detach().item()
                    ### erase useless dimensions
                    label  = label.squeeze();
                    target = target.squeeze();                                         
                    total_loss += loss
                    total_cat_loss += loss_cat
                    total_reg_loss += loss_reg
                else:
                    loss,loss_cat,loss_reg = 0,0,0;
                    total_loss += loss
                    total_cat_loss += loss_cat
                    total_reg_loss += loss_reg

                num_batches += 1
                count += num_examples

                ### classification accuracy
                if pred_cat_output.shape[0] == num_examples and pred_reg.shape[0] == num_examples:

                    correct = (pred_cat == label).sum().item()
                    total_correct += correct
                    ### regression spread
                    residual_reg = pred_reg - target;
                    abs_err = residual_reg.abs().sum().item();
                    sum_abs_err += abs_err;
                    sqr_err = residual_reg.square().sum().item()
                    sum_sqr_err += sqr_err

                    ### monitor results
                    tq.set_postfix({
                        'Loss': '%.5f' % loss,
                        'AvgLoss': '%.5f' % (total_loss / num_batches),
                        'Acc': '%.5f' % (correct / num_examples),
                        'AvgAcc': '%.5f' % (total_correct / count),
                        'MSE': '%.5f' % (sqr_err / num_examples),
                        'AvgMSE': '%.5f' % (sum_sqr_err / count),
                        'MAE': '%.5f' % (abs_err / num_examples),
                        'AvgMAE': '%.5f' % (sum_abs_err / count),                        
                    })

                if tb_helper:
                    if tb_helper.custom_fn:
                        with torch.no_grad():
                            tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=num_batches,
                                                mode='eval' if for_training else 'test')

                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))
    _logger.info('Evaluation class distribution: \n    %s', str(sorted(label_counter.items())))

    if tb_helper:
        tb_mode = 'eval' if for_training else 'test'
        tb_helper.write_scalars([
            ("Loss/%s (epoch)"%(tb_mode), total_loss / num_batches, epoch),
            ("Loss Cat/%s (epoch)"%(tb_mode), total_cat_loss / num_batches, epoch),
            ("Loss Reg/%s (epoch)"%(tb_mode), total_reg_loss / num_batches, epoch),
            ("Acc/%s (epoch)"%(tb_mode), total_correct / count, epoch),
            ("MSE/%s (epoch)"%(tb_mode), sum_sqr_err / count, epoch),
            ("MAE/%s (epoch)"%(tb_mode), sum_abs_err / count, epoch),
            ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode=tb_mode)

    scores_cat = np.concatenate(scores_cat).squeeze()
    scores_reg = np.concatenate(scores_reg).squeeze()
    labels  = {k: _concat(v) for k, v in labels.items()}
    targets = {k: _concat(v) for k, v in targets.items()}
    observers = {k: _concat(v) for k, v in observers.items()}

    _logger.info('Evaluation of metrics\n')
    metric_cat_results = evaluate_metrics(labels[data_config.label_names[0]], scores_cat, eval_metrics=eval_cat_metrics)    
    _logger.info('Evaluation Classification metrics: \n%s', '\n'.join(
        ['    - %s: \n%s' % (k, str(v)) for k, v in metric_cat_results.items()]))

    _logger.info('Evaluation of regression metrics\n')
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
        if scores_reg.ndim and scores_cat.ndim: 
            scores_reg = scores_reg.reshape(len(scores_reg),len(data_config.target_names))
            scores = np.concatenate((scores_cat,scores_reg),axis=1)
            gc.collect();
            return total_loss / num_batches, scores, labels, targets, observers
        else:
            gc.collect();
            return total_loss / num_batches, scores_reg, labels, targets, observers;


def evaluate_onnx_classreg(model_path, test_loader,
                           eval_cat_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix'],
                           eval_reg_metrics=['mean_squared_error', 'mean_absolute_error', 'median_absolute_error', 'mean_gamma_deviance']):

    import onnxruntime
    sess = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    gc.enable();

    data_config = test_loader.dataset.config
    label_counter = Counter()
    num_batches, total_loss, total_cat_loss, total_reg_loss, total_correct, sum_sqr_err, sum_abs_err, count = 0, 0, 0, 0, 0, 0, 0, 0
    scores_cat, scores_reg = [], []
    labels, targets, observers = defaultdict(list), defaultdict(list), defaultdict(list)
    inputs, label, pred_cat, pred_reg, loss, loss_cat, loss_reg = None, None, None, None, None, None, None;
    
    start_time = time.time()

    with tqdm.tqdm(test_loader) as tq:
        for X, y_cat, y_reg, Z in tq:
            ### input features for the model
            inputs = {k: v.numpy() for k, v in X.items()}
            label = y_cat[data_config.label_names[0]].long();
            for idx, names in enumerate(data_config.target_names):
                if idx == 0:
                    target = y_reg[names].float();
                else:
                    target = torch.column_stack((target,y_reg[names].float()))
            num_examples = max(label.shape[0],target.shape[0]);
            label_counter.update(label.cpu().numpy())
            score = sess.run([], inputs)
            score = torch.as_tensor(np.array(score)).squeeze();
            scores_cat.append(score[:,:len(data_config.label_value)]);
            scores_reg.append(score[:,len(data_config.label_value):len(data_config.label_value)+len(data_config.target_value)]);            
            ### define truth labels for classification and regression
            for k, name in enumerate(data_config.label_names):                    
                labels[name].append(_flatten_label(y_cat[name],None).cpu().numpy())
            for k, name in enumerate(data_config.target_names):
                targets[name].append(y_reg[name].cpu().numpy())                
            for k, v in Z.items():
                observers[k].append(v.cpu().numpy())

            pred_cat = score[:,:len(data_config.label_value)].argmax(1).squeeze();
            pred_reg = score[:,len(data_config.label_value):len(data_config.label_value)+len(data_config.target_value)].squeeze().float();
            count += num_examples
            num_batches += 1;
            
            if label.dim() == 1:
                label = label[:,None]
            if target.dim() == 1:
                target = target[:,None]

            ### erase uselss dimensions                                                                                                                                                            
            label  = label.squeeze();
        
            if pred_cat.shape[0] == num_examples and pred_reg.shape[0] == num_examples:
                correct = (pred_cat == label).sum().item()
                total_correct += correct
                residual_reg = pred_reg - target;
                abs_err = residual_reg.abs().sum().item();
                sum_abs_err += abs_err;
                sqr_err = residual_reg.square().sum().item()
                sum_sqr_err += sqr_err

            ### monitor results
            tq.set_postfix({
                'Acc': '%.5f' % (correct / num_examples),
                'AvgAcc': '%.5f' % (total_correct / count),
                'MSE': '%.5f' % (sqr_err / num_examples),
                'AvgMSE': '%.5f' % (sum_sqr_err / count),
                'MAE': '%.5f' % (abs_err / num_examples),
                'AvgMAE': '%.5f' % (sum_abs_err / count),                        
            })

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))
    _logger.info('Evaluation class distribution: \n    %s', str(sorted(label_counter.items())))
    
    scores_cat = np.concatenate(scores_cat).squeeze()
    scores_reg = np.concatenate(scores_reg).squeeze()
    labels  = {k: _concat(v) for k, v in labels.items()}
    targets = {k: _concat(v) for k, v in targets.items()}
    observers = {k: _concat(v) for k, v in observers.items()}

    metric_cat_results = evaluate_metrics(labels[data_config.label_names[0]], scores_cat, eval_metrics=eval_cat_metrics)    
    _logger.info('Evaluation Classification metrics: \n%s', '\n'.join(
        ['    - %s: \n%s' % (k, str(v)) for k, v in metric_cat_results.items()]))

    for idx, (name,element) in enumerate(targets.items()):
        if len(data_config.target_names) == 1:
            metric_reg_results = evaluate_metrics(element, scores_reg, eval_metrics=eval_reg_metrics)
        else:
            metric_reg_results = evaluate_metrics(element, scores_reg[:,idx], eval_metrics=eval_reg_metrics)

        _logger.info('Evaluation Regression metrics for '+name+' target: \n%s', '\n'.join(
            ['    - %s: \n%s' % (k, str(v)) for k, v in metric_reg_results.items()]))        
    
    if scores_reg.ndim and scores_cat.ndim: 
        scores_reg = scores_reg.reshape(len(scores_reg),len(data_config.target_names))
        scores = np.concatenate((scores_cat,scores_reg),axis=1)        
        gc.collect();
        return total_loss / num_batches, scores, labels, targets, observers
    else:
        gc.collect();
        return total_loss / num_batches, scores_reg, labels, targets, observers

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
