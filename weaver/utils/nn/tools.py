import numpy as np
import awkward as ak
import tqdm
import time
import torch
import ast

from collections import defaultdict, Counter
from .metrics import evaluate_metrics
from ..data.tools import _concat
from ..logger import _logger
from .utils import _flatten_label, _flatten_preds, fgsm_attack, fngm_attack


## train a classifier for which classes are condensed into a single label_name --> argmax of numpy
def train_classification(model, loss_func, opt, scheduler, train_loader, dev, epoch, steps_per_epoch=None, grad_scaler=None, tb_helper=None, network_option=None):

    model.train()
    torch.backends.cudnn.benchmark = True; 
    torch.backends.cudnn.enabled = True;

    data_config = train_loader.dataset.config
    label_counter = Counter()
    count, num_batches, total_loss, total_correct = 0, 0, 0, 0
    loss, inputs, label, weight, label_mask, model_output, preds, correct = None, None, None, None, None, None, None, None

    start_time = time.time()

    with tqdm.tqdm(train_loader) as tq:
        for X, y_cat, _, _, _, _, _, y_weight in tq:
            inputs = [X[k].to(dev,non_blocking=True) for k in data_config.input_names]
            label  = y_cat[data_config.label_names[0]].long().to(dev,non_blocking=True)
            if y_weight:
                weight = y_weight[data_config.label_sample_weight_names[0]].float().to(dev,non_blocking=True)
                weight = weight.squeeze();
            try:
                label_mask = y_cat[data_config.label_names[0] + '_mask'].bool().to(dev,non_blocking=True)
            except KeyError:
                label_mask = None
            label = _flatten_label(label,mask=label_mask)

            num_examples = label.shape[0]
            label_counter.update(label.cpu().numpy().astype(dtype=np.int32))
            label = label.to(dev,non_blocking=True)

            model.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                model_output = model(*inputs)                
                model_output, label, label_mask = _flatten_preds(model_output, label=label, mask=label_mask)
                model_output = model_output.squeeze().float();
                label  = label.squeeze();
                if label_mask:
                    label_mask = label_mask.squeeze();
                if torch.is_tensor(weight):
                    loss = loss_func(model_output, label, weight)
                else:
                    loss = loss_func(model_output, label)
                    
            if grad_scaler is None:
                loss.backward()
                opt.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt)
                grad_scaler.update()

            if scheduler and getattr(scheduler, '_update_per_step', True):
                scheduler.step()

            loss  = loss.detach().item()
            label = label.detach();
            _, preds = model_output.detach().max(1)
            num_batches += 1
            count += num_examples
            correct = (preds == label).sum().item()
            total_loss += loss
            total_correct += correct

            tq.set_postfix({
                'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                'Loss': '%.5f' % loss,
                'AvgLoss': '%.5f' % (total_loss / num_batches),
                'Acc': '%.5f' % (correct / num_examples),
                'AvgAcc': '%.5f' % (total_correct / count)})

            if tb_helper:
                tb_helper.write_scalars([
                    ("Loss/train", loss, tb_helper.batch_train_count + num_batches),
                    ("Acc/train", correct / num_examples, tb_helper.batch_train_count + num_batches),
                    ])
                if tb_helper.custom_fn:
                    with torch.no_grad():
                        tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=num_batches, mode='train')

            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))
    _logger.info('Train AvgLoss: %.5f' % (total_loss / num_batches))
    _logger.info('Train AvgAcc: %.5f'%(total_correct / count if count else 0))
    _logger.info('Train class distribution: \n    %s', str(sorted(label_counter.items())))

    if tb_helper:
        tb_helper.write_scalars([
            ("Loss/train (epoch)", total_loss / num_batches, epoch),
            ("Acc/train (epoch)", total_correct / count, epoch),
            ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode='train')
        # update the batch state
        tb_helper.batch_train_count += num_batches


## evaluate a classifier for which classes are condensed into a single label_name --> argmax of numpy
def evaluate_classification(model, test_loader, dev, epoch, for_training=True, loss_func=None, steps_per_epoch=None, tb_helper=None,
                            eval_attack=None, eps_attack=None, network_option=None, grad_scaler=None,
                            eval_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix']):

    model.eval()
    torch.backends.cudnn.benchmark = True;
    torch.backends.cudnn.enabled = True;

    data_config = test_loader.dataset.config

    label_counter = Counter()
    num_batches, count, entry_count, total_correct, total_loss = 0, 0, 0, 0, 0
    inputs, label, label_mask, model_output, weight, preds, loss, correct = None, None, None, None, None, None, None, None
    scores = []
    labels_counts = []
    labels = defaultdict(list)
    targets = defaultdict(list)
    labels_domain = defaultdict(list)
    observers = defaultdict(list)
    num_labels = len(data_config.label_value);

    start_time = time.time()

    
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y_cat, _, _, Z, _, _, y_weight in tq:
                inputs = [X[k].to(dev,non_blocking=True) for k in data_config.input_names]
                label  = y_cat[data_config.label_names[0]].long().to(dev,non_blocking=True)
                if y_weight:
                    weight = y_weight[data_config.label_sample_weight_names[0]].float().to(dev,non_blocking=True)
                    weight = weight.squeeze();
                entry_count += label.shape[0]
                try:
                    label_mask = y_cat[data_config.label_names[0] + '_mask'].bool().to(dev,non_blocking=True)
                except KeyError:
                    label_mask = None
                if not for_training and label_mask is not None:
                    labels_counts.append(np.squeeze(label_mask.numpy(force=True).sum(axis=-1).astype(dtype=np.int32)))
                label = _flatten_label(label,mask=label_mask)
                num_examples = label.shape[0]
                label_counter.update(label.numpy(force=True).astype(dtype=np.int32))
                for k, v in y_cat.items():
                    labels[k].append(_flatten_label(v,mask=label_mask).numpy(force=True).astype(dtype=np.int32))
                if not for_training:
                    for k, v in Z.items():
                        if v.numpy(force=True).dtype in (np.int16, np.int32, np.int64):
                            observers[k].append(v.numpy(force=True).astype(dtype=np.int32))
                        else:
                            observers[k].append(v.numpy(force=True).astype(dtype=np.float32))

                count += num_examples

                model_output = model(*inputs)
                model_output, label, label_mask = _flatten_preds(model_output,label=label,mask=label_mask)                
                model_output = model_output.squeeze().float();
                label = label.squeeze();
                
                if model_output.shape[0] == num_examples:
                    scores.append(torch.softmax(model_output,dim=1).numpy(force=True).astype(dtype=np.float32))
                else:
                    scores.append(torch.zeros(num_examples,num_labels).numpy(force=True).astype(dtype=np.float32));

                if loss_func is None :                    
                    loss = 0
                else:
                    if torch.is_tensor(weight):
                        loss = loss_func(model_output, label, weight).item()
                    else:
                        loss = loss_func(model_output, label).item()
                
                num_batches += 1
                _, preds = model_output.max(1)
                correct = (preds == label).sum().item()
                total_loss += loss;
                total_correct += correct

                tq.set_postfix({
                    'Loss': '%.5f' % loss,
                    'AvgLoss': '%.5f' % (total_loss / num_batches),
                    'Acc': '%.5f' % (correct / num_examples),
                    'AvgAcc': '%.5f' % (total_correct / count)})

                if tb_helper:
                    if tb_helper.custom_fn:
                        with torch.no_grad():
                            tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=num_batches,
                                                mode='eval' if for_training else 'test')

                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))
    _logger.info('Evaluation AvgLoss: %.5f' % (total_loss / num_batches))
    _logger.info('Evaluation AvgAcc: %.5f'%(total_correct / count if count else 0))
    _logger.info('Evaluation class distribution: \n    %s', str(sorted(label_counter.items())))

    if tb_helper:
        tb_mode = 'eval' if for_training else 'test'
        tb_helper.write_scalars([
            ("Loss/%s (epoch)" % tb_mode, total_loss / num_batches, epoch),
            ("Acc/%s (epoch)" % tb_mode, total_correct / count, epoch),
            ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode=tb_mode)

    scores = np.concatenate(scores).squeeze()
    labels = {k: _concat(v) for k, v in labels.items()}
    observers = {k: _concat(v) for k, v in observers.items()}

    metric_results = evaluate_metrics(labels[data_config.label_names[0]], scores, eval_metrics=eval_metrics)
    _logger.info('Evaluation metrics: \n%s', '\n'.join(
        ['    - %s: \n%s' % (k, str(v)) for k, v in metric_results.items()]))

    if for_training:
        return total_loss / num_batches
    else:
        # convert 2D labels/scores
        if len(scores) != entry_count:
            if len(labels_counts):
                labels_counts = np.concatenate(labels_counts)
                scores = ak.unflatten(scores, labels_counts)
                for k, v in labels.items():
                    labels[k] = ak.unflatten(v, labels_counts)
            else:
                assert(count % entry_count == 0)
                scores = scores.reshape((entry_count, int(count / entry_count), -1)).transpose((1, 2))
                for k, v in labels.items():
                    labels[k] = v.reshape((entry_count, -1))
        return total_loss / num_batches, scores, labels, targets, labels_domain, observers

## evaluate a classifier for which classes are condensed into a single label_name --> argmax of numpy --> use ONNX instead of pytorch
def evaluate_onnx_classification(model_path, test_loader, eval_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix']):

    import onnxruntime
    sess = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    data_config = test_loader.dataset.config

    label_counter = Counter()
    total_correct, count = 0, 0, 0
    scores = []
    labels = defaultdict(list)
    labels_domain = defaultdict(list)
    targets = defaultdict(list)
    observers = defaultdict(list)
    inputs, label, score, preds, correct = None, None, None, None, None
    start_time = time.time();

    with tqdm.tqdm(test_loader) as tq:
        for X, y_cat, _, _, Z, _, _ in tq:
            inputs = {k: v.numpy(force=True).astype(dtype=np.float32) for k, v in X.items()}
            label = y_cat[data_config.label_names[0]].long()
            num_examples = label.shape[0]
            label_counter.update(label.cpu.numpy().astype(dtype=np.int32))
            for k, v in y_cat.items():
                labels[k].append(v.numpy(force=True).astype(dtype=np.int32))
            for k, v in Z.items():
               if v.numpy(force=True).dtype in (np.int16, np.int32, np.int64):
                  observers[k].append(v.numpy(force=True).astype(dtype=np.int32))
               else:
                  observers[k].append(v.numpy(force=True).astype(dtype=np.float32))
            score = sess.run([], inputs)
            score = torch.as_tensor(np.array(score)).squeeze();
            scores.append(score.numpy(force=True).astype(dtype=np.float32))
            preds = score.squeeze().float().argmax(1)
            label = label.squeeze();
            correct = (preds == label).sum()
            total_correct += correct
            count += num_examples

            tq.set_postfix({
                'Acc': '%.5f' % (correct / num_examples),
                'AvgAcc': '%.5f' % (total_correct / count)})

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))
    _logger.info('Evaluation class distribution: \n    %s', str(sorted(label_counter.items())))

    scores = np.concatenate(scores)
    labels = {k: _concat(v) for k, v in labels.items()}
    observers = {k: _concat(v) for k, v in observers.items()}

    metric_results = evaluate_metrics(labels[data_config.label_names[0]], scores, eval_metrics=eval_metrics)
    _logger.info('Evaluation metrics: \n%s', '\n'.join(
        ['    - %s: \n%s' % (k, str(v)) for k, v in metric_results.items()]))

    return total_correct / count, scores, labels, targets, labels_domain, observers


## train a regression with possible multi-dimensional target i.e. a list of 1D functions (target_names) 
def train_regression(model, loss_func, opt, scheduler, train_loader, dev, epoch, steps_per_epoch=None, grad_scaler=None, tb_helper=None):
    
    model.train()   
    torch.backends.cudnn.benchmark = True;
    torch.backends.cudnn.enabled = True;
    
    data_config = train_loader.dataset.config

    num_batches, total_loss, sum_abs_err, sum_sqr_err, count = 0, 0, 0, 0, 0
    loss, inputs, target, model_output, preds = None, None, None, None, None
    start_time = time.time()

    with tqdm.tqdm(train_loader) as tq:
       for X, _, y_reg, _, _, _, _, _ in tq:
         inputs = [X[k].to(dev,non_blocking=True) for k in data_config.input_names]
         for idx, names in enumerate(data_config.target_names):
            if idx == 0:
               target = y_reg[names].float();
            else:
               target = torch.column_stack((target,y_reg[names].float()))
            num_examples = target.shape[0]
            target = target.to(dev,non_blocking=True)
            model.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
               model_output = model(*inputs)
               model_output = model_output.squeeze().float();
               target = target.squeeze();
               loss = loss_func(model_output, target)
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
            num_batches += 1
            count += num_examples
            total_loss += loss
            preds = model_output.detach().float()
            target = target.detach();
            e = preds - target;
            abs_err = e.abs().sum().item()
            sum_abs_err += abs_err
            sqr_err = e.square().sum().item()
            sum_sqr_err += sqr_err

            tq.set_postfix({
                'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                'Loss': '%.5f' % loss,
                'AvgLoss': '%.5f' % (total_loss / num_batches),
                'MSE': '%.5f' % (sqr_err / num_examples),
                'AvgMSE': '%.5f' % (sum_sqr_err / count),
                'MAE': '%.5f' % (abs_err / num_examples),
                'AvgMAE': '%.5f' % (sum_abs_err / count),
            })

            if tb_helper:
                tb_helper.write_scalars([
                    ("Loss/train", loss, tb_helper.batch_train_count + num_batches),
                    ("MSE/train", sqr_err / num_examples, tb_helper.batch_train_count + num_batches),
                    ("MAE/train", abs_err / num_examples, tb_helper.batch_train_count + num_batches),
                    ])
                if tb_helper.custom_fn:
                    with torch.no_grad():
                        tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=num_batches, mode='train')

            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))
    _logger.info('Train AvgLoss: %.5f, AvgMSE: %.5f, AvgMAE: %.5f' %
                 (total_loss / num_batches, sum_sqr_err / count, sum_abs_err / count))

    if tb_helper:
        tb_helper.write_scalars([
            ("Loss/train (epoch)", total_loss / num_batches, epoch),
            ("MSE/train (epoch)", sum_sqr_err / count, epoch),
            ("MAE/train (epoch)", sum_abs_err / count, epoch),
            ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode='train')
        # update the batch state
        tb_helper.batch_train_count += num_batches


def evaluate_regression(model, test_loader, dev, epoch, for_training=True, loss_func=None, steps_per_epoch=None, tb_helper=None,
                        eval_metrics=['mean_squared_error', 'mean_absolute_error', 'median_absolute_error', 'mean_gamma_deviance']):

   model.eval()
   torch.backends.cudnn.benchmark = True;
   torch.backends.cudnn.enabled = True;
       
   data_config = test_loader.dataset.config

   total_loss, num_batches, sum_sqr_err, sum_abs_err, count = 0, 0, 0, 0, 0
   scores = []
   labels, targets, labels_domain, observers = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
   inputs, target, model_output, preds, loss = None, None, None, None, None
   start_time = time.time()

   with torch.no_grad():
      with tqdm.tqdm(test_loader) as tq:
         for X, _, y_reg, _, Z, _, _, _ in tq:
            inputs = [X[k].to(dev,non_blocking=True) for k in data_config.input_names]
            for idx, names in enumerate(data_config.target_names):
               if idx == 0:
                  target = y_reg[names].float();
               else:
                  target = torch.column_stack((target,y_reg[names].float()))
            num_examples = target.shape[0]
            target = target.to(dev,non_blocking=True)
            for k, v in y_reg.items():
               targets[k].append(v.numpy(force=True).astype(dtype=np.float32))
               if not for_training:
                  for k, v in Z.items():
                     if v.numpy(force=True).dtype in (np.int16, np.int32, np.int64):
                        observers[k].append(v.numpy(force=True).astype(dtype=np.int32))
                     else:
                        observers[k].append(v.numpy(force=True).astype(dtype=np.float32))

               model_output = model(*inputs)
               model_output = model_output.squeeze().float();
               scores.append(model_output.numpy(force=True).astype(dtype=np.float32))      

               target =  target.squeeze();
               loss = 0 if loss_func is None else loss_func(model_output, target).item()

               num_batches += 1
               count += num_examples
               total_loss += loss
                
               e = model_output - target
               abs_err = e.abs().sum().item()
               sum_abs_err += abs_err
               sqr_err = e.square().sum().item()
               sum_sqr_err += sqr_err

               tq.set_postfix({
                  'Loss': '%.5f' % loss,
                  'AvgLoss': '%.5f' % (total_loss / num_batches),
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

   if tb_helper:
      tb_mode = 'eval' if for_training else 'test'
      tb_helper.write_scalars([
         ("Loss/%s (epoch)" % tb_mode, total_loss / num_batches, epoch),
         ("MSE/%s (epoch)" % tb_mode, sum_sqr_err / count, epoch),
         ("MAE/%s (epoch)" % tb_mode, sum_abs_err / count, epoch),
      ])
      if tb_helper.custom_fn:
         with torch.no_grad():
            tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode=tb_mode)

   scores = np.concatenate(scores).squeeze();
   targets = {k: _concat(v) for k, v in targets.items()}
   observers = {k: _concat(v) for k, v in observers.items()}        

   for idx, (name,element) in enumerate(targets.items()):
      if len(data_config.target_names) == 1:
         metric_results = evaluate_metrics(element, scores, eval_metrics=eval_metrics)
      else:
         metric_results = evaluate_metrics(element, scores[:,idx], eval_metrics=eval_metrics)

      _logger.info('Evaluation metrics: \n%s', '\n'.join(
         ['    - %s: \n%s' % (k, str(v)) for k, v in metric_results.items()]))        

   if for_training:
      return total_loss / num_batches
   else:
      # convert 2D targets/scores
      scores = scores.reshape(len(scores),len(data_config.target_names))
      return total_loss / num_batches, scores, labels, targets, labels_domain, observers
        
## evaluate regression via ONNX
def evaluate_onnx_regression(model_path, test_loader, 
                             eval_metrics=['mean_squared_error', 'mean_absolute_error', 'median_absolute_error',
                                           'mean_gamma_deviance']):
   import onnxruntime
   sess = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])

   data_config = test_loader.dataset.config

   num_batches, total_loss, sum_sqr_err, sum_abs_err, count = 0, 0, 0, 0, 0
   scores = []
   labels, targets, labels_domain, observers = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
   inputs, target, score, preds, loss = None, None, None, None, None

   start_time = time.time()
   with tqdm.tqdm(test_loader) as tq:
      for X, _, y_reg, _, Z, _, _, _ in tq:
         inputs = {k: v.numpy().astype(dtype=np.float32) for k, v in X.items()}
         for idx, names in enumerate(data_config.target_names):
            if idx == 0:
               target = y[names].float();
            else:
               target = torch.column_stack((target,y[names].float()))
         num_examples = target.shape[0]            
         for k, v in y.items():
            targets[k].append(v.numpy(force=True).astype(dtype=np.float32))
         for k, v in Z.items():
            if v.numpy(force=True).dtype in (np.int16, np.int32, np.int64):
               observers[k].append(v.numpy(force=True).astype(dtype=np.int32))
            else:
               observers[k].append(v.numpy(force=True).astype(dtype=np.float32))

         score = sess.run([], inputs)
         score = torch.as_tensor(np.array(score)).squeeze()
         scores.append(score.numpy(force=True).astype(dtype=np.float32))
         preds = score.squeeze().float();
         target = target.squeeze();

         num_batches += 1;
         count += num_examples

         e = preds - target
         abs_err = e.abs().sum().item()
         sum_abs_err += abs_err
         sqr_err = e.square().sum().item()
         sum_sqr_err += sqr_err

         tq.set_postfix({
            'MSE': '%.5f' % (sqr_err / num_examples),
            'AvgMSE': '%.5f' % (sum_sqr_err / count),
            'MAE': '%.5f' % (abs_err / num_examples),
            'AvgMAE': '%.5f' % (sum_abs_err / count),
         })

   time_diff = time.time() - start_time
   _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))

   scores = np.concatenate(scores)
   targets = {k: _concat(v) for k, v in targets.items()}
   observers = {k: _concat(v) for k, v in observers.items()}        

   for idx, (name,element) in enumerate(targets.items()):
      if len(data_config.target_names) == 1:
         metric_results = evaluate_metrics(element, scores, eval_metrics=eval_metrics)
      else:
         metric_results = evaluate_metrics(element, scores[:,idx], eval_metrics=eval_metrics)
      _logger.info('Evaluation metrics: \n%s', '\n'.join(
         ['    - %s: \n%s' % (k, str(v)) for k, v in metric_results.items()]))
        
   scores = scores.reshape(len(scores),len(data_config.target_names))
   return total_loos/num_batches, scores, labels, targets, labels_domain, observers


## train classification + regssion into a total loss --> best training epoch decided on the loss function
def train_classreg(model, loss_func, opt, scheduler, train_loader, dev, epoch, steps_per_epoch=None, grad_scaler=None, tb_helper=None, network_option=None):
    
    model.train()
    torch.backends.cudnn.benchmark = True;
    torch.backends.cudnn.enabled = True;
    
    data_config = train_loader.dataset.config
   
    num_batches, total_loss, total_cat_loss, total_reg_loss, count = 0, 0, 0, 0, 0
    label_counter = Counter()
    total_correct, sum_sqr_err = 0, 0
    inputs, target, label, model_output, label_mask = None, None, None, None, None;
    loss, loss_cat, loss_reg, pred_cat, pred_reg, residual_reg, correct = None, None, None, None, None, None, None;
    loss_contrastive, model_output_contrastive, total_contrastive_loss = None, None, 0;

    num_labels  = len(data_config.label_value);
    if type(data_config.target_value) == dict:
        num_targets = sum(len(dct) if type(dct) == list else 1 for dct in data_config.target_value.values())
    else:
        num_targets = len(data_config.target_value);

    network_options = None;
    if network_option:
        network_options = {k: ast.literal_eval(v) for k, v in network_option}

    start_time = time.time()

    with tqdm.tqdm(train_loader) as tq:
        for X, y_cat, y_reg, _, _, _, _, _ in tq:
            ### input features for the model
            inputs = [X[k].to(dev,non_blocking=True) for k in data_config.input_names]
            ### build classification true labels (numpy argmax)
            label = y_cat[data_config.label_names[0]].long().to(dev,non_blocking=True)
            try:
                label_mask = y_cat[data_config.label_names[0] + '_mask'].bool().to(dev,non_blocking=True)
            except KeyError:
                label_mask = None;
            label = _flatten_label(label,mask=label_mask)
            label_counter.update(label.numpy(force=True).astype(dtype=np.int32))
            ### build regression targets
            for idx, names in enumerate(data_config.target_names):
                if idx == 0:
                    target = y_reg[names].float();
                else:
                    target = torch.column_stack((target,y_reg[names].float()))
            target = target.to(dev,non_blocking=True)
            ### Number of samples in the batch
            num_examples = max(label.shape[0],target.shape[0]);
            ### loss minimization
            model.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=grad_scaler is not None):            
                label = label.squeeze();
                target = target.squeeze();
                ### evaluate the model
                if network_options and network_options.get('use_contrastive',False):
                    model_output, model_output_contrastive = model(*inputs)         
                    model_output_contrastive = model_output_contrastive.squeeze().float();
                else:
                    model_output = model(*inputs)
                model_output_cat = model_output[:,:num_labels];
                model_output_reg = model_output[:,num_labels:num_labels+num_targets];
                model_output_cat, label, label_mask = _flatten_preds(model_output_cat,label=label,mask=label_mask)
                model_output_cat = model_output_cat.squeeze().float();
                model_output_reg = model_output_reg.squeeze().float();
            
            ### evaluate loss function
            if network_options and network_options.get('use_contrastive',False):                
                loss, loss_cat, loss_reg, loss_contrastive = loss_func(model_output_cat,label,model_output_reg,target,model_output_contrastive);
            else:
                loss, loss_cat, loss_reg = loss_func(model_output_cat,label,model_output_reg,target);

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
            total_loss += loss
            total_cat_loss += loss_cat;
            total_reg_loss += loss_reg;
            if loss_contrastive:
                loss_contrastive = loss_contrastive.detach().item()
                total_contrastive_loss += loss_contrastive;
            num_batches += 1
            count += num_examples;
            
            ## take the classification prediction and compare with the true labels            
            label = label.detach()
            target = target.detach()
            _, pred_cat = model_output_cat.detach().max(1)
            correct  = (pred_cat == label).sum().item()
            total_correct += correct

            ## take the regression prediction and compare with true targets
            pred_reg = model_output_reg.detach().float();
            residual_reg = pred_reg - target;            
            sqr_err = residual_reg.square().sum().item()
            sum_sqr_err += sqr_err
            
            ### monitor metrics
            if network_options and network_options.get('use_contrastive',False):
                postfix = {
                    'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                    'Loss': '%.3f' % (total_loss / num_batches if num_batches else 0),
                    'LossCat': '%.3f' % (total_cat_loss / num_batches if num_batches else 0),
                    'LossReg': '%.3f' % (total_reg_loss / num_batches if num_batches else 0),
                    'LossCont': '%.3f' % (total_contrastive_loss / num_batches if num_batches else 0),
                    'AvgAccCat': '%.3f' % (total_correct / count if count else 0),
                    'AvgMSE': '%.3f' % (sum_sqr_err / count if count else 0),
                }
            else:
                postfix = {
                    'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                    'AvgLoss': '%.3f' % (total_loss / num_batches if num_batches else 0),
                    'AvgLossCat': '%.3f' % (total_cat_loss / num_batches if num_batches else 0), 
                    'AvgLossReg': '%.3f' % (total_reg_loss / num_batches if num_batches else 0),
                    'AvgAccCat': '%.3f' % (total_correct / count if count else 0),
                    'AvgMSE': '%.3f' % (sum_sqr_err / count if count else 0)
                }
            tq.set_postfix(postfix);

            if tb_helper:
                tb_helper.write_scalars([
                    ("Loss/train", loss, tb_helper.batch_train_count + num_batches),
                    ("Acc/train", correct / num_examples, tb_helper.batch_train_count + num_batches),
                    ("MSE/train", sqr_err / num_examples, tb_helper.batch_train_count + num_batches),
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
    _logger.info('Train AvgLoss Reg: %.5f'% (total_reg_loss / num_batches))
    if network_options and network_options.get('use_contrastive',False):
        _logger.info('Train AvgLoss Contrastive: %.5f'%(total_contrastive_loss / num_batches if num_batches else 0))
    _logger.info('Train AvgAcc: %.5f'%(total_correct / count))
    _logger.info('Train AvgMSE: %.5f'%(sum_sqr_err / count))
    _logger.info('Train class distribution: \n %s', str(sorted(label_counter.items())))

    if tb_helper:
        tb_helper.write_scalars([
            ("Loss/train (epoch)", total_loss / num_batches, epoch),
            ("Loss Cat/train (epoch)", total_cat_loss / num_batches, epoch),
            ("Loss Reg/train (epoch)", total_reg_loss / num_batches, epoch),
            ("Acc/train (epoch)", total_correct / count, epoch),
            ("MSE/train (epoch)", sum_sqr_err / count, epoch),
        ])
        if tb_helper.custom_fn:
         with torch.no_grad():
            tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode='train')
        # update the batch state
        tb_helper.batch_train_count += num_batches

## evaluate classification + regression task
def evaluate_classreg(model, test_loader, dev, epoch, for_training=True, loss_func=None, steps_per_epoch=None, tb_helper=None,
                      eval_attack=None, eps_attack=None, network_option=None, grad_scaler=None,
                      eval_cat_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix'],
                      eval_reg_metrics=['mean_squared_error', 'mean_absolute_error', 'median_absolute_error', 'mean_gamma_deviance']):
    
    model.eval()
    torch.backends.cudnn.benchmark = True;
    torch.backends.cudnn.enabled = True;

    data_config = test_loader.dataset.config
    label_counter = Counter()
    total_loss, total_cat_loss, total_reg_loss, num_batches, total_correct, sum_sqr_err, entry_count, count = 0, 0, 0, 0, 0, 0, 0, 0;
    inputs, label, target,  model_output, pred_cat_output, pred_reg, loss, loss_cat, loss_reg, label_mask = None, None, None, None, None , None, None, None, None, None;
    inputs_grad_sign, inputs_attack, model_output_attack, network_options = None, None, None, None;
    num_batches_attack, total_attack_loss, count_attack, residual_attack, sum_residual_attack = 0, 0, 0, 0, 0;
    scores_cat, scores_reg, scores_attack = [], [], [];
    labels, targets, labels_domain, observers = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list);

    num_labels  = len(data_config.label_value);
    if type(data_config.target_value) == dict:
        num_targets = sum(len(dct) if type(dct) == list else 1 for dct in data_config.target_value.values())
    else:
        num_targets = len(data_config.target_value);

    if network_option:
        network_options = {k: ast.literal_eval(v) for k, v in network_option}
        
    start_time = time.time()
    ### epsilons for Attack                                                                                                                                                                   
    input_eps_min = [];
    input_eps_max = [];
    for keys, vars in data_config.input_dicts.items():
        input_eps_min.append(torch.Tensor([data_config.preprocess_params[var]['eps_min'] if data_config.preprocess_params[var]['eps_min'] is not None else float(0.) for var in vars]));
        input_eps_max.append(torch.Tensor([data_config.preprocess_params[var]['eps_max'] if data_config.preprocess_params[var]['eps_max'] is not None else float(0.) for var in vars]));

    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y_cat, y_reg, _, Z, _, _, _ in tq:
                ### input features for the model
                inputs = [X[k].to(dev,non_blocking=True) for k in data_config.input_names]
                ### build classification true labels
                label  = y_cat[data_config.label_names[0]].long().to(dev,non_blocking=True)
                try:
                    label_mask = y_cat[data_config.label_names[0] + '_mask'].bool().to(dev,non_blocking=True)
                except KeyError:
                    label_mask = None
                label  = _flatten_label(label,mask=label_mask)
                label_counter.update(label.numpy(force=True).astype(dtype=np.int32))
                ### build regression targets
                for idx, names in enumerate(data_config.target_names):
                    if idx == 0:
                        target = y_reg[names].float();
                    else:
                        target = torch.column_stack((target,y_reg[names].float()))
                target = target.to(dev,non_blocking=True)
                ### update counters
                num_examples = max(label.shape[0],target.shape[0]);
                entry_count += num_examples

                ### define truth labels for classification and regression
                for k, name in enumerate(data_config.label_names):                    
                    labels[name].append(_flatten_label(y_cat[name],None).numpy(force=True).astype(dtype=np.int32))
                for k, name in enumerate(data_config.target_names):
                    targets[name].append(y_reg[name].numpy(force=True).astype(dtype=np.float32))                
                ### observers
                if not for_training:
                    for k, v in Z.items():
                        if v.numpy(force=True).dtype in (np.int16, np.int32, np.int64):
                            observers[k].append(v.numpy(force=True).astype(dtype=np.int32))
                        else:
                            observers[k].append(v.numpy(force=True).astype(dtype=np.float32))

                ### evaluate model
                num_attack_examples = 0;
                if eval_attack:
                    num_attack_examples = max(label.shape[0],target.shape[0]);                    
                    torch.set_grad_enabled(True);
                    for idx,element in enumerate(inputs):        
                        element.requires_grad = True;
                        model.save_grad_inputs = True;
                else:
                    model.save_grad_inputs = False;

                model.zero_grad(set_to_none=True);
                if network_options and network_options.get('use_contrastive',False):
                    model_output, _ = model(*inputs)
                else:
                    model_output = model(*inputs)

                ### build classification and regression outputs
                label  = label.squeeze();
                target = target.squeeze();
                model_output_cat = model_output[:,:num_labels];
                model_output_reg = model_output[:,num_labels:num_labels+num_targets];
                model_output_cat, label, label_mask = _flatten_preds(model_output_cat,label=label,mask=label_mask)
                model_output_cat = model_output_cat.squeeze().float();
                model_output_reg = model_output_reg.squeeze().float();

                ### save scores
                if model_output_cat.shape[0] == num_examples and model_output_reg.shape[0] == num_examples:
                    scores_cat.append(torch.softmax(model_output_cat,dim=1).numpy(force=True).astype(dtype=np.float32));
                    scores_reg.append(model_output_reg.numpy(force=True).astype(dtype=np.float32))
                else:
                    scores_cat.append(torch.zeros(num_examples,num_labels).numpy(force=True).astype(dtype=np.float32));
                    if num_targets > 1:
                        scores_reg.append(torch.zeros(num_examples,num_targets).numpy(force=True).astype(dtype=np.float32));
                    else:
                        scores_reg.append(torch.zeros(num_examples).numpy(force=True).astype(dtype=np.float32));                        

                ## create adversarial testing attack features and evaluate the model
                if eval_attack:
                    loss, _ , _, _ = loss_func(model_output_cat,label,model_output_reg,target);
                    loss.backward();
                    ## produce gradient signs and features                                                                                                                                     
                    if network_options and network_options.get('use_norm_gradient',False):
                        inputs_grad = [None if element.grad is None else element.grad.data.detach().to(dev,non_blocking=True) for idx,element in enumerate(inputs)]
                        torch.set_grad_enabled(False);
                        inputs_attack = [element.detach().to(dev,non_blocking=True) if inputs_grad[idx] is None else fngm_attack(element,inputs_grad[idx],eps_attack,input_eps_min[idx].to(dev,non_blocking=True),input_eps_max[idx].to(dev,non_blocking=True)).detach().to(dev,non_blocking=True) for idx,element in enumerate(inputs)]
                    else:
                        inputs_grad = [None if element.grad is None else element.grad.data.detach().sign().to(dev,non_blocking=True) for idx,element in enumerate(inputs)]
                        torch.set_grad_enabled(False);
                        inputs_attack = [element.detach().to(dev,non_blocking=True) if inputs_grad[idx] is None else fgsm_attack(element.detach(),inputs_grad[idx],eps_attack,input_eps_min[idx].to(dev,non_blocking=True),input_eps_max[idx].to(dev,non_blocking=True)).detach().to(dev,non_blocking=True) for idx,element in enumerate(inputs)]
                    model.zero_grad(set_to_none=True)
                    model_output_attack = model(*inputs_attack)
                    model_output_attack = model_output_attack[:,:num_labels];
                    model_output_attack = _flatten_preds(model_output_attack,label=label,mask=label_mask).squeeze().float();
                    scores_attack.append(torch.softmax(model_output_attack,dim=1).detach().numpy(force=True).astype(dtype=np.float32));

                ### evaluate loss function
                if loss_func != None:
                    ### true labels and true target 
                    if network_options and network_options.get('use_contrastive',False):
                        loss, loss_cat, loss_reg, _ = loss_func(model_output_cat,label,model_output_reg,target)
                    else:
                        loss, loss_cat, loss_reg = loss_func(model_output_cat,label,model_output_reg,target)

                    loss = loss.item()
                    loss_cat = loss_cat.item()
                    loss_reg = loss_reg.item()
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
                if model_output_cat.shape[0] == num_examples and model_output_reg.shape[0] == num_examples:
                    _,pred_cat = model_output_cat.max(1)
                    correct = (pred_cat == label).sum().item()
                    total_correct += correct
                    ### regression spread
                    pred_reg = model_output_reg.float()
                    residual_reg = pred_reg - target;
                    sqr_err = residual_reg.square().sum().item()
                    sum_sqr_err += sqr_err
                    
                ## fast gradient attack loss residual w.r.t. nominal
                residual_attack = 0;
                if eval_attack:
                    num_batches_attack += 1;
                    model_output_attack = model_output_attack.detach();
                    if (torch.is_tensor(label) and torch.is_tensor(model_output_cat) and torch.is_tensor(model_output_attack) and 
                        np.iterable(label) and np.iterable(model_output_attack) and np.iterable(model_output_cat)):
                        if model_output_cat.shape == model_output_attack.shape:
                            count_attack += num_attack_examples;
                            residual_attack = torch.nn.functional.kl_div(
                                input=torch.log_softmax(model_output_attack,dim=1),
                                target=torch.softmax(model_output_cat,dim=1),
                                reduction='sum')/model_output_attack.size(dim=1);
                            sum_residual_attack += residual_attack;
                            
                ### monitor results
                tq.set_postfix({
                    'Loss': '%.5f' % loss,
                    'AvgLoss': '%.5f' % (total_loss / num_batches),
                    'Acc': '%.5f' % (correct / num_examples),
                    'AvgAcc': '%.5f' % (total_correct / count),
                    'MSE': '%.5f' % (sqr_err / num_examples),
                    'AvgMSE': '%.5f' % (sum_sqr_err / count),
                    'Attack':  '%.5f' % (residual_attack / num_attack_examples if num_attack_examples else 0),
                    'AvgAttack': '%.5f' % (sum_residual_attack / count_attack if count_attack else 0)
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
    _logger.info('Eval AvgLoss: %.5f'% (total_loss / num_batches))
    _logger.info('Eval AvgLoss Cat: %.5f'% (total_cat_loss / num_batches))
    _logger.info('Eval AvgLoss Reg: %.5f'% (total_reg_loss / num_batches))
    _logger.info('Eval AvgAccCat: %.5f'%(total_correct / count if count else 0))
    _logger.info('Eval AvgMSE: %.5f'%(sum_sqr_err / count if count else 0))
    _logger.info('Eval AvgAttack: %.5f'%(sum_residual_attack / count_attack if count_attack else 0))    
    _logger.info('Eval class distribution: \n    %s', str(sorted(label_counter.items())))

    if tb_helper:
        tb_mode = 'eval' if for_training else 'test'
        tb_helper.write_scalars([
            ("Loss/%s (epoch)"%(tb_mode), total_loss / num_batches, epoch),
            ("Loss Cat/%s (epoch)"%(tb_mode), total_cat_loss / num_batches, epoch),
            ("Loss Reg/%s (epoch)"%(tb_mode), total_reg_loss / num_batches, epoch),
            ("AccCat/%s (epoch)"%(tb_mode), total_correct / count if count else 0, epoch),
            ("MSE/%s (epoch)"%(tb_mode), sum_sqr_err / count if count else 0, epoch),
            ("Attack/train (epoch)", sum_residual_fgsm / count_fgsm if count_fgsm else 0, epoch),            
        ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode=tb_mode)
               
    scores_cat = np.concatenate(scores_cat).squeeze()
    scores_reg = np.concatenate(scores_reg).squeeze()
    if eval_attack:
        scores_attack = np.concatenate(scores_attack).squeeze()

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

    if for_training:
        return total_loss / num_batches;
    else:
        if scores_reg.ndim and scores_cat.ndim: 
            scores_reg = scores_reg.reshape(len(scores_reg),len(data_config.target_names))
            scores = np.concatenate((scores_cat,scores_reg),axis=1)
            if eval_attack:
                return total_loss / num_batches, scores, labels, targets, labels_domain, observers, scores_attack
            else:
                return total_loss / num_batches, scores, labels, targets, labels_domain, observers
        else:
            if eval_attack:
                return total_loss / num_batches, scores_reg, labels, targets, labels_domain, observers, scores_attack;
            else:
                return total_loss / num_batches, scores_reg, labels, targets, labels_domain, observers;

def evaluate_onnx_classreg(model_path, test_loader,
                           eval_cat_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix'],
                           eval_reg_metrics=['mean_squared_error', 'mean_absolute_error', 'median_absolute_error', 'mean_gamma_deviance']):

   import onnxruntime
   sess = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])

   data_config = test_loader.dataset.config
   label_counter = Counter()
   num_batches, total_loss, total_cat_loss, total_reg_loss, total_correct, sum_sqr_err, count = 0, 0, 0, 0, 0, 0, 0
   scores_cat, scores_reg = [], []
   labels, targets, observers, labels_domain = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
   inputs, label, pred_cat, pred_reg, loss, loss_cat, loss_reg = None, None, None, None, None, None, None;
    
   start_time = time.time()

   num_labels  = len(data_config.label_value);
   num_targets = len(data_config.target_value);

   with tqdm.tqdm(test_loader) as tq:
      for X, y_cat, y_reg, _, Z, _, _, _ in tq:
         ### input features for the model
         inputs = {k: v.numpy() for k, v in X.items()}
         label = y_cat[data_config.label_names[0]].long();
         label_counter.update(label.numpy(force=True).astype(dtype=np.int32))
         for idx, names in enumerate(data_config.target_names):
            if idx == 0:
               target = y_reg[names].float();
            else:
               target = torch.column_stack((target,y_reg[names].float()))
         num_examples = max(label.shape[0],target.shape[0]);
         ### define truth labels for classification and regression
         for k, name in enumerate(data_config.label_names):                    
            labels[name].append(_flatten_label(y_cat[name],None).numpy(force=True).astype(dtype=np.int32))
            for k, name in enumerate(data_config.target_names):
               targets[name].append(y_reg[name].numpy(force=True).astype(dtype=np.float32))                
            for k, v in Z.items():
               if v.numpy(force=True).dtype in (np.int16, np.int32, np.int64):
                  observers[k].append(v.numpy(force=True).astype(dtype=np.int32))
               else:
                  observers[k].append(v.numpy(force=True).astype(dtype=np.float32))
            ### evaluate the network
            score = sess.run([], inputs)
            score = torch.as_tensor(np.array(score)).squeeze();
            scores_cat.append(score[:,:num_labels].numpy(force=True).astype(dtype=np.float32));
            scores_reg.append(score[:,num_labels:num_labels+num_targets].numpy(force=True).astype(dtype=np.float32));
            pred_cat = score[:,:num_labels].squeeze().float().argmax(1);
            pred_reg = score[:,num_labels:num_labels+num_targets].squeeze().float();
            count += num_examples
            num_batches += 1;

            if pred_cat.shape[0] == num_examples and pred_reg.shape[0] == num_examples:
               correct = (pred_cat == label).sum().item()
               total_correct += correct
               residual_reg = pred_reg - target;
               sqr_err = residual_reg.square().sum().item()
               sum_sqr_err += sqr_err

            ### monitor results
            tq.set_postfix({
               'Acc': '%.5f' % (correct / num_examples),
               'AvgAcc': '%.5f' % (total_correct / count),
               'MSE': '%.5f' % (sqr_err / num_examples),
               'AvgMSE': '%.5f' % (sum_sqr_err / count),
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
      return total_loss / num_batches, scores, labels, targets, labels_domain, observers
   else:
      return total_loss / num_batches, scores_reg, labels, targets, labels_domain, observers

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
