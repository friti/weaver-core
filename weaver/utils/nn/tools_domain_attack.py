import numpy as np
import awkward as ak
import ast
import tqdm
import time
import torch
import gc
import torch._dynamo

from collections import defaultdict, Counter
from collections.abc import Iterable  
from .metrics import evaluate_metrics
from ..data.tools import _concat
from ..logger import _logger
from .utils import _flatten_label, _flatten_preds, fgsm_attack, fngm_attack

## train classification + regssion into a total loss --> best training epoch decided on the loss function
def train_classreg(model, loss_func, opt, scheduler, train_loader, dev, epoch, compile_model=None,
                   steps_per_epoch=None, grad_scaler=None, tb_helper=None, frac_attack=None, epoch_start_attack=None, eps_attack=None, frac_batch_attack=None, network_option=None):

    if compile_model:
        torch._dynamo.config.suppress_errors = True
        model = torch.compile(model, mode='max-autotune');
        
    model.train()
    torch.backends.cudnn.benchmark = True;
    torch.backends.cudnn.enabled = True;
    gc.enable();

    data_config = train_loader.dataset.config

    label_cat_counter = Counter()
    inputs, target, label_cat, label_domain, model_output, model_output_cat, model_output_reg, model_output_domain = None, None, None, None, None, None, None, None;
    num_batches, total_loss, total_cat_loss, total_reg_loss, total_domain_loss, count_cat, count_domain = 0, 0, 0, 0, 0, 0, 0;
    total_cat_correct, total_domain_correct, sum_sqr_err = 0, 0 ,0;
    loss, loss_cat, loss_reg, loss_domain, pred_cat, pred_reg, pred_domain, residual_reg, correct_cat, correct_domain = None, None, None, None, None, None, None, None, None, None;
    loss_contrastive, model_output_contrastive, model_output_contrastive_da, total_contrastive_loss = None, None, None, 0;
    inputs_attack, model_output_attack, loss_attack = None, None, None;
    num_batches_attack, total_attack_loss, count_attack, residual_attack, sum_residual_attack = 0, 0, 0, 0, 0;
    use_attack, network_options = False, None;
    
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

    if network_option:
        network_options = {k: ast.literal_eval(v) for k, v in network_option}

    ### epsilons for attack
    input_eps_min = [];
    input_eps_max = [];
    for keys, vars in data_config.input_dicts.items():
        input_eps_min.append(torch.Tensor([data_config.preprocess_params[var]['eps_min'] if data_config.preprocess_params[var]['eps_min'] is not None else float(0.) for var in vars]));
        input_eps_max.append(torch.Tensor([data_config.preprocess_params[var]['eps_max'] if data_config.preprocess_params[var]['eps_max'] is not None else float(0.) for var in vars]));

    start_time = time.time()

    with tqdm.tqdm(train_loader) as tq:
        for X, y_cat, y_reg, y_domain, _, y_cat_check, y_domain_check in tq:

            ## decide if this batch goes to Attack
            model.save_grad_inputs = False;
            inputs_attack = None;
            use_attack = False;
            rand_val = np.random.uniform(low=0,high=1);
            
            if eps_attack and frac_attack and frac_batch_attack and rand_val < frac_attack and epoch >= epoch_start_attack:
                model.save_grad_inputs = True;
                use_attack = True;
                nrows_selected = int(y_cat[data_config.label_names[0]].size(dim=0)*frac_batch_attack);
            else:
                nrows_selected = y_cat[data_config.label_names[0]].size(dim=0);
                
            inputs = [X[k][0:nrows_selected].to(dev,non_blocking=True) for k in data_config.input_names]
            if use_attack:
                for idx,element in enumerate(inputs):        
                    element.requires_grad = True
                    
            label_cat = y_cat[data_config.label_names[0]][0:nrows_selected].long()
            cat_check = y_cat_check[data_config.labelcheck_names[0]][0:nrows_selected].long()
            index_cat = cat_check.nonzero();
            label_cat = label_cat[index_cat];
            label_cat = _flatten_label(label_cat,None)

            ### build regression targets
            for idx, (k, v) in enumerate(y_reg.items()):
                if idx == 0:
                    target = v[0:nrows_selected].float();
                else:
                    target = torch.column_stack((target,v[0:nrows_selected].float()))
            target = target[index_cat];

            ### build domain true labels (numpy argmax)
            for idx, (k, v) in enumerate(y_domain.items()):
                if idx == 0:
                    label_domain = v[0:nrows_selected].long();
                else:
                    label_domain = torch.column_stack((label_domain,v[0:nrows_selected].long()))
                
            ### store indexes to separate classification+regression events from DA
            for idx, (k, v) in enumerate(y_domain_check.items()):
                if idx == 0:
                    label_domain_check = v[0:nrows_selected].long();
                    index_domain_all = v[0:nrows_selected].long().nonzero();
                else:
                    label_domain_check = torch.column_stack((label_domain_check,v[0:nrows_selected].long()))
                    index_domain_all = torch.cat((index_domain_all,v[0:nrows_selected].long().nonzero()),0)
            
            label_domain = label_domain[index_domain_all];
            label_domain_check = label_domain_check[index_domain_all];
            label_domain = label_domain.squeeze()
            label_domain_check = label_domain_check.squeeze()            

            ### Number of samples in the batch
            num_cat_examples = max(label_cat.shape[0],target.shape[0]);
            num_domain_examples = label_domain.shape[0];

            ### validity checks
            label_cat_np = label_cat.cpu().numpy().astype(dtype=np.int32)
            if np.iterable(label_cat_np):
                label_cat_counter.update(label_cat_np)
            else:
                _logger.info('label_cat not iterable --> shape %s'%(str(label_cat_np.shape)))

            index_domain = defaultdict(list)
            for idx, (k,v) in enumerate(y_domain_check.items()):
                if num_domains == 1:
                    index_domain[k] = label_domain_check.nonzero();
                    label_domain_np = label_domain[index_domain[k]].squeeze().cpu().numpy().astype(dtype=np.int32)
                    if np.iterable(label_domain_np):
                        label_domain_counter[idx].update(label_domain_np)
                    else:
                        _logger.info('label_domain not iterable --> shape %s'%(str(label_domain_np.shape)))
                else:
                    index_domain[k] = label_domain_check[:,idx].nonzero();
                    label_domain_np = label_domain[index_domain[k],idx].squeeze().cpu().numpy().astype(dtype=np.int32);
                    if np.iterable(label_domain_np):
                        label_domain_counter[idx].update(label_domain_np)
                    else:
                        _logger.info('label_domain %d not iterable --> shape %s'%(idx,str(label_domain_np.shape)))
                        
            ## send to device
            label_cat = label_cat.to(dev,non_blocking=True)
            label_domain = label_domain.to(dev,non_blocking=True)
            label_domain_check = label_domain_check.to(dev,non_blocking=True)
            target = target.to(dev,non_blocking=True)            
            
            ### loss minimization
            model.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=grad_scaler is not None):                
                ## prepare the tensor
                label_cat = label_cat.squeeze();
                label_domain = label_domain.squeeze();
                label_domain_check = label_domain_check.squeeze();
                target = target.squeeze();
                ## evaluate the model
                if network_options and network_options.get('use_contrastive',False):
                    if  network_options.get('use_contrastive_domain',False):
                        model_output, model_output_contrastive, model_output_contrastive_da = model(*inputs)
                        model_output_contrastive = model_output_contrastive[index_cat].squeeze().float();
                        model_output_contrastive_da = model_output_contrastive_da[index_domain_all].squeeze().float();
                    else:
                        model_output, model_output_contrastive = model(*inputs)
                        model_output_contrastive = model_output_contrastive[index_cat].squeeze().float();
                else:
                    model_output  = model(*inputs)
                model_output_cat = model_output[:,:num_labels]
                model_output_reg = model_output[:,num_labels:num_labels+num_targets];
                model_output_domain = model_output[:,num_labels+num_targets:num_labels+num_targets+num_labels_domain]
                model_output_cat = _flatten_preds(model_output_cat,None);
                model_output_cat = model_output_cat[index_cat].squeeze().float();
                model_output_reg = model_output_reg[index_cat].squeeze().float();
                model_output_domain = model_output_domain[index_domain_all].squeeze().float();

                ## Attack part
                if use_attack:                    
                    num_attack_examples = max(label_cat.shape[0],target.shape[0]);
                    ## compute the loss function in order to obtain the gradient
                    if network_options and network_options.get('use_contrastive',False):
                        loss, _, _, _, _, _ = loss_func(model_output_cat,label_cat,model_output_reg,target,model_output_domain,label_domain,label_domain_check);
                    else:
                        loss, _, _, _, _ = loss_func(model_output_cat,label_cat,model_output_reg,target,model_output_domain,label_domain,label_domain_check);
                    if grad_scaler is None:
                        loss.backward(retain_graph=True);
                    else:
                        grad_scaler.scale(loss).backward(retain_graph=True);                        
                    ## produce gradient signs and features
                    if network_options and network_options.get('use_norm_gradient',False):
                        inputs_grad = [None if element.grad is None else element.grad.data.detach().to(dev,non_blocking=True) for idx,element in enumerate(inputs)]
                        inputs_attack = [element.detach().to(dev,non_blocking=True) if inputs_grad[idx] is None else fngm_attack(element,inputs_grad[idx],eps_attack,input_eps_min[idx].to(dev,non_blocking=True),input_eps_max[idx].to(dev,non_blocking=True)).detach().to(dev,non_blocking=True) for idx,element in enumerate(inputs)]
                    else:
                        inputs_grad = [None if element.grad is None else element.grad.data.detach().sign().to(dev,non_blocking=True) for idx,element in enumerate(inputs)]
                        inputs_attack = [element.detach().to(dev,non_blocking=True) if inputs_grad[idx] is None else fgsm_attack(element.detach(),inputs_grad[idx],eps_attack,input_eps_min[idx].to(dev,non_blocking=True),input_eps_max[idx].to(dev,non_blocking=True)).detach().to(dev,non_blocking=True) for idx,element in enumerate(inputs)]
                    ## take out the gradients cause were only used to generate the features    
                    model.zero_grad(set_to_none=True)
                    ## infere the model to get the output on Attack inputs
                    if network_options and network_options.get('use_contrastive',False):
                        if network_options.get('use_contrastive_domain',False):
                            model_output_attack, _, _ = model(*inputs_attack)
                        else:
                            model_output_attack, _ = model(*inputs_attack)                            
                    else:
                        model_output_attack = model(*inputs_attack)
                    model_output_attack = model_output_attack[:,:num_labels];
                    model_output_attack = _flatten_preds(model_output_attack,None);
                    model_output_attack = model_output_attack[index_cat].squeeze().float();
                    ## compute the full loss
                    if network_options and network_options.get('use_contrastive',False):
                        if network_options.get('use_contrastive_da',False):
                            loss, loss_cat, loss_reg, loss_domain, loss_attack, loss_contrastive = loss_func(model_output_cat,label_cat,model_output_reg,target,model_output_domain,label_domain,label_domain_check,model_output_attack,model_output_cat,model_output_contrastive,model_output_contrastive_da);
                        else:
                            loss, loss_cat, loss_reg, loss_domain, loss_attack, loss_contrastive = loss_func(model_output_cat,label_cat,model_output_reg,target,model_output_domain,label_domain,label_domain_check,model_output_attack,model_output_cat,model_output_contrastive);
                    else:
                        loss, loss_cat, loss_reg, loss_domain, loss_attack = loss_func(model_output_cat,label_cat,model_output_reg,target,model_output_domain,label_domain,label_domain_check,model_output_attack,model_output_cat);                        
                else:
                    if network_options and network_options.get('use_contrastive',False):
                        if network_options.get('use_contrastive_da',False):
                            loss, loss_cat, loss_reg, loss_domain, loss_attack, loss_contrastive = loss_func(model_output_cat,label_cat,model_output_reg,target,model_output_domain,label_domain,label_domain_check,input_cont=model_output_contrastive,input_cont_da=model_output_contrastive_da);
                        else:
                            loss, loss_cat, loss_reg, loss_domain, loss_attack, loss_contrastive = loss_func(model_output_cat,label_cat,model_output_reg,target,model_output_domain,label_domain,label_domain_check,input_cont=model_output_contrastive);                            
                    else:
                        loss, loss_cat, loss_reg, loss_domain, loss_attack = loss_func(model_output_cat,label_cat,model_output_reg,target,model_output_domain,label_domain,label_domain_check);
                        
            ### back propagation
            if grad_scaler is None:
                loss.backward();
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
            if loss_contrastive:
                loss_contrastive = loss_contrastive.detach().item()
                total_contrastive_loss += loss_contrastive;
                
            ## take the classification prediction and compare with the true labels            
            label_cat = label_cat.detach()
            label_domain = label_domain.detach()
            target = target.detach()
            model_output_cat = model_output_cat.detach()
            model_output_reg = model_output_reg.detach()
            model_output_domain = model_output_domain.detach()
            ##
            correct_cat = 0;
            sqr_err = 0;
            if torch.is_tensor(label_cat) and torch.is_tensor(model_output_cat) and np.iterable(label_cat) and np.iterable(model_output_cat):
                _, pred_cat = model_output_cat.max(1);
                pred_reg = model_output_reg.float();
                if pred_cat.shape == label_cat.shape and pred_reg.shape == target.shape:
                    correct_cat = (pred_cat == label_cat).sum().item()
                    total_cat_correct += correct_cat
                    count_cat += num_cat_examples;
                    residual_reg = pred_reg - target;            
                    sqr_err = residual_reg.square().sum().item()
                    sum_sqr_err += sqr_err
            ## fast gradient attack loss residual w.r.t. nominal
            residual_attack = 0;
            if use_attack:
                num_batches_attack += 1;
                if loss_attack:
                    loss_attack = loss_attack.detach().item()
                    total_attack_loss += loss_attack;
                model_output_attack = model_output_attack.detach();
                if (torch.is_tensor(label_cat) and torch.is_tensor(model_output_cat) and torch.is_tensor(model_output_attack) and 
                    np.iterable(label_cat) and np.iterable(model_output_attack) and np.iterable(model_output_cat)):
                    if model_output_cat.shape == model_output_attack.shape:
                        count_attack += num_attack_examples;
                        if network_options and network_options.get('use_mmd_loss',False):
                            residual_attack = loss_func.MMDLoss(torch.softmax(model_output_cat,dim=1),torch.softmax(model_output_attack,dim=1));
                        else:
                            residual_attack = torch.nn.functional.kl_div(
                                input=torch.log_softmax(model_output_attack,dim=1),
                                target=torch.softmax(model_output_cat,dim=1),
                                reduction='sum')/model_output_attack.size(dim=1);
                        sum_residual_attack += residual_attack;
            ## single domain region
            if num_domains == 1:
                if torch.is_tensor(label_domain) and torch.is_tensor(model_output_domain) and np.iterable(label_domain) and np.iterable(model_output_domain):
                    _, pred_domain = model_output_domain.max(1);
                    if pred_domain.shape == label_domain.shape:
                        correct_domain = (pred_domain == label_domain).sum().item()
                        total_domain_correct += correct_domain
                        count_domain += num_domain_examples;
            ## multiple domain regions
            else:
                correct_domain = 0;
                for idx, (k,v) in enumerate(y_domain_check.items()):                    
                    id_dom = idx*ldomain[idx];
                    label  = label_domain[index_domain[k],idx].squeeze()
                    if not torch.is_tensor(label) or not np.iterable(label): continue;
                    pred_domain = model_output_domain[index_domain[k],id_dom:id_dom+ldomain[idx]].squeeze();
                    if not torch.is_tensor(pred_domain) or not np.iterable(pred_domain): continue;
                    _, pred_domain = pred_domain.max(1);
                    if pred_domain.shape != label.shape: continue;
                    correct_domain += (pred_domain == label).sum().item()
                total_domain_correct += correct_domain
                count_domain += num_domain_examples;

            ### monitor metrics
            if network_options and network_options.get('use_contrastive',False):
                postfix = {
                    'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                    'Loss': '%.3f' % (total_loss / num_batches if num_batches else 0),
                    'LossCat': '%.3f' % (total_cat_loss / num_batches if num_batches else 0),
                    'LossReg': '%.3f' % (total_reg_loss / num_batches if num_batches else 0),
                    'LossDom': '%.3f' % (total_domain_loss / num_batches if num_batches else 0),
                    'LossCont': '%.3f' % (total_contrastive_loss / num_batches if num_batches else 0),
                    'LossAttack': '%.3f' % (total_attack_loss / num_batches_attack if num_batches_attack else 0),
                    'AvgAccCat': '%.3f' % (total_cat_correct / count_cat if count_cat else 0),
                    'AvgAccDom': '%.3f' % (total_domain_correct / (count_domain) if count_domain else 0),
                    'AvgMSE': '%.3f' % (sum_sqr_err / count_cat if count_cat else 0),
                }
                if network_options.get('use_mmd_loss',False):
                    postfix['AvgAttack'] = '%.3f' % (sum_residual_attack / num_batches_attack if num_batches_attack else 0)
                else:
                    postfix['AvgAttack'] = '%.3f' % (sum_residual_attack / count_attack if count_attack else 0)
            else:
                postfix = {
                    'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                    'AvgLoss': '%.3f' % (total_loss / num_batches if num_batches else 0),
                    'AvgLossCat': '%.3f' % (total_cat_loss / num_batches if num_batches else 0),
                    'AvgLossReg': '%.3f' % (total_reg_loss / num_batches if num_batches else 0),
                    'AvgLossDom': '%.3f' % (total_domain_loss / num_batches if num_batches else 0),
                    'AvgLossAttack': '%.3f' % (total_attack_loss / num_batches_attack if num_batches_attack else 0),
                    'AvgAccCat': '%.3f' % (total_cat_correct / count_cat if count_cat else 0),
                    'AvgAccDom': '%.3f' % (total_domain_correct / (count_domain) if count_domain else 0),
                    'AvgMSE': '%.3f' % (sum_sqr_err / count_cat if count_cat else 0)
                }
                if network_options and network_options.get('use_mmd_loss',False):
                    postfix['AvgAttack'] = '%.3f' % (sum_residual_attack / num_batches_attack if num_batches_attack else 0)
                else:
                    postfix['AvgAttack'] = '%.3f' % (sum_residual_attack/count_attack if count_attack else 0)

                
            ## add monitoring of the lambdas and slacks
            if hasattr(loss_func,'lambdas') and len(loss_func.lambdas):
                expression = "";
                for lmbda in loss_func.lambdas:
                    expression += '%.3f, '%(lmbda)
                postfix['Lamdas'] = expression
            if hasattr(loss_func,'slacks') and len(loss_func.slacks):
                expression = "";
                for lmbda in loss_func.slacks:
                    expression += '%.3f, '%(lmbda)
                postfix['Slacks'] = expression
                
            tq.set_postfix(postfix);
                
            if tb_helper:
                tb_help = [
                    ("Loss/train", loss, tb_helper.batch_train_count + num_batches),
                    ("AccCat/train", correct_cat / num_cat_examples if num_cat_examples else 0, tb_helper.batch_train_count + num_batches),
                    ("AccDomain/train", correct_domain / (num_domain_examples) if num_domain_examples else 0, tb_helper.batch_train_count + num_batches),
                    ("MSE/train", sqr_err / num_examples_cat if num_examples_cat else 0, tb_helper.batch_train_count + num_batches)
                ]

                if network_options and network_options.get('use_mmd_loss',False):
                    tb_help.append(("Attack/train", sum_residual_attack / num_batches_attack if num_batches_attack else 0, tb_helper.batch_train_count + num_batches));
                else:
                    tb_help.append(("Attack/train", sum_residual_attack / count_attack if count_attack else 0, tb_helper.batch_train_count + num_batches));
                    
                tb_helper.write_scalars(tb_help);
                    
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
    if network_options and network_options.get('use_contrastive',False):
        _logger.info('Train AvgLoss Contrastive: %.5f'%(total_contrastive_loss / num_batches if num_batches else 0))
    _logger.info('Train AvgLoss Attack: %.5f'% (total_attack_loss / num_batches_attack if num_batches_attack else 0))
    _logger.info('Train AvgAccCat: %.5f'%(total_cat_correct / count_cat if count_cat else 0))
    _logger.info('Train AvgAccDomain: %.5f'%(total_domain_correct / count_domain if count_domain else 0))        
    _logger.info('Train AvgMSE: %.5f'%(sum_sqr_err / count_cat if count_cat else 0))
    if network_options and network_options.get('use_mmd_loss',False):        
        _logger.info('Train AvgAttack: %.5f' % (sum_residual_attack / num_batches_attack if num_batches_attack else 0))
    else:
        _logger.info('Train AvgAttack: %.5f' % (sum_residual_attack / count_attack if count_attack else 0))
    _logger.info('Train class distribution: \n %s', str(sorted(label_cat_counter.items())))
    _logger.info('Train domain distribution: \n %s', ' '.join([str(sorted(i.items())) for i in label_domain_counter]))
                
    if tb_helper:
        tb_help = [
            ("Loss/train (epoch)", total_loss / num_batches, epoch),
            ("Loss Cat/train (epoch)", total_cat_loss / num_batches, epoch),
            ("Loss Domain/train (epoch)", total_domain_loss / num_batches, epoch),
            ("Loss Reg/train (epoch)", total_reg_loss / num_batches, epoch),
            ("Loss Attack/train (epoch)", total_attack_loss / num_batches_attack if num_batches_attack else 0, epoch),
            ("AccCat/train (epoch)", total_cat_correct / count_cat if count_cat else 0, epoch),
            ("AccDomain/train (epoch)", total_domain_correct / count_domain if count_domain else 0, epoch),
            ("MSE/train (epoch)", sum_sqr_err / count_cat if count_cat else 0, epoch),            
        ]
        if network_options and network_options.get('use_mmd_loss',False):
            tb_help.append(("Attack/train (epoch)", sum_residual_attack / count_attack if num_batches_attack else 0, epoch));
        else:
            tb_help.append(("Attack/train (epoch)", sum_residual_attack / count_attack if count_attack else 0, epoch));
        tb_helper.write_scalars(tb_help);
        
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode='train')
        # update the batch state
        tb_helper.batch_train_count += num_batches

    torch.cuda.empty_cache()
    gc.collect();

## evaluate classification + regression task
def evaluate_classreg(model, test_loader, dev, epoch, for_training=True, loss_func=None, steps_per_epoch=None, tb_helper=None, grad_scaler=None, compile_model=None,
                      frac_attack=None, epoch_start_attack=None, eval_attack=None, eps_attack=None, network_option=None,
                      eval_cat_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix'],
                      eval_reg_metrics=['mean_squared_error', 'mean_absolute_error', 'median_absolute_error', 'mean_gamma_deviance']):

    if compile_model:
        import torch._dynamo
        model = torch.compile(model, mode='max-autotune')

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
    inputs_attack, model_output_attack = None, None;
    pred_cat, pred_domain, pred_reg, correct_cat, correct_domain = None, None, None, None, None;
    loss, loss_cat, loss_domain, loss_reg, loss_attack = None, None, None, None, None;
    num_batches_attack, total_attack_loss, count_attack, residual_attack, sum_residual_attack = 0, 0, 0, 0, 0;
    scores_cat, scores_reg, scores_attack, indexes_cat = [], [], [], [];
    scores_domain  = defaultdict(list); 
    labels_cat, labels_domain, targets, observers = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list);
    indexes_domain = defaultdict(list); 
    index_offset, network_options = 0, None;

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

    if network_option:
        network_options = {k: ast.literal_eval(v) for k, v in network_option}

    ### epsilons for Attack
    input_eps_min = [];
    input_eps_max = [];
    for keys, vars in data_config.input_dicts.items():
        input_eps_min.append(torch.Tensor([data_config.preprocess_params[var]['eps_min'] if data_config.preprocess_params[var]['eps_min'] is not None else float(0.) for var in vars]));
        input_eps_max.append(torch.Tensor([data_config.preprocess_params[var]['eps_max'] if data_config.preprocess_params[var]['eps_max'] is not None else float(0.) for var in vars]));

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
                label_cat_np = label_cat.cpu().numpy().astype(dtype=np.int32)
                if np.iterable(label_cat_np):
                    label_cat_counter.update(label_cat_np)
                else:
                    _logger.info('label_cat not iterable --> shape %s'%(str(label_cat_np.shape)))

                index_domain = defaultdict(list)
                for idx, (k,v) in enumerate(y_domain_check.items()):
                    if num_domains == 1:
                        index_domain[k] = label_domain_check.nonzero();
                        label_domain_np = label_domain[index_domain[k]].squeeze().cpu().numpy().astype(dtype=np.int32)
                        if np.iterable(label_domain_np):
                            label_domain_counter[idx].update(label_domain_np);
                        else:
                            _logger.info('label_domain not iterable --> shape %s'%(str(label_domain_np.shape)))
                    else:
                        index_domain[k] = label_domain_check[:,idx].nonzero();
                        label_domain_np = label_domain[index_domain[k],idx].squeeze().cpu().numpy().astype(dtype=np.int32);
                        if np.iterable(label_domain_np):
                            label_domain_counter[idx].update(label_domain_np);
                        else:
                            _logger.info('label_domain %d not iterable --> shape %s'%(idx,str(label_domain_np.shape)))

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
                        if v.cpu().numpy().dtype in (np.int16, np.int32, np.int64):
                            observers[k].append(v.cpu().numpy().astype(dtype=np.int32))
                        else:
                            observers[k].append(v.cpu().numpy().astype(dtype=np.float32))

                for idx, (k, v) in enumerate(y_domain.items()):
                    if not for_training:
                        labels_domain[k].append(v.squeeze().cpu().numpy().astype(dtype=np.int32))
                        indexes_domain[k].append((index_offset+index_domain[list(y_domain_check.keys())[idx]]).cpu().numpy().astype(dtype=np.int32));
                    else:
                        labels_domain[k].append(v[index_domain[list(y_domain_check.keys())[idx]]].squeeze().cpu().numpy().astype(dtype=np.int32))

                ### evaluate model enabling gradient
                num_attack_examples = 0;
                model.save_grad_inputs = False;
                inputs_attack = None;
                use_attack = False;
                torch.set_grad_enabled(False);
                
                if for_training:
                    rand_val = np.random.uniform(low=0,high=1);
                    if eps_attack and frac_attack and rand_val < frac_attack and epoch >= epoch_start_attack:
                        use_attack = True;
                else:
                    if eps_attack:
                        use_attack = True;
                    
                if eval_attack and use_attack:
                    num_attack_examples = max(label_cat.shape[0],target.shape[0]);                    
                    torch.set_grad_enabled(True);
                    model.save_grad_inputs = True;
                    for idx,element in enumerate(inputs):        
                        element.requires_grad = True;
                    
                model.zero_grad(set_to_none=True);
                if network_options and network_options.get('use_contrastive',False):
                    if network_options.get('use_contrastive_domain',False):
                        model_output, _, _ = model(*inputs)
                    else:
                        model_output, _ = model(*inputs)                        
                else:                    
                    model_output = model(*inputs)
                model_output_cat = model_output[:,:num_labels]
                model_output_reg = model_output[:,num_labels:num_labels+num_targets];
                model_output_domain = model_output[:,num_labels+num_targets:num_labels+num_targets+num_labels_domain]
                model_output_cat = _flatten_preds(model_output_cat,None);
                label_cat = label_cat.squeeze();
                label_domain = label_domain.squeeze();
                label_domain_check = label_domain_check.squeeze();
                target = target.squeeze();
                
                ### in validation only filter interesting events
                if for_training:                        
                    model_output_cat = model_output_cat[index_cat];
                    model_output_reg = model_output_reg[index_cat];
                    model_output_domain = model_output_domain[index_domain_all];                
                    ### adjsut outputs
                    model_output_cat = model_output_cat.squeeze().float();
                    model_output_reg = model_output_reg.squeeze().float();
                    model_output_domain = model_output_domain.squeeze().float();
                    ### save the scores
                    scores_cat.append(torch.softmax(model_output_cat,dim=1).detach().cpu().numpy().astype(dtype=np.float32));
                    scores_reg.append(model_output_reg.detach().cpu().numpy().astype(dtype=np.float32));
                    for idx, name in enumerate(y_domain.keys()):
                        id_dom = idx*ldomain[idx];
                        score_domain = model_output_domain[:,id_dom:id_dom+ldomain[idx]];
                        scores_domain[name].append(torch.softmax(score_domain[index_domain[list(y_domain_check.keys())[idx]]].squeeze(),dim=1).detach().cpu().numpy().astype(dtype=np.float32));
                else:                    
                    model_output_cat = model_output_cat.float();
                    model_output_reg = model_output_reg.float();
                    model_output_domain = model_output_domain.float();                    
                    scores_cat.append(torch.softmax(model_output_cat,dim=1).detach().cpu().numpy().astype(dtype=np.float32));
                    scores_reg.append(model_output_reg.detach().cpu().numpy().astype(dtype=np.float32));
                    for idx, name in enumerate(y_domain.keys()):
                        id_dom = idx*ldomain[idx];
                        score_domain = model_output_domain[:,id_dom:id_dom+ldomain[idx]];
                        scores_domain[name].append(torch.softmax(score_domain.squeeze(),dim=1).detach().cpu().numpy().astype(dtype=np.float32));
                        
                    model_output_cat = model_output_cat[index_cat];
                    model_output_reg = model_output_reg[index_cat];
                    model_output_domain = model_output_domain[index_domain_all];                        
                    ### adjsut outputs
                    model_output_cat = model_output_cat.squeeze().float();
                    model_output_reg = model_output_reg.squeeze().float();
                    model_output_domain = model_output_domain.squeeze().float();

                ## create adversarial testing attack features and evaluate the model
                if eval_attack and use_attack:
                    ## first forward and bkg pass for gradients
                    if network_options and network_options.get('use_contrastive',False):
                        loss, _, _, _, _, _ = loss_func(model_output_cat,label_cat,model_output_reg,target,model_output_domain,label_domain,label_domain_check);
                    else:
                        loss, _, _, _, _ = loss_func(model_output_cat,label_cat,model_output_reg,target,model_output_domain,label_domain,label_domain_check);
                    if grad_scaler is None:
                        loss.backward();
                    else:
                        grad_scaler.scale(loss).backward()
                    ## generation of adversarial sample
                    if network_options and network_options.get('use_norm_gradient',False):
                        inputs_grad = [None if element.grad is None else element.grad.data.detach().to(dev,non_blocking=True) for idx,element in enumerate(inputs)]
                        torch.set_grad_enabled(False);
                        inputs_attack = [element.detach().to(dev,non_blocking=True) if inputs_grad[idx] is None else fngm_attack(element,inputs_grad[idx],eps_attack,input_eps_min[idx].to(dev,non_blocking=True),input_eps_max[idx].to(dev,non_blocking=True)).detach().to(dev,non_blocking=True) for idx,element in enumerate(inputs)]
                    else:
                        inputs_grad = [None if element.grad is None else element.grad.data.detach().sign().to(dev,non_blocking=True) for idx,element in enumerate(inputs)]
                        torch.set_grad_enabled(False);
                        inputs_attack = [element.detach().to(dev,non_blocking=True) if inputs_grad[idx] is None else fgsm_attack(element.detach(),inputs_grad[idx],eps_attack,input_eps_min[idx].to(dev,non_blocking=True),input_eps_max[idx].to(dev,non_blocking=True)).detach().to(dev,non_blocking=True) for idx,element in enumerate(inputs)]
                    ## erase the gradient
                    model.zero_grad(set_to_none=True)
                    ## infere the model
                    if network_options and network_options.get('use_contrastive',False):
                         if network_options.get('use_contrastive_domain',False):
                             model_output_attack, _, _ = model(*inputs_attack);
                         else:
                             model_output_attack, _ = model(*inputs_attack);
                    else:
                        model_output_attack = model(*inputs_attack);
                    model_output_attack = model_output_attack[:,:num_labels];
                    model_output_attack = _flatten_preds(model_output_attack,None);
                    model_output_attack = model_output_attack[index_cat].squeeze().float();
                    if not for_training:
                        scores_attack.append(torch.softmax(model_output_attack,dim=1).detach().cpu().numpy().astype(dtype=np.float32));
                    
                ### evaluate loss function
                num_batches += 1
                index_offset += (num_cat_examples+num_domain_examples)

                if loss_func != None:
                    if eval_attack and use_attack:
                        if network_options and network_options.get('use_contrastive',False):
                            loss, loss_cat, loss_reg, loss_domain, loss_attack, _ = loss_func(model_output_cat,label_cat,model_output_reg,target,model_output_domain,label_domain,label_domain_check,model_output_attack,model_output_cat);
                        else:
                            loss, loss_cat, loss_reg, loss_domain, loss_attack = loss_func(model_output_cat,label_cat,model_output_reg,target,model_output_domain,label_domain,label_domain_check,model_output_attack,model_output_cat);
                            
                    else:
                        if network_options and network_options.get('use_contrastive',False):
                            loss, loss_cat, loss_reg, loss_domain, loss_attack, _ = loss_func(model_output_cat,label_cat,model_output_reg,target,model_output_domain,label_domain,label_domain_check);
                        else:
                            loss, loss_cat, loss_reg, loss_domain, loss_attack = loss_func(model_output_cat,label_cat,model_output_reg,target,model_output_domain,label_domain,label_domain_check);      
                            
                    if loss: loss = loss.item()
                    if loss_cat: loss_cat = loss_cat.item()
                    if loss_reg: loss_reg = loss_reg.item()
                    if loss_domain: loss_domain = loss_domain.item()
                    if loss_attack: loss_attack = loss_attack.item()
                else:
                    loss,loss_cat,loss_reg,loss_domain,loss_attack = 0,0,0,0,0;
                                    
                if loss: total_loss += loss
                if loss_cat: total_cat_loss += loss_cat
                if loss_reg: total_reg_loss += loss_reg
                if loss_domain: total_domain_loss += loss_domain                

                ## prediction + metric for classification
                if np.iterable(label_cat) and torch.is_tensor(label_cat) and np.iterable(model_output_cat) and torch.is_tensor(model_output_cat):
                    _, pred_cat = model_output_cat.max(1);
                    pred_reg = model_output_reg.float();
                    if pred_cat.shape == label_cat.shape and pred_reg.shape == target.shape:
                        correct_cat = (pred_cat == label_cat).sum().item()
                        count_cat += num_cat_examples
                        total_cat_correct += correct_cat
                        residual_reg = pred_reg - target;
                        sqr_err = residual_reg.square().sum().item()
                        sum_sqr_err += sqr_err

                ## fast gradient attack loss residual w.r.t. nominal
                residual_attack = 0;
                if eval_attack and use_attack:
                    num_batches_attack += 1;
                    total_attack_loss += loss_attack;
                    model_output_attack = model_output_attack.detach();
                    if (torch.is_tensor(label_cat) and torch.is_tensor(model_output_cat) and torch.is_tensor(model_output_attack) and 
                        np.iterable(label_cat) and np.iterable(model_output_attack) and np.iterable(model_output_cat)):
                        if model_output_cat.shape == model_output_attack.shape:
                            count_attack += num_attack_examples;
                            if network_options and network_options.get('use_mmd_loss',False):
                                residual_attack = loss_func.MMDLoss(torch.softmax(model_output_cat,dim=1),torch.softmax(model_output_attack,dim=1));
                            else:
                                residual_attack = torch.nn.functional.mse_loss(
                                    input=torch.softmax(model_output_attack,dim=1).gather(1,label_cat.view(-1,1)),
                                    target=torch.softmax(model_output_cat,dim=1).gather(1,label_cat.view(-1,1)),
                                    reduction='sum');
                            sum_residual_attack += residual_attack;
                ## single domain region                                                                                                                                                          
                if num_domains == 1:
                    if torch.is_tensor(label_domain) and torch.is_tensor(model_output_domain) and np.iterable(label_domain) and np.iterable(model_output_domain):
                        _, pred_domain = model_output_domain.max(1);
                        if pred_domain.shape == label_domain.shape:
                            correct_domain = (pred_domain == label_domain).sum().item()
                            total_domain_correct += correct_domain
                            count_domain += num_domain_examples                
                ## multiple domains
                else:
                    correct_domain = 0;
                    for idx, (k,v) in enumerate(y_domain_check.items()):
                        id_dom = idx*ldomain[idx];
                        label = label_domain[index_domain[k],idx].squeeze()
                        if not torch.is_tensor(label) or not np.iterable(label): continue;
                        pred_domain = model_output_domain[index_domain[k],id_dom:id_dom+ldomain[idx]].squeeze()
                        if not torch.is_tensor(pred_domain) or not np.iterable(pred_domain): continue;
                        _, pred_domain = pred_domain.max(1);
                        if pred_domain.shape != label.shape: continue;
                        correct_domain += (pred_domain == label).sum().item()
                    total_domain_correct += correct_domain
                    count_domain += num_domain_examples                
                                    
                ### monitor results
                postfix = {
                    'AvgLoss': '%.3f' % (total_loss / num_batches if num_batches else 0),
                    'AvgLossCat': '%.3f' % (total_cat_loss / num_batches if num_batches else 0),
                    'AvgLossReg': '%.3f' % (total_reg_loss / num_batches if num_batches else 0),
                    'AvgLossDom': '%.3f' % (total_domain_loss / num_batches if num_batches else 0),
                    'AvgLossAttack': '%.3f' % (total_attack_loss / num_batches_attack if num_batches_attack else 0),
                    'AvgAccCat': '%.3f' % (total_cat_correct / count_cat if count_cat else 0),
                    'AvgAccDom': '%.3f' % (total_domain_correct / (count_domain) if count_domain else 0),
                    'AvgMSE': '%.3f' % (sum_sqr_err / count_cat if count_cat else 0),
                }

                if network_options and network_options.get('use_mmd_loss',False):
                    postfix['AvgAttack'] = '%.3f' % (sum_residual_attack / num_batches_attack if num_batches_attack else 0)
                else:
                    postfix['AvgAttack'] = '%.3f' % (sum_residual_attack / count_attack if count_attack else 0)

                tq.set_postfix(postfix);

                if tb_helper:
                    tb_help = [
                        ("Loss/train", loss, tb_helper.batch_train_count + num_batches),
                        ("AccCat/train", correct_cat / num_cat_examples if num_cat_examples else 0, tb_helper.batch_train_count + num_batches),
                        ("AccDomain/train", correct_domain / (num_domain_examples) if num_domain_examples else 0, tb_helper.batch_train_count + num_batches),
                        ("MSE/train", sqr_err / num_examples_cat if num_examples_cat else 0, tb_helper.batch_train_count + num_batches),
                    ]
                    if network_options and network_options.get('use_mmd_loss',False):
                        tb_help.append(("Attack/train", residual_attack / num_batches_attack if num_batches_attack else 0, tb_helper.batch_train_count + num_batches))
                    else:
                        tb_help.append(("Attack/train", residual_attack / num_attack_examples if num_attack_examples else 0, tb_helper.batch_train_count + num_batches))
                        
                    tb_helper.write_scalars(tb_help);
                    if tb_helper.custom_fn:
                        tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=num_batches, mode='eval' if for_training else 'test')
                        
                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count_cat+count_domain, (count_cat+count_domain) / time_diff))
    _logger.info('Eval AvgLoss: %.5f'% (total_loss / num_batches))
    _logger.info('Eval AvgLoss Cat: %.5f'% (total_cat_loss / num_batches))
    _logger.info('Eval AvgLoss Domain: %.5f'% (total_domain_loss / num_batches))
    _logger.info('Eval AvgLoss Reg: %.5f'% (total_reg_loss / num_batches))
    _logger.info('Eval AvgLoss Attack: %.5f'% (total_attack_loss / num_batches_attack if num_batches_attack else 0))
    _logger.info('Eval AvgAccCat: %.5f'%(total_cat_correct / count_cat if count_cat else 0))
    _logger.info('Eval AvgAccDomain: %.5f'%(total_domain_correct / (count_domain) if count_domain else 0))
    _logger.info('Eval AvgMSE: %.5f'%(sum_sqr_err / count_cat if count_cat else 0))
    if network_options and network_options.get('use_mmd_loss',False):        
        _logger.info('Eval AvgAttack %.5f' % (sum_residual_attack / num_batches_attack if num_batches_attack else 0))
    else:
        _logger.info('Eval AvgAttack %.5f' % (sum_residual_attack / count_attack if count_attack else 0))
    _logger.info('Eval class distribution: \n    %s', str(sorted(label_cat_counter.items())))
    _logger.info('Eval domain distribution: \n %s', ' '.join([str(sorted(i.items())) for i in label_domain_counter]))

    if tb_helper:
        tb_mode = 'eval' if for_training else 'test'
        tb_help = [
            ("Loss/%s (epoch)"%(tb_mode), total_loss / num_batches, epoch),
            ("Loss Cat/%s (epoch)"%(tb_mode), total_cat_loss / num_batches, epoch),
            ("Loss Reg/%s (epoch)"%(tb_mode), total_reg_loss / num_batches, epoch),
            ("Loss Domain/%s (epoch)"%(tb_mode), total_domain_loss / num_batches, epoch),
            ("Loss Attack/train (epoch)", total_attack_loss / num_batches_attack if num_batches_attack else 0, epoch),
            ("AccCat/%s (epoch)"%(tb_mode), total_cat_correct / count_cat if count_cat else 0, epoch),
            ("AccDomain/%s (epoch)"%(tb_mode), total_domain_correct / count_domain if count_domain else 0, epoch),
            ("MSE/%s (epoch)"%(tb_mode), sum_sqr_err / count_cat if count_cat else 0, epoch)
        ]
        if network_options and network_options.get('use_mmd_loss',False):
            tb_help.append(("Attack/train Attack (epoch)", (sum_residual_attack / num_batches_attack if num_batches_attack else 0), epoch));
        else:
            tb_help.append(("Attack/train Attack (epoch)", (sum_residual_attack / count_attack if count_attack else 0), epoch));
        tb_helper.write_scalars(tb_help);

        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode=tb_mode)

    #####
    scores_cat = np.concatenate(scores_cat).squeeze()
    scores_reg = np.concatenate(scores_reg).squeeze()
    scores_domain = {k: _concat(v) for k, v in scores_domain.items()}
    if eval_attack and not for_training:
        scores_attack = np.concatenate(scores_attack).squeeze()
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
        if eval_attack:
            return total_loss / num_batches, scores, labels_cat, targets, labels_domain, observers, scores_attack
        else:
            return total_loss / num_batches, scores, labels_cat, targets, labels_domain, observers

def evaluate_onnx_classreg(model_path, test_loader,
                           eval_cat_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix'],
                           eval_reg_metrics=['mean_squared_error', 'mean_absolute_error', 'median_absolute_error', 'mean_gamma_deviance']):

    import onnxruntime
    sess = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    gc.enable();

    data_config = test_loader.dataset.config
    label_cat_counter = Counter()
    total_loss, num_batches, total_cat_correct, sum_sqr_err, count_cat, count_domain = 0, 0, 0, 0, 0, 0;
    labels_cat, labels_domain, targets, observers = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    scores_cat, scores_reg, scores_domain = [], [], []
    scores_domain  = defaultdict(list); 
    pred_cat, pred_reg = None, None;
    inputs, label_cat, label_domain, target, model_output, model_output_cat, model_output_reg = None, None, None, None, None , None, None;
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
            label_cat_np = label_cat.cpu().numpy().astype(dtype=np.int32);
            if np.iterable(label_cat_np):
                label_cat_counter.update(label_cat_np)
            index_domain = defaultdict(list)
            for idx, (k,v) in enumerate(y_domain_check.items()):
                if num_domains == 1:
                    index_domain[k] = label_domain_check.nonzero();
                    label_domain_np = label_domain[index_domain[k]].squeeze().cpu().numpy().astype(dtype=np.int32);
                    if np.iterable(label_domain_np):
                        label_domain_counter[idx].update(label_domain_np);
                else:
                    index_domain[k] = label_domain_check[:,idx].nonzero();
                    label_domain_np = label_domain[index_domain[k],idx].squeeze().cpu().numpy().astype(dtype=np.int32);
                    if np.iterable(label_domain_np):
                        label_domain_counter[idx].update(label_domain_np);

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
                if v.cpu().numpy().dtype in (np.int16, np.int32, np.int64):
                    observers[k].append(v.cpu().numpy().astype(dtype=np.int32))
                else:
                    observers[k].append(v.cpu().numpy().astype(dtype=np.float32))                    
            for idx, (k, v) in enumerate(y_domain.items()):
                labels_domain[k].append(v.squeeze().cpu().numpy().astype(dtype=np.int32))
                indexes_domain[k].append((index_offset+index_domain[list(y_domain_check.keys())[idx]]).cpu().numpy().astype(dtype=np.int32));
  
            ### output of the mode
            model_output = sess.run([], inputs)
            model_output = torch.as_tensor(np.array(model_output)).squeeze();            
            model_output_cat = model_output[:,:num_labels]
            model_output_reg = model_output[:,num_labels:num_labels+num_targets];
            model_output_cat = _flatten_preds(model_output_cat,None)
            label_cat = label_cat.squeeze();
            label_domain = label_domain.squeeze();
            target = target.squeeze();

            scores_cat.append(torch.softmax(model_output_cat,dim=1).cpu().numpy().astype(dtype=np.float32));
            scores_reg.append(model_output_reg.cpu().numpy().astype(dtype=np.float32));
            
            model_output_cat = model_output_cat[index_cat];
            model_output_reg = model_output_reg[index_cat];

            ### adjsut outputs
            model_output_cat = model_output_cat.squeeze().float();
            model_output_reg = model_output_reg.squeeze().float();

            num_batches += 1
            index_offset += (num_cat_examples+num_domain_examples)

            ## prediction + metric for classification
            if np.iterable(label_cat) and torch.is_tensor(label_cat) and np.iterable(model_output_cat) and torch.is_tensor(model_output_cat):
                _, pred_cat = model_output_cat.max(1);
                correct_cat = (pred_cat == label_cat).sum().item()
                count_cat += num_cat_examples
                total_cat_correct += correct_cat
                ## prediction + metric for regression
                pred_reg = model_output_reg.float();
                residual_reg = pred_reg - target;
                sqr_err = residual_reg.square().sum().item()
                sum_sqr_err += sqr_err

            count_domain += num_domain_examples;
                
            ### monitor results
            tq.set_postfix({
                'AccCat': '%.5f' % (correct_cat / num_cat_examples if num_cat_examples else 0),
                'AvgAccCat': '%.5f' % (total_cat_correct / count_cat if count_cat else 0),
                'MSE': '%.5f' % (sqr_err / num_cat_examples if num_cat_examples else 0),
                'AvgMSE': '%.5f' % (sum_sqr_err / count_cat if count_cat else 0),
            })
                
    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count_cat+count_domain, (count_cat+count_domain) / time_diff))
    _logger.info('Eval AvgAccCat: %.5f'%(total_cat_correct / count_cat if count_cat else 0))
    _logger.info('Eval AvgMSE: %.5f'%(sum_sqr_err / count_cat if count_cat else 0))
    _logger.info('Evaluation class distribution: \n    %s', str(sorted(label_cat_counter.items())))
    _logger.info('Train domain distribution: \n %s', ' '.join([str(sorted(i.items())) for i in label_domain_counter]))
    
    scores_cat = np.concatenate(scores_cat).squeeze()
    scores_reg = np.concatenate(scores_reg).squeeze()
    indexes_cat = np.concatenate(indexes_cat).squeeze()
    indexes_domain = {k: _concat(v) for k, v in indexes_domain.items()}
    labels_cat  = {k: _concat(v) for k, v in labels_cat.items()}
    labels_domain  = {k: _concat(v) for k, v in labels_domain.items()}
    targets = {k: _concat(v) for k, v in targets.items()}
    observers = {k: _concat(v) for k, v in observers.items()}

    metric_cat_results = evaluate_metrics(labels_cat[data_config.label_names[0]][indexes_cat].squeeze(),scores_cat[indexes_cat].squeeze(),eval_metrics=eval_cat_metrics)            
    _logger.info('Evaluation Classification metrics: \n%s', '\n'.join(
        ['    - %s: \n%s' % (k, str(v)) for k, v in metric_cat_results.items()]))

    for idx, (name,element) in enumerate(targets.items()):
        if num_targets == 1:
            metric_reg_results = evaluate_metrics(element[indexes_cat].squeeze(), scores_reg[indexes_cat].squeeze(), eval_metrics=eval_reg_metrics)
        else:
            metric_reg_results = evaluate_metrics(element[indexes_cat].squeeze(), scores_reg[indexes_cat,idx].squeeze(), eval_metrics=eval_reg_metrics)
            _logger.info('Evaluation Regression metrics for '+name+' target: \n%s', '\n'.join(
                ['    - %s: \n%s' % (k, str(v)) for k, v in metric_reg_results.items()]))        
            
    scores_reg = scores_reg.reshape(len(scores_reg),num_targets);
    scores_domain = torch.zeros(scores_cat.shape[0],num_labels_domain);
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
