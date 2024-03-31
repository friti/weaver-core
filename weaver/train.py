#!/usr/bin/env python
import os
import ast
import sys
import shutil
import glob
import argparse
import functools
import numpy as np
import math
import torch
import gc
from torch.utils.data import DataLoader
from utils.logger import _logger, _configLogger
from utils.dataset import SimpleIterDataset
from utils.import_tools import import_module

parser = argparse.ArgumentParser()
parser.add_argument('--weaver-mode', type=str, default='class', choices=['class', 'reg', 'classreg', 'classregdomain','preprocess','classregdomainattack'],  # TODO: add more  
                    help='class: classification task, reg: regression task, classreg: classification+regression,' 
                    'classregdomain: classification+regression with domain adversarial, classregdomainattack: class+reg+domain+attack adversarial,'
                    'preprocess: only run re-weight step and produce the new yaml file'
                )
parser.add_argument('--data-config', type=str, default='', help='data config YAML file')
parser.add_argument('--extra-selection', type=str, default=None,
                    help='Additional selection requirement, will modify `selection` to `(selection) & (extra)` on-the-fly')
parser.add_argument('--extra-test-selection', type=str, default=None,
                    help='Additional test-time selection requirement, will modify `test_time_selection` to `(test_time_selection) & (extra)` on-the-fly')
parser.add_argument('--data-train', nargs='*', default=[],
                    help='training files; supported syntax:'
                         ' (a) plain list, `--data-train /path/to/a/* /path/to/b/*`;'
                         ' (b) (named) groups [Recommended], `--data-train a:/path/to/a/* b:/path/to/b/*`,'
                         ' the file splitting (for each dataloader worker) will be performed per group,'
                         ' and then mixed together, to ensure a uniform mixing from all groups for each worker.'
                    )
parser.add_argument('--data-val', nargs='*', default=[], help='validation files; when not set, will use training files and split by `--train-val-split`')
parser.add_argument('--data-test', nargs='*', default=[],
                    help='testing files; supported syntax:'
                         ' (a) plain list, `--data-test /path/to/a/* /path/to/b/*`;'
                         ' (b) keyword-based, `--data-test a:/path/to/a/* b:/path/to/b/*`, will produce output_a, output_b;'
                         ' (c) split output per N input files, `--data-test a%%10:/path/to/a/*`, will split per 10 input files.')
parser.add_argument('--data-fraction', type=float, default=1,
                    help='fraction of events to load from each file; for training, the events are randomly selected for each epoch')
parser.add_argument('--file-fraction', type=float, default=1,
                    help='fraction of files to load; for training, the files are randomly selected for each epoch')
parser.add_argument('--fetch-by-files', action='store_true', default=False,
                    help='When enabled, will load all events from a small number (set by ``--fetch-step-train``) of files for each data fetching. '
                         'Otherwise (default), load a small fraction of events from all files each time, which helps reduce variations in the sample composition.')
parser.add_argument('--fetch-step-preprocess', type=float, default=0.01,
                    help='fraction of events to load each time from every file (when ``--fetch-by-files`` is disabled); '
                         'Or: number of files to load each time (when ``--fetch-by-files-preprocess`` is enabled). Shuffling & sampling is done within these events, so set a large enough value.')
parser.add_argument('--fetch-step-train', type=float, default=0.01,
                    help='fraction of events to load each time from every file (when ``--fetch-by-files`` is disabled); '
                         'Or: number of files to load each time (when ``--fetch-by-files-train`` is enabled). Shuffling & sampling is done within these events, so set a large enough value.')
parser.add_argument('--fetch-step-val', type=float, default=0.01,
                    help='fraction of events to load each time from every file (when ``--fetch-by-files`` is disabled); '
                         'Or: number of files to load each time (when ``--fetch-by-files`` is enabled). Shuffling & sampling is done within these events, so set a large enough value.')
parser.add_argument('--fetch-step-test', type=float, default=0.01,
                    help='fraction of events to load each time from every file (when ``--fetch-by-files`` is disabled); '
                         'Or: number of files to load each time (when ``--fetch-by-files`` is enabled). Shuffling & sampling is done within these events, so set a large enough value.')
parser.add_argument('--in-memory', action='store_true', default=False,
                    help='load the whole dataset (and perform the preprocessing) only once and keep it in memory for the entire run')
parser.add_argument('--train-val-split', type=float, default=0.8,
                    help='training/validation split fraction')
parser.add_argument('--remake-weights', action='store_true', default=False,
                     help='remake weights for sampling (reweighting), use existing ones in the previous auto-generated data config YAML file')
parser.add_argument('--lr-finder', type=str, default=None,
                    help='run learning rate finder instead of the actual training; format: ``start_lr, end_lr, num_iters``')
parser.add_argument('--tensorboard', type=str, default=None,
                    help='create a tensorboard summary writer with the given comment')
parser.add_argument('--tensorboard-custom-fn', type=str, default=None,
                    help='the path of the python script containing a user-specified function `get_tensorboard_custom_fn`, '
                         'to display custom information per mini-batch or per epoch, during the training, validation or test.')
parser.add_argument('--network-config', type=str, default='',
                    help='network architecture configuration file; the path must be relative to the current dir')
parser.add_argument('--network-option', nargs=2, action='append', default=[],
                    help='options to pass to the model class constructor, e.g., `--network-option use_counts False`')
parser.add_argument('--model-prefix', type=str, default='models/{auto}/network',
                    help='path to save or load the model; for training, this will be used as a prefix, so model snapshots '
                         'will saved to `{model_prefix}_epoch-%%d_state.pt` after each epoch, and the one with the best '
                         'validation metric to `{model_prefix}_best_epoch_state.pt`; for testing, this should be the full path '
                         'including the suffix, otherwise the one with the best validation metric will be used; '
                         'for training, `{auto}` can be used as part of the path to auto-generate a name, '
                         'based on the timestamp and network configuration')
parser.add_argument('--load-model-weights', type=str, default=None,
                    help='initialize model with pre-trained weights')
parser.add_argument('--exclude-model-weights', type=str, default=None,
                    help='comma-separated regex to exclude matched weights from being loaded, e.g., `a.fc..+,b.fc..+`')
parser.add_argument('--num-epochs', type=int, default=20,
                    help='number of epochs')
parser.add_argument('--steps-per-epoch', type=int, default=None,
                    help='number of steps (iterations) per epochs; '
                         'if neither of `--steps-per-epoch` or `--samples-per-epoch` is set, each epoch will run over all loaded samples')
parser.add_argument('--steps-per-epoch-val', type=int, default=None,
                    help='number of steps (iterations) per epochs for validation; '
                         'if neither of `--steps-per-epoch-val` or `--samples-per-epoch-val` is set, each epoch will run over all loaded samples')
parser.add_argument('--samples-per-epoch', type=int, default=None,
                    help='number of samples per epochs; '
                         'if neither of `--steps-per-epoch` or `--samples-per-epoch` is set, each epoch will run over all loaded samples')
parser.add_argument('--samples-per-epoch-val', type=int, default=None,
                    help='number of samples per epochs for validation; '
                         'if neither of `--steps-per-epoch-val` or `--samples-per-epoch-val` is set, each epoch will run over all loaded samples')
parser.add_argument('--optimizer', type=str, default='ranger', choices=['adam', 'adamW', 'radam', 'ranger'],  # TODO: add more
                    help='optimizer for the training')
parser.add_argument('--optimizer-option', nargs=2, action='append', default=[],
                    help='options to pass to the optimizer class constructor, e.g., `--optimizer-option weight_decay 1e-4`')
parser.add_argument('--lr-scheduler', type=str, default='flat+decay',
                    choices=['none', 'steps', 'flat+decay', 'flat+linear', 'flat+cos', 'one-cycle', 'custom'],
                    help='learning rate scheduler')
parser.add_argument('--lr-epochs', type=int, default=20,
                    help='number of epochs to be considered by lr optimizer')
parser.add_argument('--lr-epoch-to-start-decay', type=int, default=2,
                    help='in the decay lr-scheduler start decaying the lr from this epoch, while in flat+decay or flat+linear start decay from lr-epochs-1/lr-epoch-to-start-decay')
parser.add_argument('--lr-decay-rate', type=int, default=0.7,
                    help='used in the decay lr-scheduler')
parser.add_argument('--warmup-steps', type=int, default=0,
                    help='number of warm-up steps, only valid for `flat+linear` and `flat+cos` lr schedulers')
parser.add_argument('--load-epoch', type=int, default=None,
                    help='used to resume interrupted training, load model and optimizer state saved in the `epoch-%%d_state.pt` and `epoch-%%d_optimizer.pt` files')
parser.add_argument('--load-best-metric', type=float, default=None,
                    help='best metric of the resumed training')
parser.add_argument('--start-lr', type=float, default=5e-3,
                    help='start learning rate')
parser.add_argument('--batch-size-train', type=int, default=128,
                    help='batch size')
parser.add_argument('--batch-size-val', type=int, default=128,
                    help='batch size')
parser.add_argument('--batch-size-test', type=int, default=128,
                    help='batch size')
parser.add_argument('--use-amp', action='store_true', default=False,
                    help='use mixed precision training (fp16)')
parser.add_argument('--persistent-workers', action='store_true', default=False,
                    help='make workers persistent')
parser.add_argument('--gpus', type=str, default='0',
                    help='device for the training/testing; to use CPU, set to empty string (""); to use multiple gpu, set it as a comma separated list, e.g., `1,2,3,4`')
parser.add_argument('--num-workers-train', type=int, default=1,
                    help='number of threads to load the training dataset; memory consumption and disk access load increases (~linearly) with this numbers')
parser.add_argument('--num-workers-val', type=int, default=1,
                    help='number of threads to load the validation dataset (when provided via --data-val otherwise use num-workers-train); memory consumption and disk access load increases (~linearly) with this numbers')
parser.add_argument('--num-workers-test', type=int, default=1,
                    help='number of threads to load the testing dataset; memory consumption and disk access load increases (~linearly) with this numbers')
parser.add_argument('--max-resample', type=int, default=10,
                    help='re-sampling factor for classification/regression events')
parser.add_argument('--predict', action='store_true', default=False,                    
                    help='run prediction instead of training')
parser.add_argument('--eps-attack', type=float, default=None,                    
                    help='value of the epsilon parameter in adv attack')
parser.add_argument('--frac-attack', type=float, default=None,                    
                    help='fraction of batches for adv attack')
parser.add_argument('--frac-batch-attack', type=float, default=0.5,                    
                    help='when adv attack is enabled, fraction of batch events dedidcate to it')
parser.add_argument('--epoch-start-attack', type=int, default=0,                    
                    help='Epoch from which start the adv attack')
parser.add_argument('--eval-attack', action='store_true', default=False,
                    help='Add adv attack varied scores in the output files')
parser.add_argument('--use-mdmm-constraints', action='store_true', default=False,
                    help='add mdmm parameters / costraints to the optimizer parameters')
parser.add_argument('--predict-output', type=str,
                    help='path to save the prediction output, support `.root` and `.parquet` format')
parser.add_argument('--export-onnx', type=str, default=None,
                    help='export the PyTorch model to ONNX model and save it at the given path (path must ends w/ .onnx); '
                         'needs to set `--data-config`, `--network-config`, and `--model-prefix` (requires the full model path)')
parser.add_argument('--onnx-opset', type=int, default=17,
                    help='ONNX opset version.')
parser.add_argument('--copy-inputs', action='store_true', default=False,
                    help='copy input files to the current dir (can help to speed up dataloading when running over remote files, e.g., from EOS)')
parser.add_argument('--log', '-l', type=str, default='',
                    help='path to the log file; `{auto}` can be used as part of the path to auto-generate a name, based on the timestamp and network configuration')
parser.add_argument('--profile', action='store_true', default=False,
                    help='run the profiler')
parser.add_argument('--backend', type=str, choices=['gloo', 'nccl', 'mpi'], default=None,
                    help='backend for distributed training')
parser.add_argument('--cross-validation', type=str, default=None,
                    help='enable k-fold cross validation; input format: `variable_name%k`')
parser.add_argument('--start-from-fold', type=int, default=0,
                    help='restart from a fold != 0')
parser.add_argument('--compile-model', action='store_true', default=False,
                    help='turn-on torch model compilation')
parser.add_argument("--local-rank", default=None, type=int,
                    help='local rank for DistributedDataParallel')

def to_filelist(args, mode='train'):

    if mode == 'train':
        flist = args.data_train
    elif mode == 'val':
        flist = args.data_val
    else:
        raise NotImplementedError('Invalid mode %s' % mode)

    # keyword-based: 'a:/path/to/a b:/path/to/b'
    file_dict = {}
    for f in flist:
        ## with xrootd a file by file list is provided
        if 'root:' in f:
            name, files = '_', f
            if name in file_dict:
                file_dict[name].append(files)
            else:
                file_dict[name] = []
                file_dict[name].append(files)
        else:
            if ':' in f:
                name, fp = f.split(':')
            else:
                name, fp = '_', f
            files = glob.glob(fp)
            if name in file_dict:
                file_dict[name] += files
            else:
                file_dict[name] = files
        # sort files
        for name, files in file_dict.items():
            file_dict[name] = sorted(files)

    if args.local_rank is not None:
        if mode == 'train':
            local_world_size = int(torch.distributed.get_world_size())
            new_file_dict = {}
            for name, files in file_dict.items():
                new_files = files[args.local_rank::local_world_size]
                assert(len(new_files) > 0)
                np.random.shuffle(new_files)
                new_file_dict[name] = new_files
            file_dict = new_file_dict

    if args.copy_inputs:
        import tempfile
        tmpdir = tempfile.mkdtemp()
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
        new_file_dict = {name: [] for name in file_dict}
        for name, files in file_dict.items():
            for src in files:
                dest = os.path.join(tmpdir, src.lstrip('/'))
                if not os.path.exists(os.path.dirname(dest)):
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(src, dest)
                _logger.info('Copied file %s to %s' % (src, dest))
                new_file_dict[name].append(dest)
            if len(files) != len(new_file_dict[name]):
                _logger.error('Only %d/%d files copied for %s file group %s',
                              len(new_file_dict[name]), len(files), mode, name)
        file_dict = new_file_dict

    filelist = sum(file_dict.values(), [])
    assert(len(filelist) == len(set(filelist)))
    return file_dict, filelist

def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy("file_system")

def train_load(args):
    """
    Loads the training data.
    :param args:
    :return: train_loader, val_loader, data_config, train_inputs
    """

    train_file_dict, train_files = to_filelist(args, 'train')
    if args.data_val:
        val_file_dict, val_files = to_filelist(args, 'val')
        train_range = val_range = (0, 1)
    else:
        val_file_dict, val_files = train_file_dict, train_files
        train_range = (0, args.train_val_split)
        val_range = (args.train_val_split, 1)

    _logger.info('Using %d files for training, range: %s' % (len(train_files), str(train_range)))
    _logger.info('Using %d files for validation, range: %s' % (len(val_files), str(val_range)))

    if args.in_memory and (args.steps_per_epoch is None or args.steps_per_epoch_val is None):
        raise RuntimeError('Must set --steps-per-epoch when using --in-memory!')

    ## create training dataset
    train_data = SimpleIterDataset(
        train_file_dict, args.data_config, for_training=True,
        load_range_and_fraction=(train_range, args.data_fraction),
        extra_selection=args.extra_selection,
        remake_weights = args.remake_weights,
        file_fraction=args.file_fraction,
        fetch_by_files=args.fetch_by_files,
        fetch_step=args.fetch_step_train if not args.fetch_by_files else 1,
        infinity_mode=args.steps_per_epoch is not None,
        in_memory=args.in_memory,
        max_resample=args.max_resample,
        name='train' + ('' if args.local_rank is None else '_rank%d' % args.local_rank)
    )

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size_train, drop_last=True, pin_memory=True,
        num_workers=min(args.num_workers_train, int(len(train_files) * args.file_fraction)),
        persistent_workers=args.num_workers_train > 0 and (args.steps_per_epoch is not None or args.persistent_workers),
        worker_init_fn=set_worker_sharing_strategy
    )

    ## create validation dataset
    val_data = SimpleIterDataset(
        val_file_dict, args.data_config, for_training=True,
        load_range_and_fraction=(val_range, args.data_fraction),
        extra_selection=args.extra_selection,
        file_fraction=args.file_fraction,
        fetch_by_files=args.fetch_by_files,
        fetch_step=args.fetch_step_val if not args.fetch_by_files else 1,
        infinity_mode=args.steps_per_epoch_val is not None,
        in_memory=args.in_memory,
        max_resample=args.max_resample,
        name='val' + ('' if args.local_rank is None else '_rank%d' % args.local_rank)
    )

    val_loader = DataLoader(
        val_data, batch_size=args.batch_size_val, drop_last=True, pin_memory=True,
        num_workers=min(args.num_workers_val if args.data_val else args.num_workers_train, int(len(val_files) * args.file_fraction)),
        persistent_workers= (args.num_workers_val > 0 if args.data_val else args.num_workers_train > 0) and (args.steps_per_epoch_val is not None or args.persistent_workers),
        worker_init_fn=set_worker_sharing_strategy
    )

    data_config = train_data.config
    train_input_names = train_data.config.input_names
    train_label_names = train_data.config.label_names
    train_target_names = train_data.config.target_names

    return train_loader, val_loader, data_config, train_input_names, train_label_names, train_target_names

def preprocess_load(args):

    """
    Loads all events for pre-processing.
    """

    preprocess_file_dict, preprocess_files = to_filelist(args, 'train')
    preprocess_range = (0,1)
    _logger.info('Using %d files for pre-processing, range: %s' % (len(preprocess_files), str(preprocess_range)))

    if args.in_memory and (args.steps_per_epoch is Nsone or args.steps_per_epoch_val is None):
        raise RuntimeError('Must set --steps-per-epoch when using --in-memory!')

    preprocess_data = SimpleIterDataset(
        preprocess_file_dict, args.data_config, for_training=True,
        load_range_and_fraction=(preprocess_range, args.data_fraction),
        extra_selection=args.extra_selection,
        file_fraction=args.file_fraction,
        fetch_by_files=args.fetch_by_files,
        fetch_step=args.fetch_step_preprocess if not args.fetch_by_files else 1,
        infinity_mode=args.steps_per_epoch is not None,
        in_memory=args.in_memory,
        max_resample=1,
        name='preprocess' + ('' if args.local_rank is None else '_rank%d' % args.local_rank)
    )

    return preprocess_data;


def test_load(args):

    """
    Loads the test data.
    :param args:
    :return: test_loaders, data_config
    """
    # keyword-based --data-test: 'a:/path/to/a b:/path/to/b'
    # split --data-test: 'a%10:/path/to/a/*'
    file_dict = {}
    split_dict = {}

    for f in args.data_test:
        ## special case of xrootd path to single files
        if 'root:' in f:
            name, files = '_', f
            if name in file_dict:
                file_dict[name].append(files)
            else:
                file_dict[name] = []
                file_dict[name].append(files)
        else:
            if ':' in f:
                name, fp = f.split(':')
                if '%' in name:
                    name, split = name.split('%')
                    split_dict[name] = int(split)
            else:
                name, fp = '', f
            files = glob.glob(fp)            
            if name in file_dict:
                file_dict[name] += files
            else:
                file_dict[name] = files

    # sort files
    for name, files in file_dict.items():
        file_dict[name] = sorted(files)

    # apply splitting
    for name, split in split_dict.items():
        files = file_dict.pop(name)
        for i in range((len(files) + split - 1) // split):
            file_dict[f'{name}_{i}'] = files[i * split:(i + 1) * split]

    def get_test_loader(name):
        filelist = file_dict[name]
        _logger.info('Running on test file group %s with %d files:\n...%s', name, len(filelist), '\n...'.join(filelist))

        num_workers = min(args.num_workers_test, len(filelist))

        test_data = SimpleIterDataset(
            {name: filelist}, args.data_config, for_training=False,
            load_range_and_fraction=((0, 1), args.data_fraction),
            extra_selection=args.extra_test_selection,
	    file_fraction=args.file_fraction,
            fetch_by_files=args.fetch_by_files,
            fetch_step=args.fetch_step_test if not args.fetch_by_files else 1,
            in_memory=args.in_memory,
            name='test_' + name            
        )

        test_loader = DataLoader(
            test_data, batch_size=args.batch_size_test, drop_last=False, pin_memory=True,
            num_workers = min(args.num_workers_test, len(filelist)),
            worker_init_fn=set_worker_sharing_strategy
        )        
        return test_loader;

    test_loaders = {name: functools.partial(get_test_loader, name) for name in file_dict}
    data_config = SimpleIterDataset({}, args.data_config, for_training=False).config
    return test_loaders, data_config


def onnx(args):
    """
    Saving model as ONNX.
    :param args:
    :return:
    """
    assert (args.export_onnx.endswith('.onnx'))
    model_path = args.model_prefix
    _logger.info('Exporting model %s to ONNX' % model_path)

    from utils.dataset import DataConfig
    data_config = DataConfig.load(args.data_config, load_observers=False, load_reweight_info=False)
    model, model_info, _ = model_setup(args, data_config)
    if "domain" in args.weaver_mode:
        model.load_state_dict(torch.load(model_path, map_location='cpu'),strict=False)
    else:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

    model = model.cpu()
    model.eval()

    if not os.path.dirname(args.export_onnx):
        args.export_onnx = os.path.join(os.path.dirname(model_path), args.export_onnx)
    os.makedirs(os.path.dirname(args.export_onnx), exist_ok=True)
    inputs = tuple(
        torch.ones(model_info['input_shapes'][k], dtype=torch.float32) for k in model_info['input_names'])
    torch.onnx.export(model, inputs, args.export_onnx,
                      input_names=model_info['input_names'],
                      output_names=model_info['output_names'],
                      dynamic_axes=model_info.get('dynamic_axes', None),
                      opset_version=args.onnx_opset)
    _logger.info('ONNX model saved to %s', args.export_onnx)

    preprocessing_json = os.path.join(os.path.dirname(args.export_onnx),args.export_onnx.replace(".onnx",".json"))
    network_options = {k: ast.literal_eval(v) for k, v in args.network_option}
    data_config.export_json(preprocessing_json,network_options.get("add_da_inference",False))
    _logger.info('Preprocessing parameters saved to %s', preprocessing_json)


def flops(model, model_info,device='cpu'):
    """
    Count FLOPs and params.
    :param args:
    :param model:
    :param model_info:
    :return:
    """
    from utils.flops_counter import get_model_complexity_info
    import copy

    model = copy.deepcopy(model).to(device)
    model.eval()

    inputs = tuple(
        torch.ones(model_info['input_shapes'][k], dtype=torch.float32, device=device) for k in model_info['input_names'])

    macs, params = get_model_complexity_info(model, inputs, as_strings=True, print_per_layer_stat=True, verbose=True)
    _logger.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    _logger.info('{:<30}  {:<8}'.format('Number of parameters: ', params))


def profile(args, model, model_info, device):
    """
    Profile.
    :param model:
    :param model_info:
    :return:
    """
    import copy
    from torch.profiler import profile, record_function, ProfilerActivity

    model = copy.deepcopy(model)
    model = model.to(device)
    model.eval()

    inputs = tuple(
        torch.ones((args.batch_size_train,) + model_info['input_shapes'][k][1:],
                   dtype=torch.float32).to(device) for k in model_info['input_names'])
    for x in inputs:
        print(x.shape, x.device)

    def trace_handler(p):
        output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=50)
        print(output)
        p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=2,
            warmup=2,
            active=6,
            repeat=2),
        on_trace_ready=trace_handler
    ) as p:
        for idx in range(100):
            model(*inputs)
            p.step()

def optim(args, model, device, loss_func=None):
    """
    Optimizer and scheduler.
    :param args:
    :param model:
    :return:
    """
    optimizer_options = {k: ast.literal_eval(v) for k, v in args.optimizer_option}
    _logger.info('Optimizer options: %s' % str(optimizer_options))

    names_lr_mult = []
    if 'weight_decay' in optimizer_options or 'lr_mult' in optimizer_options:
        # https://github.com/rwightman/pytorch-image-models/blob/master/timm/optim/optim_factory.py#L31
        import re
        decay, no_decay = {}, {}
        names_no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or (
                    hasattr(model, 'no_weight_decay') and name in model.no_weight_decay()):
                no_decay[name] = param
                names_no_decay.append(name)
            else:
                decay[name] = param

        decay_1x, no_decay_1x = [], []
        decay_mult, no_decay_mult = [], []
        mult_factor = 1
        if 'lr_mult' in optimizer_options:
            pattern, mult_factor = optimizer_options.pop('lr_mult')
            for name, param in decay.items():
                if re.match(pattern, name):
                    decay_mult.append(param)
                    names_lr_mult.append(name)
                else:
                    decay_1x.append(param)
            for name, param in no_decay.items():
                if re.match(pattern, name):
                    no_decay_mult.append(param)
                    names_lr_mult.append(name)
                else:
                    no_decay_1x.append(param)
            assert(len(decay_1x) + len(decay_mult) == len(decay))
            assert(len(no_decay_1x) + len(no_decay_mult) == len(no_decay))
        else:
            decay_1x, no_decay_1x = list(decay.values()), list(no_decay.values())
        wd = optimizer_options.pop('weight_decay', 0.)
        parameters = [
            {'params': no_decay_1x, 'weight_decay': 0.},
            {'params': decay_1x, 'weight_decay': wd},
            {'params': no_decay_mult, 'weight_decay': 0., 'lr': args.start_lr * mult_factor},
            {'params': decay_mult, 'weight_decay': wd, 'lr': args.start_lr * mult_factor},
        ]
        _logger.info('Parameters excluded from weight decay:\n - %s', '\n - '.join(names_no_decay))
        if len(names_lr_mult):
            _logger.info('Parameters with lr multiplied by %s:\n - %s', mult_factor, '\n - '.join(names_lr_mult))
    else:
        parameters = model.parameters()

    ## replicating what is done in https://github.com/the-moliver/mdmm/blob/master/mdmm/mdmm.py#L154
    if args.use_mdmm_constraints and hasattr(loss_func,'lambdas') and hasattr(loss_func,'slacks'):
        params = parameters;
        parameters = [
            {'params': params, 'lr': args.start_lr},
            {'params': loss_func.lambdas, 'lr': -args.start_lr},
            {'params': loss_func.slacks, 'lr': args.start_lr}
        ];
        
    if args.optimizer == 'ranger':
        from utils.nn.optimizer.ranger import Ranger
        opt = Ranger(parameters, lr=args.start_lr, **optimizer_options);
    elif args.optimizer == 'adam':
        opt = torch.optim.Adam(parameters, lr=args.start_lr, **optimizer_options)
    elif args.optimizer == 'adamW':
        opt = torch.optim.AdamW(parameters, lr=args.start_lr, **optimizer_options)
    elif args.optimizer == 'radam':
        opt = torch.optim.RAdam(parameters, lr=args.start_lr, **optimizer_options)

    # load previous training and resume if `--load-epoch` is set
    if args.load_epoch is not None:
        _logger.info('Resume training from epoch %d' % args.load_epoch)
        _logger.info('Open model state file '+args.model_prefix+'_epoch-%d_state.pt' % args.load_epoch)
        model_state = torch.load(args.model_prefix + '_epoch-%d_state.pt' % args.load_epoch, map_location=device)
        if isinstance(model,(torch.nn.parallel.DistributedDataParallel,torch.nn.DataParallel)):
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)

        _logger.info('Open optimizer state file '+args.model_prefix+'_epoch-%d_optimizer.pt' % args.load_epoch)
        opt_state_file = args.model_prefix + '_epoch-%d_optimizer.pt' % args.load_epoch
        if os.path.exists(opt_state_file):
            opt_state = torch.load(opt_state_file, map_location=device)
            opt.load_state_dict(opt_state)
        else:
            _logger.warning('Optimizer state file %s NOT found!' % opt_state_file)
            
    scheduler = None
    if args.lr_finder is None:
        if args.lr_scheduler == 'steps':
            lr_step = round(args.lr_epochs / args.lr_epoch_to_start_decay)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                opt, milestones=[lr_step, 2 * lr_step], gamma=0.25, last_epoch=-1 if args.load_epoch is None else args.load_epoch)
            scheduler._update_per_step = False;
        elif args.lr_scheduler == 'flat+decay':
            num_decay_epochs = max(1, round(args.lr_epochs / args.lr_epoch_to_start_decay))
            milestones = list(range(args.lr_epochs - num_decay_epochs, args.lr_epochs))
            gamma = 0.01 ** (1. / num_decay_epochs)
            if len(names_lr_mult):
                def get_lr(epoch): return gamma ** max(0, epoch - milestones[0] + 1)  # noqa
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    opt, (lambda _: 1, lambda _: 1, get_lr, get_lr), last_epoch= -1 if args.load_epoch is None else args.load_epoch)
            else:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    opt, milestones=milestones, gamma=gamma, last_epoch= -1 if args.load_epoch is None else args.load_epoch)
            scheduler._update_per_step = False;
        elif args.lr_scheduler == 'custom':
            scheduler1 = torch.optim.lr_scheduler.ConstantLR(
                opt,args.start_lr,args.lr_epoch_to_start_decay, last_epoch= -1 if args.load_epoch is None else args.load_epoch)
            learnfunc  = lambda epoch: args.lr_decay_rate ** max(0,epoch-args.lr_epoch_to_start_decay+1);
            scheduler2 = torch.optim.lr_scheduler.LambdaLR(opt, learnfunc, last_epoch= -1 if args.load_epoch is None else args.load_epoch)
            scheduler  = torch.optim.lr_scheduler.ChainedScheduler([scheduler1,scheduler2])
            scheduler._update_per_step = False;
        elif args.lr_scheduler == 'flat+linear' or args.lr_scheduler == 'flat+cos':
            total_steps = args.lr_epochs * args.steps_per_epoch
            warmup_steps = args.warmup_steps
            flat_steps = total_steps * 0.7 - 1
            min_factor = 0.001
            def lr_fn(step_num):
                if step_num > total_steps:
                    raise ValueError(
                        "Tried to step {} times. The specified number of total steps is {}".format(
                            step_num + 1, total_steps))
                if step_num < warmup_steps:
                    return 1. * step_num / warmup_steps
                if step_num <= flat_steps:
                    return 1.0
                pct = (step_num - flat_steps) / (total_steps - flat_steps)
                if args.lr_scheduler == 'flat+linear':
                    return max(min_factor, 1 - pct)
                else:
                    return max(min_factor, 0.5 * (math.cos(math.pi * pct) + 1))
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    opt, lr_fn, last_epoch= -1 if args.load_epoch is None else args.load_epoch)
            scheduler._update_per_step = True  # mark it to update the lr every step, instead of every epoch
        elif args.lr_scheduler == 'one-cycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=args.start_lr, epochs=args.lr_epochs, steps_per_epoch=args.steps_per_epoch, pct_start=0.3,
                anneal_strategy='cos', div_factor=25.0, last_epoch= -1 if args.load_epoch is None else args.load_epoch)
            scheduler._update_per_step = True  # mark it to update the lr every step, instead of every epoch
    return opt, scheduler


def model_setup(args, data_config,device='cpu'):
    """
    Loads the model
    :param args:
    :param data_config:
    :return: model, model_info, network_module, network_options
    """
    network_module = import_module(args.network_config, name='_network_module')
    network_options = {k: ast.literal_eval(v) for k, v in args.network_option}
    _logger.info('Network options: %s' % str(network_options))
    if args.export_onnx:
        network_options['for_inference'] = True
    if args.use_amp:
        network_options['use_amp'] = True
    model, model_info = network_module.get_model(data_config, **network_options)
    if args.load_model_weights:
        model_state = torch.load(args.load_model_weights, map_location='cpu')
        if args.exclude_model_weights:
            import re
            exclude_patterns = args.exclude_model_weights.split(',')
            _logger.info('The following weights will not be loaded: %s' % str(exclude_patterns))
            key_state = {}
            for k in model_state.keys():
                key_state[k] = True
                for pattern in exclude_patterns:
                    if re.match(pattern, k):
                        key_state[k] = False
                        break
            model_state = {k: v for k, v in model_state.items() if key_state[k]}
        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
        _logger.info('Model initialized with weights from %s\n ... Missing: %s\n ... Unexpected: %s' %
                     (args.load_model_weights, missing_keys, unexpected_keys))

    flops(model, model_info,device='cpu')
    # loss function
    loss_func = network_module.get_loss(data_config, **network_options)
    _logger.info('Using loss function %s with options %s' % (loss_func, network_options))

    return model, model_info, loss_func


def iotest(args, data_loader):
    """
    Io test
    :param args:
    :param data_loader:
    :return:
    """
    from tqdm.auto import tqdm
    from collections import defaultdict
    from utils.data.tools import _concat
    _logger.info('Start running IO test')
    monitor_info = defaultdict(list)

    for X, y_cat, y_reg, y_quant, Z in tqdm(data_loader):
        for k, v in Z.items():
            monitor_info[k].append(v)
    monitor_info = {k: _concat(v) for k, v in monitor_info.items()}
    if monitor_info:
        monitor_output_path = 'weaver_monitor_info.parquet'
        try:
            import awkward as ak
            ak.to_parquet(ak.Array(monitor_info), monitor_output_path, compression='LZ4', compression_level=4)
            _logger.info('Monitor info written to %s' % monitor_output_path, color='bold')
        except Exception as e:
            _logger.error('Error when writing output parquet file: \n' + str(e))

def save_root(args, output_path, data_config, scores, labels, targets, labels_domain, observers, scores_attack=np.array([])):
    """
    Saves as .root
    :param data_config:
    :param scores:
    :param labels
    :param observers
    :return:
    """
    from utils.data.fileio import _write_root
    output = {}

    if args.weaver_mode == "class":
        for idx, label_name in enumerate(data_config.label_value):
            output[label_name] = (labels[data_config.label_names[0]] == idx)
            output['score_' + label_name] = scores[:,idx]
    elif args.weaver_mode == "reg":
        for idx, target_name in enumerate(data_config.target_value):
            output['score_' + target_name] = scores[:,idx]
    elif args.weaver_mode == "classreg":
        for idx, label_name in enumerate(data_config.label_value):
            output[label_name] = (labels[data_config.label_names[0]] == idx)
            output['score_' + label_name] = scores[:,idx]
        for idx, target_name in enumerate(data_config.target_value):
            output['score_' + target_name] = scores[:,len(data_config.label_value)+idx]
    elif args.weaver_mode == "classregdomain" or args.weaver_mode == "classregdomainattack":
        for idx, label_name in enumerate(data_config.label_value):
            output[label_name] = (labels[data_config.label_names[0]] == idx)
            output['score_' + label_name] = scores[:,idx]
        for idx, target_name in enumerate(data_config.target_value):
            output['score_' + target_name] = scores[:,len(data_config.label_value)+idx]
        if type(data_config.label_domain_value) == dict:
            for idx, (k,v) in enumerate(data_config.label_domain_value.items()):
                for idy, label_name in enumerate(v):
                    output[label_name] = (labels_domain[k] == idy)
                    output['score_' + label_name] = scores[:,len(data_config.label_value)+len(data_config.target_value)+idx*len(v)+idy]    
        else:
            for idx, label_name in enumerate(data_config.label_domain_value):
                output[label_name] = (labels_domain[data_config.label_domain_names[0]] == idx)
                output['score_' + label_name] = scores[:,len(data_config.label_value)+len(data_config.target_value)+idx]    

    else:
        _logger.warning("Weaver mode not recognized when saving output file --> abort")
        sys.exit(0);

    ### save adv attack scores
    if  scores_attack.any():
        for idx, label_name in enumerate(data_config.label_value):
            output['score_' + label_name + "_attack"] = scores_attack[:,idx]

    ## break if nothing appears in the output
    if not output:
        _logger.warning("Weaver mode not recognized when saving output file --> abort")
        sys.exit(0);

    ## check output informatuion
    for k, v in labels.items():
        if k == data_config.label_names[0]:
            continue
        if v.ndim > 1:
            _logger.warning('Ignoring %s, not a 1d array.', k)
            continue
        output[k] = v

    for k, v in targets.items():
        if v.ndim > 1:
            _logger.warning('Ignoring %s, not a 1d array.', k)
            continue
        output[k] = v

    for k, v in labels_domain.items():
        if k == data_config.label_domain_names[0]:
            continue
        if v.ndim > 1:
            _logger.warning('Ignoring %s, not a 1d array.', k)
            continue
        output[k] = v

    for k, v in observers.items():
        if v.ndim > 1:
            _logger.warning('Ignoring %s, not a 1d array.', k)
            continue
        output[k] = v

    _write_root(output_path, output)


def save_parquet(args, output_path, scores, labels, targets, labels_domain, observers):
    """
    Saves as parquet file
    :param scores:
    :param labels:
    :param targets:
    :param observers:
    :return:
    """
    import awkward as ak
    output = {'scores': scores}
    output.update(labels)
    output.update(targets)
    output.update(labels_domain)
    output.update(observers)
    try:
        ak.to_parquet(ak.Array(output), output_path, compression='LZ4', compression_level=4)
        _logger.info('Written output to %s' % output_path, color='bold')
    except Exception as e:
        _logger.error('Error when writing output parquet file: \n' + str(e))


def _main(args):

    _logger.info('args:\n - %s', '\n - '.join(str(it) for it in args.__dict__.items()))

    # export to ONNX 
    if args.export_onnx:
        onnx(args);
        sys.exit(0);

    if args.file_fraction < 1:
        _logger.warning('Use of `file-fraction` is not recommended in general -- prefer using `data-fraction` instead.')

    # classification/regression mode
    if args.weaver_mode == "class":
        _logger.info('Running in classification mode')
        from utils.nn.tools import train_classification as train
        from utils.nn.tools import evaluate_classification as evaluate
        from utils.nn.tools import evaluate_onnx_classification as evaluate_onnx
    elif args.weaver_mode == "reg":
        _logger.info('Running in regression mode')
        from utils.nn.tools import train_regression as train
        from utils.nn.tools import evaluate_regression as evaluate
        from utils.nn.tools import evaluate_onnx_regression as evaluate_onnx
    elif args.weaver_mode == "classreg":
        _logger.info('Running in combined regression + classification mode')
        from utils.nn.tools import train_classreg as train
        from utils.nn.tools import evaluate_classreg as evaluate
        from utils.nn.tools import evaluate_onnx_classreg as evaluate_onnx
    elif args.weaver_mode == "classregdomain":
        _logger.info('Running in combined regression + classification mode with domain adaptation')
        from utils.nn.tools_domain_attack import train_classreg as train
        from utils.nn.tools_domain_attack import evaluate_classreg as evaluate
        from utils.nn.tools_domain_attack import evaluate_onnx_classreg as evaluate_onnx
    elif args.weaver_mode == "classregdomainattack":
        _logger.info('Running in combined regression + classification mode with domain adaptation and attack')
        from utils.nn.tools_domain_attack import train_classreg as train
        from utils.nn.tools_domain_attack import evaluate_classreg as evaluate
        from utils.nn.tools_domain_attack import evaluate_onnx_classreg as evaluate_onnx

    # training/testing mode
    training_mode = not args.predict

    # device detection
    if args.gpus:
        gpus = [int(i) for i in args.gpus.split(',')]
        ngpus = len(gpus);
        if args.backend is not None and ngpus > 1:
            gpus = [args.local_rank]
            torch.cuda.set_device(args.local_rank)
            dev = torch.device(args.local_rank)
            torch.distributed.init_process_group(backend=args.backend,init_method='env://')
            _logger.info(f'Using distributed PyTorch with {args.backend} backend')
        else:
            dev = torch.device(gpus[0])
    else:
        gpus = None
        dev = torch.device('cpu')
        try:
            if torch.backends.mps.is_available():
                dev = torch.device('mps')
        except AttributeError:
            pass

    if args.tensorboard:
        from utils.nn.tools import TensorboardHelper
        tb = TensorboardHelper(tb_comment=args.tensorboard, tb_custom_fn=args.tensorboard_custom_fn)
    else:
        tb = None

    # load data
    if args.weaver_mode == "preprocess":
        preprocess_data = preprocess_load(args);
        sys.exit(0);
    else:
        if training_mode:
            train_loader, val_loader, data_config, train_input_names, train_label_names, train_target_names = train_load(args)
        else:
            test_loaders, data_config = test_load(args)

    ## setup the model
    model, model_info, loss_func = model_setup(args, data_config, device=dev)
    
    if args.profile:
        profile(args, model, model_info, device=dev)
        sys.exit(0);
        
    # note: we should always save/load the state_dict of the original model, not the one wrapped by nn.DataParallel
    # so we do not convert it to nn.DataParallel now
    grad_scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    if training_mode:

        model = model.to(dev)
        if args.compile_model and ngpus:
            model = torch.compile(model);
            
        # DistributedDataParallel
        if args.backend is not None and ngpus > 1: 
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=gpus, output_device=args.local_rank)
        else:
            # DataParallel
            if gpus is not None and ngpus > 1:
                model = torch.nn.DataParallel(model, device_ids=gpus)

        # optimizer & learning rate
        opt, scheduler = optim(args, model, dev, loss_func=loss_func)

        # lr finder: keep it after all other setups
        if args.lr_finder is not None:
            start_lr, end_lr, num_iter = args.lr_finder.replace(' ', '').split(',')
            from utils.lr_finder import LRFinder
            lr_finder = LRFinder(model, opt, loss_func, device=dev, input_names=train_input_names,
                                 label_names=train_label_names+train_target_names)
            lr_finder.range_test(train_loader, start_lr=float(start_lr), end_lr=float(end_lr), num_iter=int(num_iter))
            lr_finder.plot(output='lr_finder.png')  # to inspect the loss-learning rate graph
            sys.exit(0);

        # training loop
        best_val_metric = np.inf;
        for epoch in range(args.num_epochs):
            
            if args.load_epoch is not None:

                if epoch <= args.load_epoch:
                    _logger.info('Skip Epoch #%d in training' % epoch)
                    continue

                if args.load_best_metric is not None:
                    best_val_metric = args.load_best_metric
                    
            _logger.info('-' * 50)
            _logger.info('Epoch #%d training' % epoch)

            if "attack" in args.weaver_mode:
                train(model, loss_func, opt, scheduler, train_loader, dev, epoch, steps_per_epoch=args.steps_per_epoch, grad_scaler=grad_scaler, tb_helper=tb, network_option=args.network_option,
                      eps_attack=args.eps_attack, epoch_start_attack=args.epoch_start_attack, frac_attack=args.frac_attack, frac_batch_attack=args.frac_batch_attack);
            else:
                train(model, loss_func, opt, scheduler, train_loader, dev, epoch, steps_per_epoch=args.steps_per_epoch, grad_scaler=grad_scaler, tb_helper=tb, network_option=args.network_option);
                
            if args.model_prefix and (args.backend is None or args.local_rank == 0):
                dirname = os.path.dirname(args.model_prefix)
                if dirname and not os.path.exists(dirname):
                    os.makedirs(dirname)
                state_dict = model.module.state_dict() if isinstance(
                    model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else model.state_dict()
                torch.save(state_dict, args.model_prefix + '_epoch-%d_state.pt' % epoch)
                torch.save(opt.state_dict(), args.model_prefix + '_epoch-%d_optimizer.pt' % epoch)
                
                _logger.info('Epoch #%d validating' % epoch)

            if "attack" in args.weaver_mode:                
                val_metric = evaluate(model, val_loader, dev, epoch, loss_func=loss_func, steps_per_epoch=args.steps_per_epoch_val, grad_scaler=grad_scaler, tb_helper=tb,
                                      network_option=args.network_option, eval_attack=args.eval_attack, eps_attack=args.eps_attack, epoch_start_attack=args.epoch_start_attack, frac_attack=args.frac_attack)
            else:
                val_metric = evaluate(model, val_loader, dev, epoch, loss_func=loss_func, steps_per_epoch=args.steps_per_epoch_val, grad_scaler=grad_scaler, tb_helper=tb,  
                                      network_option=args.network_option, eval_attack=args.eval_attack)
                
            is_best_epoch = (val_metric < best_val_metric)

            if is_best_epoch:
                best_val_metric = val_metric
                if args.model_prefix and (args.backend is None or args.local_rank == 0):
                    shutil.copy2(args.model_prefix + '_epoch-%d_state.pt' %
                                 epoch, args.model_prefix + '_best_epoch_state.pt')
            _logger.info('Epoch #%d: Current validation metric: %.5f (best: %.5f)' %
                         (epoch, val_metric, best_val_metric), color='bold')    
            _logger.info('Best validation metric: %f',best_val_metric)

            if scheduler and not getattr(scheduler, '_update_per_step', False):
                scheduler.step()

            gc.collect();
            torch.cuda.empty_cache();
                
    if args.data_test:

        model = model.to(dev)
        if args.compile_model and ngpus:
            model = torch.compile(model);

        if args.backend is not None and args.local_rank != 0:
            sys.exit(0);

        if not args.model_prefix.endswith('.onnx'):

            model_path = args.model_prefix if args.model_prefix.endswith(
                '.pt') else args.model_prefix + '_best_epoch_state.pt'
            _logger.info('Loading model %s for eval' % model_path)
             
            if args.backend is not None and ngpus > 1:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=gpus, output_device=args.local_rank)
                model.module.load_state_dict(torch.load(model_path, map_location=dev))
            else:
                if gpus is not None and ngpus > 1:
                    model = torch.nn.DataParallel(model, device_ids=gpus)
                    model.module.load_state_dict(torch.load(model_path, map_location=dev))
                else:
                    model.load_state_dict(torch.load(model_path, map_location=dev))
                    
        for name, get_test_loader in test_loaders.items():
            test_loader = get_test_loader()
            # run prediction
            if args.model_prefix.endswith('.onnx'):
                _logger.info('Loading model %s for eval' % args.model_prefix)
                test_metric, scores, labels, targets, labels_domain, observers = evaluate_onnx(
                    args.model_prefix, test_loader)
            else:
                if args.eval_attack:
                    test_metric, scores, labels, targets, labels_domain, observers, scores_attack = evaluate(
                        model, test_loader, dev, loss_func=loss_func, epoch=None, for_training=False, tb_helper=tb, grad_scaler=grad_scaler,
                        eps_attack=args.eps_attack, eval_attack=args.eval_attack, network_option=args.network_option)
                else:
                    test_metric, scores, labels, targets, labels_domain, observers = evaluate(
                        model, test_loader, dev, loss_func=loss_func, epoch=None, for_training=False, tb_helper=tb, grad_scaler=grad_scaler,
                        network_option=args.network_option)
            _logger.info('Test metric %.5f' % test_metric, color='bold')

            if args.predict_output and scores.ndim:
                if not os.path.dirname(args.predict_output):
                    predict_output = os.path.join(
                        os.path.dirname(args.model_prefix),
                        'predict_output', args.predict_output)
                else:
                    predict_output = args.predict_output
                os.makedirs(os.path.dirname(predict_output), exist_ok=True)
                if name == '':
                    output_path = predict_output
                else:
                    base, ext = os.path.splitext(predict_output)
                    output_path = base + '_' + name + ext

                if output_path.endswith('.root'):
                    if args.eval_attack:
                        save_root(args, output_path, data_config, scores, labels, targets, labels_domain, observers, scores_attack)
                    else:
                        save_root(args, output_path, data_config, scores, labels, targets, labels_domain, observers)
                else:
                    save_parquet(args, output_path, scores, labels, targets, domains, labels_domain, observers)
                _logger.info('Written output to %s' % output_path, color='bold')

            gc.collect();
            torch.cuda.empty_cache();

def main():

    args = parser.parse_args()
    if args.samples_per_epoch is not None:
        if args.steps_per_epoch is None:
            args.steps_per_epoch = args.samples_per_epoch // args.batch_size+train
        else:
            raise RuntimeError('Please use either `--steps-per-epoch` or `--samples-per-epoch`, but not both!')

    if args.samples_per_epoch_val is not None:
        if args.steps_per_epoch_val is None:
            args.steps_per_epoch_val = args.samples_per_epoch_val // args.batch_size_train
        else:
            raise RuntimeError('Please use either `--steps-per-epoch-val` or `--samples-per-epoch-val`, but not both!')

    if args.steps_per_epoch_val is None and args.steps_per_epoch is not None:
        args.steps_per_epoch_val = round(args.steps_per_epoch * (1 - args.train_val_split) / args.train_val_split)
    if args.steps_per_epoch_val is not None and args.steps_per_epoch_val < 0:
        args.steps_per_epoch_val = None

    if '{auto}' in args.model_prefix or '{auto}' in args.log:
        import hashlib
        import time
        model_name = time.strftime('%Y%m%d-%H%M%S') + "_" + os.path.basename(args.network_config).replace('.py', '')
        if len(args.network_option):
            model_name = model_name + "_" + hashlib.md5(str(args.network_option).encode('utf-8')).hexdigest()
        model_name += '_{optim}_lr{lr}_batch{batch}'.format(lr=args.start_lr,
                                                            optim=args.optimizer, batch=args.batch_size_train)
        args._auto_model_name = model_name
        args.model_prefix = args.model_prefix.replace('{auto}', model_name)
        args.log = args.log.replace('{auto}', model_name)
        print('Using auto-generated model prefix %s' % args.model_prefix)

    if args.local_rank == None:
        args.local_rank = None if args.backend is None else int(os.environ.get("LOCAL_RANK","0"))
        
    stdout = sys.stdout
    if args.local_rank is not None:
        args.log += '.%03d' % args.local_rank
        if args.local_rank != 0:
            stdout = None
    _configLogger('weaver', stdout=stdout, filename=args.log)

    if args.cross_validation:
        model_dir, model_fn = os.path.split(args.model_prefix)
        if args.predict_output:
            predict_output_base, predict_output_ext = os.path.splitext(args.predict_output)
        load_model = args.load_model_weights or None
        var_name, kfold = args.cross_validation.split('%')
        kfold = int(kfold)
        for i in range(args.start_from_fold,kfold):
            _logger.info(f'\n=== Running cross validation, fold {i} of {kfold} ===')
            args.model_prefix = os.path.join(f'{model_dir}_fold{i}', model_fn)
            if args.predict_output:
                args.predict_output = f'{predict_output_base}_fold{i}' + predict_output_ext
            args.extra_selection = f'{var_name}%{kfold}!={i}'
            args.extra_test_selection = f'{var_name}%{kfold}=={i}'
            if load_model and '{fold}' in load_model:
                args.load_model_weights = load_model.replace('{fold}', f'fold{i}')
            _main(args)
    else:
        _main(args)

if __name__ == '__main__':

    main()
