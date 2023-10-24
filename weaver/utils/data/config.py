import numpy as np
import yaml
import copy

from ..logger import _logger
from .tools import _get_variable_names


def _as_list(x):
    if x is None:
        return None
    elif isinstance(x, (list, tuple)):
        return x
    else:
        return [x]


def _md5(fname):
    '''https://stackoverflow.com/questions/3431825/generating-an-md5-checksum-of-a-file'''
    import hashlib
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


class DataConfig(object):
    r"""Data loading configuration.
    """

    def __init__(self, print_info=True, **kwargs):

        opts = {
            'treename': None,
            'branch_magic': None,
            'file_magic': None,
            'selection': None,
            'test_time_selection': None,
            'preprocess': {'method': 'manual', 'data_fraction': 0.1, 'params': None},
            'new_variables': {},
            'inputs': {},
            'labels': {},
            'targets': {},
            'labels_domain': {},
            'observers': [],
            'monitor_variables': [],
            'weights': None,
        }
        for k, v in kwargs.items():
            if v is not None:
                if isinstance(opts[k], dict):
                    opts[k].update(v)
                else:
                    opts[k] = v
        # only information in ``self.options'' will be persisted when exporting to YAML
        self.options = opts
        if print_info:
            _logger.debug(opts)

        self.selection = opts['selection']
        self.test_time_selection = opts['test_time_selection'] if opts['test_time_selection'] else self.selection
        self.var_funcs = copy.deepcopy(opts['new_variables'])
        # preprocessing config
        self.preprocess = opts['preprocess']
        self._auto_standardization = opts['preprocess']['method'].lower().startswith('auto')
        self._missing_standardization_info = False
        self.preprocess_params = opts['preprocess']['params'] if opts['preprocess']['params'] is not None else {}
        # inputs
        self.input_names = tuple(opts['inputs'].keys())
        self.input_dicts = {k: [] for k in self.input_names}
        self.input_shapes = {}
        for k, o in opts['inputs'].items():
            self.input_shapes[k] = (-1, len(o['vars']), o['length'])
            for v in o['vars']:
                v = _as_list(v)
                self.input_dicts[k].append(v[0])

                if opts['preprocess']['params'] is None:

                    def _get(idx, default):
                        try:
                            return v[idx]
                        except IndexError:
                            return default

                    params = {'length': o['length'],
                              'pad_mode': o.get('pad_mode', 'constant').lower(),
                              'center': _get(1, 'auto' if self._auto_standardization else None),
                              'scale': _get(2, 1),
                              'min': _get(3, -5),
                              'max': _get(4, 5),
                              'eps_min': _get(5,None),
                              'eps_max': _get(6,None),
                              'pad_value': _get(7,0)
                    }
                    
                    if v[0] in self.preprocess_params and params != self.preprocess_params[v[0]]:
                        raise RuntimeError(
                            'Incompatible info for variable %s, had: \n  %s\nnow got:\n  %s' %
                            (v[0], str(self.preprocess_params[v[0]]), str(params)))
                    if k.endswith('_mask') and params['pad_mode'] != 'constant':
                        raise RuntimeError('The `pad_mode` must be set to `constant` for the mask input `%s`' % k)
                    if params['center'] == 'auto':
                        self._missing_standardization_info = True
                    self.preprocess_params[v[0]] = params
        # labels
        if opts['labels']:
            self.label_type = opts['labels']['type']
            self.label_value = opts['labels']['value']
            if self.label_type == 'simple':
                assert(isinstance(self.label_value, list))
                self.label_names = ('_label_',)
                self.labelcheck_names = ('_labelcheck_',)
                label_exprs = ['ak.to_numpy(%s)' % k for k in self.label_value]
                self.var_funcs['_label_'] = 'np.argmax(np.stack([%s], axis=1), axis=1)' % (','.join(label_exprs))
                self.var_funcs['_labelcheck_'] = 'np.sum(np.stack([%s], axis=1), axis=1)' % (','.join(label_exprs))
            else:
                self.label_names = tuple(self.label_value.keys())
                self.var_funcs.update(self.label_value)
                self.labelcheck_names = None;
        else:
            self.label_names = tuple();
            self.label_type  = None;
            self.label_value = None;
            self.labelcheck_names = None;

        ## domain
        if opts['labels_domain']:
            self.label_domain_type = opts['labels_domain']['type']
            self.label_domain_value = opts['labels_domain']['value']
            self.label_domain_loss_weight = None
            if self.label_domain_type == 'simple':
                assert(isinstance(self.label_domain_value, list))
                self.label_domain_names = ('_label_domain_',)
                self.labelcheck_domain_names = ('_labelcheck_domain_',)
                label_exprs = ['ak.to_numpy(%s)' % k for k in self.label_domain_value]
                self.var_funcs['_label_domain_'] = 'np.argmax(np.stack([%s], axis=1), axis=1)' % (','.join(label_exprs))
                self.var_funcs['_labelcheck_domain_'] = 'np.sum(np.stack([%s], axis=1), axis=1)' % (','.join(label_exprs))
            else:
                self.label_domain_names = tuple(self.label_domain_value.keys())                
                self.labelcheck_domain_names = ()
                self.label_domain_loss_weight = opts['labels_domain']['loss_weight']            
                label_check_exprs = [];
                for key, value in self.label_domain_value.items():
                    label_exprs = ['ak.to_numpy(%s)' % k for k in value]
                    label_check_exprs.append(label_exprs);                    
                    self.var_funcs[key] = 'np.argmax(np.stack([%s], axis=1), axis=1)' % (','.join(label_exprs))            
                    self.labelcheck_domain_names += (key.replace('label','labelcheck'),);
                    self.var_funcs[key.replace('label','labelcheck')] = 'np.sum(np.stack([%s], axis=1), axis=1)' % (','.join(label_exprs))            
                self.var_funcs['_labelcheck_domain_'] = 'np.sum(np.stack([%s], axis=1), axis=1)' % (','.join(','.join(value) for value in label_check_exprs))                
        else:
            self.label_domain_names = tuple();
            self.label_domain_type  = None;
            self.label_domain_value = None;
            self.labelcheck_domain_names = None;
            self.label_domain_loss_weight = None;
                
        # targets
        if opts['targets']:
            self.target_type = opts['targets']['type']
            self.target_value = opts['targets']['value']
            if 'quantile' in opts['targets']:
                self.target_quantile = opts['targets']['quantile']
            else:
                self.target_quantile = None;
            self.target_names = tuple(self.target_value.keys())
            self.var_funcs.update(self.target_value)
        else:
            self.target_names = tuple();
            self.target_type  = None;
            self.target_value = None;
            self.target_quantile = None;

        self.basewgt_name = '_basewgt_'
        self.weight_name = None        
        if opts['weights'] is not None:
            self.weight_name = 'weight_'
            self.use_precomputed_weights = opts['weights']['use_precomputed_weights']
            if self.use_precomputed_weights:
                self.var_funcs[self.weight_name] = '*'.join(opts['weights']['weight_branches'])
            else:
                ## re-weight
                self.reweight_method = opts['weights']['reweight_method']
                self.reweight_basewgt = opts['weights'].get('reweight_basewgt', None)
                if self.reweight_basewgt:
                    self.var_funcs[self.basewgt_name] = self.reweight_basewgt
                self.reweight_branches = tuple(opts['weights']['reweight_vars'].keys())
                self.reweight_bins = tuple(opts['weights']['reweight_vars'].values())
                self.reweight_classes = tuple(opts['weights']['reweight_classes'])
                self.class_weights = opts['weights'].get('class_weights', None)
                if self.class_weights is None:
                    self.class_weights = np.ones(len(self.reweight_classes))                    
                self.reweight_threshold = opts['weights'].get('reweight_threshold', 10)
                self.reweight_discard_under_overflow = opts['weights'].get('reweight_discard_under_overflow', True)
                self.reweight_hists = opts['weights'].get('reweight_hists', None)
                if self.reweight_hists is not None:
                    for k, v in self.reweight_hists.items():
                        self.reweight_hists[k] = np.array(v, dtype='float32')
            ## domain part
            if 'domain_classes' in opts['weights']:
                self.domain_classes = tuple(opts['weights']['domain_classes'])
                self.domain_weights = opts['weights'].get('domain_weights', None)
                if self.domain_weights is None:
                    self.domain_weights = np.ones(len(self.domain_classes))
            else:
                self.domain_classes = None
                self.domain_weights = None

        # observers
        self.observer_names = tuple(opts['observers'])
        # monitor variables
        self.monitor_variables = tuple(opts['monitor_variables'])
        # Z variables: returned as `Z` in the dataloader (use monitor_variables for training, observers for eval)
        self.z_variables = self.observer_names if len(self.observer_names) > 0 else self.monitor_variables
        # remove self mapping from var_funcs
        for k, v in self.var_funcs.items():
            if k == v:
                del self.var_funcs[k]

        if print_info:
            def _log(msg, *args, **kwargs):
                _logger.info(msg, *args, color='lightgray', **kwargs)
            _log('preprocess config: %s', str(self.preprocess))
            _log('selection: %s', str(self.selection))
            _log('test_time_selection: %s', str(self.test_time_selection))
            _log('var_funcs:\n - %s', '\n - '.join(str(it) for it in self.var_funcs.items()))
            _log('input_names: %s', str(self.input_names))
            _log('input_dicts:\n - %s', '\n - '.join(str(it) for it in self.input_dicts.items()))
            _log('input_shapes:\n - %s', '\n - '.join(str(it) for it in self.input_shapes.items()))
            _log('preprocess_params:\n - %s', '\n - '.join(str(it) for it in self.preprocess_params.items()))
            if self.label_names: 
                _log('label_names: %s', str(self.label_names))
            if self.target_names: 
                _log('target_names: %s', str(self.target_names))
            if self.target_quantile:
                _log('target_quantile: %s',' '.join([str(elem) for elem in self.target_quantile])) 
            if self.label_domain_names: 
                _log('label_domain_names: %s', str(self.label_domain_names))
            if self.label_domain_loss_weight:
                _log('self.label_domain_loss_weight: %s',' '.join([str(elem) for elem in self.label_domain_loss_weight]))
            _log('observer_names: %s', str(self.observer_names))
            _log('monitor_variables: %s', str(self.monitor_variables))
            if opts['weights'] is not None:
                if self.use_precomputed_weights:
                    _log('weight: %s' % self.var_funcs[self.weight_name])
                else:
                    for k in ['reweight_method', 'reweight_basewgt', 'reweight_branches', 'reweight_bins',
                              'reweight_classes', 'class_weights', 'reweight_threshold',  'reweight_discard_under_overflow',
                              'domain_classes', 'domain_weights']:
                        _log('%s: %s' % (k, getattr(self, k)))

        # parse config
        self.keep_branches = set()
        aux_branches = set()
        # selection
        if self.selection:
            aux_branches.update(_get_variable_names(self.selection))
        # test time selection
        if self.test_time_selection:
            aux_branches.update(_get_variable_names(self.test_time_selection))
        # var_funcs
        self.keep_branches.update(self.var_funcs.keys())
        for expr in self.var_funcs.values():
            aux_branches.update(_get_variable_names(expr))
        # inputs
        for names in self.input_dicts.values():
            self.keep_branches.update(names)
        # labels
        self.keep_branches.update(self.label_names)
        # targets
        self.keep_branches.update(self.target_names)
        # labels
        self.keep_branches.update(self.label_domain_names)
        # weight
        if self.weight_name:
            self.keep_branches.add(self.weight_name)
            if not self.use_precomputed_weights:
                aux_branches.update(self.reweight_branches)
                aux_branches.update(self.reweight_classes)
        # observers
        self.keep_branches.update(self.observer_names)
        # monitor variables
        self.keep_branches.update(self.monitor_variables)
        # keep and drop
        self.drop_branches = (aux_branches - self.keep_branches)
        self.load_branches = (aux_branches | self.keep_branches) - set(self.var_funcs.keys()) - {self.weight_name, }
        if print_info:
            _logger.debug('drop_branches:\n  %s', ','.join(self.drop_branches))
            _logger.debug('load_branches:\n  %s', ','.join(self.load_branches))

    def __getattr__(self, name):
        return self.options[name]

    def dump(self, fp):
        with open(fp, 'w') as f:
            yaml.safe_dump(self.options, f, sort_keys=False)

    @classmethod
    def load(cls, fp, load_observers=True, load_reweight_info=True, extra_selection=None, extra_test_selection=None):
        with open(fp) as f:
            _opts = yaml.safe_load(f)
            options = copy.deepcopy(_opts)
        if not load_observers:
            options['observers'] = None
        if not load_reweight_info:
             options['weights'] = None
        if extra_selection:
            options['selection'] = '(%s) & (%s)' % (_opts['selection'], extra_selection)
        if extra_test_selection:
            if 'test_time_selection' not in options or options['test_time_selection'] is None:
                options['test_time_selection'] = '(%s) & (%s)' % (_opts['selection'], extra_test_selection)
            else:
                options['test_time_selection'] = '(%s) & (%s)' % (_opts['test_time_selection'], extra_test_selection)
        return cls(**options)

    def copy(self):
        return self.__class__(print_info=False, **copy.deepcopy(self.options))

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy()

    def export_json(self, fp):
        import json
        ## class+reg+domain        
        if self.target_names and self.label_names and self.label_domain_names:
            j = {'output_names':self.label_value+list(self.target_value.keys()), 'input_names':self.input_names}
        ## class+reg
        elif self.target_names and self.label_names and not self.label_domain_names:
            j = {'output_names':self.label_value+list(self.target_value.keys()), 'input_names':self.input_names}
        ## class
        elif not self.target_names and not self.label_domain_names and self.label_names:
            j = {'output_names':self.label_value, 'input_names':self.input_names}
        ## regression
        elif self.target_names and not self.label_names and not self.label_domain_names:
            j = {'output_names':list(self.target_value.keys()), 'input_names':self.input_names}

        for k, v in self.input_dicts.items():
            j[k] = {'var_names': v, 'var_infos': {}}
            for var_name in v:
                j[k]['var_length'] = self.preprocess_params[var_name]['length']
                info = self.preprocess_params[var_name]
                j[k]['var_infos'][var_name] = {
                    'median': 0 if info['center'] is None else info['center'],
                    'norm_factor': info['scale'],
                    'replace_inf_value': 0,
                    'lower_bound': -1e32 if info['center'] is None else info['min'],
                    'upper_bound': 1e32 if info['center'] is None else info['max'],
                    'pad': info['pad_value']
                }
        with open(fp, 'w') as f:
            json.dump(j, f, indent=2)
