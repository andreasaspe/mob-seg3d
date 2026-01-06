
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from model import UnetWithUncertainty, get_model_from_base_kwargs, LinearTauScheduler, ConstantScheduler, PerVoxelCovarianceHead
import os
from torch.amp import autocast, GradScaler
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'nnunet'))
from utils import get_data_loader
import warnings
import pickle
import numpy as np
from argparse import ArgumentParser, Namespace
import ast
from dataloaders import HackyEvalLoader

import ast
import copy

warnings.filterwarnings('ignore', category=UserWarning)

torch.set_num_interop_threads(1)
def dice_score_binary(prediction, target, smooth = 1e-6, reduction = 'None'):

    prediction = prediction.argmax(1)
    intersection = torch.sum(prediction * target)
    out = (2. * intersection + smooth) / (torch.sum(target) + torch.sum(prediction) + smooth)
    return out.detach().cpu()

def dice_score(prediction, target, num_classes, smooth=1e-6):
    """
    prediction: [B, C, H, W] logits or probabilities
    target:     [B, H, W] class indices
    """
    pred_classes = prediction.argmax(1)  # [B, H, W]
    dice_per_class = []

    for c in range(num_classes):
        pred_c = (pred_classes == c).float()
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()

        dice_c = (2 * intersection + smooth) / (union + smooth)
        dice_per_class.append(dice_c)

    return torch.mean(torch.stack(dice_per_class))  # or mean over classes if needed

class Logger:
    def __init__(self, write_path, experiment_name = ""):

        self.write_path = write_path
        os.makedirs(write_path, exist_ok=True)

        self.save_path = os.path.join(self.write_path, f"{experiment_name}_results.pkl")
        self.latest_key = None

    @staticmethod
    def handle_results_dtype(results):
        new_results = {}
        for key, val in results.items():
            if isinstance(val, torch.Tensor):
                val = val.detach().cpu().numpy()
            
            if isinstance(val, np.ndarray):
                if np.prod(val.shape) == 1:
                    val = val.item()

            new_results[key] = val 

        return new_results    

    def write(self, results: dict, key: str):
        
        
        if os.path.isfile(self.save_path):
            results_pcl = pickle.load(open(self.save_path, 'rb'))
        else:
            results_pcl = {}
        
        results = self.handle_results_dtype(results)
        if key not in results_pcl:
            results_pcl[key] = []
        results_pcl[key].append(results)

        with open(self.save_path, 'wb') as handle:
            pickle.dump(results_pcl, handle, protocol = pickle.HIGHEST_PROTOCOL)
        
        print("Wrote results to", self.save_path)

class PerformanceKeeper:
    def __init__(self, performances):
        self.performances = performances
    
    def __add__(self, other):
        new_performances = {key: val + other.performances[key] for key, val in self.performances.items()}
        return PerformanceKeeper(new_performances)
    
    def __sub__(self, other):
        new_performances = {key: val - other.performances[key] for key, val in self.performances.items()}
        return PerformanceKeeper(new_performances)
    
    def __mul__(self, scalar):
        new_performances = {key: val * scalar for key, val in self.performances.items()}
        return PerformanceKeeper(new_performances)

    def __div__(self, scalar):
        new_performances = {key: val / scalar for key, val in self.performances.items()}
        return PerformanceKeeper(new_performances)

    def __truediv__(self, scalar):
        new_performances = {key: val /scalar for key, val in self.performances.items()}
        return PerformanceKeeper(new_performances)

    def __repr__(self):
        return repr(self.performances)
    
    def __str__(self):
        return str(self.performances)
    

class PerformanceHolder:
    def __init__(self,):

        self.performances = {}

    def update(self, key, performance):
        self.performances[key] = PerformanceKeeper(performance)
    
    def to_list(self, keys = None):
        keys = self.performances.keys() if keys is None else keys
        values = [self.performances[key] for key in keys]
        return values
        
    def mean(self, keys = None):
        values = self.to_list(keys = keys)
        mean = values[0]
        for elem in values[1:]:
            mean = mean + elem
        return mean * 1/len(values)
    
    def mean_with_other(self, other):
        combined_keys = set(self.performances.keys()).intersection(set(other.performances.keys()))
        return self.mean(keys = combined_keys)
    

        
class Trainer(object):
    def __init__(self, model_kwargs, training_kwargs):

        self.model_kwargs = model_kwargs
        self.training_kwargs = training_kwargs
        self.model_kwargs['loss_kwargs'] = training_kwargs['loss_kwargs']
        self.get_eval_loader_from_saved = training_kwargs.get('eval_loader_data_path', "")
        self.get_test_loader_from_saved = training_kwargs.get('test_loader_data_path', "")

        self.recreate_dataset = training_kwargs.pop('recreate_dataset', False)
        self.dataset_name_or_id = self.model_kwargs.pop('dataset_name_or_id')
        self.train_loader, self.eval_loader, self.num_batches_train, self.num_batches_eval = (
            None, None, 100, 100
        )

        self.tau_scheduler = None
        if training_kwargs.get('tau_schedule', False):
            self.tau_scheduler = None
        
        self.output_dir = training_kwargs['output_dir']
        print(self.output_dir)
        self.prepare_output_dir()

        self.model, self.new_modules = self.get_model(**model_kwargs)
        self.optimizer = self.setup_optimizer()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = None
        self.train_loader, self.eval_loader, self.test_loader = self.setup_data()
        self.num_iterations_per_epoch = training_kwargs.get('num_iterations_per_epoch', 250)
        self.num_val_iterations = training_kwargs.get('num_val_iterations', 50)
        self.last_saved_model_path = ""
        
        self.basis_model_results_keeper = None
        self.trained_models_results_keeper = None
        self.last_performance_diff = None
        self.gradient_accumulation = self.training_kwargs.get('gradient_accumulation', False)

        if self.test_loader is None:
            print('Test loader is None, switching to validation loader for final evals')
        
            self.test_loader = self.eval_loader
        

    def setup_optimizer(self,):
        """
        Can add potential functionality for different learning rates here
        :return:
        """
        lr = self.training_kwargs['lr']
        weight_decay = self.training_kwargs['weight_decay']
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        return optimizer

    def setup_data(self, ):

        dataloaders = get_data_loader(dataset_name_or_id = self.dataset_name_or_id)
        self.test_loader = None
        if len(dataloaders) == 3:
            self.train_loader, self.eval_loader, self.test_loader = dataloaders
        else:
            self.train_loader, self.eval_loader = dataloaders

        if self.get_eval_loader_from_saved:
            self.eval_loader = HackyEvalLoader(self.get_eval_loader_from_saved,self.eval_loader, recreate=self.recreate_dataset)
            
        if self.get_test_loader_from_saved:
            self.test_loader = HackyEvalLoader(self.get_test_loader_from_saved, self.test_loader, recreate=self.recreate_dataset)
        
        return self.train_loader, self.eval_loader, self.test_loader

    def prepare_output_dir(self, ):
        os.makedirs(self.output_dir, exist_ok=True)

    def get_model(self, **model_kwargs):
        checkpoint_path = model_kwargs.pop('checkpoint_path', "")
        
        model = get_model_from_base_kwargs(**model_kwargs)
        unmatched_modules = None
        
        if not checkpoint_path:
            print("Starting from untrained model")
        else:
            print("Starting from checkpoint {}".format(checkpoint_path))
            own_state_dict = model.state_dict()
            pretrained_state_dict = torch.load(checkpoint_path, weights_only=False)
            
            if 'network_weights' in pretrained_state_dict:
                pretrained_state_dict = pretrained_state_dict['network_weights']

            if any('decoder.encoder' in key for key in own_state_dict.keys()):
                new_state_dict = {}
                for key in pretrained_state_dict.keys():
                    new_key = key
                    if 'encoder' in key:
                        new_key = key.replace('encoder', 'decoder.encoder')
                    new_state_dict[new_key] = pretrained_state_dict[key]
                
            else:
                new_state_dict = pretrained_state_dict
            
            unmatched_modules = model.load_state_dict(pretrained_state_dict, strict=False)
        
        print(unmatched_modules)
        return model, unmatched_modules

    def train_iter(self, input_volume, target):

        input_volume = input_volume.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True).long()
        self.optimizer.zero_grad(set_to_none=True)

        #with autocast(device_type='cuda'):
        scaler = None
        if self.gradient_accumulation:
            scaler = self.scaler
        
        output = self.model(input_volume, targets=target, scaler = scaler)
        
        if scaler is None:
            self.scaler.scale(output.loss).backward()
        
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return output

    def handle_various_input(self,elem):
        
        input_volume, target = elem['data'], elem['target']
        
        target = target[0].squeeze(1)
        if isinstance(input_volume, list):
            if not all(vol.shape[0] == 1 for vol in input_volume):
                input_volume = [vol.unsqueeze(0) for vol in input_volume]
            input_volume = torch.cat(input_volume, dim = 0)
        
        if isinstance(target, list):
            if not all(tar.shape[0] == 1 for tar in target):
                target = [tar.unsqueeze(0) for tar in target]
            target = torch.cat(target, dim = 0)
        
        return input_volume, target


    def train_one_epoch(self, epoch, last_performance = None):

        self.model.train()
        self.scaler = GradScaler(device='cuda')

        running_metrics = {}
    
        train_loader_bar = tqdm(range(self.num_iterations_per_epoch), desc = f'Training epoch {epoch}, performance diff: {self.last_performance_diff}')
        for step in train_loader_bar:
            elem = next(self.train_loader)
            input_volume, target = self.handle_various_input(elem)
            output = self.train_iter(input_volume, target)

            for key, val in output.loss_decomp.items():
                if key not in running_metrics:
                    running_metrics[key] = []
                running_metrics[key].append(val)

            train_loader_bar.set_postfix({key: f"{sum(val)/len(val):0.3f}" for key, val in running_metrics.items()})
        
        print({key: f"{sum(val)/len(val):0.3f}" for key, val in running_metrics.items()})
    
    @torch.no_grad()
    def run_evaluation_new(self, eval_loader = None, basis_model_only = False, pass_to_keeper = True):

        if eval_loader is None:
            eval_loader = self.eval_loader

        if not isinstance(eval_loader, HackyEvalLoader):
            return self.run_evaluation(eval_loader=eval_loader, basis_model_only=basis_model_only)
    
        self.model.eval()
        
        num_eval_steps = len(eval_loader)
        eval_loader_bar = tqdm(desc = 'Running evaluation', total = num_eval_steps)
        performance_holder = PerformanceHolder()

        if basis_model_only:
            self.model.basis_model_only()
        
        ce_forward = nn.CrossEntropyLoss()
        
        if hasattr(self.model.decoder, 'cov_weighting_head'):
            self.model.decoder.cov_weighting_head.track_chosen_indices(True)
            self.model.decoder.cov_weighting_head.hard = False
        
        results = {'dice_score': [], 'NLL':[]}
        for i in range(num_eval_steps):
            elem = eval_loader[i]
            input_volume, target = self.handle_various_input(elem)
            input_volume = input_volume.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True).long()
            output = self.model(input_volume, target, reduction = 'mean')

            if i == 0:
                for loss_type in output.loss_decomp.keys():
                    results[loss_type] = []
            
            for loss_type, val in output.loss_decomp.items():
                    results[loss_type].append(val)
            
            dice = dice_score(output.mu, target, num_classes=output.mu.shape[1])
            results['dice_score'].append(dice)
            results['NLL'].append(self.negative_log_likelihood(output.mu, target))
            performance = {key: val[-1] for key, val in results.items()}
            performance_holder.update(elem['keys'][0].item(), performance)
            eval_loader_bar.update(1)
        
        metrics = {key: sum(val) / len(val) for key, val in results.items()}
        
        if hasattr(self.model.decoder, 'cov_weighting_head'):
            if hasattr(self.model.decoder.cov_weighting_head, 'basis_counts'):
                metrics['basis_indices'] = self.model.decoder.cov_weighting_head.basis_counts
                
            else:
                metrics['basis_indices'] = {
                    val: self.model.decoder.cov_weighting_head.chosen_indices.count(val) for val in np.unique(
                        self.model.decoder.cov_weighting_head.chosen_indices
                    )
                }
            
            self.model.decoder.cov_weighting_head.hard = False            
            self.model.decoder.cov_weighting_head.track_chosen_indices(False)
        
            print(metrics['basis_indices'])

        if pass_to_keeper:
            if basis_model_only:
                self.basis_model_results_keeper = performance_holder
            else:
                self.trained_models_results_keeper = performance_holder
        
        self.model.basis_model_only(False)
        return metrics, performance_holder.mean(), performance_holder
    
    @staticmethod
    def negative_log_likelihood(probs, target):
        target_expanded = target.unsqueeze(1)  # [B, 1, H, W, D]

        # Step 2: gather probs for the correct class
        # gather along class dim (dim=1)
        true_probs = probs.gather(1, target_expanded)  # [B, 1, H, W, D]

        # Step 3: remove the singleton class dimension
        true_probs = true_probs.squeeze(1)  # [B, H, W, D]

        # Step 4: compute NLL
        nll = -torch.log(true_probs + 1e-8).mean()
        return nll.item()

    @torch.no_grad()
    def run_evaluation(self, eval_loader = None, basis_model_only = False):

        if eval_loader is None:
            eval_loader = self.eval_loader

        max_iter = 1000
        self.model.eval()
        num_eval_steps = len(eval_loader.generator._data.identifiers)
        eval_loader_bar = tqdm(desc = 'Running evaluation', total = len(eval_loader.generator._data.identifiers))
        performance_holder = PerformanceHolder()
        
        results = {'dice_score': [], 'cross_entropy_mean': []}
        ce_forward = nn.CrossEntropyLoss()
        if basis_model_only:
            self.model.basis_model_only()
        
        has_been_identified = {key: False for key in eval_loader.generator._data.identifiers}
        wrong_hits_counter = 0
        step = 0
        while step < max_iter:
            elem = next(eval_loader)
            if has_been_identified[elem['keys'][0].item()]:
                wrong_hits_counter += 1
                continue
        
            input_volume, target = self.handle_various_input(elem)
            input_volume = input_volume.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True).long()
            output = self.model(input_volume, target, reduction = 'mean')
            
            if step == 0:
                for loss_type in output.loss_decomp.keys():
                    results[loss_type] = []
            for loss_type, val in output.loss_decomp.items():
                    results[loss_type].append(val)
            
            
            dice = dice_score(output.mu, target, num_classes=output.mu.shape[1])
            results['dice_score'].append(dice)
            results['NLL'].append(self.negative_log_likelihood(output.mu, target))
            performance = {key: val[-1] for key, val in results.items()}
            performance_holder.update(elem['keys'][0].item(), performance)
            has_been_identified[elem['keys'][0].item()] = True

            eval_loader_bar.update(1)
            step += 1
            if all(has_been_identified.values()):
                break
        
        print("Number of wrong hits by this stupid code was:", wrong_hits_counter)
        assert len(set(performance_holder.performances.keys()) - set(has_been_identified.keys())) == 0
        
        metrics = {key: sum(val) / len(val) for key, val in results.items()}

        if basis_model_only:
            self.basis_model_results_keeper = performance_holder
        else:
            self.trained_models_results_keeper = performance_holder
        self.model.basis_model_only(False)
        return metrics, performance_holder.mean()


    def save_model(self, epoch, experiment_name = ""):

        model_save_path = os.path.join(self.output_dir, 'exp_{}_model_epoch_{}.pth'.format(experiment_name, epoch))
        torch.save(self.model.state_dict(), model_save_path)
        print("Saved model to", os.path.join(self.output_dir, 'exp_{}_model_epoch_{}.pth'.format(experiment_name, epoch)))

        if self.last_saved_model_path:
            os.remove(self.last_saved_model_path)
        self.last_saved_model_path = model_save_path

    def finalize(self):

        self.train_loader.__del__()

        if not isinstance(self.eval_loader, HackyEvalLoader):
            self.eval_loader.__del__()


    def train(self, experiment_name = ""):

        logger = Logger(self.output_dir, experiment_name=experiment_name)

        test_logger = Logger(self.output_dir, experiment_name=f"{experiment_name}_test")

        performance = {'dice_score': -1}
        best_performance = 0
        self.model.to(self.device)
    
        basis_metrics, mean_perf, _ = self.run_evaluation_new(basis_model_only=True)

        if self.test_loader is not None:
            basis_metrics_test, mean_perf_test, basis_performance_holder_test = self.run_evaluation_new(
            eval_loader=self.test_loader, basis_model_only=True, pass_to_keeper=False
            )

        print(basis_metrics)
        print(mean_perf)
        print("#"*20)
        print(basis_metrics_test)
        print(mean_perf_test)

        logger.write(basis_metrics, key = 'init_metrics')
        logger.write(self.training_kwargs, 'training_kwargs')

        for epoch in range(self.training_kwargs['num_epochs']):
            self.train_one_epoch(epoch, last_performance = performance)
            performance, mean_perf,_ = self.run_evaluation_new()
            
            basis_model_performance = self.basis_model_results_keeper.mean_with_other(
                self.trained_models_results_keeper
            )
            trained_model_performance = self.trained_models_results_keeper.mean_with_other(
                self.basis_model_results_keeper
            )

            performance_diff = trained_model_performance - basis_model_performance
            self.last_performance_diff = performance_diff.performances
        
            if performance['dice_score'].item() > best_performance:
                self.save_model(epoch, experiment_name=experiment_name)
                best_performance = performance['dice_score'].item()
                test_performance, test_mean_performance, test_performance_holder = self.run_evaluation_new(eval_loader=self.test_loader, pass_to_keeper=False)

                basis_model_test_performance = basis_performance_holder_test.mean_with_other(test_performance_holder)
                trained_model_test_performance = test_performance_holder.mean_with_other(basis_performance_holder_test)
                performance_diff_test = trained_model_test_performance - basis_model_test_performance
                test_logger.write(performance_diff_test.performances, key = 'performance_diff')
                test_logger.write(test_performance, key = 'training_performance')
                print('performance diff test:', performance_diff_test)

            logger.write(performance_diff.performances, key = 'performance_diff')
            logger.write(performance, key = 'training_performance')
        

def run_weighting_grid_search(args):
    model_kwargs = {
        'checkpoint_path': 'path/to/checkpoint/',
        'loss_kwargs': dict(),
        'path_to_base': '/path/to/info_dict<dataset_id>.pkl'    }

    training_kwargs = {
        'num_epochs': 10,
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'output_dir': '/path/to/ndseg_output',
        'num_iterations_per_epoch': 250, 
        'num_val_iterations': 130,
    }

    base_output_dir = '/path/to/ndseg_output/grid_search'
    ce_values = [0.9, 1.0, 1.1]
    dice_values = [0.2, 0.5]
    kl_values = [1e-5, 1e-4, 1e-3]

    for ce in ce_values:
        for di in dice_values:
            for kl in kl_values:
                loss_kwargs = {
                        'lambda_ce':ce,
                        'lambda_dice':di,
                        'lambda_nll': 1.0,
                        'lambda_kl': kl
                    }

                training_kwargs['loss_kwargs'] = loss_kwargs
                experiment_name = 'grid_search_loss'
                outdir = os.path.join(base_output_dir, f"ce_{ce}_dice_{di}_kl_{kl}")
                training_kwargs['output_dir'] = outdir
                run_experiment(model_kwargs=model_kwargs, training_kwargs=training_kwargs, experiment_name=experiment_name)


def run_experiment(model_kwargs, training_kwargs, experiment_name, num_runs = 1):

    import copy
    for i in range(num_runs):
        model_kwargs_, training_kwargs_ = copy.deepcopy(model_kwargs), copy.deepcopy(training_kwargs)
        trainer = Trainer(model_kwargs=model_kwargs_, training_kwargs=training_kwargs_)
        trainer.train(experiment_name=f"{experiment_name}_run_{i}")
        trainer.finalize()

def get_trainer(model_kwargs, training_kwargs):
    trainer = Trainer(model_kwargs=model_kwargs, training_kwargs=training_kwargs)
    return trainer




DATASET_TO_PATHS = {
    'pancreas': {'checkpoint_path': '/scratch/awias/data/nnUNet_mob-seg3d/nnUNet_results/Dataset001_TotalSegmentatorPancreas/nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans__3d_fullres/fold_0/checkpoint_final.pth',
                 'path_to_base': '/home/pjtka/ndsegment/mob-seg3d/info_dict_1.pkl',
                 'dataset_name_or_id': '001'},
    'gallbladder': {'checkpoint_path': 'path/to/checkpoint/',
                 'path_to_base': '/path/to/info_dict<dataset_id>.pkl',
                 'dataset_name_or_id': '<dataset_id>'},
    'duodenum': {'checkpoint_path': 'path/to/checkpoint/',
                 'path_to_base': '/path/to/info_dict<dataset_id>.pkl', 
                 'dataset_name_or_id': '<dataset_id>'},
    'adrenal_left': {'checkpoint_path': 'path/to/checkpoint/',
                 'path_to_base': '/path/to/info_dict<dataset_id>.pkl', 
                 'dataset_name_or_id': '<dataset_id>'},
    
    'combined': {'checkpoint_path': 'path/to/checkpoint/',
                 'path_to_base': '/path/to/info_dict<dataset_id>.pkl', 
                 'dataset_name_or_id': '<dataset_id>'}
    }


DATASET_TO_SAVE_PATHS = {
    'pancreas': {'eval_loader_data_path': '/scratch/pjtka/mob-segref-test/val',
                 'test_loader_data_path': ''},
    'gallbladder': {'eval_loader_data_path': '/path/to/gallbladder_validation',
                    'test_loader_data_path': '/path/to/gallbladder_test'},

    'duodenum': {'eval_loader_data_path': '/path/to/duodenum_validation',
                   'test_loader_data_path': '/path/to/duodenum_test'},
    
    'adrenal_left': {'eval_loader_data_path': '/path/to/adrenal_left_validation',
                   'test_loader_data_path': '/path/to/duodenum_left_test'},
    'combined': {'eval_loader_data_path':  '/path/to/combined_validation',
                 'test_loader_data_path': '/path/to/combined_test'}
}


def get_model_kwargs_from_dataset(dataset, current_kwargs):
    
    from pprint import pprint
    dataset_specifics = DATASET_TO_PATHS[dataset]
    for key, val in dataset_specifics.items():
        current_kwargs[key] = val 
    
    print("Using the following paths")
    pprint(dataset_specifics)

    return current_kwargs

def get_training_kwargs_from_dataset(dataset, current_kwargs):

    from pprint import pprint
    dataset_specifics = DATASET_TO_SAVE_PATHS[dataset]
    for key, val in dataset_specifics.items():
        current_kwargs[key] = val 
    
    print("Using the following paths")
    pprint(dataset_specifics)

    return current_kwargs


def run_ppt(args):
    model_kwargs = {
        'checkpoint_path': 'path/to/checkpoint/',
        'loss_kwargs': {
                        'lambda_ce':1.0,
                        'lambda_dice':1.0,
                        'lambda_nll': 1.0,
                        'lambda_kl': 1e-4
                    },
        'path_to_base': '/path/to/info_dict<dataset_id>.pkl',
        'num_samples_train': 5,
        'num_samples_inference': 30,
        'sample_type': 'torch'
    }

    model_kwargs = get_model_kwargs_from_dataset(args.dataset, model_kwargs)
    training_kwargs = {
        'num_epochs': 20,
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'output_dir': args.outdir,
        'num_iterations_per_epoch': 250,
        'num_val_iterations': 5,
        'loss_kwargs': {
                        'lambda_ce':1.0,
                        'lambda_dice':1.0,
                        'lambda_nll': 1.0,
                        'lambda_kl': 1e-4
                    },
        
        'eval_loader_data_path': '/path/to/pancreas_validation',
        'test_loader_data_path': '/path/to/pancreas_test',
        'recreate_dataset': args.recreate
    }

    training_kwargs = get_training_kwargs_from_dataset(args.dataset, training_kwargs)

    run_experiment(model_kwargs=model_kwargs, training_kwargs=training_kwargs, experiment_name=args.exp_name, num_runs = args.num_runs)
    

def run_diag(args):
    model_kwargs = {
        'checkpoint_path': 'path/to/checkpoint/',
        'loss_kwargs': {
                        'lambda_ce':1.0,
                        'lambda_dice':1.0,
                        'lambda_nll': 1.0,
                        'lambda_kl': 1e-4
                    },
        'path_to_base': '/path/to/info_dict<dataset_id>.pkl',
        'num_samples_train': 5,
        'num_samples_inference': 30,
        'sample_type': 'diagonal'
    }
    model_kwargs = get_model_kwargs_from_dataset(args.dataset, model_kwargs)
    
    training_kwargs = {
        'num_epochs': 20,
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'output_dir': args.outdir,
        'num_iterations_per_epoch': 250,
        'num_val_iterations': 5,
        'loss_kwargs': {
                        'lambda_ce':1.0,
                        'lambda_dice':1.0,
                        'lambda_nll': 1.0,
                        'lambda_kl': 1e-4
                    },
        
        'eval_loader_data_path': '/path/to/pancreas_validation',
        'test_loader_data_path': '/path/to/pancreas_test',
        'recreate_dataset': args.recreate
    }

    training_kwargs = get_training_kwargs_from_dataset(args.dataset, training_kwargs)
    run_experiment(model_kwargs=model_kwargs, training_kwargs=training_kwargs, experiment_name=args.exp_name, num_runs = args.num_runs)
    


def run_basic(args):
    model_kwargs = {
        'checkpoint_path': 'path/to/checkpoint/',
        'loss_kwargs': {
                        'lambda_ce':1.0,
                        'lambda_dice':1.0,
                        'lambda_nll': 1.0,
                        'lambda_kl': 1e-4
                    },
        'path_to_base': '/path/to/info_dict<dataset_id>.pkl',
        'num_samples_train': 1,
        'num_samples_inference': 30,
        'cov_weighting_kwargs': {
            'sample_type': 'ours' if not args.proper else 'basic_proper'
        }
    }

    model_kwargs = {
        'checkpoint_path': 'path/to/checkpoint/',
        'loss_kwargs': {
                        'lambda_ce':1.0,
                        'lambda_dice':1.0,
                        'lambda_nll': 1.0,
                        'lambda_kl': 1e-4
                    },
        'path_to_base': '/path/to/info_dict<dataset_id>.pkl',
        'num_samples_train': 1,
        'num_samples_inference': 30,
        'cov_weighting_kwargs': {
        }
    }

    model_kwargs = get_model_kwargs_from_dataset(args.dataset, model_kwargs)

    training_kwargs = {
        'num_epochs': 20,
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'output_dir': args.outdir,
        'num_iterations_per_epoch': 250,
        'num_val_iterations': 5,
        'loss_kwargs': {
                        'lambda_ce':1.0,
                        'lambda_dice':1.0,
                        'lambda_nll': 1.0,
                        'lambda_kl': 1e-4
                    },
        
        'eval_loader_data_path': '/path/to/gallbladder_validation',
        'test_loader_data_path': '/path/to/gallbladder_test',
        'recreate_dataset': args.recreate
    }
    
    training_kwargs = get_training_kwargs_from_dataset(args.dataset, training_kwargs)

    run_experiment(model_kwargs=model_kwargs, training_kwargs=training_kwargs, experiment_name=args.exp_name, num_runs = args.num_runs)




def run_multiple_bases(args):

    from model import PartionedCovHead
    model_kwargs = {
        'checkpoint_path': 'path/to/checkpoint/',
        'loss_kwargs': {
                        'lambda_ce':1.0,
                        'lambda_dice':1.0,
                        'lambda_nll': 1.0,
                        'lambda_kl': 5*1e-4
                    },
        'path_to_base': '/path/to/info_dict<dataset_id>.pkl',
        'model_type': 'weighted_basis',
        'cov_weighting_kwargs': {
            'num_bases': 3,
            'sample_type': 'partitioned' if not args.proper else 'partitioned_proper',
            'class': PartionedCovHead
        },
        'num_samples_train': 5,
        'num_samples_inference': 30
    }

    model_kwargs = get_model_kwargs_from_dataset(args.dataset, model_kwargs)

    training_kwargs = {
        'num_epochs': 20,
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'output_dir': args.outdir,
        'num_iterations_per_epoch': 250,
        'num_val_iterations': 5,
        'loss_kwargs': {
                        'lambda_ce': 1.0,
                        'lambda_dice': 1.0,
                        'lambda_nll': 1.0,
                        'lambda_kl': 1e-4
                    },

        'tau_schedule': {'type': 'constant', 'max': 2, 'min': 0.1},

        'eval_loader_data_path': '/path/to/pancreas_validation',
        'test_loader_data_path': '/path/to/pancreas_test',
        'gradient_accumulation': False,
        'recreate_dataset': args.recreate
    }

    training_kwargs = get_training_kwargs_from_dataset(args.dataset, training_kwargs)
    run_experiment(model_kwargs=model_kwargs, training_kwargs=training_kwargs, experiment_name=args.exp_name, num_runs = args.num_runs)


def run_multiple_bases_multiple_partitions(args):
    from model import PartionedCovHead

    base_args = copy.deepcopy(args)
    for num_bases in [2,4,5]:
        args = copy.deepcopy(base_args)
        args.exp_name = f"{args.exp_name}_num_bases_{num_bases}"        
        model_kwargs = {
            'checkpoint_path': 'path/to/checkpoint/',
            'loss_kwargs': {
                            'lambda_ce':1.0,
                            'lambda_dice':1.0,
                            'lambda_nll': 1.0,
                            'lambda_kl': 5*1e-4
                        },
            'path_to_base': '/path/to/info_dict<dataset_id>.pkl',
            'model_type': 'weighted_basis',
            'cov_weighting_kwargs': {
                'num_bases': num_bases,
                'sample_type': 'partitioned' if not args.proper else 'partitioned_proper',
                'class': PartionedCovHead
            },
            'num_samples_train': 5,
            'num_samples_inference': 30
        }

        model_kwargs = get_model_kwargs_from_dataset(args.dataset, model_kwargs)

        training_kwargs = {
            'num_epochs': 20,
            'lr': 1e-4,
            'weight_decay': 1e-4,
            'output_dir': args.outdir,
            'num_iterations_per_epoch': 250,
            'num_val_iterations': 5,
            'loss_kwargs': {
                            'lambda_ce': 1.0,
                            'lambda_dice': 1.0,
                            'lambda_nll': 1.0,
                            'lambda_kl': 1e-4
                        },

            'eval_loader_data_path': '/path/to/pancreas_validation',
            'test_loader_data_path': '/path/to/pancreas_test',
            'gradient_accumulation': False,
            'recreate_dataset': args.recreate
        }

        training_kwargs = get_training_kwargs_from_dataset(args.dataset, training_kwargs)
        run_experiment(model_kwargs=model_kwargs, training_kwargs=training_kwargs, experiment_name=args.exp_name, num_runs = args.num_runs)

def run_per_voxel_multi_basis(args):
    model_kwargs = {
        'checkpoint_path': 'path/to/checkpoint/',
        'loss_kwargs': {
                        'lambda_ce':1.0,
                        'lambda_dice':1.0,
                        'lambda_nll': 1.0,
                        'lambda_kl': 5*1e-4,
                        'lambda_weight': 0.0
                    },
        'path_to_base': '/path/to/info_dict<dataset_id>.pkl',
        'model_type': 'weighted_basis',
        'cov_weighting_kwargs': {
            'num_bases': 3,
            'sample_type': 'per_voxel_multi_basis',
            'class': PerVoxelCovarianceHead
        },
        'num_samples_train': 5,
        'num_samples_inference': 30
    }

    model_kwargs = get_model_kwargs_from_dataset(args.dataset, model_kwargs)

    training_kwargs = {
        'num_epochs': 20,
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'output_dir': args.outdir,
        'num_iterations_per_epoch': 250,
        'num_val_iterations': 5,
        'loss_kwargs': {
                        'lambda_ce': 1.0,
                        'lambda_dice': 1.0,
                        'lambda_nll': 1.0,
                        'lambda_kl': 1e-4
                    },

        'tau_schedule': {'type': 'constant', 'max': 2, 'min': 0.1},

        'eval_loader_data_path': '/path/to/pancreas_validation',
        'test_loader_data_path': '/path/to/pancreas_test',
        'gradient_accumulation': False,
        'recreate_dataset': args.recreate
    }

    training_kwargs = get_training_kwargs_from_dataset(args.dataset, training_kwargs)
    run_experiment(model_kwargs=model_kwargs, training_kwargs=training_kwargs, experiment_name=args.exp_name, num_runs = args.num_runs)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--exp_type', type = str, default='basic', nargs="+")
    parser.add_argument('--exp_name', type = str, default='basic', nargs="+")
    parser.add_argument('--num_runs', type = int, default=1)
    parser.add_argument('--outdir', type = str, default='/path/to/ndseg_output', nargs="+")
    parser.add_argument('--recreate', default=False, type = ast.literal_eval)
    parser.add_argument('--dataset', type = str, default='gallbladder')
    parser.add_argument('--proper', type = ast.literal_eval, default=False)

    args = parser.parse_args()

    print(args)

    exp_type_to_func = {
        'basic': run_basic,
        'grid': run_weighting_grid_search,
        'multi_basis': run_multiple_bases,
        'ppt': run_ppt,
        'diag': run_diag,
        'num_bases': run_multiple_bases_multiple_partitions,
        'per_voxel': run_per_voxel_multi_basis
    }
    assert len(args.exp_type) == len(args.exp_name), 'We require new name for each experiment type otherwise it will overwrite'

    if not isinstance(args.outdir, list):
        args.outdir = [args.outdir]

    
    args.recreate = [args.recreate] + [False] * (len(args.exp_type) -1)
    
    if len(args.outdir) == 1:
        args.outdir = args.outdir * len(args.exp_type)

    iterator = zip(args.exp_type, args.exp_name, args.outdir, args.recreate)
    
    for exp_type, exp_name, outdir, recreate in iterator:
        run_args = Namespace(**{'exp_type': exp_type, 'exp_name': exp_name, 'outdir': outdir, 'num_runs': args.num_runs, 'dataset': args.dataset, 'recreate': recreate, 'proper': args.proper})
        exp_type_to_func[exp_type](run_args)    
    
        
        













