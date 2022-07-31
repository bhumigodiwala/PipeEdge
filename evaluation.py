""" Evaluate accuracy on ImageNet dataset of PipeEdge """
import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Normalize, Resize, Compose
from transformers import DeiTFeatureExtractor, ViTFeatureExtractor
from pipeedge.quantization.hook import forward_hook_quant_encode_with_clamp, forward_pre_hook_quant_decode
from pipeedge.quantization.clamp_op import clamp
from runtime import _get_default_quant
import model_cfg

class ReportAccuracy():
    def __init__(self, batch_size, output_dir, model_name, partition, quant) -> None:
        self.current_acc = 0.0
        self.total_acc = 0.0
        self.correct = 0
        self.tested_batch = 0
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.partition = partition
        self.quant = quant
        self.model_name = model_name.split('/')[1]

    def update(self, pred, target):
        self.correct = pred.eq(target.view(1, -1).expand_as(pred)).float().sum()
        self.current_acc = self.correct / self.batch_size
        self.total_acc = (self.total_acc * self.tested_batch + self.current_acc)/(self.tested_batch+1)
        self.tested_batch += 1

    def report(self,):
        print(f"The accuracy so far is: {100*self.total_acc:.2f}")
        file_name = os.path.join(self.output_dir, self.model_name, "result_"+self.partition+"_"+str(self.quant)+".txt")
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, 'a') as f:
            f.write(f"{100*self.total_acc:.2f}\n")

class ViTFeatureExtractorTransforms:
    def __init__(self, feature_extractor):
        transform = []

        if feature_extractor.do_resize:
            transform.append(Resize([feature_extractor.size, feature_extractor.size]))

        transform.append(ToTensor())

        if feature_extractor.do_normalize:
            transform.append(Normalize(feature_extractor.image_mean, feature_extractor.image_std))

        self.transform = Compose(transform)

    def __call__(self, x):
        return self.transform(x)

def _make_shard(model_name, model_file, stage_layers, stage, q_bits):
    shard = model_cfg.module_shard_factory(model_name, model_file, stage_layers[stage][0],
                                            stage_layers[stage][1], stage)
    shard.register_buffer('quant_bits', q_bits)
    shard.eval()
    return shard

def _forward_model(input_tensor, model_shards, stage_layers, is_clamp=True, model_name='google/vit-base-patch16-224'):
    num_shards = len(model_shards)
    temp_tensor = input_tensor
    for idx in range(num_shards):
        shard = model_shards[idx]

        # decoder
        if idx != 0:
            temp_tensor = forward_pre_hook_quant_decode(shard, temp_tensor)

        # forward
        if isinstance(temp_tensor[0], tuple):
            temp_tensor = temp_tensor[0]
        temp_tensor = shard(temp_tensor)

        # encoder
        if idx != num_shards-1:
            # defination of position where distribution diverse
            if model_name == 'google/vit-base-patch16-224':
                half_point = 19
            # clamp
            if is_clamp:
                # x,skip
                clamp_types=[]
                if stage_layers[idx][1] % 4 in [1]:
                    if stage_layers[idx][1] < half_point:
                        clamp_types = ['normal', 'normal'] # manully selected ['good', 'good']
                    else:
                        clamp_types = ['normal', 'normal'] # manully selected ['good', 'normal']
                elif stage_layers[idx][1] % 4 in [2,0]:
                    if stage_layers[idx][1] < half_point:
                        clamp_types = ['normal', 'normal'] # manully selected ['good', 'good']
                    else:
                        clamp_types = ['normal', 'normal']
                else:
                    if stage_layers[idx][1] <= half_point:
                        clamp_types = ['good', 'normal'] # manully selected ['GeLU', 'good']
                    else:
                        clamp_types = ['good', 'normal'] # manully selected ['GeLU', 'normal']
                # do clamp
                temp_tensors_list = []
                temp_clamp_list = []
                if len(temp_tensor) == 2:
                    for tensor_idx, tensor in enumerate(temp_tensor):
                        result_tensor, clamp_value = clamp(tensor, shard.quant_bits[1].item(), layer_type=clamp_types[tensor_idx])
                        temp_tensors_list.append(result_tensor)
                        temp_clamp_list.append(clamp_value)
                    temp_tensor = tuple(temp_tensors_list)
                elif len(temp_tensor) == 1:
                    temp_tensor, clamp_value = clamp(tensor, shard.quant_bits[1].item(), layer_type=clamp_types[0])
                    temp_clamp_list.append(clamp_value)
            # do encoder
            temp_tensor = (forward_hook_quant_encode_with_clamp(shard, None, temp_tensor, temp_clamp_list),)
    return temp_tensor

def evaluation(args):
    """ Evaluation main func"""
    # localize parameters
    imagenet_root = args.imagenet_root
    batch_size = args.batch_size
    num_workers = args.num_workers
    partition = args.partition
    quant = args.quant
    output_dir = args.output_dir
    model_name = args.model_name
    model_file = args.model_file
    is_clamp = args.clamp
    if model_file is None:
        model_file = model_cfg.get_model_default_weights_file(model_name)

    # load dataset
    if model_name in ['facebook/deit-base-distilled-patch16-224',
                        'facebook/deit-small-distilled-patch16-224',
                        'facebook/deit-tiny-distilled-patch16-224']:
        feature_extractor = DeiTFeatureExtractor.from_pretrained(model_name)
    else:
        feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

    val_transform = ViTFeatureExtractorTransforms(feature_extractor)
    val_dataset = ImageFolder(os.path.join(imagenet_root, 'val'),
                            transform = val_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle=False,
        pin_memory=True
    )

    # model config
    parts = [int(i) for i in partition.split(',')]
    num_shards = len(parts)//2
    stage_layers = [(parts[i], parts[i+1]) for i in range(0, len(parts), 2)]
    if quant:
        stage_quant = [int(i) for i in quant.split(',')]
    else:
        stage_quant = _get_default_quant(len(stage_layers))

    # model construct
    model_shards = []
    q_bits = []
    for stage in range(num_shards):
        q_bits = torch.tensor((0 if stage == 0 else stage_quant[stage - 1], stage_quant[stage]))
        model_shards.append(_make_shard(model_name, model_file, stage_layers, stage, q_bits))

    # run inference
    acc_reporter = ReportAccuracy(batch_size, output_dir, model_name, partition, stage_quant[0])
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(val_loader):
            output = _forward_model(input, model_shards, stage_layers, is_clamp)
            _, pred = output.topk(1)
            pred = pred.t()
            acc_reporter.update(pred, target)
            acc_reporter.report()
    print(f"Final Accuracy: {100*acc_reporter.total_acc}; Quant Bitwidth: {stage_quant}")


if __name__ == "__main__":
    """Main function."""
    parser = argparse.ArgumentParser(description="Pipeline Parallelism Runtime",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Model options
    parser.add_argument("-m", "--model-name", type=str, default="google/vit-base-patch16-224",
                        choices=model_cfg.get_model_names(),
                        help="the neural network model for loading")
    parser.add_argument("-M", "--model-file", type=str,
                        help="the model file, if not in working directory")
    # Runtime configs
    parser.add_argument("-n", "--num-workers", default=4, type=int,
                        help="the number of worker threads for the dataloder")
    parser.add_argument("-I", "--imagenet-root", type=str,
                        default="/project/jpwalter_148/hnwang/datasets/ImageNet/",
                        help="the root directory of the imagenet")
    parser.add_argument("-b", "--batch-size", default=1, type=int, help="batch size")
    parser.add_argument("-q", "--quant", type=str,
                        help="comma-delimited list of quantization bits to use after each stage")
    parser.add_argument("-pt", "--partition", type=str, default= '1,43,44,48',
                        help="comma-delimited list of start/end layer pairs, e.g.: '1,24,25,48'; "
                             "single-node default: all layers in the model")
    parser.add_argument("-o", "--output-dir", type=str, default="/home1/haonanwa/projects/PipeEdge/results")
    parser.add_argument("--clamp", action="store_true", default=False, help="whether do clamp for quantization")
    args = parser.parse_args()

    evaluation(args)
