import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from transformers import DeiTFeatureExtractor, ViTFeatureExtractor
sys.path.append('..')
import model_cfg
from evaluation import ViTFeatureExtractorTransforms, _make_shard
from runtime import _get_default_quant
from edgepipe.quantization.basic_op import _quant_op
from edgepipe.quantization.clamp_op import clamp


def clamp_wrap(x, skip, layer_id, shard, model_name = 'google/vit-base-patch16-224'):
    if isinstance(x, np.ndarray):
        x = torch.Tensor(x)
    if isinstance(skip, np.ndarray):
        skip = torch.Tensor(skip)
    if model_name == 'google/vit-base-patch16-224':
        half_point = 19
    # x,skip
    clamp_types=[] 
    if layer_id % 4 in [1]:
        if layer_id < half_point:
            clamp_types = ['normal', 'normal'] # ['good', 'good']
        else:
            clamp_types = ['normal', 'normal'] # ['good', 'normal']
    elif layer_id % 4 in [2,0]:
        if layer_id < half_point:
            clamp_types = ['normal', 'normal'] # ['good', 'good']
        else:
            clamp_types = ['normal', 'normal'] # ['normal', 'normal']
    else:
        if layer_id <= half_point:
            clamp_types = ['GeLU', 'good']
        else:
            clamp_types = ['GeLU', 'good'] # ['GeLU', 'normal']

    # do clamp
    temp_x, clamp_value_x = clamp(x, shard.quant_bits[1].item(), layer_type=clamp_types[0])
    if skip is not None:
        temp_skip, clamp_value_skip = clamp(skip, shard.quant_bits[1].item(), layer_type=clamp_types[1])

    if isinstance(x, torch.Tensor):
        x = x.numpy()
    if isinstance(skip, torch.Tensor):
        skip = skip.numpy()
    return (temp_x, temp_skip), (clamp_value_x, clamp_value_skip)    
        

def mse(ori_x,x, is_norm=False):
    if isinstance(ori_x, torch.Tensor):
        ori_x = ori_x.numpy()
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    mse = (np.square(ori_x-x)).mean()
    if is_norm:
        mse = mse/((ori_x)**2).mean()
    return mse

def quant_tensor(input_data, quant_bit):
    """
        The input to the encoder should be a torch.Tensor
        We first cast it to a np.array, then do everything else
    """
    if quant_bit == 0:
        return input_data
    
    if isinstance(input_data, torch.Tensor):
        input_data = input_data.numpy()
    shape = input_data.shape
    # ensure the input is scaled to [0,1],
    shift = input_data.min()
    input_data = input_data - shift
    scale_factor = input_data.max()
    rescale_input = input_data/scale_factor
    # quant
    quant_result, _ = _quant_op(rescale_input, quant_bit)
    quant_result = quant_result*scale_factor + shift

    # return a np.array
    return quant_result


def run_one_iteration(args, input_tensor, model_shards, parts, is_quant=True, is_plot=True, save_tensor=False,  bins=100, save_dir = './dist_imgs'):
    num_shards = len(model_shards)
    temp_tensor = input_tensor
    is_gelu = parts[1]%4==3
    MSE_X, MSE_SKIP = [], []

    for idx in range(num_shards):
        shard = model_shards[idx]
        quant_bit = shard.quant_bits.tolist()[1]
        file_name = 'pt_' + str(parts[1]) + '_q_' + str(quant_bit)

        # forward
        if idx != num_shards-1:
            x, skip = shard(temp_tensor)

            if isinstance(x, torch.Tensor):
                x = x.numpy()
            if isinstance(skip, torch.Tensor):
                skip = skip.numpy()
            ori_x = np.copy(x)
            ori_skip = np.copy(skip)

            _, (clamp_value_x, clamp_value_skip) = clamp_wrap(ori_x, ori_skip, parts[1], shard)
            if is_plot:
                if is_gelu:
                    fig = plt.figure(figsize=(6,7))
                    plt_cfg = 230
                    # split x into - and +
                    x_pos, x_neg = [],[]
                    for num in np.nditer(x):
                        if num > 0.0:
                            x_pos.append(num)
                        elif num < 0.0:
                            x_neg.append(num)
                        else:
                            x_pos.append(num)
                            x_neg.append(num)
                    x_pos = np.array(x_pos)
                    x_neg = np.array(x_neg)
                    # plot x- , x+ , and skip
                    plt.subplot(plt_cfg+1)
                    hist_x, bins_x, _ = plt.hist(x_pos, bins=bins)
                    plt.vlines(clamp_value_x, 0, hist_x.max(), linestyles ="dashed", colors ="r")
                    plt.title('origin x(+)')

                    plt.subplot(plt_cfg+2)
                    plt.hist(x_neg, bins=bins)
                    plt.title('origin x(-)')
                    
                    plt.subplot(plt_cfg+3)
                    hist_skip, bins_skip, _ = plt.hist(skip.flatten(), bins=bins)
                    plt.vlines(clamp_value_skip, 0, hist_skip.max(), linestyles ="dashed", colors ="r")
                    plt.vlines(-clamp_value_skip, 0, hist_skip.max(), linestyles ="dashed", colors ="r")
                    plt.title('origin skip')
                else:
                    fig = plt.figure(figsize=(9,7))
                    plt_cfg = 220
                    # plot the original x and skip
                    plt.subplot(plt_cfg+1)
                    hist_x, bins_x, _ = plt.hist(x.flatten(), bins=bins, log=True)
                    plt.vlines(clamp_value_x, 0, hist_x.max(), linestyles ="dashed", colors ="r")
                    plt.vlines(-clamp_value_x, 0, hist_x.max(), linestyles ="dashed", colors ="r")
                    plt.title('origin x')

                    plt.subplot(plt_cfg+2)
                    hist_skip, bins_skip, _ = plt.hist(skip.flatten(), bins=bins, log=True)
                    plt.vlines(clamp_value_skip, 0, hist_skip.max(), linestyles ="dashed", colors ="r")
                    plt.vlines(-clamp_value_skip, 0, hist_skip.max(), linestyles ="dashed", colors ="r")
                    plt.title('origin skip')

            # save tensors
            if save_tensor:
                np.save(file_name+'_x', x)
                np.save(file_name+'_skip', skip)

            # quant
            if is_quant:
                # clamp 
                (x,skip), _ = clamp_wrap(x, skip, parts[1], shard)
                print(f"clamp value for x is: {clamp_value_x}")
                print(f"clamp value for skip is: {clamp_value_skip}")
                x = quant_tensor(x, quant_bit)
                if skip is not None:
                    skip = quant_tensor(skip, quant_bit)
            
            # save quantized tensors
            if save_tensor:
                np.save(file_name+'_x_q', x)
                np.save(file_name+'_skip_q', skip)

            # histogram of quant
            if is_plot:
                if is_gelu:
                    # split x into - and +
                    x_pos, x_neg = [],[]
                    for num in np.nditer(x):
                        if num > 0.0:
                            x_pos.append(num)
                        elif num < 0.0:
                            x_neg.append(num)
                        else:
                            x_pos.append(num)
                            x_neg.append(num)
                    x_pos = np.array(x_pos)
                    x_neg = np.array(x_neg)
                    # plot x- , x+ , and skip
                    plt.subplot(plt_cfg+4)
                    hist_x_pos, _, _ = plt.hist(x_pos, bins=bins)
                    plt.vlines(clamp_value_x, 0, hist_x_pos.max(), linestyles ="dashed", colors ="r")
                    plt.plot()
                    plt.title('quantized x(+)')
                    plt.subplot(plt_cfg+5)
                    plt.hist(x_neg, bins=bins)
                    plt.title('quantized x(-)')
                    plt.subplot(plt_cfg+6)
                    plt.hist(skip.flatten(), bins=bins)
                    plt.vlines(clamp_value_skip, 0, skip.max(), linestyles ="dashed", colors ="r")
                    plt.title('quantized skip')
                else:
                    # plot the original x and skip
                    plt.subplot(plt_cfg+3)
                    hist_x, bins_x, _ = plt.hist(x.flatten(), bins=bins, log=True)
                    plt.vlines(clamp_value_x, 0, hist_x.max(), linestyles ="dashed", colors ="r")
                    plt.vlines(-clamp_value_x, 0, hist_x.max(), linestyles ="dashed", colors ="r")
                    plt.title(f'quantized x, MSE={mse(ori_x, x, is_norm=True)}')
                    plt.subplot(plt_cfg+4)
                    hist_skip, bins_skip, _ = plt.hist(skip.flatten(), bins=bins, log=True)
                    plt.vlines(clamp_value_skip, 0, hist_skip.max(), linestyles ="dashed", colors ="r")
                    plt.vlines(-clamp_value_skip, 0, hist_skip.max(), linestyles ="dashed", colors ="r")
                    plt.title(f'quantized skip, MSE={mse(ori_skip, skip)}')

                if args.show_fig:
                    plt.show()
                plt.subplots_adjust(hspace=0.5)
                plt.suptitle(file_name, fontsize=20)
                plt.savefig(os.path.join(save_dir, file_name), dpi=300)

            # compute MSE after quant
            if is_quant:
                MSE_X.append(mse(ori_x, x, is_norm=True))
                MSE_SKIP.append(mse(ori_skip, skip))

            # restore as tensor
            x = torch.from_numpy(x)
            if skip is not None:
                skip = torch.from_numpy(skip)
            temp_tensor = (x,skip)
        else:
            x = shard(temp_tensor)
            skip = None

    result = x
    return result, [MSE_X, MSE_SKIP]

def main(args):
    """ main func"""
    # localize parameters
    imagenet_root = args.imagenet_root
    batch_size = args.batch_size
    num_workers = args.num_workers
    partition = args.partition
    quant = args.quant
    output_dir = args.output_dir
    model_name = args.model_name
    model_file = args.model_file
    if model_file is None:
        model_file = os.path.join('..', model_cfg.get_model_default_weights_file(model_name))
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
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with torch.no_grad():
        mse_output_list = []
        mse_x_list = []
        mse_skip_list = []

        for batch_idx, (input, target) in enumerate(val_loader):
            if batch_idx == args.image_id:
            
                if args.baseline:
                    output,_ = run_one_iteration(args, input, model_shards, parts, is_quant=False, is_plot=False)
                    _, pred = output.topk(1)
                    pred = pred.t()
                    # if pred == target:
                    #     ori_pred_correct = 1
                    #     print("prediction correct.")
                    # else:
                    #     ori_pred_correct = 0
                    #     print("prediction wrong!")
                output_q, quant_mse_lists = run_one_iteration(args, input, model_shards, parts, is_quant=True, is_plot=args.plot, save_tensor=args.save_tensor, save_dir = output_dir)
                _, pred_q = output_q.topk(1)
                pred_q = pred_q.t()
                if pred_q == target:
                    q_pred_correct = 1
                    print("prediction after quant correct.")
                else:
                    q_pred_correct = 0
                    print("prediction after quant wrong!")

                if args.baseline and args.view_mse:
                    # save results
                    mse_output_list.append(mse(output, output_q))
                    mse_x_list.append(*quant_mse_lists[0])
                    mse_skip_list.append(*quant_mse_lists[1])
                    print(f"MSE of Output for image {batch_idx}: {mse_output_list[-1]}")
                    # line = str(stage_quant[0])+"\t "\
                    #         + str(parts[1])+"\t "\
                    #         + str(*quant_mse_lists[0])+"\t "\
                    #         + str(*quant_mse_lists[1])+"\t "\
                    #         + str(mse(output, output_q))+"\t "\
                    #         + str(ori_pred_correct)+"\t "\
                    #         + str(q_pred_correct)+"\n"
                    # with open("q_vs_pt_mse.txt", 'a') as f:
                    #     f.write(line)

        if args.view_mse:
            avg_mse_output = np.array(mse_output_list).mean()
            avg_mse_x = np.array(mse_x_list).mean()
            avg_mse_skip = np.array(mse_skip_list).mean()
            with open("mse_output_avg_on_dataset.txt", 'a+') as f:
                # write_line = str(stage_quant[0])+"\t " + str(parts[1])+"\t " + str(avg_mse_output)+"\t \n"
                write_line = str(stage_quant[0])+"\t "\
                        + str(parts[1])+"\t "\
                        + str(avg_mse_x)+"\t "\
                        + str(avg_mse_skip)+"\t "\
                        + str(avg_mse_output)+"\n"
                f.write(write_line)

    return

if __name__ == "__main__":
    """Main function."""
    #########################################################
    #                 Check Enviroment Settings             #
    #########################################################
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
    parser.add_argument("-i", "--image-id", type=int,
                        default=0,
                        help="iterate untill the specified id of the image in ImageNet")
    parser.add_argument("--batch-size", default=1, type=int, help="batch size")
    parser.add_argument("-q", "--quant", type=str,
                        help="comma-delimited list of quantization bits to use after each stage")
    parser.add_argument("-pt", "--partition", type=str, default= '1,43,44,48',
                        help="comma-delimited list of start/end layer pairs, e.g.: '1,24,25,48'; "
                             "single-node default: all layers in the model")
    parser.add_argument("-o", "--output-dir", type=str, default="/home1/haonanwa/projects/EdgePipe/results/dist_imgs_log")
    parser.add_argument("-b", "--baseline", action='store_true', help="run baseline model without quantization to check and compare output")
    parser.add_argument("-p", "--plot", action='store_true', help="run baseline model without quantization to check and compare output")
    parser.add_argument("--save-tensor", action='store_true', help="store tensor of partition boundary to npy file")
    parser.add_argument("--view-mse", action='store_true', help="view the average mse of the output before and after quant")
    parser.add_argument("--show-fig", action='store_true', help="view the plot of the distribution")
    args = parser.parse_args()

    main(args)