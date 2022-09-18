"""clamp functions used for clipping quantization methods"""
import torch

clamp_factor = {
    "normal":{
        "laplace": {
            2: 2.83,
            3: 3.89,
            4: 5.03,
            5: 6.20,
            6: 7.41,
            8: 9.90,
            16: 20.27
        },
        "gaussian": {
            2: 0.4308,
            3: 0.4400,
            4: 0.4414,
            5: 0.4417,
            6: 0.4420,
            8: 0.4421,
            16:0.4421
        }
    },
    "GeLU":{
        "laplace": {
            2: 3.897,
            3: 5.029,
            4: 6.205,
            5: 7.41,
            6: 8.646,
            8: 11.163,
            16: 21.59
        },
        "gaussian": {
            2: 0.4308,
            3: 0.4400,
            4: 0.4414,
            5: 0.4417,
            6: 0.4420,
            8: 0.4421,
            16:0.4421
        }
    },
    "good":{
        "laplace": {
            2: 3.897,
            3: 5.029,
            4: 6.205,
            5: 7.41,
            6: 8.646,
            8: 11.163 
        },
        "gaussian": {
            2: 0.4308,
            3: 0.4400,
            4: 0.4414,
            5: 0.4417,
            6: 0.4420,
            8: 0.4421,
            16:0.4421
        }
    }
}


def clamp(input, bit, layer_type='normal', mode='laplace'):
    """clamp a input tensor"""
    if layer_type == 'normal':
        variance = torch.var(input, unbiased = False) #, mean = torch.var_mean(input, unbiased = False)
    elif layer_type == 'GeLU':
        # Special case for GeLU layer
        # Distribution after GeLU only has half of bell curve
        # Assuming mean = 0, and ignore the influence of negtive small values
        variance = torch.pow(input, 2).sum()/torch.numel(input)
    elif layer_type == 'good':
        variance = 2*(torch.pow(input, 2).sum()/torch.numel(input))

    if mode == 'laplace':
            dist_parameter = torch.sqrt(0.5*variance)
    else:
        # TODO
        dist_parameter = variance
    optimal_clamp_range = clamp_factor[layer_type][mode][bit] * dist_parameter
    # clamp
    result = torch.where(torch.abs(input)<optimal_clamp_range, input, optimal_clamp_range)

    return result, optimal_clamp_range

if __name__ == '__main__':
    input = torch.randn(3,4)*10
    print(input)
    print(clamp(input, 2))