"""clamp functions used for clipping quantization methods"""
import torch

_CLAMP_FACTOR_LAPLACE = {
    2: 2.83,
    3: 3.89,
    4: 5.03,
    5: 6.20,
    6: 7.41,
    8: 9.90,
    16: 20.27
}

_CLAMP_FACTOR_GELU = {
    2: 3.897,
    3: 5.029,
    4: 6.205,
    5: 7.41,
    6: 8.646,
    8: 11.163,
    16: 21.59
}

PLOT_BINS = 200

def normalize(a, axis=-1, order=1):
    l2 = torch.atleast_1d(torch.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / torch.unsqueeze(l2, axis)

def estimate_laplace(x):
    return torch.abs(x).sum()/x.nelement()

def Laplace(x, mu, beta):
    return torch.exp(-torch.abs(x-mu)/beta)/(2*beta)

def mse(ori_x, x, is_norm=False):
    mse = torch.square(ori_x - x).mean()
    if is_norm:
        mse = mse/((ori_x)**2).mean()
    return mse

def fitting_with_search(norm_hist_x, theo_xaxi, fitting_parameter, search_steps=100):
    fitting_curve = Laplace(theo_xaxi, 0.0, fitting_parameter)
    norm_fitting_curve = normalize(fitting_curve)
    search_direction = -1.0 if norm_hist_x.max() > norm_fitting_curve.max() else 1.0
    
    b_real = 1/(2*norm_hist_x.max())
    b_fitting = 1/(2*norm_fitting_curve.max())
    step_len = torch.abs(b_real - b_fitting) * fitting_parameter / (search_steps * b_fitting)
    best_parameter, best_mse = 0.0, 100000.0
    for i in range(search_steps+1):
        temp_parameter = fitting_parameter + i * step_len * search_direction
        fitting_hist = normalize(Laplace(theo_xaxi, 0.0, temp_parameter))
        temp_mse = mse(norm_hist_x, fitting_hist)
        if best_mse > temp_mse:
            best_mse = temp_mse
            best_parameter = temp_parameter
    
    return best_parameter


def clamp_banner2019_gelu(tensor: torch.Tensor, bit: int) -> torch.Tensor:
    """Like `clamp_banner2019_laplace` but modified for a GeLU layer output."""
    # Special case for GeLU layer
    # Distribution after GeLU only has half of bell curve
    # Assuming mean = 0, and ignore the influence of negtive small values
    variance = 2* torch.pow(tensor, 2).sum()/torch.numel(tensor)
    dist_parameter = torch.sqrt(0.5*variance)
    optimal_clamp_range = _CLAMP_FACTOR_GELU[bit] * dist_parameter
    result = tensor.clamp(min = -optimal_clamp_range, max = optimal_clamp_range)
    return result


def clamp_banner2019_laplace(tensor: torch.Tensor, bit: int) -> torch.Tensor:
    """Clamp tensor with a Laplace distribution - based on Banner et. al.'s NIPS 2019 paper."""
    # "Post training 4-bit quantization of convolutional networks for rapid-deployment"
    variance = torch.var(tensor, unbiased = False)
    dist_parameter = torch.sqrt(0.5*variance)
    optimal_clamp_range = _CLAMP_FACTOR_LAPLACE[bit] * dist_parameter
    result = tensor.clamp(min = -optimal_clamp_range, max = optimal_clamp_range)
    return result

def clamp_wang2022_DSACIQ(tensor: torch.Tensor, bit: int) -> torch.Tensor:
    """Clamp tensor with directed search method - based on our paper."""
    # gather info. of input tensor
    hist_x, bins_x = torch.histogram(tensor, bins = PLOT_BINS)
    theo_xaxi = torch.linspace(bins_x.min(), bins_x.max(), PLOT_BINS)
    norm_hist_x = torch.flatten(normalize(hist_x))
    fitting_parameter = estimate_laplace(tensor)
    best_parameter = fitting_with_search(norm_hist_x, theo_xaxi, fitting_parameter)
    optimal_clamp_range = _CLAMP_FACTOR_LAPLACE[bit] * best_parameter
    result = tensor.clamp(min = -optimal_clamp_range, max = optimal_clamp_range)
    return result