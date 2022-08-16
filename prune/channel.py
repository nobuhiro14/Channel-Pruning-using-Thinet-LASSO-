import math
import torch
import numpy as np
from sklearn.linear_model import Lasso
from scipy.spatial import distance

num_pruned_tolerate_coeff = 1.1


def channel_selection(inputs, module, sparsity=0.5, method='greedy'):
    """
    Among the input channels of the current module, select the most important channel.
    Find the input channel that can most closely match the existing output.
    :param inputs: torch.Tensor, input features map
    :param module: torch.nn.module, layer
    :param sparsity: float, 0 ~ 1 how many prune channel of output of this layer
    :param method: str, how to select the channel
    :return:
        list of int, indices of channel to be selected and pruned
    """
    num_channel = inputs.size(1)  # number of channels
    num_pruned = int(math.ceil(num_channel * sparsity))  # Number of channels to be deleted according to the input sparsity
    if num_pruned %4 != 0:
        gap = num_pruned%4 
        num_pruned -= gap 
    num_stayed = num_channel - num_pruned

    print('num_pruned', num_pruned)
    if method == 'greedy':
        indices_pruned = []
        while len(indices_pruned) < num_pruned:
            min_diff = 1e10
            min_idx = 0
            for idx in range(num_channel):
                if idx in indices_pruned:
                    continue
                indices_try = indices_pruned + [idx]
                inputs_try = torch.zeros_like(inputs)
                inputs_try[:, indices_try, ...] = inputs[:, indices_try, ...]
                output_try = module(inputs_try)
                output_try_norm = output_try.norm(2)
                if output_try_norm < min_diff:
                    min_diff = output_try_norm
                    min_idx = idx
            indices_pruned.append(min_idx)

        print('indices_pruned !!! ', indices_pruned)

        indices_stayed = list(set([i for i in range(num_channel)]) - set(indices_pruned))

    elif method == 'greedy_GM':
        indices_stayed = []
        while len(indices_stayed) < num_stayed:
            max_farthest_channel_norm = 1e-10
            farthest_channel_idx = 0

            for idx in range(num_channel):
                if idx in indices_stayed:
                    continue
                indices_try = indices_stayed + [idx]
                inputs_try = torch.zeros_like(inputs)
                inputs_try[:, indices_try, ...] = inputs[:, indices_try, ...]
                output_try = module(inputs_try).view(num_channel,-1).cpu().detach().numpy()
                similar_matrix = distance.cdist(output_try, output_try,'euclidean')
                similar_sum = np.sum(np.abs(similar_matrix), axis=0)
                similar_large_index = similar_sum.argsort()[-1]
                farthest_channel_norm= np.linalg.norm(similar_sum[similar_large_index])

                if max_farthest_channel_norm < farthest_channel_norm :
                    max_farthest_channel_norm = farthest_channel_norm
                    farthest_channel_idx = idx

            print(farthest_channel_idx)
            indices_stayed.append(farthest_channel_idx)

        print('indices_stayed !!! ', indices_stayed)

        indices_pruned = list(set([i for i in range(num_channel)]) - set(indices_stayed))

    elif method == 'lasso':
        y = module(inputs)

        if module.bias is not None:  # bias.shape = [N]
            bias_size = [1] * y.dim()  # bias_size: [1, 1, 1, 1]
            bias_size[1] = -1  # [1, -1, 1, 1]
            bias = module.bias.view(bias_size)  # bias.view([1, -1, 1, 1] = [1, N, 1, 1])
            y -= bias  # Subtract the amount of bias from the output feature (y - b)
        else:
            bias = 0.
        y = y.view(-1).data.cpu().numpy()  # flatten all of outputs
        y_channel_spread = []
        for i in range(num_channel):
            x_channel_i = torch.zeros_like(inputs)
            x_channel_i[:, i, ...] = inputs[:, i, ...]
            y_channel_i = module(x_channel_i) - bias
            y_channel_spread.append(y_channel_i.data.view(-1, 1))
        y_channel_spread = torch.cat(y_channel_spread, dim=1).cpu()

        alpha = 1e-7
        solver = Lasso(alpha=alpha, warm_start=True, selection='random', random_state=0)

        # choice_idx = np.random.choice(y_channel_spread.size()[0], 2000, replace=False)
        # selected_y_channel_spread = y_channel_spread[choice_idx, :]
        # new_output = y[choice_idx]
        #
        # del y_channel_spread, y

        # Gradually increase the alpha value until the desired number of channels are deleted.
        alpha_l, alpha_r = 0, alpha
        num_pruned_try = 0
        while num_pruned_try < num_pruned:
            alpha_r *= 2
            solver.alpha = alpha_r
            # solver.fit(selected_y_channel_spread, new_output)
            solver.fit(y_channel_spread,y)
            num_pruned_try = sum(solver.coef_ == 0)

        # After finding an alpha that is sufficiently pruned, 
        # the left and right sides of the alpha value are narrowed 
        # to find a more accurate alpha value.
        num_pruned_max = int(num_pruned)
        while True:
            alpha = (alpha_l + alpha_r) / 2
            solver.alpha = alpha
            # solver.fit(selected_y_channel_spread, new_output)
            solver.fit(y_channel_spread,y)
            num_pruned_try = sum(solver.coef_ == 0)

            if num_pruned_try > num_pruned_max:
                alpha_r = alpha
            elif num_pruned_try < num_pruned:
                alpha_l = alpha
            else:
                break

        # Finally, convert lasso coeff to index
        indices_stayed = np.where(solver.coef_ != 0)[0].tolist()
        indices_pruned = np.where(solver.coef_ == 0)[0].tolist()

    else:
        raise NotImplementedError

    inputs = inputs.cuda()
    module = module.cuda()

    return indices_stayed, indices_pruned  # Returns the index of the selected channel


def module_surgery(module ,attached_module,next_module, indices_stayed):
    """
    Select less important filter
    :param module: torch.nn.module, module of the Conv layer (be pruned for filters)
    :param attached_module: torch.nn.module, series of modules following the this layer (like BN)
    :param next_module: torch.nn.module, module of the next layer (be pruned for channels)
    :param indices_stayed: list of int, indices of channels and corresponding filters to be pruned
    :return:
        void
    """

    num_channels_stayed = len(indices_stayed)

    if module is not None:
        if isinstance(module, torch.nn.Conv2d):
            module.out_channels = num_channels_stayed
        elif isinstance(module, torch.nn.Linear):
            module.out_features = num_channels_stayed
        else:
            raise NotImplementedError

        # redesign module structure (delete filters)
        new_weight = module.weight[indices_stayed, ...].clone()
        del module.weight
        module.weight = torch.nn.Parameter(new_weight)
        if module.bias is not None:
            new_bias = module.bias[indices_stayed, ...].clone()
            del module.bias
            module.bias = torch.nn.Parameter(new_bias)

    # redesign BN module
    if attached_module is not None:
        if isinstance(attached_module, torch.nn.modules.BatchNorm2d):
            attached_module.num_features = num_channels_stayed
            running_mean = attached_module.running_mean[indices_stayed, ...].clone()
            running_var = attached_module.running_var[indices_stayed, ...].clone()
            new_weight = attached_module.weight[indices_stayed, ...].clone()
            new_bias = attached_module.bias[indices_stayed, ...].clone()
            del attached_module.running_mean, attached_module.running_var, attached_module.weight, attached_module.bias
            attached_module.running_mean, attached_module.running_var = running_mean, running_var
            attached_module.weight, attached_module.bias = torch.nn.Parameter(new_weight), torch.nn.Parameter(new_bias)

    # redesign next module structure (modify input channels)
    if next_module is not None:
        if isinstance(next_module, torch.nn.Conv2d):
            next_module.in_channels = num_channels_stayed
        elif isinstance(next_module, torch.nn.Linear):
            next_module.in_features = num_channels_stayed
        new_weight = next_module.weight[:, indices_stayed, ...].clone()
        del next_module.weight
        next_module.weight = torch.nn.Parameter(new_weight)


def weight_reconstruction(module, inputs, outputs, use_gpu=False):
    """
    After pruning of the previous layer is performed, the weight of the next layer is a
    djusted using the pruned output of the current layer.
    The least square is solved by setting the output of the pruned layer to X and 
    the output value of the next layer (in this case, the original model's output value) to Y.
    The parameter obtained in this way will be used as the weight of the next layer.
    reconstruct the weight of the next layer to the one being pruned
    :param module: torch.nn.module, module of the this layer
    :param inputs: torch.Tensor, new input feature map of the this layer
    :param outputs: torch.Tensor, original output feature map of the this layer
    :param use_gpu: bool, whether done in gpu
    :return:
        void
    """
    if use_gpu:
        inputs = inputs.cuda()
        module = module.cuda()

    if module.bias is not None:
        bias_size = [1] * outputs.dim()
        bias_size[1] = -1
        outputs -= module.bias.view(bias_size)  #Subtract the amount of bias from the output feature (y - b)
    if isinstance(module, torch.nn.Conv2d):
        unfold = torch.nn.Unfold(kernel_size=module.kernel_size, dilation=module.dilation,
                                 padding=module.padding, stride=module.stride)
        if use_gpu:
            unfold = unfold.cuda()
        unfold.eval()
        x = unfold(inputs)  # Spread one patch (reception field) into a three-dimensional array with columns (N * KKC * L(number of fields))
        x = x.transpose(1, 2)  # tensor transpose (N * KKC * L) -> (N * L * KKC)
        num_fields = x.size(0) * x.size(1)
        x = x.reshape(num_fields, -1)  # x: (NL * KKC)
        y = outputs.view(outputs.size(0), outputs.size(1), -1)  # feature map (N * C * WH)
        y = y.transpose(1, 2)  # tensor transpose (N * C * HW) -> (N * HW * C), L == HW
        y = y.reshape(-1, y.size(2))  # y: (NHW * C),  (NHW) == (NL)

        if x.size(0) < x.size(1) or use_gpu is False:
            x, y = x.cpu(), y.cpu()
    #         x,y = x.cpu(), y.cpu()

    param, residuals, rank, s = np.linalg.lstsq(x.detach().cpu().numpy(),y.detach().cpu().numpy(),rcond=-1)
    # param, _ = torch.lstsq(y, x)
    if use_gpu:
        # param = param.cuda()
        param = torch.from_numpy(param).cuda()

    param = param[0:x.size(1), :].clone().t().contiguous().view(y.size(1), -1)
    if isinstance(module, torch.nn.Conv2d):
        param = param.view(module.out_channels, module.in_channels, *module.kernel_size)
    del module.weight
    module.weight = torch.nn.Parameter(param)