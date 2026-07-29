# FILE NAME: layers.py
from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # Reshape the input x into a 2D matrix of shape (N, D)
    # N = number of samples, D = product of remaining dimensions.
    x_reshaped = x.reshape(x.shape[0], -1)

    # Compute the matrix multiplication and add the bias
    out = np.dot(x_reshaped, w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # Gradient with respect to bias b
    db = np.sum(dout, axis=0)

    # Reshape the input x for matrix multiplication
    x_reshaped = x.reshape(x.shape[0], -1)

    # Gradient with respect to weights w
    dw = np.dot(x_reshaped.T, dout)

    # Gradient with respect to input x
    dx_reshaped = np.dot(dout, w.T)
    dx = dx_reshaped.reshape(x.shape)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # The ReLU function simply sets all negative values to zero.
    out = np.maximum(0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # The gradient of ReLU is 1 for x > 0 and 0 for x <= 0.
    # The gradient passes through only where the input was positive.
    dx = dout
    dx[x <= 0] = 0
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # Step 1: Calculate mean of the minibatch
        sample_mean = np.mean(x, axis=0)

        # Step 2: Calculate variance of the minibatch
        sample_var = np.var(x, axis=0)

        # Step 3: Normalize the data
        x_normalized = (x - sample_mean) / np.sqrt(sample_var + eps)

        # Step 4: Scale and shift
        out = gamma * x_normalized + beta

        # Store intermediates for the backward pass
        cache = (x, sample_mean, sample_var, x_normalized, gamma, eps)

        # Update the running mean and variance for test time
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # Normalize the data using the running mean and variance
        x_normalized = (x - running_mean) / np.sqrt(running_var + eps)

        # Scale and shift
        out = gamma * x_normalized + beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # Unpack intermediates from the cache
    x, sample_mean, sample_var, x_normalized, gamma, eps = cache
    N, D = dout.shape

    # --- Step-by-step backpropagation ---

    # Step 1: Gradient of beta (dbeta)
    # This is the sum of dout over the batch dimension.
    dbeta = np.sum(dout, axis=0)

    # Step 2: Gradient of gamma (dgamma)
    # This is the sum of dout multiplied by the normalized x.
    dgamma = np.sum(dout * x_normalized, axis=0)

    # Step 3: Gradient with respect to the normalized input (dx_normalized)
    # The output is out = gamma * x_normalized + beta.
    dx_normalized = dout * gamma

    # Step 4: Gradient with respect to the variance (dvar)
    # This involves backpropagating through the 1/sqrt(var + eps) term.
    x_minus_mean = x - sample_mean
    inv_std = 1.0 / np.sqrt(sample_var + eps)
    dvar = np.sum(dx_normalized * x_minus_mean * -0.5 * (inv_std ** 3), axis=0)

    # Step 5: Gradient with respect to the mean (dmean)
    # This has two parts: one from normalizing x and one from the variance calculation.
    dmean = np.sum(dx_normalized * -inv_std, axis=0) + dvar * np.mean(-2.0 * x_minus_mean, axis=0)

    # Step 6: Gradient with respect to the input x (dx)
    # The gradient dx is the sum of gradients from three paths in the computation graph.
    dx = (dx_normalized * inv_std) + (dvar * 2.0 * x_minus_mean / N) + (dmean / N)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # Step 1: Calculate mean for each sample across its features (axis=1)
    mean = np.mean(x, axis=1, keepdims=True)

    # Step 2: Calculate variance for each sample across its features (axis=1)
    var = np.var(x, axis=1, keepdims=True)

    # Step 3: Normalize the data
    x_normalized = (x - mean) / np.sqrt(var + eps)

    # Step 4: Scale and shift
    out = gamma * x_normalized + beta

    # Store intermediates for the backward pass
    cache = (x, mean, var, x_normalized, gamma, eps)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # Unpack intermediates from the cache
    x, mean, var, x_normalized, gamma, eps = cache
    N, D = x.shape

    # --- Step-by-step backpropagation (parallel to batchnorm_backward) ---

    # Step 1: Gradient of beta (dbeta)
    # Sum over the batch dimension.
    dbeta = np.sum(dout, axis=0)

    # Step 2: Gradient of gamma (dgamma)
    # Sum over the batch dimension.
    dgamma = np.sum(dout * x_normalized, axis=0)

    # Step 3: Gradient with respect to the normalized input (dx_normalized)
    dx_normalized = dout * gamma

    # Step 4: Gradient with respect to the variance (dvar)
    # Sum over the feature dimension (axis=1).
    x_minus_mean = x - mean
    inv_std = 1.0 / np.sqrt(var + eps)
    dvar = np.sum(dx_normalized * x_minus_mean * -0.5 * (inv_std ** 3), axis=1, keepdims=True)

    # Step 5: Gradient with respect to the mean (dmean)
    # Sum over the feature dimension (axis=1).
    dmean = np.sum(dx_normalized * -inv_std, axis=1, keepdims=True) + dvar * np.mean(-2.0 * x_minus_mean, axis=1,
                                                                                     keepdims=True)

    # Step 6: Gradient with respect to the input x (dx)
    # Distribute gradients, dividing by D (number of features) instead of N.
    dx = (dx_normalized * inv_std) + (dvar * 2.0 * x_minus_mean / D) + (dmean / D)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # Step 1: Create a mask with the same shape as x.
        # Each element is 1 with probability p, and 0 otherwise.
        mask = (np.random.rand(*x.shape) < p)

        # Step 2: Apply the mask to the input x.
        # Step 3: Scale the output by 1/p. This is the "inverted" part.
        out = x * mask / p
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # During test time, we do nothing and just pass the input through.
        # The scaling was already handled during the training phase.
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # Get the keep probability
        p = dropout_param['p']

        # The gradient is only passed through the neurons that were not dropped.
        # We also apply the same scaling factor (1/p) from the forward pass.
        dx = dout * mask / p
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.


    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # Unpack dimensions and parameters
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride = conv_param.get('stride', 1)
    pad = conv_param.get('pad', 0)

    # Calculate output dimensions
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride

    # Initialize the output volume with zeros
    out = np.zeros((N, F, H_out, W_out))

    # Pad the input volume 'x' on the spatial dimensions (Height and Width)
    # The padding format is ((pad_before, pad_after), ...) for each axis.
    # We don't pad the N (batch) or C (channel) dimensions.
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')

    # Naive implementation using nested loops
    for n in range(N):  # Iterate over each image in the batch
        for f in range(F):  # Iterate over each filter
            for i in range(H_out):  # Iterate over the height of the output
                for j in range(W_out):  # Iterate over the width of the output
                    # Define the current slice of the input volume
                    h_start = i * stride
                    h_end = h_start + HH
                    w_start = j * stride
                    w_end = w_start + WW

                    # Extract the receptive field from the padded input
                    receptive_field = x_padded[n, :, h_start:h_end, w_start:w_end]

                    # Perform the convolution: element-wise product and sum
                    conv_sum = np.sum(receptive_field * w[f])

                    # Add the bias and store it in the output volume
                    out[n, f, i, j] = conv_sum + b[f]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # Unpack cache and parameters
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride = conv_param.get('stride', 1)
    pad = conv_param.get('pad', 0)

    # Calculate output dimensions from the upstream derivative shape
    _, _, H_out, W_out = dout.shape

    # Initialize gradients with zeros
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    # Create padded versions of x and dx for easier computation
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    dx_padded = np.pad(dx, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')

    # --- Gradient Calculation ---

    # Loop over each image, filter, and output position
    for n in range(N):  # Iterate over each image in the batch
        for f in range(F):  # Iterate over each filter
            # Gradient for bias 'db' is the sum of the upstream derivatives
            # for that filter's entire feature map.
            db[f] += np.sum(dout[n, f])

            for i in range(H_out):  # Iterate over the height of the output
                for j in range(W_out):  # Iterate over the width of the output
                    # Define the current slice of the input volume
                    h_start = i * stride
                    h_end = h_start + HH
                    w_start = j * stride
                    w_end = w_start + WW

                    # Extract the receptive field from the padded input
                    receptive_field = x_padded[n, :, h_start:h_end, w_start:w_end]

                    # Get the specific upstream derivative for this position
                    d_out = dout[n, f, i, j]

                    # Update gradients
                    # Gradient for weights 'dw': multiply the receptive field by the upstream derivative.
                    # This accumulates the gradient for the filter at every position it was applied.
                    dw[f] += receptive_field * d_out

                    # Gradient for input 'dx': multiply the filter weights by the upstream derivative.
                    # This "scatters" the gradient back to the input region that produced the output.
                    dx_padded[n, :, h_start:h_end, w_start:w_end] += w[f] * d_out

    # Unpad the gradient for dx to match the original input shape
    dx = dx_padded[:, :, pad:-pad, pad:-pad]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # Unpack dimensions and pooling parameters
    N, C, H, W = x.shape
    pool_height = pool_param.get('pool_height', 2)
    pool_width = pool_param.get('pool_width', 2)
    stride = pool_param.get('stride', 2)

    # Calculate output dimensions
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride

    # Initialize the output volume with zeros
    out = np.zeros((N, C, H_out, W_out))

    # Naive implementation using nested loops
    for n in range(N):  # Iterate over each image in the batch
        for c in range(C):  # Iterate over each channel
            for i in range(H_out):  # Iterate over the height of the output
                for j in range(W_out):  # Iterate over the width of the output
                    # Define the current pooling window
                    h_start = i * stride
                    h_end = h_start + pool_height
                    w_start = j * stride
                    w_end = w_start + pool_width

                    # Extract the pooling window from the input
                    pooling_window = x[n, c, h_start:h_end, w_start:w_end]

                    # Find the maximum value in the window and store it in the output
                    out[n, c, i, j] = np.max(pooling_window)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # Get input dimensions
    N, C, H, W = x.shape

    # 1. Reshape the input from (N, C, H, W) to (N*H*W, C).
    # This treats all spatial locations for a given channel as a single feature vector.
    # First, move the channel axis to the end: (N, H, W, C)
    x_transposed = x.transpose(0, 2, 3, 1)
    # Then, reshape into a 2D matrix
    x_reshaped = x_transposed.reshape(-1, C)

    # 2. Call the standard batch normalization function on the reshaped data.
    out_reshaped, cache = batchnorm_forward(x_reshaped, gamma, beta, bn_param)

    # 3. Reshape the output back to the original input shape (N, C, H, W).
    # First, reshape back to (N, H, W, C)
    out_transposed = out_reshaped.reshape(N, H, W, C)
    # Then, move the channel axis back to its original position
    out = out_transposed.transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # Get the original input dimensions from the upstream derivative
    N, C, H, W = dout.shape

    # 1. Reshape the upstream derivative dout from (N, C, H, W) to (N*H*W, C).
    # This matches the shape of the output from the vanilla batchnorm_forward.
    # First, move the channel axis to the end: (N, H, W, C)
    dout_transposed = dout.transpose(0, 2, 3, 1)
    # Then, reshape into a 2D matrix
    dout_reshaped = dout_transposed.reshape(-1, C)

    # 2. Call the standard batch normalization backward function.
    # It will return gradients for dx, dgamma, and dbeta.
    # dgamma and dbeta are already in the correct shape (C,).
    dx_reshaped, dgamma, dbeta = batchnorm_backward(dout_reshaped, cache)

    # 3. Reshape the gradient dx back to the original input shape (N, C, H, W).
    # First, reshape back to (N, H, W, C)
    dx_transposed = dx_reshaped.reshape(N, H, W, C)
    # Then, move the channel axis back to its original position
    dx = dx_transposed.transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # Get input dimensions
    N, C, H, W = x.shape

    # 1. Reshape input from (N, C, H, W) to (N, G, C/G, H, W).
    # This explicitly separates the C channels into G groups.
    x_reshaped = x.reshape(N, G, C // G, H, W)

    # 2. Compute mean and variance for each group.
    # Normalization is done over the channels within a group and the spatial dimensions.
    # We compute stats over axes (2, 3, 4) which correspond to (C/G, H, W).
    mean = np.mean(x_reshaped, axis=(2, 3, 4), keepdims=True)
    var = np.var(x_reshaped, axis=(2, 3, 4), keepdims=True)

    # 3. Normalize the data within each group.
    x_normalized_reshaped = (x_reshaped - mean) / np.sqrt(var + eps)

    # 4. Reshape the normalized data back to the original input shape (N, C, H, W).
    x_normalized = x_normalized_reshaped.reshape(N, C, H, W)

    # 5. Apply scale and shift. gamma and beta are of shape (C,), so they need to be
    # reshaped to (1, C, 1, 1) to broadcast correctly with the (N, C, H, W) input.
    out = gamma.reshape(1, C, 1, 1) * x_normalized + beta.reshape(1, C, 1, 1)

    # Store values needed for the backward pass
    cache = (x, gamma, G, eps, x_normalized_reshaped, mean, var)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # Unpack values from the cache
    x, gamma, G, eps, x_normalized_reshaped, mean, var = cache
    N, C, H, W = x.shape

    # --- Step 1: Compute gradients for beta and gamma ---
    # These are computed before reshaping dout.

    # Reshape gamma for broadcasting
    gamma_reshaped = gamma.reshape(1, C, 1, 1)

    # Reshape the normalized data back to (N, C, H, W) to compute dgamma
    x_normalized = x_normalized_reshaped.reshape(N, C, H, W)

    # Gradient of beta is the sum of dout over all but the channel dimension
    dbeta = np.sum(dout, axis=(0, 2, 3))

    # Gradient of gamma is the sum of (dout * x_normalized) over all but the channel dimension
    dgamma = np.sum(dout * x_normalized, axis=(0, 2, 3))

    # --- Step 2: Compute gradient for the input x ---

    # Gradient with respect to the normalized output
    dx_normalized = dout * gamma_reshaped

    # Reshape dx_normalized to the grouped shape (N, G, C/G, H, W)
    dx_normalized_reshaped = dx_normalized.reshape(N, G, C // G, H, W)

    # Reshape original input x to the grouped shape
    x_reshaped = x.reshape(N, G, C // G, H, W)

    # --- Apply the layer norm backward logic to each group ---

    # Number of elements in each group
    D_group = C // G * H * W

    # Intermediates for the backward pass
    inv_std = 1.0 / np.sqrt(var + eps)
    x_minus_mean = x_reshaped - mean

    # Gradient with respect to variance (sum over group axes)
    dvar = np.sum(dx_normalized_reshaped * x_minus_mean * -0.5 * (inv_std ** 3), axis=(2, 3, 4), keepdims=True)

    # Gradient with respect to mean (sum over group axes)
    dmean = np.sum(dx_normalized_reshaped * -inv_std, axis=(2, 3, 4), keepdims=True) + \
            dvar * np.sum(-2.0 * x_minus_mean, axis=(2, 3, 4), keepdims=True) / D_group

    # Gradient with respect to the input (grouped)
    dx_reshaped = (dx_normalized_reshaped * inv_std) + \
                  (dvar * 2.0 * x_minus_mean / D_group) + \
                  (dmean / D_group)

    # Reshape dx back to the original input shape (N, C, H, W)
    dx = dx_reshaped.reshape(N, C, H, W)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
