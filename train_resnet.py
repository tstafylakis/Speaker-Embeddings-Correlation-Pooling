    #!/usr/bin/env python
    import numpy as np
    import tensorflow as tf
    import style_eff_FS_stopgrad_mvnorm_chanfreqGM_kernel_SE_resnet_models_noatt as resnet_models

    # Data and label placeholders
    X1_p        = tf.placeholder('float32', shape=[None,None,80,1], name='X1_p') # Input tensor: Batch, Frames, frame-dim (80), input channels (1).                                                                                                  
    is_test_p   = tf.placeholder(dtype='bool', shape=[], name='is_test_p')  # Is the training (False) or the testing phase (True)?                                                                                                      
    L_c_p       = tf.placeholder('int32', shape=[None], name='L_c_out_p')   # Speaker labels as indices                                             

    n_spk = 5994 # for VoxCeleb 

    # 1st conv layer config (before the blocks/stages)
    num_filters       = 64                         
    kernel_size       = 3                          
    conv_stride       = 1                          
    first_pool_size   = None                       
    first_pool_stride = 1    

    ### Standard 34-ResNet apart from the block_filters                     
    block_sizes       = [3, 4, 6, 3]               
    block_strides     = [1, 2, 2, 2] 
    block_filters     = [64, 128, 256, 256]     
    block_freqs       = [80, 40, 20, 10] # Output num of frequencies for the 4 ResNet blocks/stages. 
    #                               ... They can be derived from the strides, but it's easier to pass them as argument.
    conv_dilation_rate = 1 # Use dilated convs in the freq axis. Set to 1 for standard
    use_bias = True # Use biases in ResNet and Pooling
    activation = tf.nn.relu # Activations in ResNet

    ### Embedding Config
    embd_dim = 256 # Embedding dimensions
    bn_in_emb = False # batch norm in embeddings.
    bn_in_emb_train = True # Trainable batch norm in embeddings. No impact if bn_in_emb == False.
    l2_norm_embd = True # True for AMM/arcface lose

    ### Squeeze and Exhitation 
    SE = True
    SE_blocks = [True,True,False,False] #I apply it only to the first 2 ResNet blocks (or stages) 
    SE_ratio = 4
    SE_type = 'standard'

    ### ResNet config
    # l2 regularizers for difference parts of the net. I just use 1e-4 for all.
    l2_reg = np.asarray([tf.keras.regularizers.l2(1e-4), tf.keras.regularizers.l2(1e-4), 
        tf.keras.regularizers.l2(1e-4), tf.keras.regularizers.l2(1e-4), tf.keras.regularizers.l2(1e-4)])
    batch_renorm = False # False, i.e. use standard batch norm

    ### AAM/arcface loss
    scale_arcface = 30.0 #fixed
    margin_arcface = 0.1 # start with 0.1 and increase it to 0.3

    ### Standard stats pooling
    include_regular_stats = False # Set True in case you want to use or to add standard stats pooling
    has_std_in_pooling = True # In case you want to use/add standard stats pooling, include std features to the mean features
    # In case you want only standard pooling, set all 4 entries in style_kernel.style_in_stats_blocks to False (see below). 

    # style_kernel: Parameters related to correlation (or style, from style-transfer) pooling
    # kernel_type:None will apply correlation with dot products as in the paper. But you may also use kernels (Only Gaussian is implemented)
    # kernel_param: A list of params specific to each kernel. Correlation does not use kernel_params.  
    # conv_new_chn_dims: new num of channels per block (stage). Set None for blocks not added in corr. pooling.
    # l2norm_chn_style:None, apply l2-norm to channels, set to None for corr. pooling
    # pownorm_chn_style:False, apply power-norm to channels, set to False for corr. pooling
    # style_in_stats_blocks: which blocks should be added to corr. pooling. If >1 they will be concatenated.
    # mask_type:, from the correlation matrices keep only the uptriangle (symmetric matrices) and remove the diag (composed of ones)
    # .... set 'uptriang' for keeping the diagonal (e.g. for covariance pooling) 
    # subtract_mean: True for correlation and covariance pooling
    # divide_std: True for correlation, False for covariance pooling.
    # droprate_chn: channel dropout rate (see paper)
    # stop_grad: For experiments with not backpropagating corr loss to the ResNet. Set to False our proposed corr pooling.
    # activation_on_cov_feat: You may try activation functions on the covariance. Set to tf.identity for no activation. 
    # apply_1D_bool: True if you want to reduce num of channels. Set to True.  
    # 1D_conv_freq_ind: 2D vs 3D projection. Set to False for 3D projection, True for 2D.


    split_freq_bands = [None,None,None,None]
    split_freq_bands[3] = [[0,2],[2,4],[4,6],[6,8],[8,10]] # The Frequency Merging I explain the the paper. 
    #It will merge the 10 consecutive freq bins to 5 freq ranges, e.g. [0,1] -> 0 , [2,3] -> 1  etc.

    style_kernel = {'kernel_type':None, 'kernel_param':[1.0], 'conv_new_chn_dims':[None, None, None, 64], 'l2norm_chn_style':None,
                  'pownorm_chn_style':False, 'style_in_stats_blocks':[False, False, False, True], 'mask_type':'uptriang_nodiag', 'apply_1D_bool':True,
                  'split_freq_bands':split_freq_bands, 'subtract_mean':True, 'divide_std':True, 'stop_grad':False, 'droprate_chn':0.25,
                  'activation_on_cov_feat':tf.identity, '1D_conv_freq_ind':False}

    model = resnet_models.resnet_embedding(bottleneck=False, embd_dim=embd_dim, n_classes=n_spk, num_filters=num_filters,
                                                           kernel_size=kernel_size, conv_stride=conv_stride,
                                                           first_pool_size=first_pool_size, first_pool_stride=first_pool_stride,
                                                           block_sizes=block_sizes, block_strides=block_strides,
                                                           resnet_version=2, data_format=None, dtype=tf.float32,
                                                           bn_in_emb=bn_in_emb, reg=l2_reg, use_bias=use_bias,                                                       
                                                           conv_dilation_rate=conv_dilation_rate, block_filters=block_filters,
                                                           SE=SE, SE_blocks=SE_blocks, SE_ratio=SE_ratio, SE_type=SE_type,
                                                           add_class_layer=True, stop_grad_softmax=False, has_std_in_pooling=has_std_in_pooling,
                                                           renorm=batch_renorm, activation=activation, bn_in_emb_train=bn_in_emb_train,
                                                           l2_norm_embd=l2_norm_embd, get_output_blocks=True, scope_name='resnet_model',
                                                           include_regular_stats=include_regular_stats, style_kernel=style_kernel,
                                                           block_freqs=block_freqs)

    embd_, C_c_, excitations_, output_blocks_ = model(X1_p, tf.math.logical_not(is_test_p), scale=scale_arcface, margin=margin_arcface, spkr_labs=L_c_p)
    #Note that margin and spkr_labs are only required during training (AMM/arcface loss requires them).
    #embd_: the embeddings
    #C_c_: the logits
    #excitations_: the weighted from the Squeeze and Exhitation in case you want to visualize them.
    #output_blocks_: the outputs from ResNet blocks.
    #for training use this averaged loss:
    loss_cf_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = L_c_p, logits = C_c_)
