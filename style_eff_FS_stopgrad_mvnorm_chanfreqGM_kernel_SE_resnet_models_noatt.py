import copy
import numpy as np
import tensorflow as tf
import resnet_tiedSE_model_uniform_init as res

_building_block_SE_v2   = res._building_block_SE_v2
conv2d_fixed_padding = res.conv2d_fixed_padding
block_layer          = res.block_layer
batch_norm           = res.batch_norm


_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_VERSION = 2
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES

class resnet_embedding(object):
  """Base class for building the Resnet Model."""

  def __init__(self, bottleneck, embd_dim, n_classes, num_filters,
               kernel_size,
               conv_stride, first_pool_size, first_pool_stride,
               block_sizes, block_strides,
               resnet_version=DEFAULT_VERSION, data_format=None,
               dtype=DEFAULT_DTYPE, conv_dilation_rate=1, 
               block_kernel_sizes=[], block_dilation_rate=[], 
               bn_in_emb=False, has_std_in_pooling=True,       
               activation=tf.nn.relu, l2_norm_embd=False,
               reg=[], use_bias=False, block_filters=[], SE_type='standard', SE=False, SE_blocks=[], SE_ratio=4,
               renorm=False, bn_in_emb_train=True, get_excitation_blocks=True, get_output_blocks=True,
               include_regular_stats=True, block_freqs=[80,40,20,10],
               style_kernel={'kernel_type':None, 'kernel_param':None, 'conv_new_chn_dims':None, 'norm_chan_style':None,
                             'style_in_stats_blocks':[False]*4, 'mask_type':'full', 'apply_1D_bool':False, 'split_freq_bands':[None]*4},
               add_class_layer=True, stop_grad_softmax=False, scope_name='resnet_model'):

    self.n_classes = n_classes
  
    if not data_format:
      data_format = 'channels_first'
      
    self.resnet_version = resnet_version
    if resnet_version not in (1, 2):
      raise ValueError(
          'Resnet version should be 1 or 2. See README for citations.')

    self.bottleneck = bottleneck
    self.block_fn = _building_block_SE_v2
          
    if len(block_filters)==0:
      block_filters = [num_filters * (2**i) for i in range(len(block_sizes))]
      
    if len(SE_blocks)==0:
      SE_blocks = [SE for i in range(len(block_sizes))]
    if dtype not in ALLOWED_TYPES:
      raise ValueError('dtype must be one of: {}'.format(ALLOWED_TYPES))

    self.data_format = data_format
    self.embd_dim = embd_dim
    self.num_filters = num_filters
    self.block_filters = block_filters
    self.SE_blocks = SE_blocks
    self.kernel_size = kernel_size
    self.conv_stride = conv_stride
    self.conv_dilation_rate = conv_dilation_rate
    self.first_pool_size = first_pool_size
    self.first_pool_stride = first_pool_stride
    self.block_sizes = block_sizes
    self.block_strides = block_strides
    if block_kernel_sizes == []:
      self.block_kernel_sizes = [3]*len(block_strides)
    else:
      self.block_kernel_sizes = block_kernel_sizes 
    if block_dilation_rate == []:      
      self.block_dilation_rate = [1]*len(block_strides)
    else:
      self.block_dilation_rate = block_dilation_rate
    self.dtype = dtype
    self.pre_activation = resnet_version == 2    
    self.bn_in_emb = bn_in_emb
    if reg == []:
      self.reg == [None, [None]**len(block_strides), None]
    else:
      self.reg = reg
    self.use_bias = use_bias
    self.SE_ratio = SE_ratio
    self.SE_type = SE_type
    self.has_std_in_pooling = has_std_in_pooling
    self.add_class_layer = add_class_layer
    self.stop_grad_softmax = stop_grad_softmax
    self.scope_name = scope_name
    self.renorm = renorm
    self.activation = activation
    self.bn_in_emb_train = bn_in_emb_train
    self.l2_norm_embd = l2_norm_embd
    self.get_excitation_blocks = get_excitation_blocks
    self.get_output_blocks = get_output_blocks
    self.style_kernel = style_kernel
    self.include_regular_stats = include_regular_stats
    if not('droprate_chn' in self.style_kernel):
        self.style_kernel['droprate_chn'] = 0.0
    if not('activation_on_cov_feat' in self.style_kernel):
        self.style_kernel['activation_on_cov_feat'] = tf.identity
    self.block_freqs = block_freqs

  def _custom_dtype_getter(self, getter, name, shape=None, dtype=DEFAULT_DTYPE,
                           *args, **kwargs):

    if dtype in CASTABLE_TYPES:
      var = getter(name, shape, tf.float32, *args, **kwargs)
      return tf.cast(var, dtype=dtype, name=name + '_cast')
    else:
      return getter(name, shape, dtype, *args, **kwargs)

  def _model_variable_scope(self):

    return tf.compat.v1.variable_scope(self.scope_name,
                                       custom_getter=self._custom_dtype_getter)
  
  # This is the function that estimates the channel-wise correlations.
  def _gram_matrix_resnet(self, input_tensor, triang_mask):
    input_shape = tf.shape(input_tensor)
    input_tensor = tf.reshape(input_tensor,[-1,input_shape[1],input_shape[2]*input_shape[3]]) #Reshaping according to freq. ranges
    if self.style_kernel['subtract_mean']:
        input_tensor -= tf.math.reduce_mean(input_tensor, axis=-1, keepdims=True)
    if self.style_kernel['divide_std']:
        input_tensor /= tf.math.reduce_std(input_tensor, axis=-1, keepdims=True) + 1e-5       
    if self.style_kernel['kernel_type'] is None: # correlation pooling
      result = tf.linalg.einsum('bci,bdi->bcd', input_tensor, input_tensor)      
    elif self.style_kernel['kernel_type']=='Gauss': # correlation pooling with Gaussian kernel, not presented in the paper
      ssq = tf.reduce_sum(tf.math.square(input_tensor),axis=-1,keepdims=True)
      result = ssq + tf.transpose(ssq, perm=[0,2,1])
      result -= 2*tf.linalg.einsum('bci,bdi->bcd', input_tensor, input_tensor)
      result = tf.keras.activations.exponential(-0.5*result/self.style_kernel['kernel_param'][0]) # self.style_kernel['kernel_param'][0] = sigma**2
    result = self.style_kernel['activation_on_cov_feat'](result)
    result = tf.transpose(result, perm = [1,2,0])
    result = tf.boolean_mask(result, triang_mask)
    result = tf.transpose(result, perm = [1,0])
    num_locations = tf.cast(input_shape[2]*input_shape[3], tf.float32)
    result = result/num_locations
    result = tf.math.sign(result)*tf.math.sqrt(tf.math.abs(result)) if self.style_kernel['pownorm_chn_style'] else tf.identity(result)
    result = tf.math.l2_normalize(result,axis=-1) if self.style_kernel['l2norm_chn_style'] else tf.identity(result)    
    return result 

  # Mask defining which elements of the correlation matrices to keep in pooling, to avoid repetitions and fixed values in diagonal.  
  def _create_triang_mask(self,nch):
    ones = tf.ones([nch,nch])
    if self.style_kernel['mask_type'] == "uptriang":
        mask = tf.cast(tf.linalg.band_part(ones, 0, -1), dtype=tf.bool) # Make a bool mask                                                                 
        triang_mask_len = int(nch*(nch+1)/2.0)
    elif self.style_kernel['mask_type'] == "uptriang_nodiag":
        mask = tf.cast(tf.linalg.band_part(ones, 0, -1) - tf.linalg.band_part(ones, 0, 0), dtype=tf.bool) # Make a bool mask                               
        triang_mask_len = int(nch*(nch-1)/2.0)
    elif self.style_kernel['mask_type'] == "diag":
        mask = tf.cast(tf.linalg.band_part(ones, 0, 0), dtype=tf.bool) # Make a bool mask                                                                  
        triang_mask_len = int(nch)
    else: #full                                                                                                                                            
        mask = tf.cast(ones, dtype=tf.bool) # Make a bool mask                                                                                             
        triang_mask_len = int(nch**2)
    return mask, triang_mask_len

  # created the 3D projection matrix (L), i.e. different projection matrix for each freq. range
  def _create_1D_conv(self): 
    conv_1D_style_lst = []
    if self.style_kernel['apply_1D_bool'] == False:
      return conv_1D_style_lst
    for b_cnt,new_dim in enumerate(self.style_kernel['conv_new_chn_dims']):
      conv_1D_fb_lst = []
      fbn = len(self.style_kernel['split_freq_bands'][b_cnt]) if self.style_kernel['split_freq_bands'][b_cnt] is not None else 1
      for f_cnt in range(fbn):
          conv_1D = tf.keras.layers.Conv2D(filters=new_dim, kernel_size=(1,1), data_format="channels_first") if new_dim is not None else None
          conv_1D_fb_lst.append(conv_1D)
      conv_1D_style_lst.append(conv_1D_fb_lst)
    return conv_1D_style_lst

  # created the 2D projection matrix (L), i.e. as single projection matrix for all freq. ranges
  def _create_1D_conv_freq_independent(self):
    conv_1D_style_lst = []
    if self.style_kernel['apply_1D_bool'] == False:
      return conv_1D_style_lst
    for b_cnt,new_dim in enumerate(self.style_kernel['conv_new_chn_dims']):
      conv_1D = tf.keras.layers.Conv2D(filters=new_dim, kernel_size=(1,1), data_format="channels_first") if new_dim is not None else None
      conv_1D_style_lst.append(conv_1D)
    return conv_1D_style_lst
  
  #Just apply the projection matrix L to the tensor
  def _apply_1D_conv(self,outputs,conv_1D):
    outputs = conv_1D(outputs) if conv_1D is not None else tf.identity(outputs)
    return outputs
  
  #It creates a list of tensors, one per frequency range 
  def _split_in_freq(self,outputs,freq_bands):
    outputs_lst = []
    if freq_bands is None:
      outputs_lst.append(outputs)
      return outputs_lst
    for fb in freq_bands:
      outputs_lst.append(outputs[:,:,:,fb[0]:fb[1]])
    return outputs_lst

  def __call__(self, inputs, training, scale=30.0, margin=0.2, spkr_labs=None):

    reg_cnt = 0
    with self._model_variable_scope():
      if self.data_format == 'channels_first':
        # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
        # This provides a large performance boost on GPU. See
        # https://www.tensorflow.org/performance/performance_guide#data_formats
        inputs = tf.transpose(a=inputs, perm=[0, 3, 1, 2])

      inputs = conv2d_fixed_padding(
          inputs=inputs, filters=self.num_filters, kernel_size=self.kernel_size,
          strides=self.conv_stride, data_format=self.data_format, dilation_rate=self.conv_dilation_rate, reg=self.reg[reg_cnt], use_bias=self.use_bias)
      inputs = tf.identity(inputs, 'initial_conv')
      reg_cnt += 1
      # We do not include batch normalization or activation functions in V2
      # for the initial conv1 because the first ResNet unit will perform these
      # for both the shortcut and non-shortcut paths as part of the first
      # block's projection. Cf. Appendix of [2].
      if self.resnet_version == 1:
        inputs = batch_norm(inputs, training, self.data_format, renorm=self.renorm)
        inputs = self.activation(inputs)

      if self.first_pool_size:
        inputs = tf.compat.v1.layers.max_pooling2d(
            inputs=inputs, pool_size=self.first_pool_size,
            strides=self.first_pool_stride, padding='SAME',
            data_format=self.data_format)
        inputs = tf.identity(inputs, 'initial_max_pool')
      
      block_reg = self.reg[reg_cnt]
      reg_cnt += 1
      excitation_blocks = []
      output_blocks = []
      
      for i, num_blocks in enumerate(self.block_sizes):
        num_filters = self.block_filters[i] # updated
        SE_ratio = self.SE_ratio if self.SE_blocks[i] else 0
        inputs, excitation_block = block_layer(
          inputs=inputs, filters=num_filters, bottleneck=self.bottleneck,
          block_fn=self.block_fn, blocks=num_blocks,
          strides=self.block_strides[i], block_kernel_sizes = self.block_kernel_sizes[i],
          block_dilation_rate = self.block_dilation_rate[i],           
          training=training, renorm=self.renorm, activation=self.activation, SE_type=self.SE_type,
          name='block_layer{}'.format(i + 1), data_format=self.data_format, reg=block_reg, use_bias=self.use_bias, SE_ratio=SE_ratio)
        print('Block cnt {}, output tensor shape: {}'.format(i,tf.shape(inputs)))
        output_blocks.append(inputs)
        if self.get_excitation_blocks:
            excitation_blocks.append(excitation_block)
            
      # Only apply the BN and ReLU for model that does pre_activation in each
      # building/bottleneck block, eg resnet V2.

      # Note that these 2 layers are not used in correlation pooling.
      if self.pre_activation:
        inputs = batch_norm(inputs, training, self.data_format, renorm=self.renorm)
        inputs = self.activation(inputs)

      reg_att = self.reg[reg_cnt]
      reg_cnt += 1

      ############ Create list of 1D convolutions for styles using conv_new_chn_dims field 
      ######  This is the 2D or 3D projection tensor L which I describe in the paper (2D if 1D_conv_freq_ind==True)
      conv_1D_style_lst = self._create_1D_conv_freq_independent() if self.style_kernel['1D_conv_freq_ind'] else self._create_1D_conv()
      
      ############ Add correlations to stats  

      stats_lst = [] # A list containing corr stats from each block and frequency range. In the paper I extract only from the last block (or stage).
      stats_len = 0
      triang_mask_lst = []

      for b_cnt,outputs_orig in enumerate(output_blocks):
          triang_mask = []
          r_DO = self.style_kernel['droprate_chn']
          if self.style_kernel['style_in_stats_blocks'][b_cnt]: # Do you want correlations from this block? 
              # noise_shape is set so that we sample only the channel dim (i.e. channelwise dropout)
              outputs = tf.cond(training, lambda: tf.nn.dropout(outputs_orig, rate=r_DO, noise_shape=[1,self.block_filters[b_cnt],1,1]),
                                lambda: tf.identity(outputs_orig))
              nch = self.style_kernel['conv_new_chn_dims'][b_cnt] if self.style_kernel['apply_1D_bool'] else self.block_filters[b_cnt]
              outputs = self._apply_1D_conv(outputs,conv_1D_style_lst[b_cnt]) if self.style_kernel['1D_conv_freq_ind'] else tf.identity(outputs)
              triang_mask, triang_mask_len = self._create_triang_mask(nch) #create the mask to keeponly unique and trainable variables
              outputs_f_lst = self._split_in_freq(outputs,self.style_kernel['split_freq_bands'][b_cnt]) # split tensor according to freq ranges. 
              for f_cnt,outputs_f in enumerate(outputs_f_lst): #for each freq range calculate correlation and append in to list
                  outputs_f = tf.stop_gradient(outputs_f) if self.style_kernel['stop_grad'] else tf.identity(outputs_f)
                  outputs_f = self._apply_1D_conv(outputs_f,conv_1D_style_lst[b_cnt][f_cnt]) if not self.style_kernel['1D_conv_freq_ind'] else tf.identity(outputs_f)
                  stats_lst.append(self._gram_matrix_resnet(outputs_f, triang_mask)) # Append the trainable variables of the freq range to the list
                  stats_len += triang_mask_len
                  
          triang_mask_lst.append(triang_mask)
      if (True in self.style_kernel['style_in_stats_blocks']):    
          stats_ = tf.concat(stats_lst,axis=1) # Concatenate the list to a single vector
          stats_.set_shape([None,stats_len])
      ############ Add regular to stats in case you want to ...  
      if self.include_regular_stats:
        if self.has_std_in_pooling:
          inputs = tf.concat([tf.reduce_mean(input_tensor=inputs, axis=2, keepdims=True),tf.math.reduce_std(input_tensor=inputs, axis=2, keepdims=True)],3)
        else:
          inputs = tf.reduce_mean(input_tensor=inputs, axis=2, keepdims=True)
        flatten_layer = tf.keras.layers.Flatten(trainable=False)
        stats_reg_ = flatten_layer(inputs)
      if True in self.style_kernel['style_in_stats_blocks'] and self.include_regular_stats:
        stats_ = tf.concat([stats_,stats_reg_],axis=1)
      elif not (True in self.style_kernel['style_in_stats_blocks']) and self.include_regular_stats: #just use standard stats pooling
        stats_ = stats_reg_

      d_reg = kernel_regularizer= self.reg[reg_cnt]
      reg_cnt += 1      

      # stats to embedding layer
      dense_ap1 = tf.keras.layers.Dense(units=self.embd_dim,trainable=True,use_bias=self.use_bias, kernel_regularizer=d_reg, bias_regularizer=d_reg, kernel_initializer=tf.compat.v1.keras.initializers.he_uniform(), bias_initializer=tf.compat.v1.keras.initializers.he_uniform())

      if self.bn_in_emb:
        bn_ap1 = tf.compat.v1.keras.layers.BatchNormalization(axis=1, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
                                                              center=self.bn_in_emb_train, scale=self.bn_in_emb_train,
                                                              trainable=True, name="bn_in_emb", renorm=self.renorm)

      self.params_up2stats = copy.copy(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.scope_name))

      if self.add_class_layer: # classification head layer
        dense_ap2 = tf.keras.layers.Dense(units=self.n_classes, trainable=True, use_bias=False, kernel_regularizer=self.reg[reg_cnt], kernel_constraint=tf.keras.constraints.UnitNorm(axis=0), name="ClassHead", kernel_initializer=tf.compat.v1.keras.initializers.he_uniform())
        reg_cnt += 1
      X_ = dense_ap1(stats_)
      X_ = bn_ap1(X_) if self.bn_in_emb else tf.identity(X_)
      X_ = tf.nn.l2_normalize(X_, -1, 1e-7) if self.l2_norm_embd else tf.identity(X_)

      self.params_up2embd = copy.copy(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.scope_name))
        
      if self.add_class_layer:
        embd_ = tf.identity(X_, 'embd_')
        C_c_ = dense_ap2(tf.stop_gradient(embd_)) if self.stop_grad_softmax else dense_ap2(embd_)
        ass_op_ = tf.assign( dense_ap2.kernel, dense_ap2.kernel_constraint(dense_ap2.kernel) )
        tf.compat.v1.add_to_collection(tf.GraphKeys.UPDATE_OPS, ass_op_ )
        # Apply margin and scale of AAM loss:
        C_c_ = tf.cond(training, lambda: self.calculate_arcface_logits(C_c_, spkr_labs, self.n_classes, scale, margin), lambda: C_c_*scale)
        for l in dense_ap1.losses:
          tf.compat.v1.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, l )
        for l in dense_ap2.losses:
          tf.compat.v1.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, l )
      else:
        C_c_=None

      return X_, C_c_, excitation_blocks, output_blocks
          
  def get_parameters(self):    
      #return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='resnet_model') orig
      return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.scope_name)
    
  def get_upd_parameters(self):
      #return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='resnet_model') orig
      return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name)
  def get_params_up2embd(self):
      return self.params_up2embd

  def get_params_up2stats(self):
      return self.params_up2stats

  def get_params_stats2embd(self):
      lst = [self.params_up2embd[i] for i in range(len(self.params_up2stats),len(self.params_up2embd))]
      return lst
  
  def calculate_arcface_logits(self, cos_t, labels, class_num, s, m):

      cos_m = tf.math.cos(m)
      sin_m = tf.math.sin(m)
      mm = sin_m * m
      threshold = tf.math.cos(np.pi - m)

      cos_t2 = tf.square(cos_t, name='cos_2')
      sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
      sin_t = tf.sqrt(sin_t2, name='sin_t')
      cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')
      cond_v = cos_t - threshold
      cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)
      keep_val = s*(cos_t - mm)
      cos_mt_temp = tf.where(cond, cos_mt, keep_val)
      mask = tf.one_hot(labels, depth=class_num, name='one_hot_mask')
      inv_mask = tf.subtract(1., mask, name='inverse_mask')
      s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t')
      output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_logits')
      return output      

  def get_regularization(self):
      r = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES, scope=self.scope_name)
      print("Regularized variables")
      for rr in r:
        print(rr)
      return sum(r)
