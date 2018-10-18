def dot_product(x, kernel):

    if K.backend() == 'tensorflow':
        # todo: check that this is correct
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class Attention(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True,
                 return_attention=False,
                 return_sequence=False,
                 **kwargs):
     
        self.supports_masking = True
        self.return_attention = return_attention
        self.return_sequence = return_sequence
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
             self.b = self.add_weight((1,),# fix the wrong implementation
   #         self.b = self.add_weight((input_shape[1],),
             
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):

        return None

    def call(self, x, mask=None):
        eij = dot_product(x, self.W)

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        weighted_input = x * K.expand_dims(a)
        
        if self.return_sequence:
            result = weighted_input
        else:
            result = K.sum(weighted_input, axis=1)

        if self.return_attention:
            return [result, a]
        return result

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            if self.return_sequence:
                return [input_shape, (input_shape[0], input_shape[1])]
            else:
                return [(input_shape[0], input_shape[-1]),
                    (input_shape[0], input_shape[1])]
        else:
            if self.return_sequence:
                return input_shape
            else:
                return input_shape[0], input_shape[-1



def dot_product_m(x, kernel):

    if K.backend() == 'tensorflow':
        # todo: check that this is correct
        return K.dot(x, kernel)
    else:
        return K.dot(x, kernel)


class MultiAttention(Layer):
    def __init__(self,
                 k=8,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True,
                 return_attention=False,
                 **kwargs):
        
        self.k = k
        self.supports_masking = True
        self.return_attention = return_attention
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(MultiAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],self.k),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
             self.b = self.add_weight((1,self.k),# fix the wrong implementation
   #         self.b = self.add_weight((input_shape[1],),
             
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):

        return None

    def call(self, x, mask=None):
        eij = dot_product_m(x, self.W)

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        weighted_input = K.repeat_elements(K.expand_dims(x, axis = -1), self.k, axis = -1) * K.expand_dims(a,axis = -2)

        result = K.permute_dimensions(K.sum(weighted_input, axis=1),[0,2,1])

        if self.return_attention:
            return [result, a]
        return result

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], self.k, input_shape[-1]),
                    (input_shape[0], self.k, input_shape[1])]
        else:
            return input_shape[0], input_shape[-1], self.k]
