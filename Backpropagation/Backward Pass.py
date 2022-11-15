#Defining indicator function for backward pass
def indicator_function(x):
  x_in = np.array(x)
  x_in[x_in>=0] = 1
  x_in[x_in<0] = 0
  return x_in


def backward_pass(all_weights, all_biases, all_f, all_h, y):
  #The derivatives of the loss with respect to the weights and biases - stored in lists
  all_dl_dweights = [None] * (K+1)
  all_dl_dbiases = [None] * (K+1)
  #The derivatives of the loss with respect to the activation and preactivations - stored in lists
  all_dl_df = [None] * (K+1)
  all_dl_dh = [None] * (K+1)
  
  
  #all_h[0] is the net input and all_f[k] in the network output.

  # Compute derivatives of loss with respect to network output
  all_dl_df[K] = np.array(d_loss_d_output(all_f[K],y))


  #Working backwards through the network.
  for layer in range(K,-1,-1):
    
    #Calculating the derivatives of biases at layer from all_dl_df[K]
    all_dl_dbiases[layer] = all_dl_df[layer]

    #Calculating the derivatives of weight at layer from all_dl_df[K] and all_h[K]
    all_dl_dweights[layer] = all_dl_df[layer] @ np.transpose(all_h[layer])

    #Calculating the derivatives of activations from weight and derivatives of next preactivations
    all_dl_dh[layer] = np.transpose(all_weights[layer]) @ (all_dl_df[layer])

    if layer > 0:
     
      #Calculating the derivatives of the pre-activation f with respect to activation h
      all_dl_df[layer-1] = indicator_function(all_f[layer-1]) * all_dl_dh[layer]

  return all_dl_dweights, all_dl_dbiases