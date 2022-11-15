def compute_network_output(net_input, all_weights, all_biases):

  #Storing number of layers
  K = len(all_weights) -1

  #Creating lists for pre-activations and activations at each layer
  all_f = [None] * (K+1)
  all_h = [None] * (K+1)

  #For convenience, we'll set 
  # all_h[0] to be the input, and all_f[K] will be the output
  all_h[0] = net_input

  #Calculating all_f[0...K-1] and all_h[1...K]
  for layer in range(K):
      
      #Updating preactivations and activations at this layer
      all_f[layer] = all_biases[layer] + (all_weights[layer] @ all_h[layer])
      all_h[layer+1] = ReLU(all_f[layer])

  #Computing the output from the last hidden layer
  all_f[K] = all_biases[K] + (all_weights[K] @ all_h[K])

  #Setting the output
  net_output = all_f[K]

  return net_output, all_f, all_h