# fmri_convs

### setup  
  
expects following directories added:  
`raw_data/` - location of data

modify `dotrc.lua` to set the following variables based on your environment variables (e.g. $USER and $HOME):  
- `dotrc.save_dir` to output location for plots and trained networks  
- `dotrc.has_debugger` to indicate whether or not you have [`fb.debugger`](https://github.com/facebook/fblualib/blob/master/fblualib/debugger/README.md) installed  

### usage
  
#### training a neural network:

`th run_train.lua` train neural network with default parameters  

`th run_train.lua --help` shows available options for training a neural network and default settings 

#### examples
  
`th run_train.lua -float_precision -ms 20 -wake -network_type fully_connected` will train a fully connected logistic regression network (# input units --> # output classes) on WAKE data  
`th run_train.lua -float_precision -ms 20 -network_type deep_max_temp_conv -kernel_widths 5 -conv_strides 1 -num_conv_filters 2 -max_pool_widths 2 -max_pool_strides 2` will train a classifier with one temporal convolutional layer (kW = 5, stride = 1), followed by a max pooling layer with (w = 2, stride = 2), followed be a logistic regression on the default data type (sleep)    

#### notes:  
  
- I only ever use float precision (`-float_precision`) these days since it hasn't changed the results and saves time and memory. One day soon I'll change it so that's the default behavior.  
- the `-run_single_subj` option is likely to break things since I haven't been using it recently  
- the `-early_termination flag` hasn't been updated to work with the `-predict_subj` flag  
- the `-loso` flag doesn't do anything currently  
- the default behavior is to only plot and save our network at the end of all our training iterations (`-max_iterations`), to enable periodic plotting and save use the `-log_period_in_hours` flag.  
- the `-iterate_smoothing` flag is only helpful is you're running things on a cluster submitted in batch mode where each job gets it's own job number.  in that case, `-iterate_smoothing` would try different smoothing values for each unique job_num
- the `-show_network` flag only works if `-predict_subj` is set to true, otherwise, you can see the network printed out in the console output and don't need to set this flag.  
- simulated data is provided that either has no signal in it (`-simulated 2`) or easily decodeable data (`-simulated 1`)  
- `kfold_driver.lua:` isn't in working order just yet, needs to get updated to accept deep_max_temp_conv parameters  

#### applying a trained neural network  

`th run_apply_network.lua -float_precision -ms 20 -saved_net_path 'path/to/saved/net/generated/by/run_train.lua/net.net'`  

### TODO  
  
- describe structure of program  
- update and document usage of kfold_driver.lua  
- explicitly specify required packages  
- clean up hooks  
