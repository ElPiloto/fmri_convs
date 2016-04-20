sleep_eeg = require 'sleep_eeg.lua'
local M = {}


local function shouldLog(timer, log_period_in_hours)
	local timeElapsed = timer:time().real --in seconds
	timeElapsed = timeElapsed / 60 / 60 --convert to hours
	if timeElapsed >= log_period_in_hours then
		return true
	end
	return false
end

M.setClassificationOptimizationHooks = function(state, subj_data, args, cmdOptions)

  ------------------------------------------------------------------------
  --populate hooks (training iteration, training complete, periodic logging)

  --training iteration hooks
  --------------------------
  local trainConfMatrix = function(state)
    sleep_eeg.hooks.confusionMatrix(state, 'train', subj_data.classnames, args.subj_data.predict_subj and 1 or nil )
  end
  table.insert(args.training.trainingIterationHooks, trainConfMatrix)

  if not args.subj_data.do_kfold_split then
    local validConfMatrix = function(state)
      sleep_eeg.hooks.confusionMatrix(state, 'valid', subj_data.classnames, args.subj_data.predict_subj and 1 or nil )
    end
    table.insert(args.training.trainingIterationHooks, validConfMatrix)
  end
  if args.training.showTest then 
    local testConfHook = function(state) 
      sleep_eeg.hooks.confusionMatrix(state, 'test', subj_data.classnames, args.subj_data.predict_subj and 1 or nil )
    end
    table.insert(args.training.trainingIterationHooks, testConfHook)
  end

  --add subject confusion matrix if we're predicting subjects
  if args.subj_data.predict_subj then
    trainConfMatrix = function(state)
      sleep_eeg.hooks.confusionMatrix(state, 'train', subj_data.subj_ids, 2)
    end
    table.insert(args.training.trainingIterationHooks, trainConfMatrix)
    if not args.subj_data.do_kfold_split then
      validConfMatrix = function(state)
        sleep_eeg.hooks.confusionMatrix(state, 'valid', subj_data.subj_ids, 2)
      end
      table.insert(args.training.trainingIterationHooks, validConfMatrix)
    end
    if args.training.showTest then 
      local testConfHook = function(state) 
        sleep_eeg.hooks.confusionMatrix(state, 'test', subj_data.subj_ids, 2)
      end
      table.insert(args.training.trainingIterationHooks, testConfHook)
    end
  end

  --valid/test losses
  if not args.subj_data.do_kfold_split then
    table.insert(args.training.trainingIterationHooks, sleep_eeg.hooks.validLoss)
  end
  if args.training.showTest then
    table.insert(args.training.trainingIterationHooks, sleep_eeg.hooks.testLoss)
  end

  --add subset conf matrices for wake condition
  if args.subj_data.wake then
    --make a closure that will pass in the 'train' arg to a "subsetConfusionMatrix"
    --which only cares about performance on a subset of all possible classes
    local trainConfSubsetMatrix = function(state)
      sleep_eeg.hooks.subsetConfusionMatrix(state, 'train', subj_data.classnames, {1,2}, args.subj_data.predict_subj and 1 or nil)--only do faces and places
    end
    table.insert(args.training.trainingIterationHooks, trainConfSubsetMatrix)

    if not args.subj_data.do_kfold_split then
    --make a closure that will pass in the 'valid' arg to subsetConfusionMatrix
      local validConfSubsetMatrix = function(state)
        sleep_eeg.hooks.subsetConfusionMatrix(state, 'valid', subj_data.classnames, {1,2}, args.subj_data.predict_subj and 1 or nil)
      end
      table.insert(args.training.trainingIterationHooks, validConfSubsetMatrix)
    end

    if args.training.showTest then
      local testSubsetConfHook = function(state) 
        sleep_eeg.hooks.subsetConfusionMatrix(state, 'test', subj_data.classnames, {1,2}, args.subj_data.predict_subj and 1 or nil)
      end
      table.insert(args.training.trainingIterationHooks, testSubsetConfHook)
    end
  end
  --misc
  table.insert(args.training.trainingIterationHooks, sleep_eeg.hooks.logWeightToUpdateNormRatio)

  if args.training.iterationsDecreaseLR > 0 then
    table.insert(args.training.trainingIterationHooks, sleep_eeg.hooks.decrease_learning_rate)
  end

  --Training Completed Hooks
  --------------------------
  args.training.trainingCompleteHooks[1] = function(state)
    return sleep_eeg.hooks.randomClassAcc(state, subj_data.num_classes)
  end

  table.insert(args.training.trainingCompleteHooks, sleep_eeg.hooks.saveForRNGSweep)

  table.insert(args.training.trainingCompleteHooks, sleep_eeg.hooks.plotForRNGSweep)

  --if string.match(cmdOptions.network_type, 'max') and not string.match(cmdOptions.network_type, 'no_max') and not cmdOptions.predict_subj and args.network.num_conv_filters
	  --and cmdOptions.num_hidden_mult == 1 then
    --table.insert(args.training.trainingCompleteHooks, sleep_eeg.hooks.getDistributionOfMaxTimepoints)
  --end

  --Periodic Logging Hooks
  --------------------------
  args.training.periodicLogHooks[1] = sleep_eeg.hooks.plotForRNGSweep

  --if string.match(cmdOptions.network_type, 'max') and not string.match(cmdOptions.network_type, 'no_max') and not cmdOptions.predict_subj 
	  --and cmdOptions.num_hidden_mult == 1 then 
    --args.training.periodicLogHooks[2] =  sleep_eeg.hooks.getDistributionOfMaxTimepoints
  --end

  if not cmdOptions.dont_save_network then
    table.insert(args.training.periodicLogHooks, sleep_eeg.hooks.saveNetwork)
  end


  --Early Termination Hook
  --------------------------
  --make a closure for our early termination fn
  if cmdOptions.early_termination > 0 and cmdOptions.early_termination <= 1 then
    args.training.earlyTerminationFn = function(state)
      return sleep_eeg.terminators.trainAndValidAvgClassAccuracyHigh(state, cmdOptions.early_termination)
    end
  end


end

M.setRegressionOptimizationHooks = function(state, subj_data, args, cmdOptions)

  ------------------------------------------------------------------------
  --populate hooks (training iteration, training complete, periodic logging)

  --training iteration hooks
  --------------------------

  --add subject confusion matrix if we're predicting subjects
  if args.subj_data.predict_subj then
    trainConfMatrix = function(state)
      sleep_eeg.hooks.confusionMatrix(state, 'train', subj_data.subj_ids, 2)
    end
    table.insert(args.training.trainingIterationHooks, trainConfMatrix)
    if not args.subj_data.do_kfold_split then
      validConfMatrix = function(state)
        sleep_eeg.hooks.confusionMatrix(state, 'valid', subj_data.subj_ids, 2)
      end
      table.insert(args.training.trainingIterationHooks, validConfMatrix)
    end
    if args.training.showTest then 
      local testConfHook = function(state) 
        sleep_eeg.hooks.confusionMatrix(state, 'test', subj_data.subj_ids, 2)
      end
      table.insert(args.training.trainingIterationHooks, testConfHook)
    end
  end

  --valid/test losses
  if not args.subj_data.do_kfold_split then
    table.insert(args.training.trainingIterationHooks, sleep_eeg.hooks.validLoss)
  end
  if args.training.showTest then
    table.insert(args.training.trainingIterationHooks, sleep_eeg.hooks.testLoss)
  end

   --misc
  table.insert(args.training.trainingIterationHooks, sleep_eeg.hooks.logWeightToUpdateNormRatio)
  if args.training.iterationsDecreaseLR > 0 then
    table.insert(args.training.trainingIterationHooks, sleep_eeg.hooks.decrease_learning_rate)
  end

  --Training Completed Hooks
  --------------------------

  table.insert(args.training.trainingCompleteHooks, sleep_eeg.hooks.saveForRNGSweep)

  table.insert(args.training.trainingCompleteHooks, sleep_eeg.hooks.plotForRNGSweep)

  --Periodic Logging Hooks
  --------------------------
  args.training.periodicLogHooks[1] = sleep_eeg.hooks.plotForRNGSweep

  if not cmdOptions.dont_save_network then
    table.insert(args.training.periodicLogHooks, sleep_eeg.hooks.saveNetwork)
  end

end

M.train = function(fullState)
  assert(torch.type(fullState) == 'sleep_eeg.State')
  local optimizers = sleep_eeg.optimizers

  local options = fullState.args.training --shortcut to our args

  --TODO: We actually do want to save the state of our optimizer, b/c some
  --optimizers have their own internal state
  if not fullState.optimizer and not fullState.optimSettings then
    --make our optimizer
    local optim, optimSettings = optimizers.getOptimizer(options.optimName, options.learningRate)
    fullState:add('optimizer', optim,false)
    fullState:add('optimSettings', optimSettings, false)
  elseif utils.nilXOR(fullState.network, fullState.criterion) then
    error([[You've managed to save one, but NOT BOTH, of the following values:
          - state.optim
          - state.optimSettings
          We can't load just one because optim depends on optimSettings and 
        it's not easy to check if they match up']])
  end

    if not fullState.trainingIteration then
      fullState:add('trainingIteration',0,true)
    end

	if not fullState.timerSinceLastLog then
		fullState:add('timerSinceLastLog',torch.Timer(),false)
	end

    --actually run the optimizer
    local shouldTerminateEarly = false
    local start = torch.tic()
    print('Starting to train...')
    while fullState.trainingIteration < options.maxTrainingIterations 
      and (not shouldTerminateEarly) do

      fullState.trainingIteration = fullState.trainingIteration + 1

	  fullState.network:training() --added for dropout functionality
      optimizers.performTrainIteration(fullState)
	  fullState.network:evaluate()

      if #options.trainingIterationHooks > 0 then
        for hookIdx = 1, #options.trainingIterationHooks do
          options.trainingIterationHooks[hookIdx](fullState)
        end
      end

      if options.earlyTerminationFn then
        shouldTerminateEarly = options.earlyTerminationFn(fullState)
        if shouldTerminateEarly then
          print('Terminating early!')
        end
      end

      --garbage collect every 100 training iterations
      if fullState.trainingIteration % 10 == 0 then
        print('10 iterations took: ' .. torch.toc(start) .. 'secs')
        start = torch.tic()
        collectgarbage()
      end

	  if options.log_period_in_hours and #options.periodicLogHooks > 0 
		  and shouldLog(fullState.timerSinceLastLog, options.log_period_in_hours) then

		print('Executing periodic logging...')
	    for hookIdx = 1, #options.periodicLogHooks do
	      options.periodicLogHooks[hookIdx](fullState)
	    end
	    fullState.timerSinceLastLog:reset()

	  end
    end

    --finally, if we have any hooks to eecute after completion
    if #options.trainingCompleteHooks > 0 then
      for hookIdx = 1, #options.trainingCompleteHooks do
        options.trainingCompleteHooks[hookIdx](fullState)
      end
    end
  end

local initArgs = function()
  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Neural Networks for EEG')
  cmd:text()
  cmd:text('Options')
  cmd:option('-simulated', -1, '-1 = no sim data, 1 = basic, 2 = no signal, 3 = basic + noise (not implemented yet)')
  cmd:option('-cuda', false, 'use cuda')
  cmd:option('-num_folds', 10, 'num folds to split data into')
  cmd:option('-fold_num', -1, 'fold number to test on')
  cmd:option('-run_single_subj',false, 'run within subject analysis NOTE: currently not implemented')
  cmd:option('-wake',false, 'if false, run sleep else run wake')
  cmd:option('-wake_test',false, 'if true, run waketest ')
  cmd:option('-optim','adam', 'optimizer to use, supported optimizers = "sgd" or "adam"')
  cmd:option('-learning_rate', 1e-5, 'learning rate for optimizer')
  cmd:option('-max_iterations', 20000, 'max number of iterations to optimize for (can still terminate early)')
  cmd:option('-early_termination', -1, '-1 = no early termination, values between 0 and 1 will terminate optimization if training and validation classification accuracy exceed this value')
  cmd:option('-network_type', 'deep_max_temp_conv', 'network type to use, valid values = "fully_connected", "max_channel_conv" and , "deep_max_temp_conv", "rnn", "deep_max_spatial_conv", or "volumetric_conv"')
  cmd:option('-rnn_type', 'vanilla', 'valid values = "vanilla" for vanilla rnn with tanh nonlinearity, or "lstm" for lstm, only applicable if -network_type = "rnn"')
  cmd:option('-dropout_prob', -1, 'Probability of input dropout.')
  cmd:option('-num_hidden_mult', 1, 'Number of hidden units specified as a multiple of the number of output units e.g. "2" would yield numHiddenUnits = 2 * numOutputUnits')
  cmd:option('-num_hidden_layers', 1, 'Number of weights between layers, always at least 1 (input --> output), greater than 1 creates hidden layers')
  cmd:option('-config_name', '', 'what we want to call this configuration of arguments; dictates the name of the folder we save data to. leaving this empty will generate directory name based on arguments passed.')
  cmd:option('-subj_index', 0, 'subject index, not ID. only valid for run_single_subj = true')
  cmd:option('-double_precision', false, 'whether or not to load data and optimize using double precision. Otherwise, use single (float) ')
  cmd:option('-SO_locked', false, 'whether or not to lock to slow-oscillation (SO). only applies if -wake is NOT set')
  cmd:option('-log_period_in_hours', -1, 'how frequently we log things in periodicLogHooks. if <= 0, never call periodicLogHooks')
  cmd:option('-dont_save_network', false, 'do not save network periodically if this flag is specified')
  cmd:option('-show_test', false, 'only generate and save test accuracy if this is true')
  cmd:option('-percent_decrease_LR', 97, 'number between 1 and 99 to decrease learning rate; only if -iterations_decrease_LR flag greater than 0')
  cmd:option('-iterations_decrease_LR', 0, 'after how many iterations should we decrease our learning rate; 0 = constant learning rate')
  cmd:option('-predict_subj', false, 'whether or not we should additionally predict subjects')
  cmd:option('-shuffle_data', false, 'whether or not we should shuffle trials e.g. for generating a random permutation ')
  cmd:option('-predict_delta_memory', false, 'whether or not we should predict change in memory instead of stimulus identity. not compatible with -predict_subj flag')
  cmd:option('-weight_loss_function', false, 'whether or not we should weight training examples inversely proportional to their image presentation number (works for sleep only) ')
  cmd:option('-class_to_subj_loss_ratio', 2, 'how many times more we care about the class loss compared to the subj loss when -predict_subj is set')
  cmd:option('-ms', 20, 'how many ms per timebins for data in the temporal domain; currently only supports 4 and 20')
  cmd:option('-spatial_chans', false, 'if true, load data where channels are laid out in a 2d spatial arrangement; otherwise, channels are represented by their channel number in an arbitrary order')
  cmd:option('-spatial_scale', 10, "if -spatial_chans set, what's the scale (in arbitrary units) of the 2D grid channels live on. 10 = 17x17 grid, smaller number gives more resolution")
  cmd:option('-ERP_diff', false, 'whether or not to use ERP_diff, only supported for sleep ERP currently')
  cmd:option('-ERP_I', false, 'whether or not use ERP_I data')
  cmd:option('-min_presentations', -1, 'number of min presentations, only valid for sleep data; -1 will use all cue presentations')
  cmd:option('-max_presentations', -1, 'number of max presentations, only valid for sleep data; -1 will use all cue presentations')
  --temporal smoothing parameters
  cmd:option('-smooth_std', -1, "for max temp conv, should we smooth output of convolution (smooth_std > 0) and if so, what's the std of our gaussian?")
  cmd:option('-smooth_width', 5, "if we're smoothing (smooth_std > 0), how many non-zero elements do we have in our gaussian filter? must be odd number >= 3")
  cmd:option('-smooth_step', 1, "how many timepoints do we slide over after performing our convolution")
  cmd:option('-iterate_smoothing', false, "should we iterate smoothing values")
  cmd:option('-hidden_act_fn', 'relu', "activation function for hidden units; valid values: 'relu', 'sigmoid', 'tanh', 'lrelu' (leaky relu), 'prelu' (parametric relu)")
  cmd:option('-show_network', false, 'whether or not to save network graph to disk for predict_subj networks; doesnt work on della')
  cmd:option('-mini_batch_size', -1, 'max number of exaples per minibatch (-1 uses a single batch)')
  --deep conv v2 parameters: less safe, more explicit
  cmd:option('-kernel_widths', '1', 'comma-separated list of convolution kernel width for deep_max_temp_conv, use -1 to disable for a given layer')
  cmd:option('-max_pool_widths', '2', 'comma-separated list of max-pool widths, use -1 to disable for a given layer, 0 to max over entire input')
  cmd:option('-conv_strides', '1', 'comma-separated list of conv strides, use -1 to disable for a given layer')
  cmd:option('-max_pool_strides', '2', 'comma-separated list of max-pool strides, use -1 to disable for given layer')
  cmd:option('-num_conv_filters', '128', 'comma-separated list of num filters per conv layer')
  --this is for spatiotemporal convolutions: regular deep conv v2 parameters are used for the width and height of the spatial conv, and these params are
  --used for
  cmd:option('-temp_kernel_widths', '1', 'comma-separated list of TEMPORAL convolution kernel width for volumetric_conv, use -1 to disable for a given layer')
  cmd:option('-temp_max_pool_widths', '2', 'comma-separated list of TEMPORAL max-pool widths, use -1 to disable for a given layer, 0 to max over entire input')
  cmd:option('-temp_conv_strides', '1', 'comma-separated list of TEMPORAL conv strides, use -1 to disable for a given layer')
  cmd:option('-temp_max_pool_strides', '2', 'comma-separated list of TEMPORAL max-pool strides, use -1 to disable for given layer')
 
  cmd:text()
  opt = cmd:parse(arg)
  opt.float_precision = not opt.double_precision 
  print(opt)
  assert(opt.smooth_width >= 3, '-smooth_width must be >= 3')
  --assert(opt.max_pool_width_prcnt >= 0 and opt.max_pool_width_prcnt <= 1,
    --'max pool width prcnt must be b/w 0 and 1')
  return opt, cmd
end

M.generalDriver = function()
  --all cmd-line options:
  -- args.network_type 'max_temp_conv', 'no_max_temp_conv', 'fully_connected'
  local utils = sleep_eeg.utils
  local cmdOptions, cmdLine = initArgs()

  --all arguments we will ever need to run this function
  --create sim data
  args = {}
  args.rng_seed = '102387'--TODO: rng state is NOT saved right now
  args.float_precision = cmdOptions.float_precision
  args.iterate_smoothing = cmdOptions.iterate_smoothing
  args.miniBatchSize = cmdOptions.mini_batch_size
  args.weight_loss_function = cmdOptions.weight_loss_function

  if args.float_precision then
	  torch.setdefaulttensortype('torch.FloatTensor')
  end

  args.subj_data = {}


  --subj data arguments
  args.subj_data.sim_type = cmdOptions.simulated
  args.subj_data.num_folds = cmdOptions.num_folds
  args.subj_data.fold_num = cmdOptions.fold_num
  args.subj_data.percent_train = 100-(100/args.subj_data.num_folds)
  args.subj_data.do_kfold_split = true
  args.subj_data.run_single_subj = cmdOptions.run_single_subj
  args.subj_data.wake = cmdOptions.wake
  args.subj_data.wake_test = cmdOptions.wake_test
  args.subj_data.predict_subj = cmdOptions.predict_subj
  args.subj_data.predict_delta_memory = cmdOptions.predict_delta_memory
  args.subj_data.shuffle_data = cmdOptions.shuffle_data
  args.subj_data.SO_locked = cmdOptions.SO_locked
  args.subj_data.temporal_resolution = cmdOptions.ms
  args.subj_data.ERP_diff = cmdOptions.ERP_diff
  args.subj_data.ERP_I = cmdOptions.ERP_I
  args.subj_data.min_presentations = cmdOptions.min_presentations
  args.subj_data.max_presentations = cmdOptions.max_presentations
  args.subj_data.spatial_chans = cmdOptions.spatial_chans
  args.subj_data.spatial_scale = cmdOptions.spatial_scale
  args.subj_data.volumetric_conv = string.match(cmdOptions.network_type,'volumetric_conv') ~= nil

  if args.subj_data.wake and args.subj_data.wake_test then
	error('both -wake and -wake_test flags specified, but highlander (there can only be one)')
  end
  if cmdOptions.run_single_subj and cmdOptions.predict_subj then
    error("Can't specify -run_single_subj AND -predict_subj flags at the same time. Spoiler alert: it's always the same subject")
  end

  args.network = {}

  --let's populate any job specific args we're sweeping over, because we need to get
  --subject_idx before we can populate subj_data
  sleep_eeg.utils.populateArgsBasedOnJobNumber(args)

  if cmdOptions.rng_seed then
    args.rng_seed = cmdOptions.rng_seed
  end
  if cmdOptions.subj_index ~= 0 then
    args.subj_data.subj_idx = cmdOptions.subj_index
  end
  if cmdOptions.fold_num ~= -1 then 
    args.subj_data.fold_num = cmdOptions.fold_num
  end
  args.subj_data.filename = sleep_eeg.utils.getDataFilenameFromArgs(args)

  --with the subj_data args specified, we go ahead and load the subject data 
  --because other argument values for the network and confusion matix depend on 
  --values that get loaded by subj_data
  local subj_data 
  if args.subj_data.run_single_subj then
    subj_data = sleep_eeg.SingleSubjData({filename = args.subj_data.filename, 
      do_kfold_split = args.subj_data.do_split_loso, percent_valid = args.subj_data.percent_valid, 
	  percent_train = args.subj_data.percent_train, subj_data_args = args.subj_data})
  else
    subj_data = sleep_eeg.CVBySubjData({filename = args.subj_data.filename, 
      do_kfold_split = args.subj_data.do_kfold_split, percent_valid = nil, percent_train =  args.subj_data.percent_train, 
      use_subjects_as_targets = args.subj_data.predict_subj, num_folds = args.subj_data.num_folds, 
      fold_number = args.subj_data.fold_num, subj_data_args = args.subj_data})
  end
  print('Loaded data from: ' .. sleep_eeg.utils.fileToURI(args.subj_data.filename))

  --network args
  local numOut = subj_data.num_classes
  if args.subj_data.predict_subj then
    numOut = subj_data.num_classes + subj_data.num_subjects
  end
  args.network.cuda = cmdOptions.cuda
  args.network.numHiddenUnits = cmdOptions.num_hidden_mult * numOut
  args.network.numHiddenLayers = cmdOptions.num_hidden_layers
  args.network.num_output_classes = subj_data.num_classes
  args.network.dropout_prob = cmdOptions.dropout_prob
  args.network.class_to_subj_loss_ratio = cmdOptions.class_to_subj_loss_ratio
  args.network.smooth_std = cmdOptions.smooth_std
  args.network.smooth_width = cmdOptions.smooth_width
  args.network.smooth_step = cmdOptions.smooth_step
  args.network.network_type = cmdOptions.network_type
  args.network.rnn_type = cmdOptions.rnn_type
  args.network.hidden_act_fn = cmdOptions.hidden_act_fn
  args.network.show_network = cmdOptions.show_network
  args.network.predict_delta_memory = cmdOptions.predict_delta_memory --this is both a subject AND a network arg
  --deep_conv params v2
  if string.find(args.network.network_type, 'conv') then
    args.network.convString = 'kW' .. cmdOptions.kernel_widths .. 'dW' ..
      cmdOptions.conv_strides .. 'pW' .. cmdOptions.max_pool_widths .. 'dPW' ..
	  cmdOptions.max_pool_strides .. 'numFilts' .. cmdOptions.num_conv_filters
    if string.match(args.network.network_type, 'volumetric_conv') then
      args.network.convString = args.network.convString .. 'kW' .. cmdOptions.temp_kernel_widths .. 'dW' ..
      cmdOptions.temp_conv_strides .. 'pW' .. cmdOptions.temp_max_pool_widths .. 'dPW' .. cmdOptions.temp_max_pool_strides 
    end
  else
    args.network.convString = ''
  end

  args.network.kernel_widths, args.network.conv_strides, 
    args.network.max_pool_widths, args.network.max_pool_strides,
	  args.network.num_conv_filters,  args.network.temp_kernel_widths, 
    args.network.temp_conv_strides, args.network.temp_max_pool_widths, 
    args.network.temp_max_pool_strides = sleep_eeg.utils.extractAndCheckConvOptions(cmdOptions)

  --training args, used by sleep_eeg.drivers.train()
  args.training = {}
  --if period <= 0, set to nil so we never try to execute periodicLogHooks
  args.training.log_period_in_hours = cmdOptions.log_period_in_hours > 0 and cmdOptions.log_period_in_hours or nil
  args.training.optimName = cmdOptions.optim
  args.training.learningRate = cmdOptions.learning_rate
  args.training.maxTrainingIterations =  cmdOptions.max_iterations
  args.training.showTest = cmdOptions.show_test
  args.training.iterationsDecreaseLR = cmdOptions.iterations_decrease_LR
  args.training.percentDecreaseLR = cmdOptions.percent_decrease_LR
  args.training.trainingIterationHooks = {} -- populated below
  args.training.earlyTerminationFn = nil --populated below just put this here so that, all args are easy to see
  args.training.trainingCompleteHooks = {}
  args.training.periodicLogHooks = {}

  if cmdOptions.config_name == '' then
    args.driver_name = sleep_eeg.utils.makeConfigName(args,cmdOptions) --if no config_name specified, make from args
  else
    args.driver_name = cmdOptions.config_name
  end

  if not args.network.predict_delta_memory then
    M.setClassificationOptimizationHooks(state, subj_data, args, cmdOptions)
  else
    M.setRegressionOptimizationHooks(state, subj_data, args, cmdOptions)
  end

  args.save_file = utils.saveFileNameFromDriversArgs(args,args.driver_name)
  --end args definition 
  ------------------------------------------------------------------------
  if args.network.cuda then
	  subj_data:cuda()
  end

  --this will get reload state if args.save_file already exists
  --otherwise, just keeps saving there
  state = sleep_eeg.State(args.save_file)

  --set random seed
  if not state.rngState then
    torch.manualSeed(args.rng_seed)
  else
    torch.setRNGState(state.rngState)
  end

  if not state.args then
    state:add('args', args, true)
  end

  if not state.data then
    state:add('data',subj_data,false)
  end

  -- if we want, we can also do convolutions across channels instead of temporally,
  -- if that's the case, let's transpose our data and just recycle the maxTempConv code
  if cmdOptions.network_type == 'max_channel_conv' then
	  subj_data:swapTemporalAndChannelDims()
  end

  --create our network and criterion  (network type determines criterion type)
  if not state.network and not state.criterion then
    print('making network started...')
    print(args.network)
    local network, criterion = {},{}
    if cmdOptions.network_type == 'deep_max_temp_conv' or cmdOptions.network_type == 'max_channel_conv' then 
      network, criterion = sleep_eeg.models.createDeepMaxTempConvClassificationNetwork(
        state.data:getTrainData(), args.network.numHiddenUnits, 
      args.network.numHiddenLayers, state.data.num_classes, 
		  args.network.dropout_prob, args.subj_data.predict_subj, 
		  state.data.num_subjects,args.network)
    elseif cmdOptions.network_type == 'deep_max_spatial_conv' then 
      network, criterion = sleep_eeg.models.createSpatialConvClassificationNetwork( 
        state.data:getTrainData(), args.network.numHiddenUnits, 
        args.network.numHiddenLayers, state.data.num_classes, 
        args.network.dropout_prob, args.subj_data.predict_subj, 
        state.data.num_subjects, args.network)
    elseif cmdOptions.network_type == 'volumetric_conv' then 
      network, criterion = sleep_eeg.models.createVolumetricConvClassificationNetwork( 
        state.data:getTrainData(), args.network.numHiddenUnits, 
        args.network.numHiddenLayers, state.data.num_classes, 
        args.network.dropout_prob, args.subj_data.predict_subj, 
        state.data.num_subjects, args.network)
    elseif cmdOptions.network_type == 'max_temp_conv' then 
      network, criterion = sleep_eeg.models.createMaxTempConvClassificationNetwork( 
        state.data:getTrainData(), args.network.numHiddenUnits, 
        args.network.numHiddenLayers, state.data.num_classes, 
        args.network.dropout_prob, args.subj_data.predict_subj, 
        state.data.num_subjects,args.network)
    elseif cmdOptions.network_type == 'fully_connected' then
      network, criterion = sleep_eeg.models.createFullyConnectedNetwork(
	  	state.data:getTrainData(), args.network.numHiddenUnits, 
      args.network.numHiddenLayers, state.data.num_classes, 
      args.network.dropout_prob, args.subj_data.predict_subj,
      state.data.num_subjects, args.network)

    elseif cmdOptions.network_type == 'rnn' then 
      network, criterion = sleep_eeg.models.createRnnNetwork( 
        state.data:getTrainData(), args.network.numHiddenUnits, 
        args.network.numHiddenLayers, state.data.num_classes, 
        args.network.dropout_prob, args.subj_data.predict_subj, 
        state.data.num_subjects,args.network)

    end
    print('making network finished...')
    state:add('network',network, true)
    state:add('criterion',criterion, true)
  elseif utils.nilXOR(state.network, state.criterion) then
    error([[You've managed to save one, but NOT BOTH, of the following values:
          - state.network
          - state.criterion
          We can't load just one because making the network determines the type of 
      criterion you need.]])
  end

  --we load params and gradParams together
  if not state.params and not state.gradParams then
    local params, gradParams = state.network:getParameters()
    state:add('params',params,true)
    state:add('gradParams',gradParams,true)
  elseif utils.nilXOR(state.params, state.gradParams) then
    error([[You've managed to save one, but NOT BOTH, of the following values:
          - state.params
          - state.gradParams
          We can't load just one because network:getParameters() manipulates both. 
          And you don't really want to call network:getParameters() on the same 
      network twice cause spooky things happen']])
  end

  --little output about our network
  print('-------------------------------------------------------------------')
  print('Network information: ')
  print(state.network)
  print('With a total of ' ..state.params:numel() .. ' parameters')

  --finally call the optimizer
  M.train(state)

end

M.generalDriver()
