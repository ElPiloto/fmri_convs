
local M = {}

local initArgs = function()
  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Neural Networks for EEG')
  cmd:text()
  cmd:text('Options')
  cmd:option('-simulated', -1, '-1 = no sim data, 1 = basic, 2 = no signal, 3 = basic + noise (not implemented yet)')
  cmd:option('-percent_train', 65, 'percent of data to use for training')
  cmd:option('-cuda', false, 'use cuda')
  cmd:option('-percent_valid', 20, 'percent of data to use for validation')
  cmd:option('-loso',false, 'leave-one-subject-out validation? NOTE: currently not implemented')
  cmd:option('-run_single_subj',false, 'run within subject analysis')
  cmd:option('-wake',false, 'if false, run sleep else run wake')
  cmd:option('-wake_test',false, 'if true, run waketest ')
  cmd:option('-optim','adam', 'optimizer to use, supported optimizers = "sgd" or "adam"')
  cmd:option('-learning_rate', 1e-5, 'learning rate for optimizer')
  cmd:option('-max_iterations', 20000, 'max number of iterations to optimize for (can still terminate early)')
  cmd:option('-early_termination', -1, '-1 = no early termination, values between 0 and 1 will terminate optimization if training and validation classification accuracy exceed this value')
  cmd:option('-network_type', 'max_temp_conv', 'network type to use, valid values = "max_temp_conv", "no_max_temp_conv", and "fully_connected"') 
  cmd:option('-dropout_prob', -1, 'Probability of input dropout.')
  cmd:option('-volumetric_conv', false, 'Whether or not data shuld be prepared for a volumetric convolution')
  cmd:option('-num_hidden_mult', 1, 'Number of hidden units specified as a multiple of the number of output units e.g. "2" would yield numHiddenUnits = 2 * numOutputUnits')
  cmd:option('-num_hidden_layers', 1, 'Number of weights between layers, always at least 1 (input --> output), greater than 1 creates hidden layers')
  cmd:option('-config_name', '', 'what we want to call this configuration of arguments; dictates the name of the folder we save data to. leaving this empty will generate directory name based on arguments passed.')
  cmd:option('-subj_index', 0, 'subject index, not ID. only valid for run_single_subj = true')
  cmd:option('-double_precision', false, 'whether or not to load data and optimize using double precision. Otherwise, use single (float) ')
  cmd:option('-SO_locked', false, 'whether or not to lock to slow-oscillation (SO). only applies if -wake is NOT set')
  cmd:option('-log_period_in_hours', -1, 'how frequently we log things in periodicLogHooks. if <= 0, never call periodicLogHooks')
  cmd:option('-dont_save_network', false, 'do not save network periodically if this flag is specified')
  cmd:option('-show_test', false, 'only generate and save test accuracy if this is true')
  cmd:option('-show_all', false, 'only generate and save accuracy on all data if this is true')
  cmd:option('-predict_subj', false, 'whether or not we should additionally predict subjects')
  cmd:option('-saved_net_path','','file path to saved network we want to apply')
  cmd:option('-ms', 20, 'how many ms per timebins for data in the temporal domain; currently only supports 4 and 20')
  cmd:option('-ERP_diff', false, 'whether or not to use ERP_diff, only supported for sleep ERP currently')
  cmd:option('-ERP_I', false, 'whether or not use ERP_I data')
  cmd:option('-spatial_chans', false, 'if true, load data where channels are laid out in a 2d spatial arrangement; otherwise, channels are represented by their channel number in an arbitrary order')
  cmd:option('-spatial_scale', 10, "if -spatial_chans set, what's the scale (in arbitrary units) of the 2D grid channels live on. 10 = 17x17 grid, smaller number gives more resolution")
  cmd:option('-min_presentations', -1, 'number of min presentations, only valid for sleep data; -1 will use all cue presentations')
  cmd:option('-max_presentations', -1, 'number of max presentations, only valid for sleep data; -1 will use all cue presentations')
  cmd:text()
  opt = cmd:parse(arg)
  opt.float_precision = not opt.double_precision 
  print(opt)
  assert(opt.saved_net_path ~= '', '-saved_net_path is not optional')
  return opt, cmd
end

--will throw error if network doesn't fit
local testDataFitsNetwork = function(subj_data, network, args)
	assert(network,'Failed to load network - came up nil')
	
	--currently only predict_subj networks have nn.gModule flag
	if args.subj_data.predict_subj then
		assert(torch.type(network) == 'nn.gModule', "Looks like this network was not built without the -predict_subj flag, but you want to apply it to predict subjects. Change flag or the network")
	else
		assert(torch.type(network) ~= 'nn.gModule', "Looks like this network was built with the -predict_subj flag, but you don't want to apply it to predict subjects. Change flag or the network")
	end

end

--currently for ERP temporal data, wake and wakeTest have 250 timepoints 
--(0 - 1000ms), while sleep has 50-1000ms, so we end up padding the remaining
--timepoints with zeros)
local padSleepDataIfNeeded = function(args, subj_data)
	local fname = args.saved_net_path 
	local isWakeOrWakeTestNet = string.match(fname,'Wake')
	local isSleepData = not (args.subj_data.wake or args.subj_data.wake_test)

	--this means we have to pad our sleep data, we currently do not handle the 
	--case where we need to trim wake data to apply a sleep classifier because 
	--i don't see any reason why we would want to do that
	if isWakeOrWakeTestNet and isSleepData and subj_data._train_data:size(2) == 238 then
		local function appendData(data) 
			local batch = data:size(1)
			local features = data:size(3)
			local numTimePointsToAdd = 250 - 238
			local paddedData = torch.Tensor():typeAs(subj_data._train_data):resize(batch,numTimePointsToAdd,features):zero()
			return torch.cat(data,paddedData,2)
		end
		subj_data._train_data = appendData(subj_data._train_data)
		subj_data._valid_data = appendData(subj_data._valid_data)
		subj_data._test_data = appendData(subj_data._test_data)
	end
end


M.run = function()
  require 'nn'
  require 'nngraph'
  local cmdOptions, cmdLine = initArgs()

  --all arguments we will ever need to run this function
  --create sim data
  args = {}
  args.rng_seed = '102387'--TODO: rng state is NOT saved right now
  args.float_precision = cmdOptions.float_precision

  if args.float_precision then
	  torch.setdefaulttensortype('torch.FloatTensor')
  end

  args.subj_data = {}


  --subj data arguments
  args.subj_data.sim_type = cmdOptions.simulated
  args.subj_data.percent_train = cmdOptions.percent_train
  args.subj_data.percent_valid = cmdOptions.percent_valid
  args.subj_data.do_split_loso = cmdOptions.loso
  args.subj_data.run_single_subj = cmdOptions.run_single_subj
  args.subj_data.wake = cmdOptions.wake
  args.subj_data.wake_test = cmdOptions.wake_test
  args.subj_data.predict_subj = cmdOptions.predict_subj
  args.cuda = cmdOptions.cuda
  args.subj_data.temporal_resolution = cmdOptions.ms
  args.subj_data.SO_locked = cmdOptions.SO_locked
  args.subj_data.ERP_diff = cmdOptions.ERP_diff
  args.subj_data.ERP_I = cmdOptions.ERP_I
  args.subj_data.min_presentations = cmdOptions.min_presentations
  args.subj_data.max_presentations = cmdOptions.max_presentations
  args.subj_data.spatial_chans = cmdOptions.spatial_chans
  args.subj_data.spatial_scale = cmdOptions.spatial_scale
  args.subj_data.volumetric_conv = cmdOptions.volumetric_conv
  if args.subj_data.wake and args.subj_data.wake_test then
	error('both -wake and -wake_test flags specified, but highlander (there can only be one)')
  end
  if cmdOptions.run_single_subj and cmdOptions.predict_subj then
    error("Can't specify -run_single_subj AND -predict_subj flags at the same time. Spoiler alert: it's always the same subject")
  end
  args.subj_data.filename = sleep_eeg.utils.getDataFilenameFromArgs(args)

  --let's populate any job specific args we're sweeping over, because we need to get
  --subject_idx before we can populate subj_data
  sleep_eeg.utils.populateArgsBasedOnJobNumber(args)

  if cmdOptions.rng_seed then
    args.rng_seed = cmdOptions.rng_seed
  end
  if cmdOptions.subj_index ~= 0 then
    args.subj_data.subj_idx = cmdOptions.subj_index
  end

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
      do_kfold_split = args.subj_data.do_split_loso, percent_valid = args.subj_data.percent_valid, 
	  percent_train = args.subj_data.percent_train, use_subjects_as_targets = args.subj_data.predict_subj, 
    subj_data_args = args.subj_data})
  end
  print('Loaded data from: ' .. sleep_eeg.utils.fileToURI(args.subj_data.filename))

  if args.cuda then
	  subj_data:cuda()
  end

  args.saved_net_path = opt.saved_net_path
  print('Loading network from: ' .. args.saved_net_path)

  temp = torch.load(args.saved_net_path)
  trainingIteration = temp.trainingIteration
  network = temp.net
  temp = nil

  --test everything
  testDataFitsNetwork(subj_data, network, args)

  --pad sleep data with zeros because it's smaller than the wake data
  padSleepDataIfNeeded(args, subj_data)

  --this has to be done because we nil out certain required fields when we save
  --the network
  sleep_eeg.utils.ghettoReinflateModel(network)

  local metrics = {'train', 'train_and_valid', 'valid'}

  --finally apply the network
  if cmdOptions.show_test then
    table.insert(metrics,'test')
  end
  if cmdOptions.show_all then
    table.insert(metrics,'all')
  end
  local completed_metrics = M.getClassAccuracy(network, subj_data, metrics, args.subj_data)

  print(completed_metrics)
  return completed_metrics, subj_data, network, args
end

--sometimes our data is a tensor, sometimes our data is a table of tensors,
--either way we need to combine them
local combineTensorsOrTable = function(data1, data2)
	local data1Type = torch.type(data1) 
	local data2Type = torch.type(data2)
	assert(data1Type == data2Type, 'Mismatched types: ' .. data1Type .. ',' ..
	  data2Type)
	if torch.isTensor(data1) then
		--concat along batch (first) dimension
		return torch.cat(data1, data2, 1)
	elseif data1Type == 'table' then --we have a table of tensors
		local combined = {}
		for k,v in pairs(data1) do
			assert(data2[k], 'data2 does not have same table fields as data2')
			assert(torch.isTensor(v) and torch.isTensor(data2[k]),
			  'Only tensors and table of tensors allowed')

			--concat along batch (first) dimension
			combined[k] = torch.cat(v, data2[k], 1)
		end
		return combined
	end
end

M.getClassAccuracy = function(network, subj_data, metrics, subj_args)
	require 'optim'

	--metrics:
	--.train
	--.valid
	--.train_and_valid
	--.test
	--.all
	local function getDataAndTargets(metric_name)
		local data, targets = {},{}
		if metric_name == 'train' then
			data = subj_data:getTrainData()
			target = subj_data:getTrainTargets()
		elseif metric_name == 'valid' then
			data = subj_data:getValidData()
			target = subj_data:getValidTargets()
		elseif metric_name == 'test' then
			data = subj_data:getTestData()
			target = subj_data:getTestTargets()
		elseif metric_name == 'train_and_valid' or metric_name == 'all' then
			--train
			data = subj_data:getTrainData()
			target = subj_data:getTrainTargets()
			--concat valid
			data = combineTensorsOrTable(data, subj_data:getValidData())
			target = combineTensorsOrTable(target, subj_data:getValidTargets())
			--conditionally concat test 
			if metric_name == 'all' then
			  data = combineTensorsOrTable(data, subj_data:getTestData())
			  target = combineTensorsOrTable(target, subj_data:getTestTargets())
			end
		end
		return data, target
	end

	--predict_subj has everything in form {classOut, subjOut}
	--not predict_subj just has classOut
	local function getClassOutput(output, outputTableIndex)
	  if outputTableIndex then
	    return output[outputTableIndex]
	  else
		return output
	  end
	end

	local completed_metrics = {}
	local function makeAndSetConfusionMatrix(modelOut, targets, metric_name,
	  labels, outputTableIndex)

	    --1st dim = batch, 2nd = classes
	    local networkOutClasses = modelOut:size(2)

		if networkOutClasses == #labels then
			--create and update confusion matrix
			local confMatrix = optim.ConfusionMatrix(labels)
			confMatrix:zero()

			confMatrix:batchAdd(modelOut, targets)
			confMatrix:updateValids()

			--add an entry with keyname metric_name, for each outputTableIndex
			local this_metric = completed_metrics[metric_name] or {}
			this_metric[outputTableIndex] = {}
			this_metric[outputTableIndex].classAcc = confMatrix.totalValid
			this_metric[outputTableIndex].confMat = confMatrix.mat
			completed_metrics[metric_name] = this_metric
		end

		--additionally, if we have a network with 5 outputs, then let's add an 
		--entry for confusion subset matrix
    if networkOutClasses == 5 then
      local dummyLabels = {labels[1], labels[2], 'Class3', 'Class4', 'Class5'}
      confMatrix = optim.SubsetConfusionMatrix(dummyLabels, {1,2})
      confMatrix:zero()

      confMatrix:batchAdd(modelOut, targets)
      confMatrix:updateValids()
      metric_name = metric_name .. 'Subset'

      local this_metric = completed_metrics[metric_name] or {}
      this_metric[outputTableIndex] = {}
      this_metric[outputTableIndex].classAcc = confMatrix.totalValid
      this_metric[outputTableIndex].confMat = confMatrix.mat
      completed_metrics[metric_name] = this_metric
    end

  end

	for _,metric_name in pairs(metrics) do
	  local data, targets = getDataAndTargets(metric_name)

	  if subj_args.predict_subj then
      local allModelOut = network:forward(data) 
      for outputTableIndex = 1, 2 do
        local labels = subj_data.classnames
        if outputTableIndex == 2 then
          labels = subj_data.subj_ids
        end

        modelOut = getClassOutput(allModelOut, outputTableIndex)
        local splitTarget = getClassOutput(targets, outputTableIndex)

        makeAndSetConfusionMatrix(modelOut, splitTarget, metric_name, labels,
          outputTableIndex)

      end
	  else
			local labels = subj_data.classnames

		    local modelOut = getClassOutput(network:forward(data), nil)
			local splitTarget = getClassOutput(targets, nil)

			makeAndSetConfusionMatrix(modelOut, splitTarget, metric_name,
			  labels, 0)
	  end

	end
	return completed_metrics
end

return M


