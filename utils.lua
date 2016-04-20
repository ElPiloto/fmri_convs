require 'paths'
local M = {}

M.fileExists = function(fname)
	local file = io.open(fname,'r')
	if file ~= nil then
		io.close(file)
		return true
	else
		return false
	end
end

M.nilXOR = function(obj1,obj2)
	--returns true if exactly one of the objects is not nil
	return not (obj1) == not not obj2
end

local test_nilXOR = function()
	assert(not M.nilXOR(1,1))
	assert(not M.nilXOR(nil,nil))
	assert(M.nilXOR(nil,1))
	assert(M.nilXOR(1,nil))
end
test_nilXOR()


M.normalizeData = function(data,mean_, std_)
  local numExamples = data:size(1)
  local std = std_ or data:std(1)
  local mean = mean_ or data:mean(1)

  -- actually normalize here
  for i = 1, numExamples do
     data[i]:add(-mean)
     data[i]:cdiv(std)
  end

  return mean, std
end


M.populateArgsBasedOnJobNumber = function(args)
  --the prefixed on the key names "e.g. Z in 'Z_subj_idx'"
  --specify the reverse of the order we want to iterate through
  --so A is the first dimension that gets iterated, Z is the last
  --obviously, this limits us to 26 options
	require 'os'
	local gridOptions = {}
 
  --if we want to run a single subject, then we also have to specify which
  --subj_idx we want to run for this job
  if args.subj_data.run_single_subj then 
    --we have 33 subjects
    gridOptions['Z_subj_idx'] = torch.range(1,33):long():totable()
  end

  if args.subj_data.do_kfold_split then
    gridOptions['A_fold_num'] = torch.range(1,args.subj_data.num_folds):long():totable()
  end

  local smooth_stds = torch.range(0.15, 0.35, 0.1):totable()
  local smooth_widths = torch.range(3,7,2):totable()
  local smooth_steps = torch.range(1,7,2):totable()

  if args.iterate_smoothing then
	  gridOptions['B_smooth_std'] = smooth_stds
	  gridOptions['C_smooth_width'] = smooth_widths
	  gridOptions['D_smooth_step'] = smooth_steps
  end

  gridOptions['Y_rng_seed'] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40}

	local job_number = os.getenv('SLURM_ARRAY_TASK_ID')
	if not job_number then
		job_number = 0
	else
		job_number = tonumber(job_number)-1
	end

	local options = sleep_eeg.param_sweep.grid({id = job_number},gridOptions)
	args.rng_seed = options['Y_rng_seed']

  if args.subj_data.run_single_subj then
	  args.subj_data.subj_idx = options['Z_subj_idx']
  end

  if args.subj_data.do_kfold_split then
    args.subj_data.fold_num = options['A_fold_num']
  end

  if args.iterate_smoothing then
    args.network.smooth_std = options['B_smooth_std']
    args.network.smooth_width = options['C_smooth_width']
	--make sure we don't step larger than the width
    args.network.smooth_step = math.min(options['D_smooth_step'],
	  options['C_smooth_width'])
  end
end

M.saveFileNameFromDriversArgs = function(args,base_name)
	--build file path
	local driverPrefix = base_name 
	local gitCommitHash = M.getGitCommitNumAndHash()
	local rngSeedString = 'rng_' .. args.rng_seed 
    local learningRateString = 'learnRate_' .. string.format("%.0e",args.training.learningRate)
	local fullPath = paths.concat(dotrc.save_dir,driverPrefix, gitCommitHash)
	if not paths.dir(fullPath) then
		paths.mkdir(fullPath)
	end
	local smoothString =  ''
	if args.network.smooth_std > 0 then
		if args.network.network_type == 'max_temp_conv' or 
			args.network.network_type == 'max_channel_conv' then
			smoothString = 'smooth' .. string.format('%.2f', args.network.smooth_std)
			.. 'Width' .. tonumber(args.network.smooth_width) ..  'Step' .. 
		args.network.smooth_step
	end
	end

	--build filename
  local filename 
  if args.subj_data.shuffle_data then
    filename  = 'SHUFFLE' 
  else
    filename = ''
  end
  filename = filename .. smoothString .. learningRateString .. '_' ..  rngSeedString 

  if not args.subj_data.do_kfold_split then --not applicable
    filename = filename .. 'train' .. args.subj_data.percent_train .. 'valid' .. args.subj_data.percent_valid
  end

  if args.training.iterationsDecreaseLR > 0 then
    filename = filename .. '_' .. args.training.percentDecreaseLR .. 'lrDecayEvery' .. args.training.iterationsDecreaseLR
  end

  filename = filename ..'.th7'

	local fullFilename = paths.concat(fullPath,filename)
	print(fullFilename)
	return  fullFilename
end


M.copyArgsRemoveUnsupportedMatioTypes = function (source)
	local tablex = require 'pl.tablex'
	local target = tablex.deepcopy(source)
	target.training.trainingIterationHooks = nil
	target.training.earlyTerminationFn = nil
	target.training.trainingCompleteHooks = nil
	return target
end




M.replaceTorchSaveWithNetSave = function(torchFilename, suffix)
	local dir = paths.dirname(torchFilename)
	local baseFilename = paths.basename(torchFilename,'.th7')
	if suffix then
		return paths.concat(dir,baseFilename .. suffix .. '.net')
	else
		return paths.concat(dir,baseFilename .. '.net')
	end
end


M.replaceTorchSaveWithMatSave = function(torchFilename)
	local dir = paths.dirname(torchFilename)
	local baseFilename = paths.basename(torchFilename,'.th7')
	return paths.concat(dir,baseFilename .. '.mat')
end

M.replaceTorchSaveWithEpsSave = function(torchFilename, suffix)
	local dir = paths.dirname(torchFilename)
	local baseFilename = paths.basename(torchFilename,'.th7')
	if suffix then
		return paths.concat(dir,baseFilename .. suffix .. '.eps')
	else
		return paths.concat(dir,baseFilename .. '.eps')
	end
end

M.replaceTorchSaveWithPngSave = function(torchFilename, suffix)
	local dir = paths.dirname(torchFilename)
	local baseFilename = paths.basename(torchFilename,'.th7')
	if suffix then
		return paths.concat(dir,baseFilename .. suffix .. '.png')
	else
		return paths.concat(dir,baseFilename .. '.png')
	end
end

M.insertDirToSaveFile = function(torchFilename, dirToAdd)
	local dir = paths.concat(paths.dirname(torchFilename), dirToAdd)
	local filename = paths.basename(torchFilename)
	if not paths.dir(dir) then
		paths.mkdir(dir)
	end
	return paths.concat(dir, filename)
end

M.matioHelper = function(filename, varNames)
  local matio = require 'matio'
  matio.use_lua_strings = true
  local loaded = {}
  for _, var in pairs(varNames) do
    loaded[var] = matio.load(filename,var)
  end
  return loaded
end

M.getUniqueStrings = function(strTable)
  local numUniqueStrings = 0
  -- keys are unique strings, values are which key number they are
  local uniqueStringIDs = {} 
  --list of which uniqueStringIDs each example belongs to
  local uniqueStringIdxs = {}
  --an ordered list of unique strings
  local uniqueStrings = {}
  --num strings: 
  local size 
  if torch.type(strTable) == 'table' then
    size = #strTable
  elseif string.find(torch.type(strTable),'Tensor') then
    assert(strTable:nDimension() == 1, 
        'this function only works with tables or tensors with one dimension')
    size = strTable:size(1)
  end

  for strIdx = 1, size do
    local currentStr = strTable[strIdx]
    if uniqueStringIDs[currentStr] == nil then -- new string
      numUniqueStrings = numUniqueStrings + 1
      uniqueStringIDs[currentStr] = numUniqueStrings 
      uniqueStrings[numUniqueStrings] = currentStr
    end
    uniqueStringIdxs[strIdx] = uniqueStringIDs[currentStr]
  end
  return uniqueStrings, uniqueStringIdxs
end

--applies to a tensor or a table of tensors
M.getDataFromTableOrTensor = function(data,idxs)
  local canCopyInBulk = false
  if idxs:max()-idxs:min()+1 == idxs:numel() then
    canCopyInBulk = true
  end

  local newData = {}
  local shouldUntableData = false
  if torch.type(data) ~= 'table' then
    data = {data}
    shouldUntableData = true
  end

  for dataIdx = 1, #data do
    if canCopyInBulk then
      newData[dataIdx] = data[dataIdx][{{idxs:min(), idxs:max()}}]
    else
      local size = data[dataIdx]:size():totable()
      size[1] = idxs:numel()
      newData[dataIdx] = 
          torch.Tensor():typeAs(data[dataIdx]):resize(unpack(size))
      for exampleIdx = 1, idxs:numel() do
        newData[dataIdx][exampleIdx] = data[dataIdx][idxs[exampleIdx]]
      end
    end
  end

  if shouldUntableData then
    return newData[1]
  end

  return newData
end

M.splitDataBasedOnLabels = function(data, labels)
  --[
  -- labels can be either a tensor of unique numbers or a table of unique 
  -- elements
  --]
  assert(data:size(1) == labels:size(1))--one label per datapoint
  assert(data:nDimension() == 2)
  local numExamples = data:size(1)
  local numFeatures = data:size(2)

  local uniqueIDs, uniqueIDXs = M.getUniqueStrings(labels)
  uniqueIDXs = torch.Tensor{uniqueIDXs}

  local numClasses = #uniqueIDs
  local maxExamples = 0
  for classIdx = 1, numClasses do
    local numClassExamples = uniqueIDXs:eq(classIdx):sum()
    assert(numClassExamples > 0, 'something fishy going on here')
    if numClassExamples > maxExamples then
      maxExamples = numClassExamples
    end
  end

  local splitData = 
      torch.Tensor():typeAs(data):resize(numClasses,maxExamples,numFeatures):zero()

  for classIdx = 1, numClasses do
    local classIndicator = uniqueIDXs:eq( classIdx )
    local numClassExamples = classIndicator:sum()
    classIndicator = torch.range(1, numExamples):long()[classIndicator]

    splitData[{classIdx,{1,numClassExamples},{}}] = 
        M.getDataFromTableOrTensor(data, classIndicator)
    --here we fill in any "extra" slots with the last value we have
    if numClassExamples < splitData:size(2) then
      for exampleIdx = numClassExamples+1, splitData:size(2) do
        splitData[{classIdx,exampleIdx}] = 
            splitData[{classIdx,numClassExamples}]
      end
    end
  end

  return splitData
end

M.getGitCommitNumAndHash = function()
	--we cache this result as soon as this script gets require'd because io.popen calls fork() which essentially copies the current process's allocated memory
	if not M.__gitCommitNumAndHash then
		local io = require 'io'
		--short version of git commit hash
		local handle = io.popen('git log -n 1 --pretty=format:"%h" ')
		local commitHash = handle:read("*a")
		handle:close()
		--tells us which commit number we are on, basically so that
		--we know the order of our commits
		local handle = io.popen('git rev-list --count HEAD')
		print(handle)
		local commitNum = handle:read("*a")
		handle:close()
		M.__gitCommitNumAndHash = commitNum:gsub("\n","") .. '_' .. commitHash:gsub("\n","")
	end
	return M.__gitCommitNumAndHash
end

M.getGitCommitNumAndHash()

M.ghettoClearStateSequential = function(model)
  --only works for nn.Container
  for m = 1, #model.modules do
    if model.modules[m].clearState then
      model.modules[m]:clearState()
    else
      model.modules[m].output = nil
      model.modules[m].gradInput = nil
      --for max pooling modules
      if model.modules[m].indices then
        model.modules[m].indices = nil
      end
    end
  end
  --finally clear out the module itself
  model.output = nil
  model.gradInput = nil
end

--this is because some networks were "delflated" 
--(using ghettoClearStateSequential) which sets fields  to nil so that they can
--be saved in a reasonable amount of space on disk.  however, once we load these
--networks, they error if the fields set to nil are not restated.
M.ghettoReinflateModel = function(model)
  for m = 1, #model.modules do
    if model.modules[m].output == nil then
      model.modules[m].output = torch.Tensor()
    end
    if model.modules[m].gradInput == nil then
      model.modules[m].gradInput = nil
    end
    --for max pooling modules
    if torch.type(model.modules[m]) == 'nn.TemporalMaxPooling' then
      if model.modules[m].indices == nil then
        model.modules[m].indices = torch.Tensor()
      end
    end
  end
  --just in case we saved a network in training mode
  model:evaluate()
end

M.fileToURI = function(file)
  --makes it so that when we print this in gnome-terminal,
  --it gets recognized as URI which we can click and open
  --from the terminal!
  return 'file://' .. file
end

M.extractAndCheckConvOptions = function(cmdOptions)

  local isVolumetric = false
  if string.match(cmdOptions.network_type, 'volumetric_conv') then
    isVolumetric = true
  end
  local kernel_widths = string.split(cmdOptions.kernel_widths,',')
  local conv_strides = string.split(cmdOptions.conv_strides,',')
  local max_pool_widths = string.split(cmdOptions.max_pool_widths,',')
  local max_pool_strides = string.split(cmdOptions.max_pool_strides,',')
  local num_conv_filters = string.split(cmdOptions.num_conv_filters,',')

  local temp_kernel_widths
  local temp_conv_strides
  local temp_max_pool_widths
  local temp_max_pool_strides
  if isVolumetric then
    temp_kernel_widths = string.split(cmdOptions.temp_kernel_widths,',')
    temp_conv_strides = string.split(cmdOptions.temp_conv_strides,',')
    temp_max_pool_widths = string.split(cmdOptions.temp_max_pool_widths,',')
    temp_max_pool_strides = string.split(cmdOptions.temp_max_pool_strides,',')
  end


  if isVolumetric then
    assert(#kernel_widths == #conv_strides and #kernel_widths == #max_pool_widths
      and #kernel_widths == #max_pool_strides and #kernel_widths == #temp_kernel_widths
      and #kernel_widths == #temp_max_pool_strides and #kernel_widths == #temp_conv_strides
      and #kernel_widths == #temp_max_pool_widths, [[Unequal number of conv/max_pool 
          parameters supplied. If we have 3 kernel_widths (meaning 3 conv layers), then
          we need to have 3 conv_strides, 3 max_pool_widths and 3 max_pool strides, 
          3 temp kernel widths, and 3 num_conv_filters.]])
  else
    assert(#kernel_widths == #conv_strides and #kernel_widths == #max_pool_widths
      and #kernel_widths == #max_pool_strides, [[Unequal number of conv/max_pool 
          parameters supplied. If we have 3 kernel_widths (meaning 3 conv layers), then
          we need to have 3 conv_strides, 3 max_pool_widths and 3 max_pool strides. 
          and 3 num_conv_filters.]])
  end

  for idx = 1, #kernel_widths do
    kernel_widths[idx] = tonumber(kernel_widths[idx])
    conv_strides[idx] = tonumber(conv_strides[idx])
    max_pool_widths[idx] = tonumber(max_pool_widths[idx])
    max_pool_strides[idx] = tonumber(max_pool_strides[idx])
    num_conv_filters[idx] = tonumber(num_conv_filters[idx])
    if isVolumetric then
      temp_kernel_widths[idx] = tonumber(temp_kernel_widths[idx])
      temp_conv_strides[idx] = tonumber(temp_conv_strides[idx])
      temp_max_pool_widths[idx] = tonumber(temp_max_pool_widths[idx])
      temp_max_pool_strides[idx] = tonumber(temp_max_pool_strides[idx])
    end
  end
  return kernel_widths, conv_strides, max_pool_widths, max_pool_strides,
    num_conv_filters, temp_kernel_widths, temp_conv_strides, temp_max_pool_widths, temp_max_pool_strides
end

M.makeConfigName = function(args, cmdOptions)
  
  local snake_to_CamelCase = function (s)
    return s:gsub("_%w", function (u) return u:sub(2,2):upper() end)
  end
  local function firstToUpper(s)
    return s:gsub("^%l", string.upper)
  end

  local name = snake_to_CamelCase(cmdOptions.network_type) .. firstToUpper(cmdOptions.optim)

  if cmdOptions.network_type == 'rnn' then
    name = snake_to_CamelCase(cmdOptions.rnn_type .. '_' .. cmdOptions.network_type) .. firstToUpper(cmdOptions.optim)
  else
    name = snake_to_CamelCase(cmdOptions.network_type) .. firstToUpper(cmdOptions.optim)
  end
  
  name = name .. args.network.convString

  if cmdOptions.dropout_prob > 0 then
  	name = name .. 'Drop' .. tostring(cmdOptions.dropout_prob)
  end
  --simulated data indicator
  if cmdOptions.simulated >= 0 then
    simString = 'Sim' .. tostring(cmdOptions.simulated)
    name = name .. simString
  end
  if cmdOptions.wake then
	name = name .. 'Wake'
  elseif cmdOptions.wake_test then
	name = name .. 'WakeTest'
  else
    if cmdOptions.SO_locked then
	  name = name .. 'SOsleep'
    else 
	  name = name .. 'Sleep'
    end
  end
--per subject indicator
  if cmdOptions.run_single_subj then 
    name = name .. 'PerSubj'
  end
  if cmdOptions.float_precision then
	  name = name .. 'Single'
  end
  if cmdOptions.predict_subj then
    name = name .. 'PredSubj' .. cmdOptions.class_to_subj_loss_ratio .. 'to1'
  end
  if cmdOptions.predict_delta_memory then
    name = name .. 'PredDeltaMem' 
  end
  if cmdOptions.weight_loss_function then
    name = name .. 'WeightLoss' 
  end
  name = name .. cmdOptions.num_hidden_mult .. 'xHidden' .. cmdOptions.num_hidden_layers 
  name = name .. '_' .. cmdOptions.ms .. 'ms'

  if cmdOptions.num_folds then
    name = name .. 'Fold' .. args.subj_data.fold_num .. 'of' .. args.subj_data.num_folds
  end

  if args.subj_data.ERP_diff then
    name = name .. 'Diff'
  end

  --for maxPresentations, we just append "maxPres"
  if args.subj_data.max_presentations >= 1 then
    name = name .. '_maxPres' .. args.subj_data.max_presentations
  end
  
  --for "ERP_I", we have to replaced cuelocked with cuelocked_I
  if args.subj_data.ERP_I then
    name = name .. "_ERP_I"
  end

  --finally append 
  name = name .. firstToUpper(cmdOptions.hidden_act_fn)

  if cmdOptions.mini_batch_size ~= -1 then
    name = name .. 'Mini' .. cmdOptions.mini_batch_size
  end

  if cmdOptions.spatial_chans then
    name = name .. 'Spatial' .. cmdOptions.spatial_scale
  end

  if cmdOptions.cuda then
	  name = name .. 'Cuda'
  end

  return name
end

M.getDataFilenameFromArgs = function(args)
  local fileName = ''
  if args.subj_data.wake then
    fileNameRoot = 'wake_ERP_cuelocked_all_' .. args.subj_data.temporal_resolution .. 'ms_1000'
  elseif args.subj_data.wake_test then 
    fileNameRoot = 'waketest_all_ERP_cuelocked_all_' .. args.subj_data.temporal_resolution .. 'ms_1000'
  else
    if args.subj_data.SO_locked then
      fileNameRoot = 'sleep_ERP_SOlocked_all_phase_SO1'
    else
      fileNameRoot = 'sleep_ERP_cuelocked_all_' .. args.subj_data.temporal_resolution .. 'ms_1000'
    end
  end

  --for ERP_diff, we just replace "ERP" with "ERP_diff"
  if args.subj_data.ERP_diff then
    fileNameRoot = fileNameRoot:gsub('ERP','ERP_diff')
  end

  --for min/maxPresentations, we just append "min/maxPres"
  if args.subj_data.min_presentations >= 1 then
    fileNameRoot = fileNameRoot .. '_minPres' .. args.subj_data.min_presentations
  end
  if args.subj_data.max_presentations >= 1 then
    fileNameRoot = fileNameRoot .. '_maxPres' .. args.subj_data.max_presentations
  end
  
  --for "ERP_I", we have to replaced cuelocked with cuelocked_I
  if args.subj_data.ERP_I then
    fileNameRoot = fileNameRoot:gsub('cuelocked','cuelocked_I')
  end

  if args.subj_data.spatial_chans then
    fileNameRoot = fileNameRoot .. '_spatial' .. args.subj_data.spatial_scale
  end

  if args.float_precision then
	  fileNameRoot = fileNameRoot .. 'Single'
  end


  if args.subj_data.sim_type > 0 then
    if args.subj_data.sim_type == 1 or args.subj_data.sim_type == 2 then
      fileNameRoot = fileNameRoot .. '_sim' ..  args.subj_data.sim_type 
    else
      error('Unknown or unimplemented simulated data type.  Only valid values are sim_type = 1 and sim_type == 2, sim_type == 3 yet to be implemented')
    end
  end


  fileName = './torch_exports/' .. fileNameRoot .. '.mat'

  return fileName
end

M.getMiniBatchTrials = function(shuffledExamples, miniBatchIdx, miniBatchSize)
  local startIdx = (miniBatchIdx-1)*miniBatchSize + 1
  local endIdx = math.min(startIdx + miniBatchSize -1, shuffledExamples:numel())
  return shuffledExamples[{{startIdx,endIdx}}]
end

M.getNumMiniBatches = function(numExamples, miniBatchSize)
  if miniBatchSize == -1 then
    return 1
  else
    return math.ceil(numExamples/miniBatchSize)
  end
end

M.indexIntoTensorOrTableOfTensors = function(source, dim, idxs)
  if torch.isTensor(source) then
    return source:index(dim, idxs)
  end
  --otherwise we have a table of tensors
 local result = {}
 for k,v in pairs(source) do
   result[k] = v:index(dim, idxs)
 end
 return result
end

return M
