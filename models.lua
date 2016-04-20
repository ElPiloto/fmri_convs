require 'nn'

local M = {}
M.deprecated = {} --deprecated

M.getConvOutputWidth = function(inWidth, kernelWidth, stride)
    print('Input: ', inWidth, 'kW: ', kernelWidth, 'stride: ', stride, 'out:', math.floor((inWidth - kernelWidth)/stride) + 1)
    return math.floor((inWidth - kernelWidth)/stride) + 1
end
--supported activationFunctions for hidden units and what we call them
M.activationFns = {['relu'] = nn.ReLU, ['tanh'] = nn.Tanh, ['sigmoid'] = nn.Sigmoid, 
  ['prelu'] = nn.PReLU, ['lrelu'] = nn.LeakyReLU}

M.createRnnNetwork = function(egInputBatch, numHiddenUnits, numHiddenLayers,
    numOutputClasses,  dropout_prob, predict_subj, numSubjects, 
    net_args)
  
  assert(require 'torch-rnn', 'Failed to load "torch-rnn" something is up')
  --assert(not net_args.cuda, '-cuda for rnn not yet supported!!!')
  assert(not predict_subj, 'predict_subj for rnn not supported')

  if net_args.cuda then 
    require 'cutorch'
    require 'cunn'
  end

  local rnn_module
  if net_args.rnn_type == 'vanilla' then
    rnn_module = nn.VanillaRNN
  elseif net_args.rnn_type == 'lstm' then
    rnn_module = nn.LSTM
  end
	local numTimePoints = egInputBatch:size(2)
	local numInputUnits = egInputBatch:size(3)
	local numTotalFeatures = numTimePoints * numInputUnits
  local model = nn.Sequential()
  if dropout_prob > 0 then
    model:add(nn.Dropout(dropout_prob))
  end
  local numPrevUnits = numInputUnits
  for hidden_idx = 1, numHiddenLayers do
    model:add(rnn_module(numPrevUnits, numHiddenUnits))
    numPrevUnits = numHiddenUnits
  end
  model:add(nn.Select(2,numTimePoints)) --grab last timepoint from  time dimension (2)


  local criterion
  if not net_args.predict_delta_memory then
    model:add(nn.Linear(numPrevUnits,numOutputClasses))
    model:add(nn.LogSoftMax())
    criterion = nn.ClassNLLCriterion()
  else
    model:add(nn.Linear(numPrevUnits,1))
    criterion = nn.MSECriterion(true)
  end

  if net_args.cuda then
    model:cuda()
    model:insert(nn.Copy('torch.CudaTensor',torch.getdefaulttensortype()),
      #model.modules+1)
  end

  return model, criterion

end

--expect egInputBatch to have dimensions = [examples, time, features]
M.createFullyConnectedNetwork = function(egInputBatch, numHiddenUnits, 
		numHiddenLayers, numOutputClasses, dropout_prob, predict_subj, 
		numSubjects, net_args)

	if net_args.cuda then
	  require 'cunn'
	end
	local numX = egInputBatch:size(2)
	local numY = egInputBatch:size(3)
	local numZ = egInputBatch:size(4)
	local numTotalFeatures = numX * numY * numZ
	assert(egInputBatch and numHiddenUnits and numHiddenLayers and numOutputClasses)
	assert(numHiddenLayers >= 0)
	dropout_prob = dropout_prob or -1
    local hiddenActivationFn = M.activationFns[net_args.hidden_act_fn] or nn.ReLU --defaults to ReLU
    if predict_subj then
		require 'nngraph'
		nngraph.setDebug = true

		local input = nn.Identity()()
		local prev = {}
		if dropout_prob > 0 then
			prev = nn.Dropout(dropout_prob)(input)
		else
			prev = input
		end
		prev = nn.View(-1):setNumInputDims(3)(prev)
		local toClasses = {}
		local toSubjects = {}
		if numHiddenLayers > 1 then
			local lastLayer = numTotalFeatures
			for hiddenLayerIdx = 1, numHiddenLayers-1 do
				prev = nn.Linear(lastLayer,numHiddenUnits)(prev)
				prev = hiddenActivationFn()(prev)
				lastLayer = numHiddenUnits
			end

			--now we split
			toClasses = nn.Linear(numHiddenUnits,numOutputClasses)(prev)
			toSubjects = nn.Linear(numHiddenUnits, numSubjects)(prev)
		else

			toClasses = nn.Linear(numTotalFeatures,numOutputClasses)(prev)
			toSubjects = nn.Linear(numTotalFeatures, numSubjects)(prev)
		end
		toClasses = nn.LogSoftMax()(toClasses)
		toSubjects = nn.LogSoftMax()(toSubjects)

		if net_args.cuda then
			toClasses = nn.Copy('torch.CudaTensor',torch.getdefaulttensortype())(toClasses)
			toSubjects = nn.Copy('torch.CudaTensor',torch.getdefaulttensortype())(toSubjects)
		end

		model = nn.gModule({input},{toClasses, toSubjects})

		if net_args.cuda then
			for moduleIdx = 1, #model.modules do
				local module = model.modules[moduleIdx]
				if torch.type(module) ~= 'nn.Copy' then
					module:cuda()
					print(module)
				end
			end
		end

		criterion = nn.ParallelCriterion()
		--weight subjects two times as important
		criterion:add(nn.ClassNLLCriterion(),net_args.class_to_subj_loss_ratio)
		criterion:add(nn.ClassNLLCriterion(),1)

		if net_args.show_network then
		  local start = torch.tic()
		  model:forward(egInputBatch[{{1,2},{},{}}])
		  print('2 examples took: ', torch.toc(start), 'secs')
		  graph.dot(model.fg, 'mlp','fully_connected')
		end

		--model:forward(egInputBatch)
		--graph.dot(model.fg, 'mlp','test2')
    assert(not net_args.predict_delta_memory, '-predict_subj and -predict_delta_memory not currently supported, although babytown easy to add if curious')
		return model, criterion
	else
		local model = nn.Sequential()
		if dropout_prob > 0 then
			model:add(nn.Dropout(dropout_prob))
		end
		model:add(nn.View(-1):setNumInputDims(3)) --flatten

		if numHiddenLayers > 1 then
			local lastLayer = numTotalFeatures
			for hiddenLayerIdx = 1, numHiddenLayers-1 do
				model:add(nn.Linear(lastLayer,numHiddenUnits))
				model:add(hiddenActivationFn())
				lastLayer = numHiddenUnits
			end

		end

		--finally logsoftmax gives us 1 numOutputClasses-way classifier

    local criterion
    if not net_args.predict_delta_memory then
			model:add(nn.Linear(numTotalFeatures,numOutputClasses))
      model:add(nn.LogSoftMax())
      criterion = nn.ClassNLLCriterion()
    else
			model:add(nn.Linear(numTotalFeatures,1))
      criterion = nn.MSECriterion(true)
    end
		if net_args.cuda then
			model:cuda()
			model:insert(nn.Copy('torch.CudaTensor',torch.getdefaulttensortype()),
			  #model.modules+1)
		end


		return model, criterion
	end

end

--expect egInputBatch to have dimensions = [examples, time, features]
M.createSpatialConvClassificationNetwork = function(...)
  local args, egInputBatch, numHiddenUnits, numPostConvHiddenLayers, 
      numOutputClasses, dropout_prob, predict_subj, numSubjects, net_args = dok.unpack(
      {...},
      'createSpatialConvClassificationNetwork',
      'Make a convolution ',
      {arg='egInputBatch',type='Tensor',help='', req=true},
      {arg='numHiddenUnits',type='number',help='num filters in conv and how ' .. 
        ' many hidden units in subsequent hidden layers', req=true},
      {arg='numPostConvHiddenLayers',type='number',help='number of hidden ' .. 
        'layers excluding the output filters we get when we do our conv', 
        req=true},
      {arg='numOutputClasses',type='number',help='', req=false, default=nil},
      {arg='dropout_prob',type='number',help='', req=false, default=-1},
      {arg='predict_subj',type='number',
        help='whether or not to predict subjects as well as classes', req=false, default=false},
      {arg='numSubjects',type='number',help='only applies if predict_subj is true', 
        req=false, default=-1},
      {arg='net_args',type='table',help='', req=true}
  )
  local hiddenActivationFn = M.activationFns[net_args.hidden_act_fn] or nn.ReLU --defaults to ReLU

  if net_args.cuda then
	  require 'cunn'
  end

	local numTimePoints = egInputBatch:size(2)
  local width = egInputBatch:size(3)
  local height = egInputBatch:size(4)

  print(numTimePoints, width, height)
	assert(egInputBatch and numHiddenUnits and numPostConvHiddenLayers)
	--if we're not going to take the max after our convolution, which collapses
	--the number of output features, then we have to have at least one hidden
	--layer
	assert(numPostConvHiddenLayers > 0)
	numOutputClasses = numOutputClasses or 2

  if predict_subj then
		require 'nngraph'
		nngraph.setDebug = true

		local input = nn.Identity()()
		local prev = {}
		if dropout_prob > 0 then
			prev = nn.Dropout(dropout_prob)(input)
		else
			prev = input
		end
     
  local prevOutputWidth = width
  local prevOutputHeight = height
  local prevNumFilters = numTimePoints

	for conv_idx = 1, #net_args.kernel_widths do
		local added_conv = false
		local kernel_width = net_args.kernel_widths[conv_idx]
		local conv_stride = net_args.conv_strides[conv_idx]
		local max_pool_width = net_args.max_pool_widths[conv_idx]
		local max_pool_stride = net_args.max_pool_strides[conv_idx]
		local num_conv_filters = net_args.num_conv_filters[conv_idx]

		--check conv params okay
		assert(prevOutputWidth >= kernel_width, string.format([[Specified conv
		  kernel width = %d for layer: %d, but the input to that layer has 
		  width %d]], conv_idx, kernel_width, prevOutputWidth))
		assert(prevOutputHeight >= kernel_width, string.format([[Specified conv
		  kernel width = %d for layer: %d, but the input to that layer has 
		  width %d]], conv_idx, kernel_width, prevOutputHeight))
		assert(conv_stride > 0, "Can't have conv_stride = 0")
		assert(num_conv_filters > 0, "Can't have num_conv_filters = 0")

		prev = nn.SpatialConvolution(prevNumFilters, num_conv_filters, 
		  kernel_width, kernel_width, conv_stride, conv_stride)(prev)
    prev = hiddenActivationFn()(prev)
		--update our widths and filters
		prevOutputWidth = M.getConvOutputWidth(prevOutputWidth, kernel_width, 
		  conv_stride)
		prevOutputHeight = M.getConvOutputWidth(prevOutputHeight, kernel_width, 
		  conv_stride)
		prevNumFilters = num_conv_filters

		if max_pool_width == 0 then --special code for max over entire input
			max_pool_width = prevOutputWidth
			max_pool_stride = 1
		end

		if max_pool_width > 1 and max_pool_stride > 0 then
			prev = nn.SpatialMaxPooling(max_pool_width, max_pool_width, 
        max_pool_stride, max_pool_stride)(prev)
		  prevOutputWidth = M.getConvOutputWidth(prevOutputWidth, 
			  max_pool_width, max_pool_stride)
		  prevOutputHeight = M.getConvOutputWidth(prevOutputHeight, 
			  max_pool_width, max_pool_stride)
		end

	end

    local prevLayerOutputs =  prevOutputHeight*prevOutputWidth*prevNumFilters --from the convNet
    prev = nn.View(-1):setNumInputDims(3)(prev)

    for hiddenLayerIdx = 2, numPostConvHiddenLayers do
      prev = nn.Linear(prevLayerOutputs,numHiddenUnits)(prev)
      prev = hiddenActivationFn()(prev)
      prevLayerOutputs = numHiddenUnits
    end

    --finally go from shared hidden to individual output paths
    local toClasses = nn.Linear(prevLayerOutputs,numOutputClasses)(prev)
    local toSubjects = nn.Linear(prevLayerOutputs, numSubjects)(prev)
    toClasses = nn.LogSoftMax()(toClasses)
    toSubjects = nn.LogSoftMax()(toSubjects)

    if net_args.cuda then
		toClasses = nn.Copy('torch.CudaTensor',torch.getdefaulttensortype())(toClasses)
		toSubjects = nn.Copy('torch.CudaTensor',torch.getdefaulttensortype())(toSubjects)
	end

    model = nn.gModule({input},{toClasses, toSubjects})

	if net_args.cuda then
		for moduleIdx = 1, #model.modules do
			local module = model.modules[moduleIdx]
			if torch.type(module) ~= 'nn.Copy' then
				module:cuda()
				print(module)
			end
	  end
	end

    criterion = nn.ParallelCriterion()
    --weight subjects two times as important
		criterion:add(nn.ClassNLLCriterion(),net_args.class_to_subj_loss_ratio)
		criterion:add(nn.ClassNLLCriterion(),1)

    if net_args.show_network then
      local start = torch.tic()
      model:forward(egInputBatch[{{1,2},{},{}}])
      print('2 examples took: ', torch.toc(start), 'secs')
      graph.dot(model.fg, 'mlp','deep_max_temp_conv')
    end

    assert(not net_args.predict_delta_memory, '-predict_subj and -predict_delta_memory not currently supported, although babytown easy to add if curious')
    return model, criterion

  else

    local model = nn.Sequential()
    if dropout_prob > 0 then
      model:add(nn.Dropout(dropout_prob))
    end

    local prevOutputWidth = width
    local prevOutputHeight = height
    local prevNumFilters = numTimePoints

    for conv_idx = 1, #net_args.kernel_widths do
      local added_conv = false
      local kernel_width = net_args.kernel_widths[conv_idx]
      local conv_stride = net_args.conv_strides[conv_idx]
      local max_pool_width = net_args.max_pool_widths[conv_idx]
      local max_pool_stride = net_args.max_pool_strides[conv_idx]
      local num_conv_filters = net_args.num_conv_filters[conv_idx]


      --check conv params okay
      assert(prevOutputWidth >= kernel_width, string.format([[Specified conv
              kernel width = %d for layer: %d, but the input to that layer has 
        width %d]], conv_idx, kernel_width, prevOutputWidth))
      assert(prevOutputHeight >= kernel_width, string.format([[Specified conv
              kernel width = %d for layer: %d, but the input to that layer has 
        width %d]], conv_idx, kernel_width, prevOutputHeight))
      assert(conv_stride > 0, "Can't have conv_stride = 0")
      assert(num_conv_filters > 0, "Can't have num_conv_filters = 0")

      model:add(nn.SpatialConvolution(prevNumFilters, num_conv_filters, 
        kernel_width, kernel_width, conv_stride, conv_stride))
      model:add(hiddenActivationFn())
      --update our widths and filters
      prevOutputWidth = M.getConvOutputWidth(prevOutputWidth, kernel_width, 
        conv_stride)
      prevOutputHeight = M.getConvOutputWidth(prevOutputHeight, kernel_width, 
        conv_stride)
      prevNumFilters = num_conv_filters

      if max_pool_width == 0 then --special code for max over entire input
        max_pool_width = prevOutputWidth
        max_pool_stride = 1
      end

      if max_pool_width > 1 and max_pool_stride > 0 then
        model:add(nn.SpatialMaxPooling(max_pool_width, max_pool_width, 
          max_pool_stride, max_pool_stride))
        prevOutputWidth = M.getConvOutputWidth(prevOutputWidth, 
          max_pool_width, max_pool_stride)
        prevOutputHeight = M.getConvOutputWidth(prevOutputHeight, 
          max_pool_width, max_pool_stride)
      end

    end

    local prevLayerOutputs =  prevOutputHeight*prevOutputWidth*prevNumFilters --from the convNet

    model:add(nn.View(-1):setNumInputDims(3)) 

    for hiddenLayerIdx = 2, numPostConvHiddenLayers do
      model:add(nn.Linear(prevLayerOutputs,numHiddenUnits))
      model:add(nn.hiddenActivationFn())
      prevLayerOutputs = numHiddenUnits
    end

    --go from last hidden layer to number of classes

    --finally logsoftmax gives us 1 numOutputClasses-way classifier
    local criterion
    if not net_args.predict_delta_memory then
      model:add(nn.Linear(prevLayerOutputs,numOutputClasses))
      model:add(nn.LogSoftMax())
      criterion = nn.ClassNLLCriterion()
    else
      model:add(nn.Linear(prevLayerOutputs,1))
      criterion = nn.MSECriterion(true)
    end

    if net_args.cuda then
      model:cuda()
      model:insert(nn.Copy('torch.CudaTensor',torch.getdefaulttensortype()),
        #model.modules+1)
    end

    return model, criterion
  end
end

--expect egInputBatch to have dimensions = [examples, time, features]
M.createVolumetricConvClassificationNetwork = function(...)
  local args, egInputBatch, numHiddenUnits, numPostConvHiddenLayers, 
      numOutputClasses, dropout_prob, predict_subj, numSubjects, net_args = dok.unpack(
      {...},
      'createVolumetricConvClassificationNetwork',
      'Make a convolution ',
      {arg='egInputBatch',type='Tensor',help='', req=true},
      {arg='numHiddenUnits',type='number',help='num filters in conv and how ' .. 
        ' many hidden units in subsequent hidden layers', req=true},
      {arg='numPostConvHiddenLayers',type='number',help='number of hidden ' .. 
        'layers excluding the output filters we get when we do our conv', 
        req=true},
      {arg='numOutputClasses',type='number',help='', req=false, default=nil},
      {arg='dropout_prob',type='number',help='', req=false, default=-1},
      {arg='predict_subj',type='number',
        help='whether or not to predict subjects as well as classes', req=false, default=false},
      {arg='numSubjects',type='number',help='only applies if predict_subj is true', 
        req=false, default=-1},
      {arg='net_args',type='table',help='', req=true}
  )
  local hiddenActivationFn = M.activationFns[net_args.hidden_act_fn] or nn.ReLU --defaults to ReLU

  if net_args.cuda then
	  require 'cunn'
  end

	local numTimePoints = egInputBatch:size(3)
  local width = egInputBatch:size(4)
  local height = egInputBatch:size(5)

  print(numTimePoints, width, height)
	assert(egInputBatch and numHiddenUnits and numPostConvHiddenLayers)
	--if we're not going to take the max after our convolution, which collapses
	--the number of output features, then we have to have at least one hidden
	--layer
	assert(numPostConvHiddenLayers > 0)
	numOutputClasses = numOutputClasses or 2

  if predict_subj then
		require 'nngraph'
		nngraph.setDebug = true

		local input = nn.Identity()()
		local prev = {}
		if dropout_prob > 0 then
			prev = nn.Dropout(dropout_prob)(input)
		else
			prev = input
		end
     
  local prevOutputWidth = width
  local prevOutputHeight = height
  local prevOutputTime = numTimePoints
  local prevNumFilters = 1

	for conv_idx = 1, #net_args.kernel_widths do
		local added_conv = false
		local kernel_width = net_args.kernel_widths[conv_idx]
		local conv_stride = net_args.conv_strides[conv_idx]
		local max_pool_width = net_args.max_pool_widths[conv_idx]
		local max_pool_stride = net_args.max_pool_strides[conv_idx]
		local num_conv_filters = net_args.num_conv_filters[conv_idx]
		local temp_kernel_width = net_args.temp_kernel_widths[conv_idx]
		local temp_conv_stride = net_args.temp_conv_strides[conv_idx]
		local temp_max_pool_width = net_args.temp_max_pool_widths[conv_idx]
		local temp_max_pool_stride = net_args.temp_max_pool_strides[conv_idx]

		--check conv params okay
		assert(prevOutputWidth >= kernel_width, string.format([[Specified conv
		  kernel width = %d for layer: %d, but the input to that layer has 
		  width %d]], conv_idx, kernel_width, prevOutputWidth))
		assert(prevOutputHeight >= kernel_width, string.format([[Specified conv
		  kernel width = %d for layer: %d, but the input to that layer has 
		  width %d]], conv_idx, kernel_width, prevOutputHeight))
		assert(prevOutputTime >= temp_kernel_width, string.format([[Specified conv
		  kernel width = %d for layer: %d, but the input to that layer has 
		  width %d]], conv_idx, temp_kernel_width, prevOutputTime))
		assert(conv_stride > 0, "Can't have conv_stride = 0")
		assert(temp_conv_stride > 0, "Can't have temp_conv_stride = 0")
		assert(num_conv_filters > 0, "Can't have num_conv_filters = 0")

		prev = nn.VolumetricConvolution(prevNumFilters, num_conv_filters, 
		  temp_kernel_width, kernel_width, kernel_width, temp_conv_stride, 
      conv_stride, conv_stride)(prev)
    prev = hiddenActivationFn()(prev)
		--update our widths and filters
		prevOutputWidth = M.getConvOutputWidth(prevOutputWidth, kernel_width, 
		  conv_stride)
		prevOutputHeight = M.getConvOutputWidth(prevOutputHeight, kernel_width, 
		  conv_stride)
		prevOutputTime = M.getConvOutputWidth(prevOutputTime, temp_kernel_width, 
		  temp_conv_stride)
		prevNumFilters = num_conv_filters

		if max_pool_width == 0 then --special code for max over entire input
			max_pool_width = prevOutputWidth
			max_pool_stride = 1
		end

		if temp_max_pool_width == 0 then --special code for max over entire input
			temp_max_pool_width = prevOutputTime
			temp_max_pool_stride = 1
		end

		if max_pool_width > 1 and max_pool_stride > 0 then
			prev = nn.VolumetricMaxPooling(temp_max_pool_width, max_pool_width, max_pool_width, 
        temp_max_pool_stride, max_pool_stride, max_pool_stride)(prev)
		  prevOutputWidth = M.getConvOutputWidth(prevOutputWidth, 
			  max_pool_width, max_pool_stride)
		  prevOutputHeight = M.getConvOutputWidth(prevOutputHeight, 
			  max_pool_width, max_pool_stride)
		  prevOutputTime = M.getConvOutputWidth(prevOutputTime, 
			  temp_max_pool_width, temp_max_pool_stride)
		end

	end

    local prevLayerOutputs =  prevOutputTime*prevOutputHeight*prevOutputWidth*prevNumFilters --from the convNet
    prev = nn.View(-1):setNumInputDims(4)(prev)

    for hiddenLayerIdx = 2, numPostConvHiddenLayers do
      prev = nn.Linear(prevLayerOutputs,numHiddenUnits)(prev)
      prev = hiddenActivationFn()(prev)
      prevLayerOutputs = numHiddenUnits
    end

    --finally go from shared hidden to individual output paths
    local toClasses = nn.Linear(prevLayerOutputs,numOutputClasses)(prev)
    local toSubjects = nn.Linear(prevLayerOutputs, numSubjects)(prev)
    toClasses = nn.LogSoftMax()(toClasses)
    toSubjects = nn.LogSoftMax()(toSubjects)

    if net_args.cuda then
		toClasses = nn.Copy('torch.CudaTensor',torch.getdefaulttensortype())(toClasses)
		toSubjects = nn.Copy('torch.CudaTensor',torch.getdefaulttensortype())(toSubjects)
	end

    model = nn.gModule({input},{toClasses, toSubjects})

	if net_args.cuda then
		for moduleIdx = 1, #model.modules do
			local module = model.modules[moduleIdx]
			if torch.type(module) ~= 'nn.Copy' then
				module:cuda()
				print(module)
			end
	  end
	end

    criterion = nn.ParallelCriterion()
    --weight subjects two times as important
		criterion:add(nn.ClassNLLCriterion(),net_args.class_to_subj_loss_ratio)
		criterion:add(nn.ClassNLLCriterion(),1)

    if net_args.show_network then
      local start = torch.tic()
      model:forward(egInputBatch[{{1,2},{},{}}])
      print('2 examples took: ', torch.toc(start), 'secs')
      graph.dot(model.fg, 'mlp','volumetric_conv')
    end

    assert(not net_args.predict_delta_memory, '-predict_subj and -predict_delta_memory not currently supported, although babytown easy to add if curious')
    return model, criterion

  else

    local model = nn.Sequential()
    if dropout_prob > 0 then
      model:add(nn.Dropout(dropout_prob))
    end

    local prevOutputWidth = width
    local prevOutputHeight = height
    local prevOutputTime = numTimePoints
    local prevNumFilters = 1

    for conv_idx = 1, #net_args.kernel_widths do
      local added_conv = false
      local kernel_width = net_args.kernel_widths[conv_idx]
      local conv_stride = net_args.conv_strides[conv_idx]
      local max_pool_width = net_args.max_pool_widths[conv_idx]
      local max_pool_stride = net_args.max_pool_strides[conv_idx]
      local num_conv_filters = net_args.num_conv_filters[conv_idx]
      local temp_kernel_width = net_args.temp_kernel_widths[conv_idx]
      local temp_conv_stride = net_args.temp_conv_strides[conv_idx]
      local temp_max_pool_width = net_args.temp_max_pool_widths[conv_idx]
      local temp_max_pool_stride = net_args.temp_max_pool_strides[conv_idx]

      --check conv params okay
      assert(prevOutputWidth >= kernel_width, string.format([[Specified conv
              kernel width = %d for layer: %d, but the input to that layer has 
        width %d]], conv_idx, kernel_width, prevOutputWidth))
      assert(prevOutputHeight >= kernel_width, string.format([[Specified conv
              kernel width = %d for layer: %d, but the input to that layer has 
        width %d]], conv_idx, kernel_width, prevOutputHeight))
      assert(prevOutputTime >= temp_kernel_width, string.format([[Specified conv
              kernel width = %d for layer: %d, but the input to that layer has 
        width %d]], conv_idx, temp_kernel_width, prevOutputTime))
      assert(conv_stride > 0, "Can't have conv_stride = 0")
      assert(temp_conv_stride > 0, "Can't have temp_conv_stride = 0")
      assert(num_conv_filters > 0, "Can't have num_conv_filters = 0")

      model:add(nn.VolumetricConvolution(prevNumFilters, num_conv_filters, 
        temp_kernel_width, kernel_width, kernel_width, 
        temp_conv_stride, conv_stride, conv_stride))

      model:add(hiddenActivationFn())
      --update our widths and filters
      prevOutputWidth = M.getConvOutputWidth(prevOutputWidth, kernel_width, 
        conv_stride)
      prevOutputHeight = M.getConvOutputWidth(prevOutputHeight, kernel_width, 
        conv_stride)
      prevOutputTime = M.getConvOutputWidth(prevOutputTime, temp_kernel_width, 
        temp_conv_stride)
      prevNumFilters = num_conv_filters

      if max_pool_width == 0 then --special code for max over entire input
        max_pool_width = prevOutputWidth
        max_pool_stride = 1
      end
      if temp_max_pool_width == 0 then --special code for max over entire input
        temp_max_pool_width = prevOutputTime
        temp_max_pool_stride = 1
      end

      if max_pool_width > 1 and max_pool_stride > 0 then
        model:add(nn.VolumetricMaxPooling(temp_max_pool_width, max_pool_width, 
            max_pool_width, temp_max_pool_stride, max_pool_stride, max_pool_stride))
        prevOutputWidth = M.getConvOutputWidth(prevOutputWidth, 
          max_pool_width, max_pool_stride)
        prevOutputHeight = M.getConvOutputWidth(prevOutputHeight, 
          max_pool_width, max_pool_stride)
        prevOutputTime = M.getConvOutputWidth(prevOutputTime, 
          temp_max_pool_width, temp_max_pool_stride)
      end

    end

    local prevLayerOutputs =  prevOutputTime*prevOutputHeight*prevOutputWidth*prevNumFilters --from the convNet

    model:add(nn.View(-1):setNumInputDims(4)) 

    for hiddenLayerIdx = 2, numPostConvHiddenLayers do
      model:add(nn.Linear(prevLayerOutputs,numHiddenUnits))
      model:add(nn.hiddenActivationFn())
      prevLayerOutputs = numHiddenUnits
    end

    --go from last hidden layer to number of classes

    --finally logsoftmax gives us 1 numOutputClasses-way classifier
    local criterion
    if not net_args.predict_delta_memory then
      model:add(nn.Linear(prevLayerOutputs,numOutputClasses))
      model:add(nn.LogSoftMax())
      criterion = nn.ClassNLLCriterion()
    else
      model:add(nn.Linear(prevLayerOutputs,1))
      criterion = nn.MSECriterion(true)
    end

    if net_args.cuda then
      model:cuda()
      model:insert(nn.Copy('torch.CudaTensor',torch.getdefaulttensortype()),
        #model.modules+1)
    end

    return model, criterion
  end
end

--expect egInputBatch to have dimensions = [examples, time, features]
-- just haven't been using this that much
M.deprecated.createSumTempConvClassificationNetwork = function(...)
  local args, egInputBatch, numHiddenUnits, numPostConvHiddenLayers, 
      numOutputClasses, dropout_prob, predict_subj, numSubjects, net_args = dok.unpack(
      {...},
      'createSumTempConvClassificationNetwork',
      'Make a convolution ',
      {arg='egInputBatch',type='Tensor',help='', req=true},
      {arg='numHiddenUnits',type='number',help='num filters in conv and how ' .. 
        ' many hidden units in subsequent hidden layers', req=true},
      {arg='numPostConvHiddenLayers',type='number',help='number of hidden ' .. 
        'layers excluding the output filters we get when we do our conv', 
        req=true},
      {arg='numOutputClasses',type='number',help='', req=false, default=nil},
      {arg='dropout_prob',type='number',help='', req=false, default=-1},
      {arg='predict_subj',type='number',
        help='whether or not to predict subjects as well as classes', req=false, default=false},
      {arg='numSubjects',type='number',help='only applies if predict_subj is true', 
        req=false, default=-1},
      {arg='net_args',type='table',help='', req=true}
  )
	local numTimePoints = egInputBatch:size(2)
	local numInputUnits = egInputBatch:size(3)
  local hiddenActivationFn = M.activationFns[net_args.hidden_act_fn] or nn.ReLU --defaults to ReLU
  print(numTimePoints, numInputUnits)
	assert(egInputBatch and numHiddenUnits and numPostConvHiddenLayers)
	--if we're not going to take the max after our convolution, which collapses
	--the number of output features, then we have to have at least one hidden
	--layer
	assert(numPostConvHiddenLayers > 0)
	numOutputClasses = numOutputClasses or 2

  if predict_subj then
		require 'nngraph'
		nngraph.setDebug = true

		local input = nn.Identity()()
		local prev = {}
		if dropout_prob > 0 then
			prev = nn.Dropout(dropout_prob)(input)
		else
			prev = input
		end
    prev = nn.TemporalConvolution(numInputUnits, numHiddenUnits, 1, 1)(prev)
    prev = hiddenActivationFn()(prev)
    --sum, instead of max, across temporal dimension
    prev = nn.Sum(2)(prev) --only works if we have batch data!!!!
    --usually we need a nn:View(-1) to collapse the singleton temporal dimension, but sum gets rid of that

    local prevLayerOutputs =  numHiddenUnits --from the convNet

    for hiddenLayerIdx = 2, numPostConvHiddenLayers do
      prev = nn.Linear(prevLayerOutputs,numHiddenUnits)(prev)
      prev = hiddenActivationFn()(prev)
      prevLayerOutputs = numHiddenUnits
    end

    --finally go from shared hidden to individual output paths
    local toClasses = nn.Linear(prevLayerOutputs,numOutputClasses)(prev)
    local toSubjects = nn.Linear(prevLayerOutputs, numSubjects)(prev)
    toClasses = nn.LogSoftMax()(toClasses)
    toSubjects = nn.LogSoftMax()(toSubjects)

    model = nn.gModule({input},{toClasses, toSubjects})

    criterion = nn.ParallelCriterion()
    --weight subjects two times as important
		criterion:add(nn.ClassNLLCriterion(),net_args.class_to_subj_loss_ratio)
		criterion:add(nn.ClassNLLCriterion(),1)

		model:forward(egInputBatch)
		graph.dot(model.fg, 'mlp','test_max_temp_conv')

    return model, criterion

  else

    local model = nn.Sequential()
    if dropout_prob > 0 then
      model:add(nn.Dropout(dropout_prob))
    end
    tempConv = nn.TemporalConvolution(numInputUnits, numHiddenUnits, 1, 1)
    model:add(tempConv)

    -- flattens from batch x 1 x numHiddens --> batch numHiddens
    -- now we have batch x numTimePoints x numHiddens --> batch x numTimePoints * numHiddens
    model:add(hiddenActivationFn())
    out = model:forward(egInputBatch)
    --sum, instead of max, across temporal dimension
    model:add(nn.Sum(2)) --only works if we have batch data!!!
    --usually we need a nn:View(-1) to collapse the singleton temporal dimension, but sum gets rid of that

    --we only want to ReLU() the output if we have hidden layers, otherwise we 
    --want linear output (aka what we already get from the conv output) that will 
    --eventually get sent to a criterion which takes the log soft max using linear 
    --output 
    --TODO: Might want to reconsider this behavior, why not have 
    --conv --> pool --> ReLU --> sigmoid?
    local prevLayerOutputs = numHiddenUnits --from the convNet

    for hiddenLayerIdx = 2, numPostConvHiddenLayers do
      model:add(nn.Linear(prevLayerOutputs,numHiddenUnits))
      model:add(hiddenActivationFn())
      prevLayerOutputs = numHiddenUnits
    end

    --go from last hidden layer to number of classes
    model:add(nn.Linear(prevLayerOutputs,numOutputClasses))

    --finally logsoftmax gives us 1 numOutputClasses-way classifier
    local criterion
    if not net_args.predict_delta_memory then
      model:add(nn.LogSoftMax())
      criterion = nn.ClassNLLCriterion()
    else
      criterion = nn.MSECriterion(true)
    end
    return model, criterion
  end
end

--expect egInputBatch to have dimensions = [examples, time, features]
--just haven't been using this that much
M.deprecated.createShallowMaxTempConvClassificationNetwork = function(...)
  local args, egInputBatch, numHiddenUnits, numPostConvHiddenLayers, 
      numOutputClasses, dropout_prob, predict_subj, numSubjects, net_args = dok.unpack(
      {...},
      'createShallowMaxTempConvClassificationNetwork',
      'Make a convolution ',
      {arg='egInputBatch',type='Tensor',help='', req=true},
      {arg='numHiddenUnits',type='number',help='num filters in conv and how ' .. 
        ' many hidden units in subsequent hidden layers', req=true},
      {arg='numPostConvHiddenLayers',type='number',help='number of hidden ' .. 
        'layers excluding the output filters we get when we do our conv', 
        req=true},
      {arg='numOutputClasses',type='number',help='', req=false, default=nil},
      {arg='dropout_prob',type='number',help='', req=false, default=-1},
      {arg='predict_subj',type='number',
        help='whether or not to predict subjects as well as classes', req=false, default=false},
      {arg='numSubjects',type='number',help='only applies if predict_subj is true', 
        req=false, default=-1},
      {arg='net_args',type='table',help='', req=true}
  )
	local numTimePoints = egInputBatch:size(2)
	local numInputUnits = egInputBatch:size(3)
  print(numTimePoints, numInputUnits)
	assert(egInputBatch and numHiddenUnits and numPostConvHiddenLayers)
	--if we're not going to take the max after our convolution, which collapses
	--the number of output features, then we have to have at least one hidden
	--layer
	assert(numPostConvHiddenLayers > 0)
	numOutputClasses = numOutputClasses or 2


  if predict_subj then
    error("-predict_subj not supported for shallow_temp_conv");
  else

    local model = nn.Sequential()
    if dropout_prob > 0 then
      model:add(nn.Dropout(dropout_prob))
    end

    tempConv = nn.TemporalConvolution(numInputUnits, numOutputClasses, 1, 1)
    model:add(tempConv)

    -- flattens from batch x 1 x numHiddens --> batch numHiddens
    -- now we have batch x numTimePoints x numHiddens --> batch x numTimePoints * numHiddens
    model:add(nn.TemporalMaxPooling(numTimePoints,1))
    model:add(nn.View(-1):setNumInputDims(2)) 


    local criterion
    if not net_args.predict_delta_memory then
      --finally logsoftmax gives us 1 numOutputClasses-way classifier
      model:add(nn.LogSoftMax())
      criterion = nn.ClassNLLCriterion()
    else
      criterion = nn.MSECriterion(true)
    end

    return model, criterion
  end
end

--expect egInputBatch to have dimensions = [examples, time, features]
M.deprecated.createMaxChannelConvClassificationNetwork = function(...)
  local args, egInputBatch, numHiddenUnits, numPostConvHiddenLayers, 
      numOutputClasses, dropout_prob, predict_subj, numSubjects, net_args = dok.unpack(
      {...},
      'createMaxChannelConvClassificationNetwork',
      'Make a convolution ',
      {arg='egInputBatch',type='Tensor',help='', req=true},
      {arg='numHiddenUnits',type='number',help='num filters in conv and how ' .. 
        ' many hidden units in subsequent hidden layers', req=true},
      {arg='numPostConvHiddenLayers',type='number',help='number of hidden ' .. 
        'layers excluding the output filters we get when we do our conv', 
        req=true},
      {arg='numOutputClasses',type='number',help='', req=false, default=nil},
      {arg='dropout_prob',type='number',help='', req=false, default=-1},
      {arg='predict_subj',type='number',
        help='whether or not to predict subjects as well as classes', req=false, default=false},
      {arg='numSubjects',type='number',help='only applies if predict_subj is true', 
        req=false, default=-1},
      {arg='net_args',type='table',help='', req=true}
  )
	local numTimePoints = egInputBatch:size(3)
	local numInputUnits = egInputBatch:size(2)
  local hiddenActivationFn = M.activationFns[net_args.hidden_act_fn] or nn.ReLU --defaults to ReLU
  print(numTimePoints, numInputUnits)
	assert(egInputBatch and numHiddenUnits and numPostConvHiddenLayers)
	--if we're not going to take the max after our convolution, which collapses
	--the number of output features, then we have to have at least one hidden
	--layer
	assert(numPostConvHiddenLayers > 0)
	numOutputClasses = numOutputClasses or 2

  if predict_subj then
		require 'nngraph'
		nngraph.setDebug = true

		local input = nn.Identity()()
		local prev = {}
		if dropout_prob > 0 then
			prev = nn.Dropout(dropout_prob)(input)
		else
			prev = input
		end
    prev = nn.Transpose({2,3})
    prev = nn.TemporalConvolution(numInputUnits, numHiddenUnits, 1, 1)(prev)
    prev = hiddenActivationFn()(prev)
    prev = nn.TemporalMaxPooling(numTimePoints, 1)(prev)
    prev = nn.View(-1):setNumInputDims(2)(prev)

    local prevLayerOutputs =  numHiddenUnits --from the convNet

    for hiddenLayerIdx = 2, numPostConvHiddenLayers do
      prev = nn.Linear(prevLayerOutputs,numHiddenUnits)(prev)
      prev = hiddenActivationFn()(prev)
      prevLayerOutputs = numHiddenUnits
    end

    --finally go from shared hidden to individual output paths
    local toClasses = nn.Linear(prevLayerOutputs,numOutputClasses)(prev)
    local toSubjects = nn.Linear(prevLayerOutputs, numSubjects)(prev)
    toClasses = nn.LogSoftMax()(toClasses)
    toSubjects = nn.LogSoftMax()(toSubjects)

    model = nn.gModule({input},{toClasses, toSubjects})

    criterion = nn.ParallelCriterion()
    --weight subjects two times as important
		criterion:add(nn.ClassNLLCriterion(),net_args.class_to_subj_loss_ratio)
		criterion:add(nn.ClassNLLCriterion(),1)

		model:forward(egInputBatch)
		graph.dot(model.fg, 'mlp','test_max_temp_conv')

    return model, criterion

  else

    local model = nn.Sequential()
    if dropout_prob > 0 then
      model:add(nn.Dropout(dropout_prob))
    end
    model:add(nn.Transpose({2,3}))
    tempConv = nn.TemporalConvolution(numInputUnits, numHiddenUnits, 1, 1)
    model:add(tempConv)

    -- flattens from batch x 1 x numHiddens --> batch numHiddens
    -- now we have batch x numTimePoints x numHiddens --> batch x numTimePoints * numHiddens
    model:add(hiddenActivationFn())
    model:add(nn.TemporalMaxPooling(numTimePoints,1))
    model:add(nn.View(-1):setNumInputDims(2))

    --we only want to ReLU() the output if we have hidden layers, otherwise we
    --want linear output (aka what we already get from the conv output) that will 
    --eventually get sent to a criterion which takes the log soft max using linear 
    --output 
    --TODO: Might want to reconsider this behavior, why not have 
    --conv --> pool --> ReLU --> sigmoid?
    local prevLayerOutputs = numHiddenUnits --from the convNet

    for hiddenLayerIdx = 2, numPostConvHiddenLayers do
      model:add(nn.Linear(prevLayerOutputs,numHiddenUnits))
      model:add(hiddenActivationFn())
      prevLayerOutputs = numHiddenUnits
    end

    --go from last hidden layer to number of classes
    model:add(nn.Linear(prevLayerOutputs,numOutputClasses))

    --finally logsoftmax gives us 1 numOutputClasses-way classifier
    model:add(nn.LogSoftMax())

    --local criterion = nn.CrossEntropyCriterion()
    local criterion = nn.ClassNLLCriterion()

    return model, criterion
  end
end

--expect egInputBatch to have dimensions = [examples, time, features]
M.createDeepMaxTempConvClassificationNetwork = function(...)
  local args, egInputBatch, numHiddenUnits, numPostConvHiddenLayers, 
      numOutputClasses, dropout_prob, predict_subj, numSubjects, net_args = dok.unpack(
      {...},
      'createMaxTempConvClassificationNetwork',
      'Make a convolution ',
      {arg='egInputBatch',type='Tensor',help='', req=true},
      {arg='numHiddenUnits',type='number',help='num filters in conv and how ' .. 
        ' many hidden units in subsequent hidden layers', req=true},
      {arg='numPostConvHiddenLayers',type='number',help='number of hidden ' .. 
        'layers excluding the output filters we get when we do our conv', 
        req=true},
      {arg='numOutputClasses',type='number',help='', req=false, default=nil},
      {arg='dropout_prob',type='number',help='', req=false, default=-1},
      {arg='predict_subj',type='number',
        help='whether or not to predict subjects as well as classes', req=false, default=false},
      {arg='numSubjects',type='number',help='only applies if predict_subj is true', 
        req=false, default=-1},
      {arg='net_args',type='table',help='', req=true}
  )
  local smooth_std = net_args.smooth_std or -1
  local smooth_width = net_args.smooth_width or 5
  local hiddenActivationFn = M.activationFns[net_args.hidden_act_fn] or nn.ReLU --defaults to ReLU

  if net_args.cuda then
	  require 'cunn'
  end

	local numTimePoints = egInputBatch:size(2)
	local numInputUnits = egInputBatch:size(3)
  print(numTimePoints, numInputUnits)
	assert(egInputBatch and numHiddenUnits and numPostConvHiddenLayers)
	--if we're not going to take the max after our convolution, which collapses
	--the number of output features, then we have to have at least one hidden
	--layer
	assert(numPostConvHiddenLayers > 0)
	numOutputClasses = numOutputClasses or 2


  local smoothModule = {}
  local shouldSmooth = false
  if smooth_std > 0 then
    local filter = nn.TemporalSmoothing.filters.makeGaussian(smooth_width, smooth_std, true)
    smoothModule = nn.TemporalSmoothing(filter, net_args.smooth_step, false, 2)
    shouldSmooth = true
  end
  if predict_subj then
		require 'nngraph'
		nngraph.setDebug = true

		local input = nn.Identity()()
		local prev = {}
		if dropout_prob > 0 then
			prev = nn.Dropout(dropout_prob)(input)
		else
			prev = input
		end
     

	local prevOutputWidth = numTimePoints
	local prevNumFilters = numInputUnits

	for conv_idx = 1, #net_args.kernel_widths do
		local added_conv = false
		local kernel_width = net_args.kernel_widths[conv_idx]
		local conv_stride = net_args.conv_strides[conv_idx]
		local max_pool_width = net_args.max_pool_widths[conv_idx]
		local max_pool_stride = net_args.max_pool_strides[conv_idx]
		local num_conv_filters = net_args.num_conv_filters[conv_idx]


		--check conv params okay
		assert(prevOutputWidth >= kernel_width, string.format([[Specified conv
		  kernel width = %d for layer: %d, but the input to that layer has 
		  width %d]], conv_idx, kernel_width, prevOutputWidth))
		assert(conv_stride > 0, "Can't have conv_stride = 0")
		assert(num_conv_filters > 0, "Can't have num_conv_filters = 0")

		prev = nn.TemporalConvolution(prevNumFilters, num_conv_filters, 
		  kernel_width, conv_stride)(prev)
        prev = hiddenActivationFn()(prev)
		--update our widths and filters
		prevOutputWidth = M.getConvOutputWidth(prevOutputWidth, kernel_width, 
		  conv_stride)
		prevNumFilters = num_conv_filters

		if max_pool_width == 0 then --special code for max over entire input
			max_pool_width = prevOutputWidth
			max_pool_stride = 1
		end

		if max_pool_width > 1 and max_pool_stride > 0 then
			prev = nn.TemporalMaxPooling(max_pool_width, max_pool_stride)(prev)
		    prevOutputWidth = M.getConvOutputWidth(prevOutputWidth, 
			  max_pool_width, max_pool_stride)
		end

	end

    local prevLayerOutputs =  prevOutputWidth*prevNumFilters --from the convNet
    prev = nn.View(-1):setNumInputDims(2)(prev)

    for hiddenLayerIdx = 2, numPostConvHiddenLayers do
      prev = nn.Linear(prevLayerOutputs,numHiddenUnits)(prev)
      prev = hiddenActivationFn()(prev)
      prevLayerOutputs = numHiddenUnits
    end

    --finally go from shared hidden to individual output paths
    local toClasses = nn.Linear(prevLayerOutputs,numOutputClasses)(prev)
    local toSubjects = nn.Linear(prevLayerOutputs, numSubjects)(prev)
    toClasses = nn.LogSoftMax()(toClasses)
    toSubjects = nn.LogSoftMax()(toSubjects)

    if net_args.cuda then
		toClasses = nn.Copy('torch.CudaTensor',torch.getdefaulttensortype())(toClasses)
		toSubjects = nn.Copy('torch.CudaTensor',torch.getdefaulttensortype())(toSubjects)
	end

    model = nn.gModule({input},{toClasses, toSubjects})

	if net_args.cuda then
		for moduleIdx = 1, #model.modules do
			local module = model.modules[moduleIdx]
			if torch.type(module) ~= 'nn.Copy' then
				module:cuda()
				print(module)
			end
	    end
	end

    criterion = nn.ParallelCriterion()
    --weight subjects two times as important
		criterion:add(nn.ClassNLLCriterion(),net_args.class_to_subj_loss_ratio)
		criterion:add(nn.ClassNLLCriterion(),1)

    if net_args.show_network then
      local start = torch.tic()
      model:forward(egInputBatch[{{1,2},{},{}}])
      print('2 examples took: ', torch.toc(start), 'secs')
      graph.dot(model.fg, 'mlp','deep_max_temp_conv')
    end

    assert(not net_args.predict_delta_memory, '-predict_subj and -predict_delta_memory not currently supported, although babytown easy to add if curious')
    return model, criterion

  else

    local model = nn.Sequential()
    if dropout_prob > 0 then
      model:add(nn.Dropout(dropout_prob))
    end

	local prevOutputWidth = numTimePoints
	local prevNumFilters = numInputUnits

	for conv_idx = 1, #net_args.kernel_widths do
		local added_conv = false
		local kernel_width = net_args.kernel_widths[conv_idx]
		local conv_stride = net_args.conv_strides[conv_idx]
		local max_pool_width = net_args.max_pool_widths[conv_idx]
		local max_pool_stride = net_args.max_pool_strides[conv_idx]
		local num_conv_filters = net_args.num_conv_filters[conv_idx]


		--check conv params okay
		assert(prevOutputWidth >= kernel_width, string.format([[Specified conv
		  kernel width = %d for layer: %d, but the input to that layer has 
		  width %d]], conv_idx, kernel_width, prevOutputWidth))
		assert(conv_stride > 0, "Can't have conv_stride = 0")
		assert(num_conv_filters > 0, "Can't have num_conv_filters = 0")

		model:add(nn.TemporalConvolution(prevNumFilters, num_conv_filters, 
		  kernel_width, conv_stride))
        model:add(hiddenActivationFn())
		--update our widths and filters
		prevOutputWidth = M.getConvOutputWidth(prevOutputWidth, kernel_width, 
		  conv_stride)
		prevNumFilters = num_conv_filters

		if max_pool_width == 0 then --special code for max over entire input
			max_pool_width = prevOutputWidth
			max_pool_stride = 1
		end

		if max_pool_width > 1 and max_pool_stride > 0 then
			model:add(nn.TemporalMaxPooling(max_pool_width, max_pool_stride))
		    prevOutputWidth = M.getConvOutputWidth(prevOutputWidth, 
			  max_pool_width, max_pool_stride)
		end

	end

    local prevLayerOutputs =  prevOutputWidth*prevNumFilters --from the convNet

    model:add(nn.View(-1):setNumInputDims(2)) 

    for hiddenLayerIdx = 2, numPostConvHiddenLayers do
      model:add(nn.Linear(prevLayerOutputs,numHiddenUnits))
      model:add(nn.hiddenActivationFn())
      prevLayerOutputs = numHiddenUnits
    end

    --go from last hidden layer to number of classes

    --finally logsoftmax gives us 1 numOutputClasses-way classifier
    local criterion
    if not net_args.predict_delta_memory then
      model:add(nn.Linear(prevLayerOutputs,numOutputClasses))
      model:add(nn.LogSoftMax())
      criterion = nn.ClassNLLCriterion()
    else
      model:add(nn.Linear(prevLayerOutputs,1))
      criterion = nn.MSECriterion(true)
    end
    if net_args.cuda then
      model:cuda()
      model:insert(nn.Copy('torch.CudaTensor',torch.getdefaulttensortype()),
        #model.modules+1)
    end

    return model, criterion
  end
end

--expect egInputBatch to have dimensions = [examples, time, features]
M.deprecated.createMaxTempConvClassificationNetwork = function(...)
  local args, egInputBatch, numHiddenUnits, numPostConvHiddenLayers, 
      numOutputClasses, dropout_prob, predict_subj, numSubjects, net_args = dok.unpack(
      {...},
      'createMaxTempConvClassificationNetwork',
      'Make a convolution ',
      {arg='egInputBatch',type='Tensor',help='', req=true},
      {arg='numHiddenUnits',type='number',help='num filters in conv and how ' .. 
        ' many hidden units in subsequent hidden layers', req=true},
      {arg='numPostConvHiddenLayers',type='number',help='number of hidden ' .. 
        'layers excluding the output filters we get when we do our conv', 
        req=true},
      {arg='numOutputClasses',type='number',help='', req=false, default=nil},
      {arg='dropout_prob',type='number',help='', req=false, default=-1},
      {arg='predict_subj',type='number',
        help='whether or not to predict subjects as well as classes', req=false, default=false},
      {arg='numSubjects',type='number',help='only applies if predict_subj is true', 
        req=false, default=-1},
      {arg='net_args',type='table',help='', req=true}
  )
  local smooth_std = net_args.smooth_std or -1
  local smooth_width = net_args.smooth_width or 5
  local hiddenActivationFn = M.activationFns[net_args.hidden_act_fn] or nn.ReLU --defaults to ReLU

  if net_args.cuda then
	  require 'cunn'
  end

	local numTimePoints = egInputBatch:size(2)
	local numInputUnits = egInputBatch:size(3)
  print(numTimePoints, numInputUnits)
	assert(egInputBatch and numHiddenUnits and numPostConvHiddenLayers)
	--if we're not going to take the max after our convolution, which collapses
	--the number of output features, then we have to have at least one hidden
	--layer
	assert(numPostConvHiddenLayers > 0)
	numOutputClasses = numOutputClasses or 2

  local smoothModule = {}
  local shouldSmooth = false
  if smooth_std > 0 then
    local filter = nn.TemporalSmoothing.filters.makeGaussian(smooth_width, smooth_std, true)
    smoothModule = nn.TemporalSmoothing(filter, net_args.smooth_step, false, 2)
    shouldSmooth = true
  end

  if predict_subj then
		require 'nngraph'
		nngraph.setDebug = true

		local input = nn.Identity()()
		local prev = {}
		if dropout_prob > 0 then
			prev = nn.Dropout(dropout_prob)(input)
		else
			prev = input
		end
    prev = nn.TemporalConvolution(numInputUnits, numHiddenUnits, 1, 1)(prev)
    prev = hiddenActivationFn()(prev)
    local prevOutputWidth = numTimePoints
    if shouldSmooth then
      prev = smoothModule(prev)
      prevOutputWidth = smoothModule:getTemporalOutputSize(numTimePoints)
    end

    local maxPoolWidth = prevOutputWidth
    if net_args.max_pool_width_prcnt then
      maxPoolWidth = math.floor(prevOutputWidth * net_args.max_pool_width_prcnt)
    end

    --prev = nn.TemporalMaxPooling(prevOutputWidth, 1)(prev)
	if maxPoolWidth > 1 then --account for the fact that max_pool_width_prcnt = 0 would give us no pooling
		prev = nn.TemporalMaxPooling(maxPoolWidth, maxPoolWidth)(prev)
		prevOutputWidth = M.getConvOutputWidth(prevOutputWidth,maxPoolWidth, maxPoolWidth)
	else
		print('Warning: no max pooling performed.  Prev width: ', 
		  prevOutputWidth, 'Pool width prcnt: ', net_args.max_pool_width_prcnt)
    end
    prev = nn.View(-1):setNumInputDims(2)(prev)

    local prevLayerOutputs =  numHiddenUnits * prevOutputWidth --from the convNet

    for hiddenLayerIdx = 2, numPostConvHiddenLayers do
      prev = nn.Linear(prevLayerOutputs,numHiddenUnits)(prev)
      prev = hiddenActivationFn()(prev)
      prevLayerOutputs = numHiddenUnits
    end

    --finally go from shared hidden to individual output paths
    local toClasses = nn.Linear(prevLayerOutputs,numOutputClasses)(prev)
    local toSubjects = nn.Linear(prevLayerOutputs, numSubjects)(prev)
    toClasses = nn.LogSoftMax()(toClasses)
    toSubjects = nn.LogSoftMax()(toSubjects)


    if net_args.cuda then
		toClasses = nn.Copy('torch.CudaTensor',torch.getdefaulttensortype())(toClasses)
		toSubjects = nn.Copy('torch.CudaTensor',torch.getdefaulttensortype())(toSubjects)
	end

    model = nn.gModule({input},{toClasses, toSubjects})

	if net_args.cuda then
		for moduleIdx = 1, #model.modules do
			local module = model.modules[moduleIdx]
			if torch.type(module) ~= 'nn.Copy' then
				module:cuda()
				print(module)
			end
	    end
	end

    criterion = nn.ParallelCriterion()
    --weight subjects two times as important
		criterion:add(nn.ClassNLLCriterion(),net_args.class_to_subj_loss_ratio)
		criterion:add(nn.ClassNLLCriterion(),1)

	if net_args.show_network then
      local start = torch.tic()
      model:forward(egInputBatch[{{1,2},{},{}}])
      print('2 examples took: ', torch.toc(start), 'secs')
      graph.dot(model.fg, 'mlp','test_max_temp_conv')
	end

    return model, criterion

  else

    local model = nn.Sequential()
    if dropout_prob > 0 then
      model:add(nn.Dropout(dropout_prob))
    end
    tempConv = nn.TemporalConvolution(numInputUnits, numHiddenUnits, 1, 1)
    model:add(tempConv)

    -- flattens from batch x 1 x numHiddens --> batch numHiddens
    -- now we have batch x numTimePoints x numHiddens --> batch x numTimePoints * numHiddens
    model:add(hiddenActivationFn())

    local prevOutputWidth = numTimePoints
    if shouldSmooth then
      model:add(smoothModule)
      prevOutputWidth = smoothModule:getTemporalOutputSize(numTimePoints)
    end

    local maxPoolWidth = prevOutputWidth
    if net_args.max_pool_outs then
      maxPoolWidth = math.floor(prevOutputWidth / net_args.max_pool_outs)
    end

	if maxPoolWidth > 1 then
		model:add(nn.TemporalMaxPooling(maxPoolWidth,maxPoolWidth))
		prevOutputWidth = M.getConvOutputWidth(prevOutputWidth,maxPoolWidth, maxPoolWidth)
	else
		print('Warning: no max pooling performed.  Prev width: ', 
		  prevOutputWidth, 'Pool width prcnt: ', net_args.max_pool_width_prcnt)
    end
    model:add(nn.View(-1):setNumInputDims(2)) 

    --we only want to ReLU() the output if we have hidden layers, otherwise we 
    --want linear output (aka what we already get from the conv output) that will 
    --eventually get sent to a criterion which takes the log soft max using linear 
    --output 
    --TODO: Might want to reconsider this behavior, why not have 
    --conv --> pool --> ReLU --> sigmoid?
    local prevLayerOutputs = prevOutputWidth * numHiddenUnits --from the convNet

    for hiddenLayerIdx = 2, numPostConvHiddenLayers do
      model:add(nn.Linear(prevLayerOutputs,numHiddenUnits))
      model:add(nn.hiddenActivationFn())
      prevLayerOutputs = numHiddenUnits
    end

    --go from last hidden layer to number of classes
    model:add(nn.Linear(prevLayerOutputs,numOutputClasses))

    --finally logsoftmax gives us 1 numOutputClasses-way classifier
    model:add(nn.LogSoftMax())

    --local criterion = nn.CrossEntropyCriterion()
    local criterion = nn.ClassNLLCriterion()
	if net_args.cuda then
		model:cuda()
		model:insert(nn.Copy('torch.CudaTensor',torch.getdefaulttensortype()),
		  #model.modules+1)
	end
    return model, criterion
  end
end

--expect egInputBatch to have dimensions = [examples, time, features]
--can just do maxTempConv with max_pool_width_prcnt == 0
M.deprecated.createNoMaxTempConvClassificationNetwork = function(...)
  local args, egInputBatch, numHiddenUnits, numPostConvHiddenLayers, 
      numOutputClasses, dropout_prob, predict_subj, numSubjects, net_args = dok.unpack(
      {...},
      'createNoMaxTempConvClassificationNetwork',
      'Make a convolution ',
      {arg='egInputBatch',type='Tensor',help='', req=true},
      {arg='numHiddenUnits',type='number',help='num filters in conv and how ' .. 
        ' many hidden units in subsequent hidden layers', req=true},
      {arg='numPostConvHiddenLayers',type='number',help='number of hidden ' .. 
        'layers excluding the output filters we get when we do our conv', 
        req=true},
      {arg='numOutputClasses',type='number',help='', req=false, default=nil},
      {arg='dropout_prob',type='number',help='', req=false, default=-1},
      {arg='predict_subj',type='number',
        help='whether or not to predict subjects as well as classes', req=false, default=false},
      {arg='numSubjects',type='number',help='only applies if predict_subj is true', 
        req=false, default=-1},
      {arg='net_args',type='table',help='', req=true}
  )
	local numTimePoints = egInputBatch:size(2)
	local numInputUnits = egInputBatch:size(3)
  local hiddenActivationFn = M.activationFns[net_args.hidden_act_fn] or nn.ReLU --defaults to ReLU
  print(numTimePoints, numInputUnits)
	assert(egInputBatch and numHiddenUnits and numPostConvHiddenLayers)
	--if we're not going to take the max after our convolution, which collapses
	--the number of output features, then we have to have at least one hidden
	--layer
	assert(numPostConvHiddenLayers > 0)
	numOutputClasses = numOutputClasses or 2

  if predict_subj then
		require 'nngraph'
		nngraph.setDebug = true

		local input = nn.Identity()()
		local prev = {}
		if dropout_prob > 0 then
			prev = nn.Dropout(dropout_prob)(input)
		else
			prev = input
		end
    prev = nn.TemporalConvolution(numInputUnits, numHiddenUnits, 1, 1)(prev)
    prev = nn.View(-1):setNumInputDims(2)(prev)
    prev = hiddenActivationFn()(prev)

    local prevLayerOutputs =  numTimePoints * numHiddenUnits --from the convNet

    for hiddenLayerIdx = 2, numPostConvHiddenLayers do
      prev = nn.Linear(prevLayerOutputs,numHiddenUnits)(prev)
      prev = hiddenActivationFn()(prev)
      prevLayerOutputs = numHiddenUnits
    end

    --finally go from shared hidden to individual output paths
    local toClasses = nn.Linear(prevLayerOutputs,numOutputClasses)(prev)
    local toSubjects = nn.Linear(prevLayerOutputs, numSubjects)(prev)
    toClasses = nn.LogSoftMax()(toClasses)
    toSubjects = nn.LogSoftMax()(toSubjects)

    model = nn.gModule({input},{toClasses, toSubjects})

    criterion = nn.ParallelCriterion()
    --weight subjects two times as important
		criterion:add(nn.ClassNLLCriterion(),net_args.class_to_subj_loss_ratio)
		criterion:add(nn.ClassNLLCriterion(),1)

		--model:forward(egInputBatch)
		--graph.dot(model.fg, 'mlp','test_no_max_temp_conv')

    return model, criterion

  else

    local model = nn.Sequential()
    if dropout_prob > 0 then
      model:add(nn.Dropout(dropout_prob))
    end
    tempConv = nn.TemporalConvolution(numInputUnits, numHiddenUnits, 1, 1)
    model:add(tempConv)

    -- flattens from batch x 1 x numHiddens --> batch numHiddens
    -- now we have batch x numTimePoints x numHiddens --> batch x numTimePoints * numHiddens
    model:add(nn.View(-1):setNumInputDims(2)) 
    model:add(hiddenActivationFn())

    --we only want to ReLU() the output if we have hidden layers, otherwise we 
    --want linear output (aka what we already get from the conv output) that will 
    --eventually get sent to a criterion which takes the log soft max using linear 
    --output 
    --TODO: Might want to reconsider this behavior, why not have 
    --conv --> pool --> ReLU --> sigmoid?
    local prevLayerOutputs = numTimePoints * numHiddenUnits --from the convNet

    for hiddenLayerIdx = 2, numPostConvHiddenLayers do
      model:add(nn.Linear(prevLayerOutputs,numHiddenUnits))
      model:add(hiddenActivationFn())
      prevLayerOutputs = numHiddenUnits
    end

    --go from last hidden layer to number of classes
    model:add(nn.Linear(prevLayerOutputs,numOutputClasses))

    --finally logsoftmax gives us 1 numOutputClasses-way classifier
    model:add(nn.LogSoftMax())

    --local criterion = nn.CrossEntropyCriterion()
    local criterion = nn.ClassNLLCriterion()

    return model, criterion
  end
end

return M
