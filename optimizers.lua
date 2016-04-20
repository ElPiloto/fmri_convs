local M = {}

M.SGD = function(learningRate)
	require 'optim'
	if learningRate == nil then
		learningRate = 0.001
	end
	optimSettings = {learningRate = learningRate, momentum = 0.9, decay = 0}
	return optim.sgd_log, optimSettings
end

M.ADAM = function(learningRate)
	require 'optim'
	if learningRate == nil then
		learningRate = 0.001
	end
	optimSettings = {learningRate = learningRate}
	return optim.adam_log, optimSettings
end

M.getOptimizer = function(name, learningRate)
	if name == 'sgd' then
		return M.SGD(learningRate)
	elseif name == 'adam' then
		return M.ADAM(learningRate)
	else
		error('Currently only supports name = "sgd" or "adam"')
	end
end


M.performTrainIteration = function(fullState)
	local args = fullState.args.training
	local trainInputs = fullState.data:getTrainData()
	local trainTargets = fullState.data:getTrainTargets()

	if not fullState.trainSetLoss then
		fullState:add('trainSetLoss', torch.FloatTensor(args.maxTrainingIterations):fill(-1.0), true)
	end

  --accomodate minibatches
  local numExamples = trainInputs:size(1)
  local shuffle = torch.randperm(numExamples):long()
  local numMiniBatches = sleep_eeg.utils.getNumMiniBatches(numExamples, fullState.args.miniBatchSize)

  fullState.trainSetLoss[fullState.trainingIteration] = 0

  local start = torch.tic()
  local batchTrainInputs
  for miniBatchIdx = 1, numMiniBatches do

    local start2
    if fullState.trainingIteration == 1 then
      start2= torch.tic()
    end

    local miniBatchTrials = sleep_eeg.utils.getMiniBatchTrials(shuffle, miniBatchIdx, fullState.args.miniBatchSize)
    --when we accum loss/accuracy over each minibatch, we want to avg according the size of the minibatch
    local miniBatchWeight = miniBatchTrials:numel()/numExamples 

    --set gradient to zero
    fullState.gradParams:zero()
	if fullState.args.network.cuda then
		batchTrainInputs = trainInputs:index(1,miniBatchTrials):cuda()
	else
		batchTrainInputs = trainInputs:index(1,miniBatchTrials)
	end
    fullState.trainModelOut = fullState.network:forward(batchTrainInputs)

    fullState.trainSetLoss[fullState.trainingIteration] = fullState.trainSetLoss[fullState.trainingIteration] + 
      fullState.criterion:forward(fullState.trainModelOut,
      sleep_eeg.utils.indexIntoTensorOrTableOfTensors(trainTargets,1,miniBatchTrials)) * miniBatchWeight

    fullState.trainSetClassAcc = 1 --evaluation.classification(trainModelOut, trainTargets)

    local criterionGradInput = fullState.criterion:backward(fullState.trainModelOut, sleep_eeg.utils.indexIntoTensorOrTableOfTensors(trainTargets,1,miniBatchTrials))

    if fullState.args.weight_loss_function and fullState.data.can_weight_loss then
        --let's get the appropriate weights for this miniBatch
        local miniBatchExampleWeights = fullState.data:getTrainExampleWeights():index(1,miniBatchTrials)
        miniBatchExampleWeights:div(miniBatchExampleWeights:sum()) --normalize so we're not working with tiny weights
        for class_idx = 1, criterionGradInput:size(2) do
            criterionGradInput[{{},class_idx}] = torch.cmul(criterionGradInput[{{},class_idx}], miniBatchExampleWeights)
        end
    end

    --actually update our network
    fullState.network:backward(batchTrainInputs, criterionGradInput)

    fullState.optimizer(function() return fullState.trainSetLoss, fullState.gradParams end,
      fullState.params, fullState.optimSettings)
    
    --todo: here we should calculate any classification errors before we clear out the outputs for the next minibatch so that we can save computation
    if fullState.trainingIteration == 1 then
      print('minibatch #: ', miniBatchIdx, ' took ', torch.toc(start2), ' seconds')
    end
  end
  if fullState.trainingIteration == 1 then
    print('One iteration through dataset took: ', torch.toc(start), 'seconds')
  end

	if fullState.trainingIteration % 100 == 0 then
		print('_________________________________')
		print('----ITERATION: ' .. fullState.trainingIteration .. '-------')
		print('Training Loss: ')
		print(fullState.trainSetLoss[fullState.trainingIteration])
		print('')
	end
end

return M
