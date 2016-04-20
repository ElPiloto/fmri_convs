--[[
    -- TODO: 
    -- add size()
    -- implement InputData interface
--]]

local SingleSubjData = torch.class('sleep_eeg.SingleSubjData', 'sleep_eeg.CVData')

-- we hardcode this because we want to never peek at the test data, but if we 
-- decrease the percent_test all nilly-willy as a parameter, then we'd be looking
-- at test data
SingleSubjData.PERCENT_TEST = 15

function SingleSubjData:__init(...)
  local args, filename, subj_idx, percent_valid, percent_train = dok.unpack(
    {...},
    'SingleSubjData',
    'Loads subject data and splits it into training, validation, and test sets',
    {arg='filename', type='string', help='MAT filename to load subj data. see ' .. 
      'SingleSubjData:__loadSubjData for variables expected in file', req=true},
		{arg='subj_idx', type = 'number', help='Subject number, NOT subject ID i.e. 4, not 115 ', req=true},
		{arg='percent_valid', type = 'number', help='Percent e.g. 15 to use for validation', req=true},
		{arg='percent_train', type ='number',help='Percent e.g. 50 to use for training', req = true}
   )

	--call our parent constructor 
	sleep_eeg.CVData.__init(self, ...)
  self._subj_idx = subj_idx
	self:__loadSubjData(filename)
  self:__splitData(percent_valid, percent_train)
end


function SingleSubjData:__loadSubjData(filename)

	--[[
	currently this loads the following fields into loadedData:
		- SUBJECT_DATA_PATH: tells where this data came from when it was being
			exported 
		- args: settings used to export this data
		- conds: table which tells us which indices belong to which classes
		- dimensions: table with the following fields:
				- freqs
				- total_num_features
				- times
				- chans 
				- shape
		- data: data tensor with [num_trials x num_chans x num_timepoints]
		- labels: (integer )class indicator tensor with [num_trials x 1] dimensions
		- subject_ids: table of string subj id, with indicies [1 to num_trials]
	--]]
  local loadedData = sleep_eeg.utils.matioHelper(filename, {'data', 'subject_ids', 'labels', 'dimensions', 'conds'})
  assert(loadedData['data'] and loadedData['subject_ids'] and 
    loadedData['labels'] and loadedData['dimensions'] and loadedData['conds'], 
    'One of the following variables is expected, but not present in file:' ..
    'data, subject_ids, data, labels, dimensions, conds')

	self._all_data = loadedData['data']:transpose(2,3)
  loadedData['data'] = nil
	self.subjectIDs = loadedData['subject_ids']
	local targets = loadedData['labels']
	self._all_targets = torch.squeeze(targets) --remove singleton dimension
	self.dimensions = loadedData['dimensions']
  self.classnames = loadedData['conds']
  
	--here we're going to make a dataframe for the trial information which
	--gives us an easy way to query trials by subject number
	local numTrials = self._all_data:size(1)
	local trialNums = torch.range(1,numTrials):long():totable()

	self.dataframe = DataFrame.new({trial = trialNums, 
			subj_id = self.subjectIDs, class = self._all_targets:totable()})

	self.subj_ids = self.dataframe:uniqueValues('subj_id')
	self.classes = self.dataframe:uniqueValues('class')
  assert(#self.classnames == #self.classes, "Please contact your local " .. 
    "fisherman cause something fishy is up:\nNumber of class names " ..
    " doesn't match up with number of unique classes")
  self.num_classes = #self.classes

end

--make training, validation and test set by looking at each
--unique combination of subj_id,class and grabbing X% for train,
--Y% for validation, and Z% for testing
function SingleSubjData:__splitData(...)
	local args, percent_valid, percent_train = dok.unpack(
		{...},
		'splitDataAcrossSubjs',
		'Make training, validation, and test set by looking at each unique combination\n'..
		'of (subj_id, class) and grab X% for train, Y% for validation, and Z% for test',
		{arg='percent_valid', type = 'number',help='Percent e.g. 15 to use for validation', req=true},
		{arg='percent_train', type ='number',help='Percent e.g. 50 to use for training', req = true}
		)
	assert(percent_valid + percent_train + SingleSubjData.PERCENT_TEST == 100, 
			"Error: Percentages don't add up to 100.  Recall test percentage " ..
			" is a static class member (SingleSubjData.PERCENT_TEST) = " ..
			SingleSubjData.PERCENT_TEST)

	--will contain all trial indices for the test
	local testSet = {}
	local trainSet = {}
	local validSet = {}

	local regularRNG = torch.getRNGState()
	torch.manualSeed('102387')

	local allTrain, allValid, allTest = torch.LongTensor(), 
			torch.LongTensor(), torch.LongTensor()

	local subj_counts = {}
	local class_counts = {}
	local total_trial_count = 0
	for subj_idx, subj_id in ipairs(self.subj_ids) do
    if subj_idx == self._subj_idx then
      self._subj_id = subj_id

      for _, class in ipairs(self.classes) do
        local queryCondition = {subj_id = subj_id, class = class}
        local trials = self.dataframe:query('inter',queryCondition, {'trial'}).trial
        local numTrials = #trials
        local numTestTrials = math.floor(#trials*SingleSubjData.PERCENT_TEST/100)
        local numNonTestTrials = numTrials - numTestTrials
        local numValidTrials = math.floor(numNonTestTrials*percent_valid/100)
        local numTrainTrials = numTrials - numTestTrials - numValidTrials
        trials = torch.LongTensor(trials) --convert for easier indexing

        --keep counts per subject
        if not subj_counts[subj_id] then
          subj_counts[subj_id] = numTrials
        else
          subj_counts[subj_id] = subj_counts[subj_id] + numTrials
        end

        --keep counts per subject
        if not class_counts[class] then
          class_counts[class] = numTrials
        else
          class_counts[class] = class_counts[class] + numTrials
        end
        --keep total counts
        total_trial_count = total_trial_count + numTrials

        --now we pick some fixed proportion to be the test set, which is always the same regardless of the rng seed specified elsewhere in the program
        local indices = torch.range(1,numTrials):long()

        --pick test
        local randomOrder = torch.randperm(numTrials):long()
        local testIdxes = 
          randomOrder:gather(1,torch.range(1,numTestTrials):long())
        testIdxes = trials:gather(1,testIdxes)

        --we do it this way so that we can still get different splits between training
        --and validation data sets for different rng seeds
        local nonTestIdxes = randomOrder:gather(1,torch.range(numTestTrials+1,numTrials):long())
        nonTestIdxes = trials:gather(1,nonTestIdxes)
        
        --pick validation/training indices
        local randomOrder = torch.randperm(nonTestIdxes:numel()):long()
        --local validIdxes = trials[{randomOrder[{{1,numValidTrials}}]:totable()}]
        
        local validIdxes = 
          randomOrder:gather(1,torch.range(1,numValidTrials):long())
        validIdxes = nonTestIdxes:gather(1,validIdxes)

        local trainIdxes = torch.range(numValidTrials+1,numNonTestTrials):long()
        trainIdxes = randomOrder:gather(1,trainIdxes)
        trainIdxes = nonTestIdxes:gather(1,trainIdxes)

        if allTrain:numel() == 0 then
          allTrain = trainIdxes
          allValid = validIdxes
          allTest = testIdxes
        else
          allTrain = torch.cat(allTrain,trainIdxes)
          allValid = torch.cat(allValid,validIdxes)
          allTest = torch.cat(allTest,testIdxes)
        end
      end
    end
	end
	
  --finally restore RNG state
  torch.setRNGState(regularRNG)

  --finally let's consolidate our data
  self._train_data = SingleSubjData.__getRows(self._all_data,  allTrain)
  -- can use gather (more efficient) for 1D data
  self._train_labels = torch.gather(self._all_targets, 1, allTrain) 

  self._valid_data = SingleSubjData.__getRows(self._all_data, allValid)
  self._valid_labels = torch.gather(self._all_targets, 1, allValid)

  self._test_data = SingleSubjData.__getRows(self._all_data, allTest)
  self._test_labels = torch.gather(self._all_targets, 1, allTest)

  --and we no longer need our self._all_data OR self.dataframe
  self._all_data = nil
  self._all_targets = nil
  self.dataframe = nil

  --and now let's do our normalization
  self._mean, self._std = sleep_eeg.utils.normalizeData(self._train_data)
  sleep_eeg.utils.normalizeData(self._valid_data, self._mean, self._std)
  sleep_eeg.utils.normalizeData(self._test_data, self._mean, self._std)
  
  self._subj_counts = subj_counts
  self._class_counts = class_counts
  self._total_trial_count = total_trial_count
  print(self:__tostring())
end

function SingleSubjData:getSubjID()
	return self._subj_id
end

function SingleSubjData:__tostring()
	local outStr = 'Subject breakdown:\n===================\n'
	for subj, count in pairs(self._subj_counts) do
		outStr = outStr .. 'Subj ' .. subj .. ': ' .. 
		string.format('%.1f', 100*count/self._total_trial_count) .. 
		'% (' .. count .. ')\n'
	end

	outStr = outStr .. 'Class breakdown:\n=================\n'
	for class, count in pairs(self._class_counts) do

		outStr = outStr .. 'Class: ' .. self.classnames[class] .. 
		': ' .. string.format('%.1f', 100*count/self._total_trial_count) 
		.. '% (' .. count .. ')\n'

	end

	outStr = outStr .. 'Split breakdown:\n=================\n'
  outStr = outStr .. 'Train: ' .. self._train_data:size(1) .. '\n'
  outStr = outStr .. 'Valid: ' .. self._valid_data:size(1) .. '\n'
  outStr = outStr .. 'Test: ' .. self._test_data:size(1) .. '\n'

	return outStr
end

function SingleSubjData.__getRows(source, idxs)
  local numIdxs = idxs:size(1)
  local sizes = source:size()
  local outputSize = {numIdxs}
  for sizeIdx = 2, #sizes do
    table.insert(outputSize,sizes[sizeIdx])
  end
  local outputSize = torch.LongStorage(outputSize)
  local output = torch.Tensor():typeAs(source):resize(outputSize)
  
  for idxIdx = 1, numIdxs do
    local sourceElement = source[idxs[idxIdx]]
    if torch.type(sourceElement) == 'number' then
      output[idxIdx] = sourceElement
    else
      output[idxIdx]:copy(sourceElement)
    end
  end
  return output
end


function SingleSubjData:getTrainData()
	return self._train_data
end

function SingleSubjData:getTrainTargets()
	return self._train_labels
end

function SingleSubjData:getTestData()
	return self._test_data
end

function SingleSubjData:getTestTargets()
	return self._test_labels
end

function SingleSubjData:getValidData()
	return self._valid_data
end

function SingleSubjData:getValidTargets()
	return self._valid_labels
end

function SingleSubjData:size(...)
	return self._train_data:size(...)
end

--takes a while to run, but is the easiest way to check that we aren't
--accidentally using trials twice
local function test_splitDataAcrossSubjs(train, test, valid)
	for i = 1, train:size(1) do
		for j = 1, test:size(1) do
			assert(train[i] ~= test[j])
			for k = 1, valid:size(1) do
				assert(train[i] ~= valid[k])
				assert(test[j] ~= valid[k])
			end
		end
	end
end

