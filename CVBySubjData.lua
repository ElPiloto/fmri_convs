--[[
    -- TODO: 
    -- add size()
    -- implement InputData interface
--]]

local CVBySubjData = torch.class('sleep_eeg.CVBySubjData', 'sleep_eeg.CVData')

-- we hardcode this because we want to never peek at the test data, but if we 
-- decrease the percent_test all nilly-willy as a parameter, then we'd be looking
-- at test data
CVBySubjData.PERCENT_TEST = 15

function CVBySubjData:__init(...)
  local args, filename, do_kfold_split, percent_valid, percent_train,
  use_subjects_as_targets, num_folds, fold_number, subj_data_args = dok.unpack(
    {...},
    'CVBySubjData',
    'Loads subject data and splits it into training, validation, and test sets',
    {arg='filename', type='string', help='MAT filename to load subj data. see ' .. 
      'CVBySubjData:__loadSubjData for variables expected in file', req=true},
    {arg='do_kfold_split',type='boolean', help='should we split our data into ' .. 
      'folds? If not, train,test,valid sets will be across all subjects into ' ..
      'a single "fold". if true, only produce train and test', req=true},
		{arg='percent_valid', type = 'number', help='Percent e.g. 15 to use for validation', req=false},
		{arg='percent_train', type ='number',help='Percent e.g. 50 to use for training', req = false},
		{arg='use_subjects_as_targets', type ='boolean', help='whether or not to return targets as {class, subjects}', req = true},
		{arg='num_folds', type ='number', help='percent of data in each fold, only required if do_kfold_split is true', req = false},
		{arg='fold_number', type ='boolean', help='which fold to test on, only required if do_kfold_split is true', req = false},
    {arg='subj_data_args', type='table', help='the table of all subj_data_args args', req = true}
   )

	--call our parent constructor 
	sleep_eeg.CVData.__init(self, ...)
	self.use_subjects_as_targets = use_subjects_as_targets
  self.predict_delta_memory = subj_data_args.predict_delta_memory
  self.shuffle_data = subj_data_args.shuffle_data
	self:__loadSubjData(filename, subj_data_args)
  --self:__initSubjIDAndClassInfo()
  if not do_kfold_split then
    self:__splitDataAcrossSubjs(percent_valid, percent_train)
  else
    self:__kFoldCrossValAcrossSubjs(num_folds, fold_number)
  end
  assert(not self.predict_delta_memory or self.extras.train.delta_memory, 'Specified that we predict the delta memory, but the loaded data does not have delta_memory field in it.')
end

function CVBySubjData:__loadSubjData(filename, subj_data_args)

	--[[
	currently this loads the following fields into loadedData:
		- SUBJECT_DATA_PATH: tells where this data came from when it was being
			exported (NOT ANYMORE)
		- args: settings used to export this data (NOT ANYMORE)
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
  local loadedData = sleep_eeg.utils.matioHelper(filename, {'data', 'subject_ids', 'labels', 'dimensions', 'conds','extras'})

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

  -- permute dimensions if our data is spatial because spatial conv modules expects
  -- last two dimensions to be "width" and "height", ultimately we want:
  -- trials x timepoints x 
  if subj_data_args.spatial_chans then
    --this is only for ERP data, we'd have to change this if we were dealing with frequency data
    self._all_data = torch.permute(self._all_data,1,4,2,3)
  end

  if subj_data_args.volumetric_conv then
    assert(subj_data_args.spatial_chans,'Can only use volumetric conv if spatial chans is true')
    --this is only for ERP data, we'd have to change this if we were dealing with frequency data
    --self._all_data = torch.permute(self._all_data,1,4,2,3)
    local trials = self._all_data:size(1)
    local times = self._all_data:size(2)
    local width = self._all_data:size(3)
    local height = self._all_data:size(4)
    --volumetric conv requires: # input planes, # times, widths, heights
    self._all_data = torch.view(self._all_data:contiguous(),trials, 1, times, width, height)
  end

    --we're going to convert subject_ids into indices so that we can use those 
	--as labels
	local subj_idx_dic = {}
	local subj_counter = 0
	local trial_counter = 0
	self._all_subj_idxs = torch.LongTensor(targets:numel())
	for k,v in ipairs(self.subjectIDs) do
	  trial_counter = trial_counter + 1
	  if not subj_idx_dic[v] then
		  subj_counter = subj_counter + 1
		  subj_idx_dic[v] = subj_counter
	  end
	  self._all_subj_idxs[trial_counter] = subj_idx_dic[v]
	end
  
	--here we're going to make a dataframe for the trial information which
	--gives us an easy way to query trials by subject number
	local numTrials = self._all_data:size(1)
	local trialNums = torch.range(1,numTrials):long():totable()

  local dfVariables = {trial = trialNums, subj_id = self.subjectIDs, class = self._all_targets:totable()}
  self.extra_fields = {}
  --add in additional variable for sleep
  if loadedData['extras']  then
    if loadedData['extras'].img_pres_num then
      self.img_pres_num = loadedData.extras.img_pres_num:squeeze()
      dfVariables['img_pres_num'] = self.img_pres_num:totable()
      table.insert(self.extra_fields, 'img_pres_num')

      --we also make a field based off of this one, that inversely weights values 
      --based on their presentation num
      self.weights_per_image = self.img_pres_num:pow(-1)
      dfVariables['weights_per_image'] = self.weights_per_image:totable()
      table.insert(self.extra_fields, 'weights_per_image')

      self.can_weight_loss = true
    end
    if loadedData['extras'].SOpow then
      --have to transpose this badboy
      self.SOpow = loadedData.extras.SOpow:squeeze()
      dfVariables['SOpow'] = self.SOpow:totable()
      table.insert(self.extra_fields, 'SOpow')
    end
    if loadedData['extras'].delta_memory then
      self.delta_memory = loadedData.extras.delta_memory:squeeze()
      dfVariables['delta_memory'] = self.delta_memory:totable()
      table.insert(self.extra_fields, 'delta_memory')
    end
    if loadedData['extras'].sound_IDs then
      self.sound_IDs = loadedData.extras.sound_IDs:squeeze()
      dfVariables['sound_IDs'] = self.sound_IDs:totable()
      table.insert(self.extra_fields, 'sound_IDs')
    end
  end

	self.dataframe = DataFrame.new(dfVariables)

	self.subj_ids = self.dataframe:uniqueValues('subj_id')
	self.classes = self.dataframe:uniqueValues('class')
  assert(#self.classnames == #self.classes, "Please contact your local " .. 
    "fisherman cause something fishy is up:\nNumber of class names " ..
    " doesn't match up with number of unique classes")
  self.num_classes = #self.classes

end

function CVBySubjData:cuda()
	require 'cutorch'
	require 'cunn'
	self._train_data  = self._train_data:cuda()
  if self._valid_data then
    self._valid_data = self._valid_data:cuda()
  end
	self._test_data = self._test_data:cuda()
end

function CVBySubjData:swapTemporalAndChannelDims()
	self._train_data  = self._train_data:transpose(2,3)
  if self._valid_data then
    self._valid_data = self._valid_data:transpose(2,3)
  end
	self._test_data = self._test_data:transpose(2,3)
end

function CVBySubjData:getTrainExampleWeights()
  return self.extras.train.weights_per_image
end

--make training, validation and test set by looking at each
--unique combination of subj_id,class and grabbing X% for train,
--Y% for validation, and Z% for testing
function CVBySubjData:__splitDataAcrossSubjs(...)
	local args, percent_valid, percent_train = dok.unpack(
		{...},
		'splitDataAcrossSubjs',
		'Make training, validation, and test set by looking at each unique combination\n'..
		'of (subj_id, class) and grab X% for train, Y% for validation, and Z% for test',
		{arg='percent_valid', type = 'number',help='Percent e.g. 15 to use for validation', req=true},
		{arg='percent_train', type ='number',help='Percent e.g. 50 to use for training', req = true}
		)
	assert(percent_valid + percent_train + CVBySubjData.PERCENT_TEST == 100, 
			"Error: Percentages don't add up to 100.  Recall test percentage " ..
			" is a static class member (CVBySubjData.PERCENT_TEST) = " ..
			CVBySubjData.PERCENT_TEST)

	--will contain all trial indices for the test
	local testSet = {}
	local trainSet = {}
	local validSet = {}

	local regularRNG = torch.getRNGState()
	torch.manualSeed('102387')
	local testRNG = torch.getRNGState()

	--torch.setRNGState(regularRNG)

	local allTrain, allValid, allTest = torch.LongTensor(), 
			torch.LongTensor(), torch.LongTensor()

	local subj_counts = {}
	local class_counts = {}
	local total_trial_count = 0
	self.num_subjects = 0
	for subj_idx, subj_id in ipairs(self.subj_ids) do
		self.num_subjects = self.num_subjects + 1
		for _, class in ipairs(self.classes) do
			local queryCondition = {subj_id = subj_id, class = class}
			local trials = self.dataframe:query('inter',queryCondition, {'trial'}).trial
			local numTrials = #trials
			local numTestTrials = math.floor(#trials*CVBySubjData.PERCENT_TEST/100)
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

			--we do it this way so that we can still get different splits between training
			--and validation data sets for different rng seeds
			local nonTestIdxes = randomOrder:gather(1,torch.range(numTestTrials+1,numTrials):long())

      --actually get the trial numbers
			testIdxes = trials:gather(1,testIdxes)
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

  --finally restore RNG state
  torch.setRNGState(regularRNG)
  self:__checkTrainValidTestSplit(allTrain, allValid, allTest)

  --this code lets us shuffle our data: either we use the REAL indices corresponding
  --to the train/valid/test trials when we select our labels/subject_ids/extra fields,
  --which i'm calling metadata, OR we shuffle the indices
  local trainMetaDataIdxes, validMetaDataIdxes, testMetaDataIdxes
  function getMetaDataIdxes(realIdxes)
    if self.shuffle_data then
      return torch.gather(realIdxes, 1, torch.randperm(realIdxes:numel()):long())
    else
      return realIdxes
    end
  end
  trainMetaDataIdxes = getMetaDataIdxes(allTrain)
  validMetaDataIdxes = getMetaDataIdxes(allValid)
  testMetaDataIdxes = getMetaDataIdxes(allTest)

  --finally let's consolidate our data
  self._train_data = CVBySubjData.__getRows(self._all_data,  allTrain)
  -- can use more gather (more efficient) for 1D data
  self._train_labels = torch.gather(self._all_targets, 1, trainMetaDataIdxes) 
  self._train_subjs = torch.gather(self._all_subj_idxs, 1, trainMetaDataIdxes) 

  self._valid_data = CVBySubjData.__getRows(self._all_data, allValid)
  self._valid_labels = torch.gather(self._all_targets, 1, validMetaDataIdxes)
  self._valid_subjs = torch.gather(self._all_subj_idxs, 1, validMetaDataIdxes) 

  self._test_data = CVBySubjData.__getRows(self._all_data, allTest)
  self._test_labels = torch.gather(self._all_targets, 1, testMetaDataIdxes)
  self._test_subjs = torch.gather(self._all_subj_idxs, 1, testMetaDataIdxes) 

  --finally re-order our extra_fields
  self.extras = {}
  self.extras.train = {}
  self.extras.test = {}
  self.extras.valid = {}
  for _,field_name in ipairs(self.extra_fields) do
    self.extras.train[field_name] = torch.gather(self[field_name], 1, trainMetaDataIdxes)
    self.extras.valid[field_name] = torch.gather(self[field_name], 1, validMetaDataIdxes)
    self.extras.test[field_name] = torch.gather(self[field_name], 1, testMetaDataIdxes)
    --clean up
    self[field_name] = nil
  end

  --and we no longer need our self._all_data OR self.dataframe
  self._all_data = nil
  self._all_targets = nil
  self._all_subj_idxs = nil
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

function CVBySubjData:__getOverlapOfIndices(one, two)
  local indices = {}
  for i = 1, one:numel() do
    for j = 1, two:numel() do
      if one[i] == two[j] then
        table.insert(indices, one[i])
      end
    end
  end
  return #indices == 0, indices
end

--train = train indices, valid = valid indices
function CVBySubjData:__checkTrainValidTestSplit(train, valid, test)
  local totalNumTrials = self._all_data:size(1)
  local filledInTrials = torch.LongTensor(totalNumTrials):zero()
  local totalDuplicates = 0
  local dupesPerSet = {0,0,0}
  local function checkTrials(trials, name, set_number)
    for t = 1, trials:numel() do
      local trial = trials[t]
      if filledInTrials[trial] ~= 0 then
        print('duplicate detected: ', trial, ' in ' .. name, 'already inserted for ', filledInTrials[trial])
        totalDuplicates = totalDuplicates + 1
        dupesPerSet[set_number] = dupesPerSet[set_number] + 1
      else
          filledInTrials[trial] = set_number
      end
    end
  end

  checkTrials(train, 'train',1)
  checkTrials(valid, 'valid',2)
  checkTrials(test, 'test',3)
  --finally check that they're all non-zero
  local nonMissedTrials = filledInTrials:nonzero()
  local totalTrialsMissed = totalNumTrials - nonMissedTrials:numel()
  print('Missed ', totalTrialsMissed, ' trials')
  print('Non-unique Duplicated trials', totalDuplicates)
  print('Duplicated trial counts for train, valid, test:')
  dupesPerSet[1] = dupesPerSet[1]/train:numel()
  dupesPerSet[2] = dupesPerSet[2]/valid:numel()
  dupesPerSet[3] = dupesPerSet[3]/test:numel()
  print(dupesPerSet)
end

function CVBySubjData:__checkTrainTestSplit(train, test)
  local totalNumTrials = self._all_data:size(1)
  local filledInTrials = torch.LongTensor(totalNumTrials):zero()
  local totalDuplicates = 0
  local dupesPerSet = {0,0}
  local function checkTrials(trials, name, set_number)
    for t = 1, trials:numel() do
      local trial = trials[t]
      if filledInTrials[trial] ~= 0 then
        print('duplicate detected: ', trial, ' in ' .. name, 'already inserted for ', filledInTrials[trial])
        totalDuplicates = totalDuplicates + 1
        dupesPerSet[set_number] = dupesPerSet[set_number] + 1
      else
          filledInTrials[trial] = set_number
      end
    end
  end

  checkTrials(train, 'train',1)
  checkTrials(test, 'test',2)
  --finally check that they're all non-zero
  local nonMissedTrials = filledInTrials:nonzero()
  local totalTrialsMissed = totalNumTrials - nonMissedTrials:numel()
  print('Missed ', totalTrialsMissed, ' trials')
  print('Non-unique Duplicated trials', totalDuplicates)
  print('Duplicated trial counts for train, test:')
  dupesPerSet[1] = dupesPerSet[1]/train:numel()
  dupesPerSet[2] = dupesPerSet[2]/test:numel()
  print(dupesPerSet)
end

function CVBySubjData:__kFoldCrossValAcrossSubjs(...)
	local args, num_folds, fold_idx  = dok.unpack(
		{...},
		'__kFoldCrossValAcrossSubjs',
		'Make data into k folds, preserving ratio of subject/class across each fold\n',
		{arg='num_folds', type = 'number',help='percent of data that goes into one fold e.g. 10', req=true},
		{arg='fold_idx', type = 'number',help='which fold to use for training and testing', req=true}
		)
  percent_in_fold = 100/num_folds
  assert(percent_in_fold >1 and percent_in_fold < 100, "Percent fold must be between 1 and 100 exclusively, ya boob!")
  assert(math.floor(100/percent_in_fold) == (100 / percent_in_fold), "num_folds must divide evenly into 100")

  local indicesBySubjClassPair = {}
  local countsBySubjClassPair = {}
  local foldSizePerSubjClassPair = {}
  local foldTrainIdxs = {}
  local foldTestIdxs = {}

  local numFolds = 100/percent_in_fold

  --local numTestTrials = math.floor(#trials*CVBySubjData.PERCENT_TEST/100)
  --local numNonTestTrials = numTrials - numTestTrials

  local regularRNG = torch.getRNGState()
	torch.manualSeed('102387')

	local subj_counts = {}
	local class_counts = {}
	local total_trial_count = 0
	self.num_subjects = 0

	for subj_idx, subj_id in ipairs(self.subj_ids) do
		self.num_subjects = self.num_subjects + 1
    indicesBySubjClassPair[subj_idx] = {}
    foldSizePerSubjClassPair[subj_idx] = {}

		for _, class in ipairs(self.classes) do

      --get data by subj/class pair
			local queryCondition = {subj_id = subj_id, class = class}
			local trials = self.dataframe:query('inter',queryCondition, {'trial'}).trial
      local numTrials = #trials

      --little book keeping
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


      --finally split into folds
      local randomOrder = torch.randperm(#trials):type('torch.LongTensor')
      indicesBySubjClassPair[subj_idx][class] = torch.LongTensor(trials):index(1,randomOrder)

      local foldSize = math.floor(#trials * percent_in_fold/100)

      --finally for each 
      for k = 1, numFolds do
        --these are the indices of the test fold, everything else will be the indices for training for this fold
        local startIdx = (k-1)*foldSize + 1
        local endIdx = math.min(#trials,startIdx + foldSize - 1)
        local testIdxs = torch.range(startIdx, endIdx):long()

        --training indices are anything before and after the testing indices
        --check "before" indices
        local trainIdxs = torch.LongTensor()
        if k > 1 then
          trainIdxs = torch.range(1,startIdx-1):long()
        end

        if endIdx+1 <= #trials then --check "after" indices
          local afterFoldIndices = torch.range(endIdx+1, #trials):long()

          if trainIdxs:numel()==0 then 
            trainIdxs = afterFoldIndices
          else
            trainIdxs = torch.cat(trainIdxs,afterFoldIndices,1)
          end
        end

        --finally store them
        if not foldTestIdxs[k] then
          foldTestIdxs[k] = indicesBySubjClassPair[subj_idx][class]:index(1,testIdxs)
        else -- we concatenate
          foldTestIdxs[k] = torch.cat(foldTestIdxs[k],indicesBySubjClassPair[subj_idx][class]:index(1,testIdxs))
        end

        if not foldTrainIdxs[k] then
          foldTrainIdxs[k] = indicesBySubjClassPair[subj_idx][class]:index(1,trainIdxs)
        else -- we concatenate
          foldTrainIdxs[k] = torch.cat(foldTrainIdxs[k],indicesBySubjClassPair[subj_idx][class]:index(1,trainIdxs))
        end

      end --finish inserting trials into folds

    end
  end

  --finally restore RNG state
  torch.setRNGState(regularRNG)
  local allTrain = foldTrainIdxs[fold_idx]
  local allTest = foldTestIdxs[fold_idx]
  self:__checkTrainTestSplit(allTrain, allTest)

  --this code lets us shuffle our data: either we use the REAL indices corresponding
  --to the train/valid/test trials when we select our labels/subject_ids/extra fields,
  --which i'm calling metadata, OR we shuffle the indices
  local trainMetaDataIdxes, testMetaDataIdxes
  function getMetaDataIdxes(realIdxes)
    if self.shuffle_data then
      return torch.gather(realIdxes, 1, torch.randperm(realIdxes:numel()):long())
    else
      return realIdxes
    end
  end
  trainMetaDataIdxes = getMetaDataIdxes(allTrain)
  testMetaDataIdxes = getMetaDataIdxes(allTest)

  --finally let's consolidate our data
  self._train_data = CVBySubjData.__getRows(self._all_data,  allTrain)
  -- can use more gather (more efficient) for 1D data
  self._train_labels = torch.gather(self._all_targets, 1, trainMetaDataIdxes) 
  self._train_subjs = torch.gather(self._all_subj_idxs, 1, trainMetaDataIdxes) 

  self._test_data = CVBySubjData.__getRows(self._all_data, allTest)
  self._test_labels = torch.gather(self._all_targets, 1, testMetaDataIdxes)
  self._test_subjs = torch.gather(self._all_subj_idxs, 1, testMetaDataIdxes) 

  --finally re-order our extra_fields
  self.extras = {}
  self.extras.train = {}
  self.extras.test = {}
  for _,field_name in ipairs(self.extra_fields) do
    self.extras.train[field_name] = torch.gather(self[field_name], 1, trainMetaDataIdxes)
    self.extras.test[field_name] = torch.gather(self[field_name], 1, testMetaDataIdxes)
    --clean up
    self[field_name] = nil
  end


  --and we no longer need our self._all_data OR self.dataframe
  self._all_data = nil
  self._all_targets = nil
  self._all_subj_idxs = nil
  self.dataframe = nil

  --and now let's do our normalization
  self._mean, self._std = sleep_eeg.utils.normalizeData(self._train_data)
  sleep_eeg.utils.normalizeData(self._test_data, self._mean, self._std)

  self._num_folds =  num_folds
  self._fold_num = fold_idx
  self._subj_counts = subj_counts
  self._class_counts = class_counts
  self._total_trial_count = total_trial_count
  self._subj_class_pair_counts = countsBySubjClassPair
  print(self:__tostring())
end

function CVBySubjData:__tostring()
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

  if self._fold_num then
    outStr = outStr .. 'Split breakdown for fold ' .. self._fold_num .. ' of ' 
      .. self._num_folds .. ' :\n=================\n'
  else
    outStr = outStr .. 'Split breakdown:\n=================\n'
  end
  outStr = outStr .. 'Train: ' .. self._train_data:size(1) .. '\n'
  if self._valid_data then
    outStr = outStr .. 'Valid: ' .. self._valid_data:size(1) .. '\n'
  end
  outStr = outStr .. 'Test: ' .. self._test_data:size(1) .. '\n'

	return outStr
end

function CVBySubjData.__getRows(source, idxs)
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



--this function will create a training, validation and test set specific to this
--cross-validation. example:
--splitDataBasedOnFold(0.75, 2) will use all data except for the 2nd subject's
--for training, and then split subject 2's data into 75% for the validation 
--and 25% for the testing. Importantly, it will always generate the same exact
--test data set.
function CVBySubjData:splitDataLOSO(prcntValidation, foldNumber)
	error('Not yet implemented')
	--assert(prcntValidation > 0 and prcntValidation < 1  
		--and foldNumber > 0 and foldNumber < self.num_subjects)
	--local prcntTrain = 1 - prcntValidation

	--local testSet = {}
	--local nonTestSet = {}

	--local rngState = torch.getRNGState()
	--torch.manualSeed(102387)

	--for class_idx = 1, self.num_classes do

	--end

	----restore random state now that we've chosen
	--torch.setRNGState(rngState)

end

function CVBySubjData:getTrainData()
	return self._train_data
end

function CVBySubjData:getTrainTargets()
  if self.predict_delta_memory then
    return self.extras.train.delta_memory
  else
    if self.use_subjects_as_targets then
      return {self._train_labels, self._train_subjs}
    else
      return self._train_labels
    end
  end
end

function CVBySubjData:getTestData()
	return self._test_data
end

function CVBySubjData:getTestTargets()
  if self.predict_delta_memory then
    return self.extras.test.delta_memory
  else
    if self.use_subjects_as_targets then
      return {self._test_labels, self._test_subjs}
    else
      return self._test_labels
    end
  end
end

function CVBySubjData:getValidData()
	return self._valid_data
end

function CVBySubjData:getValidTargets()
  if self.predict_delta_memory then
    return self.extras.valid.delta_memory
  else
    if self.use_subjects_as_targets then
      return {self._valid_labels, self._valid_subjs}
    else
      return self._valid_labels
    end
  end
end

function CVBySubjData:size(...)
	return self._train_data:size(...)
end

--NOT BEING USED
--function CVBySubjData:__combineSubjIDsAndTargets()
		--return trials_by_subj_and_class
--end

--function CVBySubjData:__initSubjIDAndClassInfo()
	--local tablex = require 'pl.tablex'
	--require 'torchx' --adds torch.group function

	----the two function calls below do the same thing, but one all_data is a table
	----and the other is atensor, hence needing different methods to use them
	--self.trials_per_subj= tablex.index_map(self.subjectIDs)
	--self.num_subjects = #self.trials_per_subj:keys()
	--self.trials_per_class = torch.group(self.targets)
	--self.num_classes = 0
	----clean up a little bit of space since torch.group returns the values as keys
	----and also as a value for that key, but we only want the indices stored for the key
	--for k,v in pairs(self.trials_per_class) do
		--self.trials_per_class[k] = v.idx
		--self.num_classes = self.num_classes + 1
	--end

	--local trials_by_subj_and_class = {}
	--local numTrials = self.targets:size(1)
	--for trialIdx = 1, numTrials do
		--local thisKey = {self.subjectIDs[trialIdx], self.targets[trialIdx]}
		--if not trials_by_subj_and_class[thisKey] then
			--trials_by_subj_and_class[thisKey] =  {trialIdx}
		--else
			--table.append(trials_by_subj_and_class[thisKey])
		--end
	--end
	--self.trials_by_subj_and_class = trials_by_subj_and_class

--end


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

