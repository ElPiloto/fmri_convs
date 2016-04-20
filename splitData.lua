local M = {}

local function copyIdxsInto(source, target, idxs)
  local numIdxs = idxs:size(1)
  for idxIdx = 1, numIdxs do
    local sourceElement = source[idxs[idxIdx]]
    if torch.type(sourceElement) == 'number' then
      target[idxIdx] = sourceElement
    else
      target[idxIdx]:copy(sourceElement)
    end
  end
end


local function concatTableOrTensor(data, newData)
  local shouldUntableData = false
  if torch.type(data) ~= 'table' then
    data = {data}
    newData = {newData}
    shouldUntableData = true
  end
  for dataIdx = 1, #newData do
    if data[dataIdx] then
      data[dataIdx] = torch.cat(data[dataIdx],newData[dataIdx], 1)
    else
      data[dataIdx] = newData[dataIdx]:clone()
    end
  end

  if shouldUntableData then
    return data[1]
  end
  return data

end


M.splitDataThreewaysPreserveSubjects = function(data, subjectIDs, 
    prcntTrain, prcntTest, rngSeed)
  assert(data and subjectIDs, 
      'You ask me to split data, but then give me no data???!!?')
  assert(prcntTrain + prcntTest < 1)
    --'You need to leave some examples for validation otherwise this FN would be ' 
    --.. ' split TWO ways not THREE ways")
  local subjectStrs, strIdxs = utils.getUniqueStrings(subjectIDs)
  strIdxs = torch.Tensor{strIdxs}

  local trainSet = {}
  local testSet = {}
  local validationSet = {}

  local numSubjs = #subjectStrs
  local numExamples = strIdxs:numel()
  for subjIdx = 1, numSubjs do
    --find which examples belong to our subject
    
    local currSubjIdxs = 
        torch.linspace(1, numExamples, numExamples)[strIdxs:eq(subjIdx)]

    --extract those examples from data
    local subjData = utils.getDataFromTableOrTensor(data,currSubjIdxs)

    --treat it like a regular splitDataThreeways call
    local trainTmp, testTmp, validationTmp = M.splitDataThreeways(subjData,
        prcntTrain, prcntTest, rngSeed)
    concatTableOrTensor(trainSet, trainTmp)
    concatTableOrTensor(testSet, testTmp)
    concatTableOrTensor(validationSet, validationTmp)
  end

  return trainSet, testSet, validationSet, subjectStrs, strIdxs
end

M.splitDataThreeways = function(data, prcntTrain, prcntTest, rngSeed)
  assert(data, 'You ask me to split data, but then give me no data???!!?')
  assert(prcntTrain + prcntTest < 1)
    --'You need to leave some examples for validation otherwise this FN would be ' 
    --.. ' split TWO ways not THREE ways")

  --make sure we don't disturb RNG state by setting RNG seed
  local rngState = torch.getRNGState()
  if rngSeed then
    torch.manualSeed(rngSeed)
  end

  local exampleData
  if torch.type(data) == 'table' then
    exampleData = data[1]
  else
    exampleData = data
    data = {exampleData}
  end

  local numExamples = exampleData:size(1)
  local randomOrder = torch.randperm(numExamples)
  local numTrainExamples = math.floor(prcntTrain*numExamples)
  local numTestExamples = math.floor(prcntTest*numExamples)
  local trainIdxs = randomOrder[{{1,numTrainExamples}}]
  local testIdxs = randomOrder[{{numTrainExamples+1,
      numTrainExamples+numTestExamples}}]
  local validationIdxs = randomOrder[{{numTrainExamples+numTestExamples+1, 
      numExamples}}]
  local numValidationExamples = validationIdxs:size(1)

  local trainSet = {}
  local testSet = {}
  local validationSet = {}
  for dataIdx = 1, #data do
    local currentData = data[dataIdx]

    local featureSize = currentData:size():totable()
    table.remove(featureSize,1) -- remove the # of examples from the size
    local trainSize
    if #featureSize == 0 then
      trainSize = {numTrainExamples}
      testSize = {numTestExamples}
      validationSize = {numValidationExamples}
    else
      trainSize = {numTrainExamples, unpack(featureSize)}
      testSize = {numTestExamples, unpack(featureSize)}
      validationSize = {numValidationExamples, unpack(featureSize)}
    end


    trainSet[dataIdx] = torch.Tensor():typeAs(currentData)
    trainSet[dataIdx]:resize(unpack(trainSize))

    testSet[dataIdx] = torch.Tensor():typeAs(currentData)
    testSet[dataIdx]:resize(unpack(testSize))

    validationSet[dataIdx] = torch.Tensor():typeAs(currentData)
    validationSet[dataIdx]:resize(unpack(validationSize))

    copyIdxsInto(currentData, trainSet[dataIdx], trainIdxs)
    copyIdxsInto(currentData, testSet[dataIdx], testIdxs)
    copyIdxsInto(currentData, validationSet[dataIdx], validationIdxs)

  end
  torch.setRNGState(rngState)

  return trainSet, testSet, validationSet
end

M.testSplitDataThreeways = function()
  local data = torch.rand(10,10)
  --test we're splitting things into right groups
  local train, test, valid = M.splitDataThreeways(data, 0.8, 0.1,1)
  assert(train[1]:size(1) == 8)
  assert(test[1]:size(1) == 1)
  assert(valid[1]:size(1) == 1)
  --test we're splitting things into right groups asymmetrically
  local train, test, valid = M.splitDataThreeways(data, 0.7, 0.2,1)
  assert(train[1]:size(1) == 7)
  assert(test[1]:size(1) == 2)
  assert(valid[1]:size(1) == 1)
  --test random number seeds give us different pairings of the data
  local train2, test2, valid2 = M.splitDataThreeways(data, 0.7, 0.2,2)
  assert(train2[1]:sum(1) ~= train[1]:sum(1))
  assert(test2[1]:sum(1) ~= test[1]:sum(1))
  assert(valid2[1]:sum(1) ~= valid[1]:sum(1))
end

--M.testSplitDataThreeways()

return M
