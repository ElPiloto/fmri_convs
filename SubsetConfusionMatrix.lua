--TODO: check if switching __replaceTarget and __replacePrediction "index()" call with
--[] operations makes this code faster

local optim = require 'optim'

local SCM, parent = torch.class('optim.SubsetConfusionMatrix', 'optim.ConfusionMatrix')

function SCM:__init(...)
  local args, allClassNames, subsetClassIdxs = dok.unpack(
	  {...},
	  'SubsetConfusionMatrix',
	  'Makes a confusion matrix that ignores certain outputs, useful when you want to enrich your classifier' ..
	  ' by adding training on extra classes, but really only care about performance on a subset of all the classses',
	  {arg='allClassNames', type ='table', help='table of class names',req = true},
	  {arg='subsetClassIdxs', type ='table', help='list-like table of class indexes to keep',req = true}
	)
	--this is for hte case where we're training a classifier on multiple classes, but 
	--we just want to consider the accuracy for a subset of those classes
	self._subsetClassNames = {}
	self._target2subsetIdxs = {} --a dictionary from original targets to subset target idx
	self._numSubclasses = #subsetClassIdxs
	self._subsetClassIdxs = torch.LongTensor(subsetClassIdxs)
	self._allClassNames = allClassNames
	local counter = 1
	for _, idx in ipairs(subsetClassIdxs) do
		self._target2subsetIdxs[idx] = counter
		table.insert(self._subsetClassNames, allClassNames[idx])
		counter = counter + 1
	end

	parent.__init(self,self._subsetClassNames)

end

function SCM:__shouldIncludeTargetIdx(targetIdx)
	assert(targetIdx)
	if self._target2subsetIdxs[target] then
		return true
	end
	return false
end

function SCM:__replacePrediction(prediction)
	--prediction = {scalar|vector|matrix}
	if type(prediction) == 'number' then
		return self._target2subsetIdxs[prediction]
	else
		assert(torch.isTensor(prediction),'Prediction must be scalar or Tensor')
		assert(prediction:nDimension() <= 2, 'Prediction must be either a vector or matrix')
		if prediction:nDimension() == 1 then
			return prediction:index(1, self._subsetClassIdxs)
		elseif prediction:nDimension() == 2 then
			return prediction:index(2, self._subsetClassIdxs)
		end
	end
	--if we've reached here, then we have some error that I haven't accounted for
	error('Unknown error')
end

function SCM:__replaceTarget(target)
	--target = {scalar|vector|matrix}
	if type(target) == 'number' then
		return self._target2subsetIdxs[target]
	end
	assert(torch.isTensor(target),'Target must be scalar or Tensor')
	assert(target:nDimension() <= 2, 'Target must be either a vector or matrix')
	if target:nDimension() == 1 then
		return target:index(1, self._subsetClassIdxs)
	elseif target:nDimension() == 2 then
		return target:index(2, self._subsetClassIdxs)
	end
	--if we've reached here, then we have some error that I haven't accounted for
	error('Unknown error')
end

function SCM:add(prediction, target)
	local p = self:__replacePrediction(prediction)
	local t = self:__replaceTarget(target)

	--if t is nil or all zeros means the target was for a class we don't care about
	if not t or (torch.isTensor(t) and t:sum() < 1) then
		return
	end
	parent.add(self, p, t)
end

function SCM:batchAdd(predictions, targets)
	for i = 1, targets:size(1) do
		local p = self:__replacePrediction(predictions[i])
		local t = self:__replaceTarget(targets[i])
		--if t is nil or all zeros that means the target was for a class we don't care about
    if t and not (torch.isTensor(t) and t:sum() < 1) then
		  parent.add(self, p, t)
    end

	end
end
