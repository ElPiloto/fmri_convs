
local CVData = torch.class('sleep_eeg.CVData', 'sleep_eeg.InputData')

function CVData:__init(...)
	--self:loadData(...)
	--self:selectFold(...)
end

function CVData:loadData()
	error('this should be implemented by a child class')
end

--function CVData:makeFolds()
	--error('this should be implemented by a child class')
--end

function CVData:selectFold()
	error('this should be implemented by a child class')
end

