--an abstract class
local InputData = torch.class('sleep_eeg.InputData')

function InputData:getExampleInput()
	error('this should be implemented by a child class')
end

function InputData:getAllData()
	error('this should be implemented by a child class')
end

function InputData:getAllTargets()
	error('this should be implemented by a child class')
end

function InputData:size()
	error('this should be implemented by a child class')
end

