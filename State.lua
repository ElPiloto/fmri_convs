local pretty = require 'pl.pretty'
local utils = sleep_eeg.utils

local State, super = torch.class('sleep_eeg.State')

function State:__init(filename)
	self._loadedFromFile = nil
	self._saveToFile = nil
	self._keysToSave = {}
  self._keysNotSaved = {}

	if filename ~= nil and type(filename) == 'string' then
		-- means one of two things:
		-- 1. we are specifying a file to initialize our state from, where that file was previously
		-- populated by a call to state:save(filename)

		-- 2. the filename currently doesn't exist, but we want to automatically save there
		self._saveToFile = filename

		--case 1 requires that we actually load things
		if utils.fileExists(filename) then
			local loaded = torch.load(filename)

			--here we go through and load them back in as key/value pairs
			for k,v in pairs(loaded) do
				self[k] = v
				self._keysToSave[k] = true
			end

			self._loadedFromFile = filename
		end
	end
end

function State:add(key, value, shouldSave)
	--key = string key for table
	--value = value stored
	--shouldSave = boolean, should we save this key when self:save() is called?
	--only adds if it doesn't already exist, meant to use with this idiom:
	--state.myKey or state:add('myKey',someValue,false)
	self[key] = value
	if shouldSave then
		self._keysToSave[key] = true
  else
    self._keysNotSaved[key] = true
	end
end

function State:save(output_file)
	-- we don't need to specify the output_file if we specified it when we created
	-- this function
	if output_file == nil then
		output_file = self._saveToFile 
				or error('Need to specify where to save to in call to State:save() OR at time or initialization i.e. "state = sleep_eeg.State(saveFileName)"')
	end
	local temp = {}
	for k,v in pairs(self._keysToSave) do
		temp[k] = self[k]
	end
	torch.save(output_file, temp)
end

function State:__tostring()
  local outStr = ''
  if self._loadedFromFile then
    outStr = 'Loaded from and saves to: ' .. sleep_eeg.utils.fileToURI(self._loadedFromFile) .. '\n'
  else
    outStr = 'Saves to: ' .. sleep_eeg.utils.fileToURI(self._saveToFile) .. '\n'
  end
  outStr = outStr .. 'Saves the following fields:\n'
  for k,v in pairs(self._keysToSave) do
    outStr = outStr .. '\t-' .. k .. '\n'
  end
  outStr = outStr .. '\nDoes NOT save the following fields:\n'
  for k,v in pairs(self._keysNotSaved) do
    outStr = outStr .. '\t-' .. k .. '\n'
  end
  return outStr

end
