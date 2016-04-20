if not dotrc then
	dofile('dotrc.lua')
end

--literally just making a namespace - not much of a coherent module here
sleep_eeg = {}

--anywhere in this module we will assume _fbd.enter() will give us a debugger
if dotrc.has_debugger then
  _fbd = require 'fb.debugger'
end

--add submodule
sleep_eeg.param_sweep = require 'ParamSweep.lua'
sleep_eeg.utils = require 'utils.lua' --note this goes first since other modules depend on it
sleep_eeg.models = require 'models.lua'
sleep_eeg.optimizers = require 'optimizers.lua'
sleep_eeg.drivers = require 'drivers.lua'
sleep_eeg.hooks = require 'hooks.lua'
sleep_eeg.terminators = require 'terminators.lua'
sleep_eeg.apply_network = require 'apply_network.lua'

--add our torch classes
dofile 'InputData.lua'
dofile 'SimData.lua'
dofile 'State.lua'
dofile 'CVData.lua'
dofile 'CVBySubjData.lua'
dofile 'SingleSubjData.lua'
dofile 'adam_log.lua'
dofile 'sgd_log.lua'
--this should get exported to its own module
dofile 'DataFrame.lua'
dofile 'SubsetConfusionMatrix.lua'
dofile 'TemporalSmoothing.lua'


--this is the end of the module definition
------------------------------------------------------------------------

return sleep_eeg
