local M = {}

--[[
                     <((((((\\\
                     /      . }\
                     ;--..--._|}
  (\                 '--/\--'  )
   \\                | '-'  :'|
    \\               . -==- .-|
     \\               \.__.'   \--._
     [\\          __.--|       //  _/'--.
     \ \\       .'-._ ('-----'/ __/      \
      \ \\     /   __>|      | '--.       |
       \ \\   |   \   |     /    /       /
        \ '\ /     \  |     |  _/       /
         \  \       \ |     | /        /
          \  \      \        /
--]]

M.trainAndValidAvgClassAccuracyHigh = function(simState, avgClassAccThresh)
	--returns whether or not we should terminate early
	local shouldTerminateEarly = false
	--these two values are defined by another hook (hooks.confusionMatrix)
	assert(simState['train_confMatrix'] and simState['valid_confMatrix'])
	assert(avgClassAccThresh)
	simState['train_confMatrix']:updateValids() --updates confusion matrix
	simState['valid_confMatrix']:updateValids() --updates confusion matrix
	--get train and valid acc (this is stored in .totalValid inside the confMatrix for some reason)
	if (simState['train_confMatrix'].totalValid > avgClassAccThresh)
		and (simState['valid_confMatrix'].totalValid > avgClassAccThresh) then
		shouldTerminateEarly = true
	end

	return shouldTerminateEarly

end

return M
