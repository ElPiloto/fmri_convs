local argcheck = require 'argcheck'

local SimData = torch.class('sleep_eeg.SimData', 'sleep_eeg.InputData')

function SimData:__init(args)
	--args = argcheck{
	   --pack=true,
	   --{name="num_examples_per_class", type="number", default=200},
	   --{name="num_classes", type="number", default=2},
	   --{name="num_channels", type="number", default=64},
	   --{name="num_timebins", type="number", default=20},
	   --{name="replay_bin_width", type="number", default=1},
	   --{name="num_replays_per_trial", type="number", default=3},
	   --{name="snr", type="number", default=0.5},
	   --{name="verbose", type="boolean", default=true}
	--}(args)
	--handle named arguments and defaults
	args.num_examples_per_class = args.num_examples_per_class or 200
	args.num_classes = args.num_classes or 2
	args.num_channels = args.num_channels or 64
	args.num_timebins = args.num_timebins or 20
	args.replay_bin_width = args.replay_bin_width or 1
	args.num_replays_per_trial = args.num_replays_per_trial or 3
	args.snr = args.snr or 0.5
	args.verbose = args.verbose or true
	args.num_validation_examples_per_class = args.num_validation_examples_per_class or 50
	args.num_test_examples_per_class = args.num_test_examples_per_class or 200

	if args.verbose then
		local old_verbose = args.verbose
		--we don't want to print the verbosity level
		args.verbose = nil
		print('Simulated data with:\n')
		print(args)
		args.verbose = old_verbose
	end
	local num_examples = args.num_examples_per_class * args.num_classes
	local num_validation_examples = args.num_validation_examples_per_class * args.num_classes
	local num_test_examples = args.num_test_examples_per_class * args.num_classes

	------------------------------------------------------------------------
	--now we generate data here:
	
	--let's make our templates using mean 0, var = 1 normal distribution
	self.template_variance = 1
	self.class_templates = torch.randn(args.num_classes,args.num_channels)

	--here we generate the noise (over which we'll stamp replay templates)
	--we always keep the same variance on our class templates, so the SNR tells us what the variance should be for our data
	self.noise_variance =  self.template_variance / args.snr

	--randn generates gaussian with mean = 0, variance = 1, which we can leverage to generate a mean 0, variance = S
	--y = a*x, where x ~ N(0,1), has var(y) = a^2 * var(x)^2
	--therefore, we should choose a = math.sqrt(var(y) / var(x)^2))
	local scaling_factor = math.sqrt(self.noise_variance)
	--make trainign data, which i call "all data" which is stupid and not descriptive and jsut wrong.
	self._all_data = torch.randn(num_examples,args.num_timebins,args.num_channels):mul(scaling_factor)
	self._all_labels = torch.LongTensor(num_examples)
	--make validation data set
	if num_validation_examples > 0 then
		self._valid_data = torch.randn(num_validation_examples,args.num_timebins,args.num_channels):mul(scaling_factor)
		self._valid_labels = torch.LongTensor(num_validation_examples)
	end
	--make test data
	if num_test_examples > 0 then
		self._test_data = torch.randn(num_test_examples,args.num_timebins,args.num_channels):mul(scaling_factor)
		self._test_labels = torch.LongTensor(num_test_examples)
	end

	if args.verbose then
		print('Sample noise variance:', torch.var(self._all_data))
	end

	--TODO: add support for wider timebins, for now jsut make sure no one tries 
	--to use tghis with a wider timebin
	assert(args.replay_bin_width == 1,'No support for replay_bin_width > 1 yet')

	local function generateData(data,labels)
		for trial_idx = 1, data:size(1) do
			local class_idx = (trial_idx-1) % args.num_classes + 1

			--TODO: Could be optimized
			local replayBins = torch.randperm(
				args.num_timebins)[{{1,args.num_replays_per_trial}}]

			--fit template into place
			for replayIdx = 1, args.num_replays_per_trial do
				--add signal
				data[{trial_idx,replayBins[replayIdx],{}}]:add(torch.rand(args.num_channels):add(self.class_templates[class_idx]))
			end

			labels[trial_idx] =  class_idx
		end


	end

	--make training date
	generateData(self._all_data, self._all_labels)

	------------------------------------------------------------------------
	--make valid data
	if num_validation_examples > 0 then
		generateData(self._valid_data,self._valid_labels)
	end

	------------------------------------------------------------------------
	--make test data
	if num_test_examples  > 0 then
		generateData(self._test_data, self._test_labels)
	end
	------------------------------------------------------------------------
	--finally normalize the training data and then apply the same normalization (with mean/std calc. from train)
	--on the test set
	local mean, std = sleep_eeg.utils.normalizeData(self._all_data)
	if num_validation_examples > 0 then
		sleep_eeg.utils.normalizeData(self._valid_data, mean, std)
	end
	if num_test_examples > 0 then
		sleep_eeg.utils.normalizeData(self._test_data, mean, std)
	end

end

function SimData:getTrainData()
	return self._all_data
end

function SimData:getTrainTargets()
	return self._all_labels
end

function SimData:getTestData()
	return self._test_data
end

function SimData:getTestTargets()
	return self._test_labels
end

function SimData:getValidData()
	return self._valid_data
end

function SimData:getValidTargets()
	return self._valid_labels
end



function SimData:size(...)
	return self._all_data:size(...)
end

