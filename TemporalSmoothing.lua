
require 'nn'

local TS, parent = torch.class('nn.TemporalSmoothing', 'nn.Module')


--static functions for making filters
TS.filters = {}

--make a truncated gaussian - here we sample a continuous gaussian
TS.filters.makeGaussian = function(...)
  local args, width, sigma, normalize, plot_filter = dok.unpack(
    {...},
    'nn.TemporalSmoothing.makeFilter',
    'will generate different filters for you',
    {arg='width', type = 'number', help = 'how many non-zero points in filter', default = 3},
    {arg='sigma', type = 'number', help = 'std dev', default = 0.25},
    {arg='normalize', type = 'number', help = 'normalize filter to sum to 1', default = false},
    {arg='plot_filter', type = 'boolean', help = 'should we plot the filter', default = false}
    )
  assert(width % 2 == 1 and width > 2, 'Width has to be odd number greater than 1')
  local mean = width/2 + 0.5
  local filter = torch.Tensor(width)

  for idx = 1, filter:numel() do
    filter[idx] = 1/(sigma*math.sqrt(2*math.pi)) *
		math.exp(-(math.pow((idx-mean)/(sigma*width),2)/2))
  end

  if normalize then
    filter:div(filter:sum())
  end

  if plot_filter then
	  require 'gnuplot'
	  gnuplot.plot(torch.range(1,width), filter)
  end

  return filter
end

function TS:__init(...)
  local args, filter, slide_size, normalize_filter, num_input_dims = dok.unpack(
    {...},
    'nn.TemporalSmoothing',
    'Convolves the input with the 1D filter specified',
    {arg='filter', type = 'table', help = 'tensor to use for smoothing', req = true},
    {arg='slide_size', type = 'number', help = 'how many points to slide when successively applying our filter; same as "dW" in nn.TemporalConvolution', default = 1},
    {arg='normalize_filter', type = 'boolean', help = 'should we normalize the filter to sum to 1', default = false},
    {arg='numInputDimensions', type = 'number', help = '1 (numTimePoints) or 2 (numTimePoints x numLayersToSmooth) ', default = 1}
    )
  assert(torch.isTensor(filter) and filter:nDimension() == 1)

  if normalize_filter then
    filter:div(filter:sum())
  end

  self._filter_size = filter:size(1)
  self._filter = filter
  self._slide_size = slide_size
  self._num_input_dims = num_input_dims

end

--tells you how big the output will be along the dimension we're smoothing,
--given that our smoothing operation does NOT zero-pad, hence input size 
--isn't the same as the output
function TS:getTemporalOutputSize(input_size)
  assert(self._filter_size and self._slide_size)
  return math.floor((input_size - self._filter_size)/self._slide_size) + 1
end

function TS:updateOutput(input)
  assert(input:nDimension() <= 3, 'Input can only be 1D (numTimePoints) or 2D (batch x numTimePoints) or (numTimePoints x numLayersToSmooth) or 3D ( batchx numTimePoints x numLayersToSmooth). Input has size:\n' .. input:size():__tostring__()) 
  local batch_mode = input:nDimension() == 2 and self._num_input_dims == 1 or input:nDimension() == 3
  if batch_mode then
    local batch_size = input:size(1)
    local output_size = self:getTemporalOutputSize(input:size(2))
    local num_layers = self._num_input_dims == 2 and input:size(3)  or 1
    if self._num_input_dims == 2 then
      self.output = torch.Tensor():typeAs(input):resize(batch_size,output_size,num_layers):zero()
    else
      self.output = torch.Tensor():typeAs(input):resize(batch_size,output_size):zero()
    end
    local indexDataFn = {}
	if self._num_input_dims == 2 then
		indexDataFn = function(batch_idx, time_idx, layer_idx)
			return {batch_idx,time_idx,layer_idx}
		end
	else
		indexDataFn = function(batch_idx, time_idx, layer_idx)
			return {batch_idx,time_idx}
		end
	end

    for b = 1, batch_size do
      for i = 1, output_size do
        for l = 1, num_layers do 
          for j = 1, self._filter_size do
            --this controls for the fact that if we have numInputDims == 2, we want to index data by [{b,out_time, layerNum}], but 
            local slide_add = (i-1)*self._slide_size
            f = self._filter[j] 
            self.output[indexDataFn(b,i,l)] = self._filter[j] * input[indexDataFn(b,slide_add+j,l)] + self.output[indexDataFn(b,i,l)]
          end
        end
      end
    end
  else
    local output_size = self:getTemporalOutputSize(input:size(1))

    local num_layers = self._num_input_dims == 2 and input:size(2) or 1
    if self._num_input_dims == 2 then
      self.output = torch.Tensor():typeAs(input):resize(output_size,num_layers):zero()
    else
      self.output = torch.Tensor():typeAs(input):resize(output_size):zero()
    end
    for i = 1, output_size do
      for l = 1, num_layers do 
        for j = 1, self._filter_size do
            --this controls for the fact that if we have numInputDims == 2, we want to index data by [{b,out_time, layerNum}], but 
            --if numInputDims == 1 then we want to index data by [{b,out_time}]
            local indexDataFn = {}
            if self._num_input_dims == 2 then
              indexDataFn = function(time_idx)
                return {time_idx,l}
              end
            else
              indexDataFn = function(time_idx)
                return {time_idx}
              end
            end

          local slide_add = (i-1)*self._slide_size
          self.output[indexDataFn(i)] = self._filter[j] * input[indexDataFn(slide_add+j)] + self.output[indexDataFn(i)]
        end
      end
    end
  end
  return self.output
end

function TS:updateGradInput(input, gradOutput)
  --assert(input:nDimension() <= 3, 'Input can only be 1D or 2D (batch mode)')
  assert(input:nDimension() <= 3, 'Input can only be 1D (numTimePoints) or 2D (batch x numTimePoints) or (numTimePoints x numLayersToSmooth) or 3D ( batchx numTimePoints x numLayersToSmooth). Input has size:\n' .. input:size():__tostring__()) 
  local batch_mode = input:nDimension() == 2 and self._num_input_dims == 1 or input:nDimension() == 3
  if batch_mode then
    local input_size = input:size(2)
    local batch_size = input:size(1)
    local output_size = self:getTemporalOutputSize(input:size(2))
    local num_layers = self._num_input_dims == 2 and input:size(3) or 1
    df_dx = torch.Tensor():typeAs(input):resize(output_size, input_size):zero()

      if self._num_input_dims == 2 then
        self.gradInput = torch.Tensor():typeAs(input):resize(batch_size, input_size, num_layers):zero()
      else
        self.gradInput = torch.Tensor():typeAs(input):resize(batch_size, input_size):zero()
      end

    --df_dx = the filter weight that x gets multiplied by to produce the out-th output in f,
    --these are the same across batches and layers (if we have them)
    for out_idx = 1, output_size do
      for filter_idx = 1, self._filter_size do
        local out_pos = (out_idx - 1)*self._slide_size + filter_idx
        df_dx[{out_idx, out_pos}] = self._filter[filter_idx]
      end
    end


    --this handles variable number of input dimensions: either 
    --num_input_dims = 2 i.e. [batch, inputs, layers] OR
    --num_input_dims = 1 i.e. [batch, inputs]
    local function dataIndexFn(batch_idx, layer_idx)
      if self._num_input_dims == 2 then
        return {{batch_idx}, {}, {layer_idx}} --[batch, inputs, layers]
      else
        return {{batch_idx}, {}} --batch x input x output
      end
    end

    --finally do: dJ/df * df/dx to get dJ/dx, where J = cost function
    --multiplication should look like: [J.num_dims, #outputs] x [# outputs, # inputs]
    for batch_idx = 1, batch_size do
      for layer_idx = 1, num_layers do
        self.gradInput[dataIndexFn(batch_idx, layer_idx)] = torch.mv(df_dx:t(),gradOutput[dataIndexFn(batch_idx, layer_idx)]:squeeze())
      end
    end

  else
    local input_size = input:size(1)
    local output_size = self:getTemporalOutputSize(input_size)
    local num_layers = self._num_input_dims == 2 and input:size(2) or 1

    local df_dx = torch.Tensor():typeAs(input):resize(input_size, output_size):zero()

    for out_idx = 1, output_size do
      for filter_idx = 1, self._filter_size do
        local out_pos = (out_idx - 1)*self._slide_size + filter_idx
        df_dx[{out_pos, out_idx}] = self._filter[filter_idx]
      end
    end

    if self._num_input_dims == 2 then
      self.gradInput = torch.Tensor():typeAs(input):resize(input_size, num_layers):zero()
    else
      self.gradInput = torch.Tensor():typeAs(input):resize(input_size):zero()
    end

	--if we have num_input_dims > 1 then we just do a matrix matrix multiplication
	if self._num_input_dims == 1 then
      self.gradInput = torch.mv(df_dx,gradOutput)
    elseif self._num_input_dims == 2 then
      self.gradInput = torch.mm(df_dx,gradOutput)
	end
  end
  return self.gradInput
end



TS.testSmoothingByEye = function()
  require 'gnuplot'
  local numTimePoints = 100
  --local inputSignal = torch.Tensor(3, numTimePoints):random()
  local inputSignal = torch.rand(50,numTimePoints)

  --our filter
  local filter = torch.Tensor{25, 50, 25}
  local filter = TS.filters.makeGaussian(5, 1, true)
  assert(filter:min() > 0, 'Somehow min filter')

  --single bump should go down
  inputSignal[{{},25}]:fill(10)
  --consecutive bumps should get stronger
  inputSignal[{{},49}]:fill(10)
  inputSignal[{{},50}]:fill(10)
  inputSignal[{{},51}]:fill(10)
  local m = nn.TemporalSmoothing(filter, 1, false, 1) --slide one-timepoint at a time, normalize to sum to 1
  local smoothedSignal = m:forward(inputSignal)
  local softMaxedSignal = nn.SoftMax():forward(smoothedSignal)
  local softMaxedInputSignal = nn.SoftMax():forward(inputSignal)

  gnuplot.raw('set multiplot layout 3,1')
  gnuplot.raw('set title "Input signal"')
  gnuplot.plot(torch.range(1,inputSignal[1]:numel()),inputSignal[1])

  gnuplot.raw('set title "Smoothed signal"')
  gnuplot.plot(torch.range(1, smoothedSignal[1]:numel()), smoothedSignal[1])

  gnuplot.raw('set title "SoftMax Over Smoothed signal"')
  gnuplot.plot(torch.range(1, softMaxedSignal[1]:numel()), softMaxedSignal[1])

  gnuplot.figure()

  gnuplot.raw('set title "Filter"')
  gnuplot.plot(torch.range(1, m._filter:numel()), m._filter)
end

TS.testSmoothing = function()
  --test 4 cases: (batch, non-batch) x (num_dims = 1 x num_dims = 2)

  -- parameters
  local precision = 1e-5
  local jac = nn.Jacobian
  ------------------------------------------------------------------------
  -- batch, num_dims =1

  -- define inputs and module
  local input = torch.Tensor(3,10):fill(2)
  local m = nn.TemporalSmoothing(torch.Tensor{0.5, 0.5},1,false,1)

  --test backprop, with Jacobian
  local err = jac.testJacobian(m,input)
  print('==> error: ' .. err)
  if err<precision then
	print('==> module OK')
  else
	print('==> error too large, incorrect implementation')
  end

  ------------------------------------------------------------------------
  -- no batch, num_dims =1

  -- define inputs and module
  local input = torch.Tensor(10):fill(2)
  local m = nn.TemporalSmoothing(torch.Tensor{0.5, 0.5},1,false,1)

  --test backprop, with Jacobian
  local err = jac.testJacobian(m,input)
  print('==> error: ' .. err)
  if err<precision then
	print('==> module OK')
  else
	print('==> error too large, incorrect implementation')
  end

  ------------------------------------------------------------------------
  -- batch, num_dims = 2

  --try with numLayersToSmooth = 3
  -- define inputs and module
  local input = torch.Tensor(3,10,5):fill(2)
  local m = nn.TemporalSmoothing(torch.Tensor{0.5, 0.5},1,false,2)

  -- test backprop, with Jacobian
  local err = jac.testJacobian(m,input)
  print('==> error: ' .. err)
  if err<precision then
    print('==> module OK')
  else
    print('==> error too large, incorrect implementation')
  end

  ------------------------------------------------------------------------
  -- no batch, num_dims = 2

  --try with numLayersToSmooth = 3, no batch
  -- define inputs and module
  local input = torch.Tensor(10,5):fill(2)
  local m = nn.TemporalSmoothing(torch.Tensor{0.5, 0.5},1,false,2)

  -- test backprop, with Jacobian
  local err = jac.testJacobian(m,input)
  print('==> error: ' .. err)
  if err<precision then
    print('==> module OK')
  else
    print('==> error too large, incorrect implementation')
  end

end
