local ParamSweep = {}
local pretty = require 'pl.pretty'
local tablex = require 'pl.tablex'

local function grid_sample(spec, id)
  local vals = spec.vals
  local id = torch.random(1,#vals)
  if vals[id] == nil then
    error('could not sample new value')
    --log.raiseError('could not sample new value')
  end
  return vals[id]
end

local function get_sample(spec,id)
  local options = {}
  -- use id+1,because it can be 0 based
  if sampler[spec.dist] == nil then
    error('Unknown distribution asked from sampler ' .. spec.dist)
  end
  local val = sampler[spec.dist](spec,id+1)
  if spec.type == nil then
    return val
  elseif spec.type == 'int' then
    return math.floor(val)
  else
    error('unknown spec.type : ' .. spec.type)
  end
end

local function get_options(sampling_options,id)
  local options = {}
  for param,spec in pairs(sampling_options) do
    if options[param] then
      error(param .. ' is specified more than once')
      error('Error parsing sampling_options')
    end
    options[param] = get_sample(spec,id)
  end
  return options
end

local function sample_job(opt, sampling_options)
  -- save the random number generator state so that parameter
  -- initialization is always same regardless of parameter sampling
  local state = torch.getRNGState()
  if not opt.id or opt.id < 0 or not sampling_options then
    --log.warning('Sampling will not continue')
    return opt
  end
  --log.info('Running random sampler for parameters for id=' .. opt.id)
  opt._originals = {}
  for name,value in pairs(get_options(sampling_options,opt.id)) do
    --log.info('Sampled : ' .. name .. ' = ' .. value)
    opt._originals[name] = opt[name]
    opt[name] = value
  end
  -- put the random number to its previous state
  torch.setRNGState(state)
  return opt
end

local function sortKeys(t)
  local keys = {}
  local len = 0
  for k,_ in pairs(t) do
    assert(type(k) == "string")
    len = len + 1
    keys[len] = k
  end
  --this makes it so that the "first" key in 
  --alphabetic order gets added last, which 
  --means we iterate through that dimension first
  local reverseStringSort = function(a,b)
    return a > b
  end
  table.sort(keys, reverseStringSort)
  return keys
end

local gridmod = {}

function gridmod.emptyParams()
  return {{}}
end

function gridmod.combine(params, name, values)
    params = params or gridmod.emptyParams()
    local result = {}
    for _, namevalues in ipairs(params) do
        for _, value in ipairs(values) do
            local p = tablex.copy(namevalues)
            p[name] = value
            result[#result + 1] = p
        end
    end
    return result
end

function gridmod.combineAll(key_to_values)
  local params = gridmod.emptyParams()
  local sorted_keys = sortKeys(key_to_values)
  for _, key in ipairs(sorted_keys) do
    local values = key_to_values[key]
    assert(values ~= nil, "values cannot be nil")
    assert(type(values) == "table", "the values of " .. key ..
           " are not ordered in a table: '" .. pretty.write(values) ..
           "'")
    assert(#values > 0, "values cannot be empty")
    params = gridmod.combine(params, key, values)
  end
  return params
end

local function grid_job(opt, grid_options)
  if not opt.id or opt.id < 0 or not grid_options then
    --log.raiseError('Invalid id ' .. opt.id .. '. Grid search cannot continue.')
    return opt
  end

  --log.info('Running grid search for parameters for id=' .. opt.id)
  local grid = gridmod.combineAll(grid_options)
  if opt.id > #grid then
    --log.raiseError('Invalid task id ' .. opt.id .. ' for grid of size ' .. #grid)
    return opt
  end

  -- Backup original parameter values.
  opt._originals = {}
  for name,value in pairs(grid[opt.id + 1]) do
    if type(value) ~= 'table' and type(value) ~= 'function' and type(value) ~= 'boolean' then
      --log.info('Selected : ' .. name .. ' = ' .. value)
      print('Selected : ' .. name .. ' = ' .. value)
    end
    opt._originals[name] = opt[name]
    opt[name] = value
  end
  return opt
end


ParamSweep.sample = sample_job
ParamSweep.grid = grid_job

return ParamSweep
