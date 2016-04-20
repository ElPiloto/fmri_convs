
local DF = torch.class('DataFrame')

--[[ TODO:
-- split union and inter into their own functions
-- get list of keys into it's own function
-- add info() method
-- expand support from table of tables to table of tensors or some combo of both
-- head()
-- tail()
-- returnKeys in query should automatically unpack if only a single key is specified
-- 
-- DONE get list of unique values for a key
-- DONE expose actual keys
-- DONE __tostr
-- ]]

function DF:__init(tidy)
	--fake args here:
	local verbose = false

   -- verbose print
   local function vprint(...) if verbose then print('<csv>',...) end end

	local start = 1
	local numEntries = nil
	self.tidy = tidy

	--in-depth argument checking, build list of keys, and get number of entries
	for k,v in pairs(self.tidy) do
		assert(type(k) == 'string', 'Invalid argument: must be table with string'
				.. ' keys, but we have encountered key of type: ' .. type(k))
		assert(type(v) == 'table', 'Invalid argument: must be table of list-like' 
				.. ' tables, but value at key=' .. k .. ' has type: ' .. type(v))
		if numEntries == nil then --let's count
			numEntries = #v
			if numEntries  == 0 then
				error('Value at key=' .. k .. ' is empty or is not an array-like'
					.. 'table')
			end
		else
			assert(#v == numEntries, 'Error: Value at key=' .. k .. ' does not'
				.. ' have the same number of entries as previous values')
		end

		--table.insert(i2key,k)
	end

	self.numEntries = numEntries

	-- query mode: build reverse index
	vprint('generating reversed index for fast queries')
	self.revidx = {}
	for var,vals in pairs(self.tidy) do
		self.revidx[var] = {}
		for i,val in ipairs(vals) do
			self.revidx[var][val] = self.revidx[var][val] or {}
			table.insert(self.revidx[var][val], i)
		end
	end

end

--return the unique values for a particular key
function DF:uniqueValues(...)
	local args, key = dok.unpack(
		{...},
		'uniqueValues',
		'Get unique values for a given key',
		{arg='key', type='string', help='the key you want to get unique values for', req=true}
	)
	assert(self.revidx[key], "key='" .. key .. " not found inside of dataframe")
	local uniqueVals = {}
	for k,v in pairs(self.revidx[key]) do
		table.insert(uniqueVals, k)
	end
	return uniqueVals
end

function DF:keys()
	local keys = {}
	for k,v in pairs(self.tidy) do
		table.insert( keys,k )
	end
	return keys
end

function DF:__tostring()
	local outStr = 'DataFrame with: ' .. self.numEntries .. ' rows and keys: \n' 
	for k,v in pairs(self.tidy) do
		outStr = outStr .. '\t' .. k .. '\n'
	end
	return outStr
end

function DF.test()
	local fake_data = {id = {'01','02', '03'}, 
		class = {1,1,2}, likes_stuff = {true, false, true}}
	df = DataFrame(fake_data)
	df:query('all')
end

-- create a function/closure that can be used to query
-- the table
function DF:query(...)
	-- usage
	local args, query, varvals, returnKeys = dok.unpack(
	{...},
	'query',
	'This closure was automatically generated to query your data.\n'
	.. 'Example of query: query(\'union\', {var1={1}, var2={2,3,4}})\n'
	.. 'this query will return a subset of the original data, where var1 = 1 OR var2 = 2 or 3 or 4 \n'
	.. '\n'
	.. 'Other example of query: query(\'inter\', {var1={1}, var2={2,3,4}})\n'
	.. 'this query will return a subset of the original data, where var1 = 1 AND var2 = 2 or 3 or 4 \n'
	.. '\n'
	.. 'Other example of query: query(\'vars\')\n'
	.. 'this will return a list of the variable names\n'
	.. '\n'
	.. 'Other example of query: query() or query(\'all\')\n'
	.. 'this query will return the complete dataset'
	,
	{arg='query',  type='string', help='query: all | help | vars | inter | union', default='all'},
	{arg='vars', type='table',  help='list of vars/vals'},
	{arg='returnKeys', type = 'table', help='list of vars to return',req =false}
	)
	if query == 'help' then
		-- help
		print(args.usage)
		return

	elseif query == 'vars' then
		-- return vars
		local vars = {}
		for k in pairs(self.tidy) do
			table.insert(vars,k)
		end
		return vars

	elseif query == 'all' then
		-- query all: return the whole thing
		return self.tidy

	else
		-- query has this form:
		-- { var1 = {'value1', 'value2'}, var2 = {'value1'} }
		-- OR
		-- { var1 = 'value1', var2 = 'value2'}
		-- convert second form into first one:
		for var,vals in pairs(varvals) do
			if type(vals) ~= 'table' then
				varvals[var] = {vals}
			end
		end
		-- find all indices that are ok
		local indices = {}
		if query == 'union' then
			for var,vals in pairs(varvals) do
				for _,val in ipairs(vals) do
					local found = self.revidx[var][val]
					if found ~= nil then
						for _,idx in ipairs(found) do
							table.insert(indices, idx)
						end
					end
				end
			end
		else -- 'inter'
			local revindices = {}
			local nvars = 0
			for var,vals in pairs(varvals) do
				for _,val in ipairs(vals) do
					local found = self.revidx[var][val]
					if found ~= nil then
						for _,idx in ipairs(found) do
							revindices[idx] = (revindices[idx] or 0) + 1
						end
					end
				end
				nvars = nvars + 1
			end
			for var,vals in pairs(varvals) do
				for _,val in ipairs(vals) do
					local found = self.revidx[var][val]
					for _,idx in ipairs(found) do
						if revindices[idx] == nvars then
							table.insert(indices, idx)
						end
					end
				end
			end
		end
		table.sort(indices, function(a,b) return a<b end)
		-- generate filtered table
		local filtered = {}
		if not returnKeys then --default return all keys
			for k in pairs(self.tidy) do
				filtered[k] = {}
			end
		else --only return keys specified in returnKeys
			for k,v in pairs(returnKeys) do
				assert(self.tidy[v], "Error: 'returnKeys' contains a key=" .. v .. " not in this dataframe")
				filtered[v] = {}
			end
		end
		for idx,i in ipairs(indices) do
			if i ~= indices[idx-1] then -- check for doubles
				for k in pairs(filtered) do
					table.insert(filtered[k], self.tidy[k][i])
				end
			end
		end
		-- return filtered table
		return filtered
	end
end


