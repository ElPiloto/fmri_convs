dotrc = {}
require 'os'
require 'paths'

--currently this is targeted at dredd,
--but for some reason luajit doesn't load
--$HOSTNAME
local export_folder_name = 'sleep_eeg_v2'
dotrc.has_debugger = true --most of my computers have fb.debugger installed
if os.getenv('USER') == 'elpiloto' then
	dotrc.save_dir = paths.concat(paths.cwd(),'output/')
elseif os.getenv('USER') == 'lpiloto' then
  if os.getenv('HOME') == '/Users/lpiloto' then
    dotrc.has_debugger = false
    dotrc.save_dir = '/Users/lpiloto/Dropbox/code/torch/sleep_eeg_v2/output/'
  elseif os.getenv('HOME') == '/usr/people/lpiloto' then --spock
    dotrc.has_debugger = false
	dotrc.save_dir = paths.concat(paths.cwd(),'output/')
  else -- we're on della 
	  dotrc.save_dir = '/tigress/lpiloto/outputs/'
  end
end
--now we make something specific for this project
dotrc.save_dir = paths.concat(dotrc.save_dir, export_folder_name)
if not paths.dirp(dotrc.save_dir) then
	paths.mkdir(dotrc.save_dir)
	assert(paths.dirp(dotrc.save_dir),'Tried to make directory ' .. dotrc.save_dir .. ' but failed.')
end

--convenience function
dotrc.prependSaveDir = function(fileOrDirName)
	return (paths.concat(dotrc.save_dir, fileOrDirName))
end




