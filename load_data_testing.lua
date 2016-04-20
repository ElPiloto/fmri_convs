sleep_eeg = require 'sleep_eeg'

filename = './torch_exports/sleep_ERP_cuelocked_all_4ms.mat'

subjData = sleep_eeg.CVBySubjData(filename)

